
#pragma once
#include "FCLayer.h"
#include <cmath>
#include <thread>
using namespace std;

FCLayer::FCLayer(FCConfig& conf)
:conf(conf.batchSize, conf.inputDim, conf.outputDim, conf.activation),
 activations(conf.batchSize * conf.outputDim), 
 zetas(conf.batchSize * conf.outputDim), 
 deltas(conf.batchSize * conf.outputDim),
 weights(conf.inputDim * conf.outputDim),
 accums_weights(conf.inputDim * conf.outputDim),
 biases(conf.outputDim),
 accums_biases(conf.outputDim),
 activationPrimeSmall(conf.batchSize * conf.outputDim),
 activationPrimeLarge(conf.batchSize * conf.outputDim)
{
	initialize();
}


void FCLayer::initialize()
{
	//Initialize weights and biases here.
	//Ensure that initialization is correctly done.
	size_t range = 30;
	size_t decimation = conf.inputDim;
	size_t size = weights.size();

	vector<myType> temp(size);
	for (size_t i = 0; i < size; ++i) {
    double val = 0;
    if(SA_VALIDATE) {
      val = ((float)(rand() % range) - range/2)/decimation;
      //val = 1./decimation; 
    } else {
      val = ((float)(rand() % range) - range/2)/decimation;
    }
		temp[i] = floatToMyType(val); 
  }

	if (partyNum == PARTY_S)
		for (size_t i = 0; i < size; ++i)
			weights[i] = temp[i];
	else if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			weights[i] = temp[i];
	else if (partyNum == PARTY_B or partyNum == PARTY_C)		
		for (size_t i = 0; i < size; ++i)
			weights[i] = 0;
		
	
	fill(biases.begin(), biases.end(), 0);
  fill(accums_weights.begin(),accums_weights.end(),0);
	fill(accums_biases.begin(), accums_biases.end(),0);
}



void FCLayer::forward(const vector<myType> &inputActivation)
{
	log_print("FC.forward");

	if (STANDALONE)
		forwardSA(inputActivation);
	else
		forwardMPC(inputActivation);
}

void FCLayer::computeOutputDelta(vector<myType>& outputData) {
  size_t rows = conf.batchSize;
  size_t columns = conf.outputDim;
  if(conf.activation == ActivationFunctions::ReLU) {
    size_t size = rows*columns;

    vector<myType> rowSum(size, 0);
    vector<myType> quotient(size, 0);

    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < columns; ++j) {
        rowSum[i*columns] += activations[i * columns + j];
      }
      for (size_t j = 1; j < columns; ++j) {
        rowSum[i*columns + j] = rowSum[i*columns];
      }
    }

    if (STANDALONE)
    {
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j)
        {
          auto index = i * columns + j;
          if (rowSum[index] != 0)
            quotient[index] = divideMyTypeSA(activations[index], rowSum[index]);
        }
      }
    }
    else
    {
      funcDivisionMPC(activations, rowSum, quotient, size);
    }

    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < columns; ++j)
      {
        auto index = i * columns + j;
        deltas[index] = quotient[index] - outputData[index];
      }
    }
  } else if (conf.activation == ActivationFunctions::Logistic || conf.activation == ActivationFunctions::Logistic4) {
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j)
        {
          auto index = i * columns + j;
          deltas[index] = activations[index] - outputData[index];
        }
      }
  }

  if(STANDALONE) {
    auto f = [](auto& vs, auto size, auto str) {
      cout << str << ": ";
      for (size_t i = 0; i < size; ++i)
        print_linear(vs[i], DEBUG_PRINT);
      cout << endl << endl;
    };
    //f(deltas, (deltas.size() > 100 ? 100 : deltas.size()), "deltas");
  }

}

void FCLayer::finishFirstDelta() {
  if (STANDALONE) {
    transform(deltas.begin(), deltas.end(), activationPrimeSmall.begin(), deltas.begin(), [](auto d, auto a) { return d*a; });
  } else {
    // Compute d(x) * activation'(x), since activation' is either 0/1 we can
    // just use select shares
    funcSelectShares3PC(deltas, activationPrimeLarge, deltas, deltas.size());
  }
}

void FCLayer::computeDelta(vector<myType>& prevDelta)
{
	log_print("FC.computeDelta");

	if (STANDALONE)
		computeDeltaSA(prevDelta);
	else
		computeDeltaMPC(prevDelta);	
}

void FCLayer::updateEquations(const vector<myType>& prevActivations)
{
	log_print("FC.updateEquations");

	if (STANDALONE)
		updateEquationsSA(prevActivations);
	else
		updateEquationsMPC(prevActivations);
}

void FCLayer::predict(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns) {

  if(conf.activation == ActivationFunctions::ReLU || columns > 1) {
    findMax(activations, max, maxIndex, rows, columns);
  } else if (conf.activation == ActivationFunctions::Logistic || conf.activation == ActivationFunctions::Logistic4) {
    if (STANDALONE) {
      for (size_t i = 0; i < rows; ++i) {
        max[i] = a[i];

        maxIndex[i] = (a[i] >= floatToMyType(.5) ? 1 : 0);
      }
    } else {
      vector<myType> ac(a.begin(), a.begin() + rows);
      // Output is 1 if activation > .5, or if activation - .5 > 0
      for(size_t i = 0; i < rows; ++i) {
        max[i] = a[i];
        if(partyNum == PARTY_A) {
          ac[i] -= floatToMyType(.5);
        }
      }
      funcRELUPrime3PC(ac, maxIndex, rows);
      if(PRIMARY) {
        //funcReconstruct2PC(a, rows, "a");
        //funcReconstruct2PC(max, rows, "max");
        //funcReconstruct2PC(maxIndex, rows, "maxIndex");
      }
    }
  }

}

void FCLayer::findMax(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns)
{
	log_print("FC.findMax");

	if (STANDALONE)
		maxSA(a, max, maxIndex, rows, columns);
	else
		maxMPC(a, max, maxIndex, rows, columns);
}





/******************************** Standalone ********************************/
void FCLayer::forwardSA(const vector<myType> &inputActivation)
{
	//zetas = inputActivation * weights + biases
	//activations = ReLU(zetas)
	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t index;

	//Matrix Multiply
	matrixMultEigen(inputActivation, weights, zetas, 
					rows, common_dim, columns, 0, 0);
	dividePlainSA(zetas, (1 << FLOAT_PRECISION));

	//Add biases and ReLU
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
		{
			index = i*columns + j; 

			zetas[index] += biases[j];
      if(conf.activation == ActivationFunctions::ReLU) {
        activationPrimeSmall[index] = (zetas[index] < LARGEST_NEG ? 1:0);
        activations[index] = activationPrimeSmall[index]*zetas[index];
      } else if (conf.activation == ActivationFunctions::Logistic) {
        // values are encoded as twos-compliment integers in an unsigned type, so normal comparison to
       // negative values does not work as expected.
       // check that -.5 < z < .5, or that z < mytype(.5) || z > mytype(max - .5)
        
	if(zetas[index] < floatToMyType(2) || zetas[index] > floatToMyType(-2)) {
          	activationPrimeSmall[index] = 0.25;
          	activations[index] = floatToMyType(0.25* myTypeToFloat(zetas[index])) + floatToMyType(.5);
        } else {
          activationPrimeSmall[index] = 0;
          // If the value is outside of that range, check to see if it is positive or not.
          activations[index] = (zetas[index] < LARGEST_NEG ? 1 : 0);
	}
          //activationPrimeSmall[index] = 1.0/(2.0 + exp(myTypeToFloat(zetas[index]))+ exp(-myTypeToFloat(zetas[index]))); 
	  //activations[index] = floatToMyType(1.0/(1.0 + exp(-myTypeToFloat(zetas[index]))));
	  //std::cout << "Zeta:" << zetas[index] << "," << exp(myTypeToFloat(zetas[index])) <<  ", Value:" << activationPrimeSmall[index] << "," << activations[index] << endl;  

        }
        //cout << "i, activation, zeta " << index << " " << activations[index] << " " << zetas[index] << endl;
		}
  if(conf.outputDim == 1) {
    auto size = zetas.size();
    auto f = [](auto& vs, auto size, auto str) {
      cout << str << ": ";
      for (size_t i = 0; i < size; ++i)
        print_linear(vs[i], DEBUG_PRINT);
      cout << endl << endl;
    };
    //f(zetas, (size > 100 ? 100 : size), "zetas");
    //f(activationPrimeSmall, (size > 100 ? 100 : size), "activationPrime");
    //f(activations, (size > 100 ? 100 : size), "activations");
  }
}


void FCLayer::computeDeltaSA(vector<myType>& prevDelta)
{
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t common_dim = conf.outputDim;

  // delta is currently the error, calculate f'(x)(\hat{y} - y)
	for (int i = 0; i < deltas.size(); ++i)
		deltas[i] = floatToMyType(myTypeToFloat(deltas[i]) * activationPrimeSmall[i]);

	matrixMultEigen(deltas, weights, prevDelta, 
					rows, common_dim, columns, 0, 1);

	dividePlainSA(prevDelta, (1 << FLOAT_PRECISION));

  if(true || conf.outputDim == 1) {
    auto size = prevDelta.size();
    auto f = [](auto& vs, auto size, auto str) {
      cout << str << ": ";
      for (size_t i = 0; i < size; ++i)
        print_linear(vs[i], DEBUG_PRINT);
      cout << endl << endl;
    };
    //f(activationPrimeSmall, (activationPrimeSmall.size() > 100 ? 100 : activationPrimeSmall.size()), "activationPrimeSmall");
    //f(activations, (activations.size() > 100 ? 100 : activations.size()), "activations");
    //f(deltas, (deltas.size() > 100 ? 100 : deltas.size()), "deltas");
    //f(weights, (weights.size() > 100 ? 100 : weights.size()), "weights");
    //f(prevDelta, (size > 100 ? 100 : size), "prevDelta");
  }
}

// If there is momentum the update equation then becomes
// w := w - \eta((1-\alpha)\nabla Q + \alpha\delta w)
// where \alpha is the momentum
void applyMomentum(vector<myType>& weights, vector<myType>& deltas, vector<myType>& prevDeltas, size_t batchSize) {
  assert(weights.size() == deltas.size() && deltas.size() == prevDeltas.size());
  size_t log_size = static_cast<size_t>(log2(batchSize));

  if (MOMENTUM > 0) {
    // divide by batch size
    if (STANDALONE) {
      for(auto& d : deltas) {
		    d = dividePlainSA(d, batchSize);
      }
    }
    else if (PRIMARY) {
      funcTruncate2PC(deltas, log_size, deltas.size(), PARTY_A, PARTY_B);
    }

    for(auto& d: deltas) {
      d = multiplyMyTypesSA(d, floatToMyType(1-MOMENTUM), FLOAT_PRECISION);
    }

    for(auto& acc: prevDeltas) {
      acc = multiplyMyTypesSA(acc, floatToMyType(MOMENTUM), FLOAT_PRECISION);
    }
    addVectors<myType>(prevDeltas, deltas, prevDeltas, prevDeltas.size());

    auto tmp = prevDeltas;
    // multiply by learning rate
    for(auto& t: tmp) {
      t = multiplyMyTypesSA(t, LEARNING_RATE, FLOAT_PRECISION);
    }
    subtractVectors<myType>(weights, tmp, weights, weights.size());
  } else {
    // divide by batch_size*learning_rate
    if (STANDALONE) {
      for(auto& d : deltas) {
		    d = dividePlainSA(multiplyMyTypesSA(d, LEARNING_RATE, FLOAT_PRECISION), batchSize);
      }
    }
    else if (PRIMARY) {
      funcTruncate2PC(deltas, log_size + LOG_LEARNING_RATE, deltas.size(), PARTY_A, PARTY_B);
    }
    subtractVectors<myType>(weights, deltas, weights, weights.size());
  }
}

void FCLayer::updateEquationsSA(const vector<myType>& prevActivations)
{
	//Update Bias
	myType sum;
  vector<myType> bias_updates(conf.outputDim);

	for (size_t i = 0; i < conf.outputDim; ++i)
	{
		bias_updates[i] = 0;
		for (size_t j = 0; j < conf.batchSize; ++j)	{
			bias_updates[i] += deltas[j * conf.outputDim + i];
    }
  }

  applyMomentum(biases, bias_updates, accums_biases, conf.batchSize);

	//Update Weights
	size_t rows = conf.inputDim;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.batchSize;
	vector<myType> deltaWeight(rows * columns);

	matrixMultEigen(prevActivations, deltas, deltaWeight, 
					rows, common_dim, columns, 1, 0);
	dividePlainSA(deltaWeight, (1 << FLOAT_PRECISION));

  applyMomentum(weights, deltaWeight, accums_weights, conf.batchSize);
  if(true || conf.outputDim == 1) {
    auto size = rows*columns;
    auto f = [](auto& vs, auto size, auto str) {
      cout << str << ": ";
      for (size_t i = 0; i < size; ++i)
        print_linear(vs[i], DEBUG_PRINT);
      cout << endl << endl;
    };
    //cout << "rows, common, columns " << rows << " " << common_dim << " " << columns << endl;
    //f(deltas, (deltas.size() > 100 ? 100 : deltas.size()), "SA deltas");
    //f(prevActivations, (prevActivations.size() > 100 ? 100 : prevActivations.size()), "SA prevActivation");
    //f(deltaWeight, (size > 100 ? 100 : size), "SA deltaWeight");
    //f(weights, (size > 100 ? 100 : size), "SA weights");
    //f(accums_weights, (size > 100 ? 100 : size), "SA accums_weights");
    //f(biases, conf.outputDim, "SA biases");
    //f(accums_biases, conf.outputDim, "SA accums_biases");
  }
}


//Chunk wise maximum of a vector of size rows*columns and max is caclulated of every 
//column number of elements. max is a vector of size rows. maxIndex contains the index of 
//the maximum value.
void FCLayer::maxSA(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns)
{
	size_t size = rows*columns;
	vector<myType> diff(size);

	for (size_t i = 0; i < rows; ++i)
	{
		max[i] = a[i*columns];
		maxIndex[i] = 0;
	}

	for (size_t i = 1; i < columns; ++i)
		for (size_t j = 0; j < rows; ++j)
		{
			if (a[j*columns + i] > max[j])
			{
				max[j] = a[j*columns + i];
				maxIndex[j] = i;
			}
		}
}



/******************************** MPC ********************************/
void FCLayer::forwardMPC(const vector<myType> &inputActivation)
{
	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows*columns;

	//Matrix Multiplication
	funcMatMulMPC(inputActivation, weights, zetas, 
				  rows, common_dim, columns, 
				  0, 0);

	//Add Biases
	if (PRIMARY)
		for(size_t r = 0; r < rows; ++r)
			for(size_t c = 0; c < columns; ++c)
				zetas[r*columns + c] += biases[c];

  if(conf.activation == ActivationFunctions::ReLU) {
      funcRELUPrime3PC(zetas, activationPrimeLarge, size);
      funcSelectShares3PC(zetas, activationPrimeLarge, activations, size);
  } else if (conf.activation == ActivationFunctions::Logistic) {
    // We approximate logistic regression as a piecewise linear function
    // { 0        x <= -.5
    // { x + .5   -.5 < x < .5
    // { 1        .5 <= x
    // Using ReLU' as a gadget, this can be calculated as
    // ReLU'([x] + .5)(([x] + .5)(ReLU'([x] - .5))) + ReLU'([x] -.5)
    // using a = ReLU'([x] + .5) and b = ReLU'([x] - .5)
    // The derivative can be calculated as
    // Logistic' = a*(1-b)
    vector<myType> zm5 = zetas;
    vector<myType> zp5 = zetas;

    if(partyNum == PARTY_A) {
      for(size_t i = 0; i < size; ++i) {
        zm5[i] -= floatToMyType(.5);
        zp5[i] += floatToMyType(.5);
      }
    }

    vector<myType> relus(2*size);
    vector<myType> zs;
    zs.reserve(2*size);
    zs.insert(zs.end(), zp5.begin(), zp5.end());
    zs.insert(zs.end(), zm5.begin(), zm5.end());
    // do all of the relus in one batch
    funcRELUPrime3PC(zs, relus, 2*size);
    vector<myType> a(relus.begin(), relus.begin() + size);
    vector<myType> b(relus.begin() + size, relus.end());

    // calculate a*(1 - b)
    vector<myType> bm(size);
    for(size_t i = 0; i < size; ++i) {
      bm[i] =  -b[i];
      if(partyNum == PARTY_A) {
        bm[i] += floatToMyType(1);
      }
    }

    funcDotProductMPC(a, bm, activationPrimeLarge, size);
    funcDotProductMPC(zp5, activationPrimeLarge, activations, size);

    addVectors<myType>(b, activations, activations, size);
  } else if (conf.activation == ActivationFunctions::Logistic4) {
    // We approximate logistic regression as a piecewise linear function
    // { 0        x <= -2
    // { .25x + .5   -2 < x < 2
    // { 1        2 <= x
    // Using ReLU' as a gadget, this can be calculated as
    // ReLU'([x] + 2)((.25[x] + .5)(ReLU'([x] - 2))) + ReLU'([x] -2)
    // using a = ReLU'([x] + 2) and b = ReLU'([x] - 2)
    // The derivative can be calculated as
    // Logistic' = a*(1-b)*.25
    //
    vector<myType> zm2 = zetas;
    vector<myType> zp2 = zetas;

    if(partyNum == PARTY_A) {
      for(size_t i = 0; i < size; ++i) {
        zm2[i] -= floatToMyType(2);
        zp2[i] += floatToMyType(2);
      }
    }

    vector<myType> relus(2*size);
    vector<myType> zs;
    zs.reserve(2*size);
    zs.insert(zs.end(), zp2.begin(), zp2.end());
    zs.insert(zs.end(), zm2.begin(), zm2.end());
    // do all of the relus in one batch
    funcRELUPrime3PC(zs, relus, 2*size);
    vector<myType> a(relus.begin(), relus.begin() + size);
    vector<myType> b(relus.begin() + size, relus.end());

    // calculate a*(1 - b)
    vector<myType> bm(size);
    for(size_t i = 0; i < size; ++i) {
      bm[i] =  -b[i];
      if(partyNum == PARTY_A) {
        bm[i] += floatToMyType(1);
      }
    }

    // multipy by .25
    if(PRIMARY) {
      funcTruncate2PC(bm, 2, bm.size(), PARTY_A, PARTY_B);
    }

    funcDotProductMPC(a, bm, activationPrimeLarge, size);
    funcDotProductMPC(zp2, activationPrimeLarge, activations, size);

    addVectors<myType>(b, activations, activations, size);
  }
}


void FCLayer::computeDeltaMPC(vector<myType>& prevDelta)
{
	//Back Propagate
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t common_dim = conf.outputDim;
	size_t size = rows*columns;

  // Compute d(x) * activation'(x), since activation' is either 0/1 we can
  // just use select shares
  funcSelectShares3PC(deltas, activationPrimeLarge, deltas, deltas.size());

	funcMatMulMPC(deltas, weights, prevDelta, 
				  rows, common_dim, columns, 0, 1);

  if(PRIMARY && conf.outputDim == 1) {
    //funcReconstruct2PC(prevDelta, (size > 100 ? 100 : size), "prevDelta");
  }
}

void FCLayer::updateEquationsMPC(const vector<myType>& prevActivations)
{
  /*
  if (PRIMARY && conf.outputDim == 1) {
    auto size = conf.inputDim*conf.outputDim;
    funcReconstruct2PC(weights, (size > 100 ? 100 : size), "weights");
    funcReconstruct2PC(accums_weights, (size > 100 ? 100 : size), "accums_weights");
    funcReconstruct2PC(biases, conf.outputDim, "biases");
    funcReconstruct2PC(accums_biases, conf.outputDim, "accums_biases");
  }
  */
	size_t rows = conf.batchSize;
  size_t log_size = static_cast<size_t>(log2(rows));
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows*columns;	
	vector<myType> bias_updates(columns, 0);

	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < columns; ++j) {
			bias_updates[j] += deltas[i*columns + j];
    }
  }

  applyMomentum(biases, bias_updates, accums_biases, conf.batchSize);

	//Update Weights 
	rows = conf.inputDim;
	columns = conf.outputDim;
	common_dim = conf.batchSize;
  size = rows*columns;

	vector<myType> deltaWeight(size);

	funcMatMulMPC(prevActivations, deltas, deltaWeight, 
   				  rows, common_dim, columns, 1, 0);

  applyMomentum(weights, deltaWeight, accums_weights, conf.batchSize);

  if (PRIMARY && conf.outputDim == 1) {
    //cout << "rows, common, columns " << rows << " " << common_dim << " " << columns << endl;
    //funcReconstruct2PC(deltas, (deltas.size() > 100 ? 100 : deltas.size()), "MPC delta");
    //funcReconstruct2PC(prevActivations, (prevActivations.size() > 100 ? 100 : prevActivations.size()), "MPC prevActivation");
    //funcReconstruct2PC(deltaWeight, (size > 100 ? 100 : size), "MPC deltaWeight");
    //funcReconstruct2PC(weights, (size > 100 ? 100 : size), "MPC weights");
    //funcReconstruct2PC(accums_weights, (size > 100 ? 100 : size), "MPC accums_weights");
    //funcReconstruct2PC(biases, conf.outputDim, "MPC biases");
    //funcReconstruct2PC(accums_biases, conf.outputDim, "MPC accums_biases");
  }
}


void FCLayer::maxMPC(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns)
{
	funcMaxMPC(a, max, maxIndex, rows, columns);
}

vector<myType>* FCLayer::getParam(vector<myType>& param)
{
	param.insert(param.begin(),weights.begin(),weights.end());
	param.insert(param.end(),biases.begin(),biases.end());
	//cout << "FCLayer.cpp: First parameter" <<param[0];
	return &param;
}

void FCLayer::outputParams(ofstream& out) {
  if (STANDALONE) {
    outputTensor(out, weights, {conf.inputDim, conf.outputDim});
    out << ", " << endl;
    outputTensor(out, biases, {conf.outputDim});
  } else {
    if (partyNum == PARTY_B) {
      sendTwoVectors<myType>(weights, biases, PARTY_A, weights.size(), biases.size());
    }
    if (partyNum == PARTY_A) {
      vector<myType> t_weights(weights.size());
      vector<myType> t_biases(biases.size());

      receiveTwoVectors(t_weights, t_biases, PARTY_B, weights.size(), biases.size());

      addVectors(weights, t_weights, t_weights, weights.size());
      addVectors(biases, t_biases, t_biases, biases.size());
      outputTensor(out, t_weights, {conf.inputDim, conf.outputDim});
      out << ", " << endl;
      outputTensor(out, t_biases, {conf.outputDim});
    }
  }

}
