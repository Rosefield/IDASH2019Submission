
#pragma once
#include "NeuralNetwork.h"
using namespace std;



NeuralNetwork::NeuralNetwork(NeuralNetConfig& config)
: inputSize(config.inputSize), outputSize(config.outputSize), batchSize(config.batchSize),
  inputData(config.inputSize*config.batchSize),
  outputData(config.outputSize*config.batchSize),
  numIterations(config.numIterations)
{
	for (auto& layer : config.layerConf)
	{
		if (layer->getType().compare("FC") == 0) {
      auto l = static_pointer_cast<FCConfig>(layer);
			layers.push_back(make_unique<FCLayer>(*l));
      if(SA_VALIDATE) {
        layersSA.push_back(make_unique<FCLayer>(*l));
      }
    }
		else {
			error("Only FC layer types currently supported");
    }
	}
}


NeuralNetwork::~NeuralNetwork()
{
	layers.clear();
}



void NeuralNetwork::forward()
{
	log_print("NN.forward");

  compareToSA("forward 0", [&](auto& ls) { ls[0]->forward(inputData); }, [&](auto& ls) { return *ls[0]->getActivation(); });

  for (size_t i = 1; i < layers.size(); ++i) {
    compareToSA(string("forward ") + std::to_string(i), [=](auto& ls) { ls[i]->forward(*(ls[i-1]->getActivation())); }, [=](auto& ls) { return *ls[i]->getActivation(); });
  }
}

void NeuralNetwork::backward()
{
	log_print("NN.backward");

	computeDelta();
	// cout << "computeDelta done" << endl;
	updateEquations();
}

void NeuralNetwork::computeDelta()
{
	log_print("NN.computeDelta");

  compareToSA("delta back", [&](auto& ls) { ls.back()->computeOutputDelta(outputData); }, [&](auto& ls) { return *ls.back()->getDelta(); });

	for (size_t i = layers.size() - 1; i > 0; --i) {
    compareToSA(string("delta ") + std::to_string(i), [=](auto& ls) { ls[i]->computeDelta(*(ls[i-1]->getDelta())); }, [=](auto& ls) { return *ls[i-1]->getDelta(); });
  }

  compareToSA("delta 0", [=](auto& ls) { ls[0]->finishFirstDelta(); }, [=](auto& ls) { return *ls[0]->getDelta(); });
}

void NeuralNetwork::updateEquations()
{
	log_print("NN.updateEquations");

	for (size_t i = layers.size() - 1; i > 0; --i) {
    compareToSA(string("weights ") + std::to_string(i), [=](auto& ls) { ls[i]->updateEquations(*(ls[i-1]->getActivation())); }, [=](auto& ls) { return *ls[i]->getWeights(); });
  }

  compareToSA(string("weights 0"), [&](auto& ls) { ls[0]->updateEquations(inputData); }, [=](auto& ls) { return *ls[0]->getWeights(); });
}

void NeuralNetwork::predict(vector<myType> &maxIndex)
{
	log_print("NN.predict");

	size_t rows = batchSize;
	size_t columns = outputSize;
	vector<myType> max(rows);

  layers.back()->predict(*(layers.back()->getActivation()), max, maxIndex, rows, columns);
}

void NeuralNetwork::getAccuracy(const vector<myType> &maxIndex, vector<size_t> &counter)
{
	log_print("NN.getAccuracy");

	size_t rows = batchSize;
	size_t columns = outputSize;
	vector<myType> max(rows), groundTruth(rows, 0);

	layers.back()->predict(outputData, max, groundTruth, rows, columns);

	if (STANDALONE)
	{
		for (size_t i = 0; i < batchSize; ++i)
		{
			counter[1]++;
      /*
      cout << "Predicted index, expected index, val " << maxIndex[i] << " " 
           << groundTruth[i] << " " << max[i] << endl;;
      */
			if (maxIndex[i] == groundTruth[i])
				counter[0]++;
		}
	}
	else
	{
		//Reconstruct things
		vector<myType> temp_maxIndex(rows), temp_groundTruth(rows);
		if (partyNum == PARTY_B)
			sendTwoVectors<myType>(maxIndex, groundTruth, PARTY_A, rows, rows);

		if (partyNum == PARTY_A)
		{
			receiveTwoVectors<myType>(temp_maxIndex, temp_groundTruth, PARTY_B, rows, rows);
			addVectors<myType>(temp_maxIndex, maxIndex, temp_maxIndex, rows);
			dividePlainSA(temp_maxIndex, (1 << FLOAT_PRECISION));
			addVectors<myType>(temp_groundTruth, groundTruth, temp_groundTruth, rows);	
			dividePlainSA(temp_groundTruth, (1 << FLOAT_PRECISION));
		}

		for (size_t i = 0; i < batchSize; ++i)
		{
			counter[1]++;
      if(partyNum == PARTY_A) {
        cout << "Predicted index, expected index " << temp_maxIndex[i] << " " 
             << temp_groundTruth[i] << endl;
      }
			if (temp_maxIndex[i] == temp_groundTruth[i])
				counter[0]++;
		}		
	}

	cout << "Rolling accuracy: " << counter[0] << " out of " 
		 << counter[1] << " (" << (counter[0]*100/counter[1]) << " %)" << endl;
}

size_t NeuralNetwork::getSize()
{
	size_t s = 0;
	for (size_t i = 0; i < layers.size(); ++i)
	{
		s += layers[i]->getSize();
	}

	return s;
}


void NeuralNetwork::outputNetwork(string fileName)
{

  std::ofstream out;

  if(fileName.size() > 0) {
    out = ofstream(fileName);
  } else {
    out = ofstream("/dev/null");
  }
  out << "[" << endl;

	for (size_t i = 0; i < layers.size(); ++i)
	{	
    layers[i]->outputParams(out);

    if(i+1 != layers.size()) {
      out << "," << endl;
    }
	}
  out << endl <<  "]";
}
 
