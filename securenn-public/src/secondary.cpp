
#include "secondary.h"
#include <iostream>
using namespace std;


//this player number
int partyNum;
//aes_key of the party
char *party_aes_key;


//For faster DGK computation
smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];


//communication
extern string * addrs;
extern BmrNet ** communicationSenders;
extern BmrNet ** communicationReceivers;
extern int base_port;


vector<vector<myType>> trainData;
vector<vector<myType>> trainLabels;
size_t trainBatchCounter = 0;

NeuralNetConfig createSimpleNN(size_t trainSize, size_t numEpochs, size_t inputWidth, size_t outputWidth) {
  size_t batchSize = 32;
  // Do 20 epochs of training
  size_t numIterations = numEpochs * ( (float)trainSize / batchSize );
  std::cout << "Creating simpleNN model with batchSize, numIterations, inputSize: " << batchSize << " " << numIterations << " " << inputWidth <<  std::endl;

  
  NeuralNetConfig config(numIterations, batchSize, inputWidth, outputWidth, "SimpleNN");

  config.addLayer(FCConfig(batchSize, inputWidth, 200, ActivationFunctions::ReLU));
  config.addLayer(FCConfig(batchSize, 200, 100, ActivationFunctions::ReLU));
  config.addLayer(FCConfig(batchSize, 100, outputWidth, ActivationFunctions::Logistic));

  return config;
}

NeuralNetConfig createSimpleNN100(size_t trainSize, size_t numEpochs, size_t inputWidth, size_t outputWidth) {
  size_t batchSize = 32;
  // Do numEpochs epochs of training
  size_t numIterations = numEpochs * ( (float)trainSize / batchSize );
  std::cout << "Creating simpleNN100 model with batchSize, numIterations, inputSize: " << batchSize << " " << numIterations << " " << inputWidth <<  std::endl;

  
  NeuralNetConfig config(numIterations, batchSize, inputWidth, outputWidth, "SimpleNN");

  config.addLayer(FCConfig(batchSize, inputWidth, 100, ActivationFunctions::ReLU));
  config.addLayer(FCConfig(batchSize, 100, 100, ActivationFunctions::ReLU));
  config.addLayer(FCConfig(batchSize, 100, outputWidth, ActivationFunctions::Logistic));

  return config;
}

NeuralNetConfig createSimplestNN(size_t trainSize, size_t numEpochs, size_t inputWidth, size_t outputWidth) {
  size_t batchSize = 32;
  // Do numEpochs epochs of training
  size_t numIterations = numEpochs * ( (float)trainSize / batchSize );
  std::cout << "Creating simplestNN model with batchSize, numIterations, inputSize: " << batchSize << " " << numIterations << " " << inputWidth <<  std::endl;

  NeuralNetConfig config(numIterations, batchSize, inputWidth, outputWidth, "SimpleNN");

  config.addLayer(FCConfig(batchSize, inputWidth, 100, ActivationFunctions::ReLU));
  config.addLayer(FCConfig(batchSize, 100, outputWidth, ActivationFunctions::Logistic4));

  return config;
}

NeuralNetConfig createConfig(std::string name, size_t numEpochs, size_t trainSize, size_t inputWidth, size_t outputWidth) {
  if(name == "SimpleNN") {
    return createSimpleNN(trainSize, numEpochs, inputWidth, outputWidth);
  } else if (name == "SimpleNN100") {
    return createSimpleNN100(trainSize, numEpochs, inputWidth, outputWidth);
  } else if (name == "SimplestNN") {
    return createSimplestNN(trainSize, numEpochs, inputWidth, outputWidth);
  }

  assert(false && "Only SimpleNN implemented");
}

NeuralNetConfig parseInputs(int argc, char* argv[])
{	
	//If this fails, change functions in tools (divide and multiply ones)
	assert((sizeof(double) == sizeof(myType)) && "sizeof(double) != sizeof(myType)");

	if (argc < 10) 
		print_usage(argv[0]);

	loadData(argv[8], argv[9]);
  assert(trainData.size() == trainLabels.size() && trainData.size() > 0);

  string networkName = argv[1];
  auto numEpochs = atoi(argv[2]);
  auto config = createConfig(networkName, numEpochs, trainData.size(), trainData[0].size(), trainLabels[0].size());
  if(argc >= 11) {
    config.outputFile = argv[10];
  }
  if(argc >= 12) {
    base_port = atoi(argv[11]);
  }

	if (strcmp(argv[3], "STANDALONE") == 0) {
		NUM_OF_PARTIES = 1;
  }
	else if (strcmp(argv[3], "3PC") == 0) {
		NUM_OF_PARTIES = 3;
  } else {
    print_usage(argv[0]);
  }

	partyNum = atoi(argv[4]);
	
	if (partyNum < 0 or partyNum > 4) {
		print_usage(argv[0]);
  }

  return config;
}


void initializeMPC()
{
	//populate offline module prime addition and multiplication tables
	for (int i = 0; i < PRIME_NUMBER; ++i)
		for (int j = 0; j < PRIME_NUMBER; ++j)
		{
			additionModPrime[i][j] = (i + j) % PRIME_NUMBER;
			multiplicationModPrime[i][j] = (i * j) % PRIME_NUMBER;
		}
}

vector<vector<myType>> readMyTypes(std::string filename) {
	myType temp;
  string line;
	ifstream f(filename);

  vector<vector<myType>> out;
  auto prevSize = 0;
  while(std::getline(f, line)) {
    vector<myType> myTypes;
    stringstream ss(line);
    while((ss >> temp)) {
      myTypes.push_back(temp);
    }
    
    if(prevSize > 0) {
      assert(prevSize == myTypes.size());
    } else {
      prevSize = myTypes.size();
    }

    out.push_back(myTypes);
  }
  f.close();
  return out;
}

vector<vector<myType>> readFloats(std::string filename) {
	float temp;
  string line;
	ifstream f(filename);

  vector<vector<myType>> out;
  auto prevSize = 0;
  while(std::getline(f, line)) {
    vector<myType> myTypes;
    stringstream ss(line);
    while((ss >> temp)) {
      myTypes.push_back(floatToMyType(temp));
    }
    
    if(prevSize > 0) {
      assert(prevSize == myTypes.size());
    } else {
      prevSize = myTypes.size();
    }

    out.push_back(myTypes);
  }
  f.close();
  return out;
}

void loadData(char* filename_train_data, char* filename_train_labels)
{
  trainData = readMyTypes(filename_train_data);
  trainLabels = readMyTypes(filename_train_labels);
}


void readMiniBatch(NeuralNetwork& net, string phase)
{
  auto inputSize = net.inputSize;
  auto outputSize = net.outputSize;
  auto batchSize = net.batchSize;

	size_t trainSize = trainData.size();
	if (phase == "TRAINING")
	{
    net.inputData.clear();
		for (int i = 0; i < batchSize; ++i) {
      auto d = trainData[(trainBatchCounter + i) % trainSize];
      net.inputData.insert(net.inputData.end(), d.begin(), d.end());
    }

    net.outputData.clear();
		for (int i = 0; i < batchSize; ++i) {
      auto d = trainLabels[(trainBatchCounter + i) % trainSize];
      net.outputData.insert(net.outputData.end(), d.begin(), d.end());
    }

		trainBatchCounter += batchSize;
	}

  if (trainBatchCounter > trainSize) {
    trainBatchCounter -= trainSize;
  }
}


void train(NeuralNetwork& net, NeuralNetConfig& config)
{
	log_print("train");

	if (!STANDALONE)
		initializeMPC();	

	for (int i = 0; i < config.numIterations; ++i)
	{

		//cout << "----------------------------------" << endl;  
		//cout << "Iteration " << i << endl;
		
		readMiniBatch(net, "TRAINING");

		net.forward();
		net.backward();
	}
}


void deleteObjects()
{
	//close connection
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i != partyNum)
		{
			delete communicationReceivers[i];
			delete communicationSenders[i];
		}
	}
	delete[] communicationReceivers;
	delete[] communicationSenders;

	delete[] addrs;
	delete[] party_aes_key;
}

