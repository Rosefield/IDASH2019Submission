
#pragma once
#include "LayerConfig.h"
#include "FCConfig.h"
#include "globals.h"
#include <iostream>
using namespace std;

class NeuralNetConfig
{
public:
	size_t numIterations = 0;
	size_t numLayers = 0;
  size_t inputSize = 0;
  size_t outputSize = 0;
  size_t batchSize = 0;
	vector<shared_ptr<LayerConfig>> layerConf;
  string name;
  string outputFile = "";

	NeuralNetConfig(size_t _numIterations, size_t batchSize, size_t inputSize, size_t outputSize, string name, string outputFile = "")
	:numIterations(_numIterations),
  batchSize(batchSize),
  inputSize(inputSize),
  outputSize(outputSize),
  name(name),
  outputFile(outputFile)
	{};

	void addLayer(FCConfig fcl) { layerConf.push_back(make_shared<FCConfig>(fcl)); };
	
	void checkNetwork() 
	{
    numLayers = layerConf.size();
		assert(layerConf.back()->getType().compare("FC") == 0 && "Last layer has to be FC");
	};
};
