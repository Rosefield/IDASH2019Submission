
#pragma once
#include "LayerConfig.h"
#include "globals.h"

using namespace std;

class FCConfig : public LayerConfig
{
public:
	size_t batchSize = 0;
	size_t inputDim = 0;
	size_t outputDim = 0;
  ActivationFunctions activation;


	FCConfig(size_t _batchSize, size_t _inputDim, size_t _outputDim, 
      ActivationFunctions _activation = ActivationFunctions::ReLU)
	:batchSize(_batchSize), 
	 inputDim(_inputDim),
	 outputDim(_outputDim),
   activation(_activation)
	{};

  string getType() override {
    return "FC";
  }
  
  size_t getSize() override{
    return inputDim * outputDim + outputDim;
  }
};
