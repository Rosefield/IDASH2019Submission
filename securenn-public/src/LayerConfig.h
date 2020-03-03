
#pragma once
#include "globals.h"
using namespace std;

class LayerConfig
{
public:
	LayerConfig() {};
  ~LayerConfig() = default;

  virtual string getType() = 0; 
  virtual size_t getSize() = 0; 
};

enum class ActivationFunctions {
  ReLU,
  Logistic,
  Logistic4
};

