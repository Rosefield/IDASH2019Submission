
#pragma once
#include "Functionalities.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;


class Layer
{
public: 
	virtual void forward(const vector<myType>& inputActivation) {};
	virtual void computeDelta(vector<myType>& prevDelta) {};
  virtual void computeOutputDelta(vector<myType>& outputs) {};
  virtual void finishFirstDelta() {};
	virtual void updateEquations(const vector<myType>& prevActivations) {};
  virtual void predict(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns) {};
	virtual void findMax(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns) {};


//Getters
	virtual vector<myType>* getActivation() {};
	virtual vector<myType>* getDelta() {};
	virtual vector<myType>* getWeights() {};
	virtual vector<myType>* getParam(vector<myType>& param) {};
	virtual size_t getSize() {};

  virtual void outputParams(ofstream& out) {};
};
