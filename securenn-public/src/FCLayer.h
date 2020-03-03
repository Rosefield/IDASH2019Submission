
#pragma once
#include "FCConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

extern int partyNum;
extern void AES_random_shuffle(vector<smallType> &vec, size_t begin_offset, size_t end_offset);


class FCLayer : public Layer
{
private:
	FCConfig conf;
	vector<myType> zetas;
	vector<myType> activations;
	vector<myType> deltas;
        vector<myType> accums_weights;
	vector<myType> accums_biases;
	vector<myType> weights;
	vector<myType> biases;
	vector<double> activationPrimeSmall;
	vector<myType> activationPrimeLarge;


public:
	//Constructor and initializer
	FCLayer(FCConfig& conf);
	void initialize();

	//Functions
	void forward(const vector<myType>& inputActivation) override;
	void computeDelta(vector<myType>& prevDelta) override;
	void computeOutputDelta(vector<myType>& output) override;
  void finishFirstDelta() override;
	void updateEquations(const vector<myType>& prevActivations) override;
  void predict(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns) override;
	void findMax(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns) override;

	//Getters
	vector<myType>* getActivation() {return &activations;};
	vector<myType>* getDelta() {return &deltas;};
	vector<myType>* getWeights() {return &weights;};
	vector<myType>* getParam(vector<myType>& param);
	size_t getSize(){return conf.getSize();};

  void outputParams(ofstream& out) override;

private:
	//Standalone
	void forwardSA(const vector<myType>& inputActivation);
	void computeDeltaSA(vector<myType>& prevDelta);
	void updateEquationsSA(const vector<myType>& prevActivations);
	void maxSA(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns);

	//MPC
	void forwardMPC(const vector<myType>& inputActivation);
	void computeDeltaMPC(vector<myType>& prevDelta);
	void updateEquationsMPC(const vector<myType>& prevActivations);
	void maxMPC(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns);
};
