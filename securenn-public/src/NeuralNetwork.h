
#pragma once
#include "NeuralNetConfig.h"
#include "FCLayer.h"
#include "tools.h"
#include "globals.h"
using namespace std;

class NeuralNetwork
{
public:
	vector<myType> inputData;
	vector<myType> outputData;
	vector<unique_ptr<Layer>> layers;
  size_t inputSize;
  size_t outputSize;
  size_t batchSize;
  size_t numIterations;

	NeuralNetwork(NeuralNetConfig& config);
	~NeuralNetwork();
	void forward();
	void backward();
	void computeDelta();
	void updateEquations();
	void predict(vector<myType> &maxIndex);
	void getAccuracy(const vector<myType> &maxIndex, vector<size_t> &counter);
	size_t getSize();
	void outputNetwork(string fileName);

	vector<unique_ptr<Layer>> layersSA;
  template <typename F, typename G>
  void NeuralNetwork::compareToSA(string prefix, F&& update, G&& getter) {
    //cout << "--------------------------" << endl;
    //cout << prefix << endl;
    update(layers);
    if (SA_VALIDATE && NUM_OF_PARTIES == 3 && PRIMARY) {
      NUM_OF_PARTIES = 1;
      update(layersSA);

      auto a1 = getter(layers);
      auto a2 = getter(layersSA);
      vector<myType> t(a1.size());
      sendVector<myType>(a1, adversary(partyNum), a1.size());
      receiveVector<myType>(t, adversary(partyNum), a1.size());
      addVectors<myType>(t, a1, t, a1.size());

      auto s = t.size() > 100 ? 100 : t.size();
      for(size_t i = 0; i < s; ++i) {
        auto diff = std::abs(myTypeToFloat(t[i] - a2[i]));
        //if(t[i] != a2[i]) {
        if(diff > .01) {
          cout << prefix << " values do not match at position " << i << " found values a1, a2, diff " << t[i] << " " << a2[i] << " " << diff << endl;
        }
      }

      NUM_OF_PARTIES = 3;
    }

  }
};
