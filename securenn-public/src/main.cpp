
#include <iostream>
#include <string>
#include "secondary.h"
#include "connect.h"
#include "AESObject.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "Functionalities.h"


using namespace std;
extern int partyNum;
int NUM_OF_PARTIES;	


AESObject* aes_common;
AESObject* aes_indep;
AESObject* aes_a_1;
AESObject* aes_a_2;
AESObject* aes_b_1;
AESObject* aes_b_2;
AESObject* aes_c_1;
ParallelAESObject* aes_parallel;



int main(int argc, char** argv)
{

/****************************** PREPROCESSING ******************************/ 
	auto config = parseInputs(argc, argv);
	string whichNetwork = config.name;

	config.checkNetwork();
	NeuralNetwork network(config);

/****************************** AES SETUP and SYNC ******************************/ 
	aes_indep = new AESObject(argv[6]);
	aes_common = new AESObject(argv[7]);
  if(THREE_PC)
  {
    aes_a_1 = new AESObject("files/keyAC");
    aes_b_1 = new AESObject("files/keyAC");
    aes_c_1 = new AESObject("files/keyAC");
    aes_a_2 = new AESObject("files/keyBC");
    aes_b_2 = new AESObject("files/keyBC");
  }
	aes_parallel = new ParallelAESObject(argv[7]);

	if (!STANDALONE)
	{
		initializeCommunication(argv[5], partyNum);
		synchronize(2000000);	
	}

	if (PARALLEL)
		aes_parallel->precompute();


/****************************** RUN NETWORK/BENCHMARKS ******************************/ 
	start_m();

	whichNetwork += " train";
	train(network, config);

	//whichNetwork += " test";
  //test(network);

	end_m(whichNetwork);

  network.outputNetwork(config.outputFile);	

	cout << "----------------------------------" << endl;  	
	cout << NUM_OF_PARTIES << "PC code, P" << partyNum << endl;
	cout << config.numIterations << " iterations, " << whichNetwork << ", batch size " << config.batchSize << endl;
	cout << "----------------------------------" << endl << endl;  


/****************************** CLEAN-UP ******************************/ 
	delete aes_common;
	delete aes_indep;
	delete aes_a_1;
	delete aes_a_2;
	delete aes_b_1;
	delete aes_b_2;
	delete aes_c_1;
	delete aes_parallel;
	// delete l3;
	if (partyNum != PARTY_S)
		deleteObjects();

	return 0;
}

