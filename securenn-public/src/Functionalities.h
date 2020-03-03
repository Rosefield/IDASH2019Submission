
#pragma once
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

extern void start_time();
extern void start_communication();
extern void end_time(string str);
extern void end_communication(string str);



void funcTruncate2PC(vector<myType> &a, size_t power, size_t size, size_t party_1, size_t party_2);
void funcXORModuloOdd2PC(vector<smallType> &bit, vector<myType> &shares, vector<myType> &output, size_t size);
void funcReconstruct2PC(const vector<myType> &a, size_t size, string str);
void funcReconstructBit2PC(const vector<smallType> &a, size_t size, string str);
void funcMatMulMPC(const vector<myType> &a, const vector<myType> &b, vector<myType> &c, 
				size_t rows, size_t common_dim, size_t columns,
			 	size_t transpose_a, size_t transpose_b);
void funcDotProductMPC(const vector<myType> &a, const vector<myType> &b, 
					   vector<myType> &c, size_t size);
void funcPrivateCompareMPC(const vector<smallType> &share_m, const vector<myType> &r, 
							  const vector<smallType> &beta, vector<smallType> &betaPrime, 
							  size_t size, size_t dim);
void funcShareConvertMPC(vector<myType> &a, size_t size);
void funcComputeMSB3PC(const vector<myType> &a, vector<myType> &b, size_t size);
void funcSelectShares3PC(const vector<myType> &a, const vector<myType> &b, vector<myType> &c, size_t size);
void funcRELUPrime3PC(const vector<myType> &a, vector<myType> &b, size_t size);
void funcRELUMPC(const vector<myType> &a, vector<myType> &b, size_t size);
void funcDivisionMPC(const vector<myType> &a, const vector<myType> &b, vector<myType> &quotient, 
						size_t size);
void funcMaxMPC(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
						size_t rows, size_t columns);
void funcMaxIndexMPC(vector<myType> &a, const vector<myType> &maxIndex, 
						size_t rows, size_t columns);
void aggregateCommunication();
