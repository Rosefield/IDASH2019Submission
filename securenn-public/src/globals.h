
#ifndef GLOBALS_H
#define GLOBALS_H

#pragma once
#include <emmintrin.h>
#include <smmintrin.h>
#include <vector>
#include <string>
#include <memory>
#include <assert.h>
#include <limits.h>
#include <ostream>
#include <fstream>
using namespace std;

//Macros
#define _aligned_malloc(size,alignment) aligned_alloc(alignment,size)
#define _aligned_free free
#define getrandom(min, max) ((rand()%(int)(((max) + 1)-(min)))+ (min))
// casting a negative float to an unsigned int sets the value to 0, convert to positive int and then
// negate the value
#define floatToMyType(a) ((a) >= 0 ? (myType)((a) * (1 << FLOAT_PRECISION)) : -(myType)(-(a) * (1 << FLOAT_PRECISION)))
#define myTypeToFloat(a) (static_cast<int64_t>((a)) / ((double)(1 << FLOAT_PRECISION)))


//AES and other globals
#define LOG_DEBUG false 
#define SA_VALIDATE false 
#define RANDOM_COMPUTE 256//Size of buffer for random elements
#define FIXED_KEY_AES "43739841701238781571456410093f43"
#define STRING_BUFFER_SIZE 256
#define true 1
#define false 0
#define DEBUG_CONST 16
#define DEBUG_INDEX 0
#define DEBUG_PRINT "FLOAT"
#define CPP_ASSEMBLY 1
#define PARALLEL true 
#define NO_CORES 4


//MPC globals
extern int NUM_OF_PARTIES;
#define STANDALONE (NUM_OF_PARTIES == 1)
#define THREE_PC (NUM_OF_PARTIES == 3)
#define PARTY_A 0
#define PARTY_B 1
#define PARTY_C 2
#define PARTY_S 4

#define PRIME_NUMBER 67
#define FLOAT_PRECISION 20
#define PRIMARY (partyNum == PARTY_A or partyNum == PARTY_B)
#define	NON_PRIMARY (partyNum == PARTY_C)
#define HELPER (partyNum == PARTY_C)
#define MPC (THREE_PC)

//Neural Network globals.
#define LOG_LEARNING_RATE 7
#define LEARNING_RATE (1 << (FLOAT_PRECISION - LOG_LEARNING_RATE))
#define MOMENTUM 0.9

#define TEST_MINI_BATCH_SIZE 32



//Typedefs and others
typedef __m128i superLongType;
typedef uint64_t myType;
typedef uint8_t smallType;

const int BIT_SIZE = (sizeof(myType) * CHAR_BIT);
const myType LARGEST_NEG = ((myType)1 << (BIT_SIZE - 1));
const myType MINUS_ONE = (myType)-1;
const smallType BOUNDARY = (256/PRIME_NUMBER) * PRIME_NUMBER;

const __m128i BIT1 = _mm_setr_epi8(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT2 = _mm_setr_epi8(2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT4 = _mm_setr_epi8(4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT8 = _mm_setr_epi8(8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT16 = _mm_setr_epi8(16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT32 = _mm_setr_epi8(32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT64 = _mm_setr_epi8(64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT128 = _mm_setr_epi8(128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);


#endif
