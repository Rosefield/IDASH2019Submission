
#pragma once
#include "TedKrovetzAesNiWrapperC.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include "AESObject.h"


using namespace std;


// AESObject::AESObject(char* filename)
// {
// 	ifstream f(filename);
// 	string str { istreambuf_iterator<char>(f), istreambuf_iterator<char>() };
// 	f.close();
// 	// cout << str << "\n";
// 	int len = str.length();
// 	// char* common_aes_key = new char[len+1];
// 	// vector<char> common_aes_key('\0', len+1);

// 	vector<char> common_aes_key(str.begin(), str.end());
// 	common_aes_key.push_back('\0');
// 	AES_set_encrypt_key(common_aes_key.data(), 256, &aes_key);
// 	rCounter = -1;
// 	// delete common_aes_key;
// }

AESObject::AESObject(char* filename)
{
	ifstream f(filename);
	string str { istreambuf_iterator<char>(f), istreambuf_iterator<char>() };
	f.close();
	int len = str.length();
  //cout << "Reading file: " << filename << endl;
  //cout << "key length: " << len << " " << str <<  endl;
  assert(len >= 32 && "make sure we actually read a key, a non-existant file will give a 0-len string");
	char common_aes_key[len+1];
	memset(common_aes_key, '\0', len+1);
	strcpy(common_aes_key, str.c_str());
	AES_set_encrypt_key((unsigned char*)common_aes_key, 256, &aes_key);
	// rCounter = -1;
	// delete common_aes_key;
}


__m128i AESObject::newRandomNumber()
{
	rCounter++;
	if (rCounter % RANDOM_COMPUTE == 0)//generate more random seeds
	{
		for (int i = 0; i < RANDOM_COMPUTE; i++)
			tempSecComp[i] = _mm_set1_epi32(rCounter+i);//not exactly counter mode - (rcounter+i,rcouter+i,rcounter+i,rcounter+i)
		AES_ecb_encrypt_chunk_in_out(tempSecComp, pseudoRandomString, RANDOM_COMPUTE, &aes_key);
	}
	return pseudoRandomString[rCounter%RANDOM_COMPUTE];
}


myType AESObject::get64Bits()
{
	myType ret;

	if (random64BitCounter == 0)
		random64BitNumber = newRandomNumber();
	

	switch(random64BitCounter) 
	{
    case 0 : 
        ret = (myType)_mm_extract_epi64(random64BitNumber, 0);
        break;       
    case 1 : 
        ret = (myType)_mm_extract_epi64(random64BitNumber, 1);
        break;
    default :
        assert(random64BitCounter < 2);
	}

	random64BitCounter++;	
	if (random64BitCounter == 2)
		random64BitCounter = 0;

	return ret;
}

smallType AESObject::get8Bits()
{
	smallType ret;

	if (random8BitCounter == 0)
		random8BitNumber = newRandomNumber();
	
	uint8_t *temp = (uint8_t*)&random8BitNumber;
	ret = (smallType)temp[random8BitCounter];

	random8BitCounter++;	
	if (random8BitCounter == 16)
		random8BitCounter = 0;

	return ret;
}

smallType AESObject::getBit()
{
	smallType ret;
	__m128i temp;

	if (randomBitCounter == 0)
		randomBitNumber = newRandomNumber();
	
	int x = randomBitCounter % 8; 
	switch(x) 
	{
    case 0 : temp = _mm_and_si128(randomBitNumber, BIT1);
             break;       
    case 1 : temp = _mm_and_si128(randomBitNumber, BIT2);
             break;
    case 2 : temp = _mm_and_si128(randomBitNumber, BIT4);
             break;       
    case 3 : temp = _mm_and_si128(randomBitNumber, BIT8);
             break;
	case 4 : temp = _mm_and_si128(randomBitNumber, BIT16);
             break;       
    case 5 : temp = _mm_and_si128(randomBitNumber, BIT32);
             break;
	case 6 : temp = _mm_and_si128(randomBitNumber, BIT64);
             break;       
    case 7 : temp = _mm_and_si128(randomBitNumber, BIT128);
             break;
	}
	uint8_t val = _mm_extract_epi8(temp, 0);
	ret = (val >> x);

	randomBitCounter++;	
	if (randomBitCounter % 8 == 0)
		randomBitNumber = _mm_srli_si128(randomBitNumber, 1);

	if (randomBitCounter == 128)
		randomBitCounter = 0;

	return ret;
}

smallType AESObject::randModPrime()
{
	smallType ret;
	
	do
	{
		ret = get8Bits();
	} while (ret >= BOUNDARY);

	return (ret % PRIME_NUMBER); 
}

smallType AESObject::randNonZeroModPrime()
{
	smallType ret;
	do
	{
		ret = randModPrime();
	} while (ret == 0);

	return ret; 
}

myType AESObject::randModuloOdd()
{
	myType ret;
	do
	{
		ret = get64Bits();
	} while (ret == MINUS_ONE);
	return ret;
}


smallType AESObject::AES_random(int i)
{
	smallType ret;
	do
	{
		ret = get8Bits();
	} while (ret >= ((256/i) * i));

	return (ret % i); 
}

void AESObject::AES_random_shuffle(vector<smallType> &vec, size_t begin_offset, size_t end_offset)
{
	vector<smallType>::iterator it = vec.begin();
	auto first = it + begin_offset;
	auto last = it + end_offset;
    auto n = last - first;

    for (auto i = n-1; i > 0; --i)
    {
        using std::swap;
        swap(first[i], first[AES_random(i+1)]);
    }
}
