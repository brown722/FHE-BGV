
#include <NTL/lzz_pXFactoring.h>
#include <fstream>
#include <sstream>
#include <sys/time.h>
//#include "NumbTh.h"
#include "timing.h"
#include "permutations.h"
#include "PAlgebra.h"
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <NTL/ZZXFactoring.h>
#include <NTL/GF2EXFactoring.h>
#include <NTL/lzz_pEXFactoring.h>
#include <NTL/GF2X.h>
#include <NTL/GF2EX.h>
#include <NTL/ZZ_pXFactoring.h>
#include <NTL/ZZ_pEX.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include "hypercube.h"
//#include "myprog.h"
#include "math.h"
#include "bluestein.h"
#include "FHE.h"
#include "EncryptedArray.h"
#include <NTL/tools.h>
#include "testBluestein.h"
#include "NewBluesteinFFT.h"
#include "NTT.h"
#include "Ctxt.h"

#define N        0xFFFFFFFF00000001ull
#define ROOT   0xE9653C8DEFA860A9ull

using namespace std;
using namespace NTL;

/****************sample Function************/
// int main(int argc, char **argv)
// {
//
//     long m=0, p=5, r=1;
//
//     long L=16;
//     long c=3;
//     long w=64;
//     long d=0;
//     long security = 128;
//
//
//     ZZX G;
//     m = FindM(security,L,c,p, d, 0, 0);
//
//     FHEcontext context(m, p, r);
//
//     buildModChain(context, L, c);
//
//     FHESecKey secretKey(context);
//
//     const FHEPubKey& publicKey = secretKey;
//
//     G = context.alMod.getFactorsOverZZ()[0];
//
//    secretKey.GenSecKey(w);
//
//    addSome1DMatrices(secretKey);
//
//    EncryptedArray ea(context, G);
//
//    long nslots = ea.size();
//
//    vector<long> v1;
//    for(int i = 0 ; i < nslots; i++) {
//        v1.push_back(i*2);
//    }
//    Ctxt ct1(publicKey);
//    ea.encrypt(ct1, publicKey, v1);
//
//
//    vector<long> v2;
//       for(int i = 0 ; i < nslots; i++) {
//           v2.push_back(i*3);
//       }
//       Ctxt ct2(publicKey);
//       ea.encrypt(ct2,publicKey,v2);
//
//       Ctxt ctSum = ct1;
//       Ctxt ctProd = ct1;
//
//       ctSum += ct1;
//       ctProd *= ct1;
//
//       vector<long> res;
//       ea.decrypt(ctSum, secretKey, res);
//       for(int i = 0; i < res.size(); i ++) {
//           cout << v1[i] << " + " << v2[i] << " = " << res[i] << endl;
//       }
//       ea.decrypt(ctProd, secretKey, res);
//       for(int i = 0; i < res.size(); i ++)
//     	  cout << v1[i] << " * " << v2[i] << " = " << res[i] << endl;
// 	return 0;
// }

/******************temporary test**********************/


//
// int main(){
//	 Vec<long> a;
//	 for (int i = 0; i < 3; i ++)
//	 {
//		 a.append(i);
//	 }
//
//	 vector< Vec<long> > aa(4);
//	 for (int i = 0; i < 4; i ++)
//	 {
//		 aa[i].SetLength(3);
//		 aa[i] = a;
//	 }
//
//	 cout<<"the 4th row 2th comlum is: "<<aa[3][1]<<endl;
//
//
//
//
//
// 	// uint32_t p;
// 	// uint64_t invP;
// 	// p = 2296577;//2296577;
// 	// invP = ModInverse(p);
//
// 	// uint64_t a;
// 	// uint64_t c;
// 	// uint64_t b;
//
// 	// a = 2252152;//2252152;
// 	// b = 2277486;//2277486;
// 	// uint64_t t;
// 	// t = ModMultiply(a, b);
//
// 	// invP = 18446744073709551615/p;
// 	// c = BarretMulMod(t, p, invP);
//
// 	// // multiplyMod (double *&a, uint64_t *&b, uint64_t *&c, long p, long length);
//
// 	// cout<<endl<<"the output of C++ is: "<<c<<endl;
//
// 	// double *aa;
// 	// uint64_t *bb;
// 	// uint64_t *cc;
// 	// uint64_t pp = 2296577;
// 	// long length = 10;
//
// 	// aa = (double *)malloc(sizeof(double)*length);
// 	// bb = (uint64_t *)malloc(sizeof(uint64_t)*length);
// 	// cc = (uint64_t *)malloc(sizeof(uint64_t)*length);
//
//
// 	// for (int i = 0; i < length; i ++)
// 	// {
// 	// 	aa[i] = 10;
// 	// 	bb[i] = 10;
// 	// }
//
// 	// multiplyMod(aa, bb, cc, pp, length);
//
// 	// cout<<endl<<"the output of GPU is:"<<endl;
// 	// for (int i = 0; i < length ; i ++)
// 	// 	cout<<cc[i]<<", ";
//
// 	// free(aa);	free(bb);	free(cc);
//
//// 	long length = 10;
//// 	uint32_t *nttInput;
//// 	nttInput = (uint32_t *)malloc(sizeof(uint32_t)*16384);//
//// 	memset(nttInput, 0, sizeof(uint32_t)*16384);
//// 	for (int i = 0; i < length; i ++)
//// 		nttInput[i] = i;
////
//// 	uint32_t *nttInputGPU;
//// 	cudaMalloc((void**)&nttInputGPU, 16384 * sizeof(uint32_t));//
//// 	cudaMemcpy(nttInputGPU, nttInput, 16384 * sizeof(uint32_t), cudaMemcpyHostToDevice);
////
//// 	uint64_t *INTT;
//// 	cudaMalloc((void**)&INTT, 16384 * sizeof(uint32_t));//cudaFree(INTT);
//// 	cudaMemcpy(INTT, nttInput, 16384 * sizeof(uint32_t), cudaMemcpyHostToDevice);
////
//// 	uint64_t *fftInput;
//// 	fftInput = (uint64_t *)malloc(sizeof(uint64_t)*16384);//
////
//// 	for (int i = 0; i < length; i ++)
//// 		fftInput[i] = i;
////
//// 	uint64_t *fftOutput;
//// 	fftOutput = (uint64_t *)malloc(sizeof(uint64_t)*16384);//
////
//// 	largeFFT(fftInput, fftOutput, 16384, 0);
//// 	free(fftInput);
////
//// 	cout<<endl<<"the output of fft is: "<<endl;
//// 	for (int i = 0; i < length; i ++)
//// 		cout<<fftOutput[i]<<", ";
//// 	cout<<endl;
////
////
//// 	uint64_t *nttOutput;
//// 	nttOutput = (uint64_t *)malloc(sizeof(uint64_t)*16384);//
////
//// 	uint32_t *inttOutput;
//// 	inttOutput = (uint32_t *)malloc(sizeof(uint32_t)*16384);//
////
//// 	uint64_t *nttOutputGPU;
//// 	cudaMalloc((void**)&nttOutputGPU, 16384 * sizeof(uint64_t));//
////
////
////
//// 	cout<<endl<<"ntting ..."<<endl;
//// 	initNtt(16384) ;
//// 	_ntt(nttOutputGPU, nttInputGPU, 0, 0,16384);
////
//// 	uint32_t *nttResultGPU;
//// 	cudaMalloc((void**)&nttResultGPU, 16384 * sizeof(uint32_t));//
////
//// 	// _intt(nttResultGPU, nttOutputGPU, 5, 0, 0, 16384);
////
//// 	cudaMemcpy(nttOutput, nttOutputGPU, 16384 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
////
//// 	cout<<endl<<"the output of ntt is: "<<endl;
//// 	for (int i = 0; i < length; i ++)
//// 		cout<<nttOutput[i]<<", ";
//// 	cout<<endl;
////
////
//// 	free(nttOutput);free(fftOutput);free(nttInput);free(inttOutput);
//// 	cudaFree(nttOutputGPU);
//// 	cudaFree(nttInputGPU);
//// 	cudaFree(INTT);
//// 	cudaFree(nttResultGPU);
//
// 	return 0;
// }

/***********************encryption test**************/

int main(int argc, char **argv)
{
	long m=0, p=5, r=1; 			// Native plaintext space
												// Computations will be 'modulo p'
	long L=8;          					// Levels
	long c=3; //3          					// Columns in key switching matrix
	long w=64;          				// Hamming weight of secret key
	long d=0;
	long security = 256;
	bool HighNoise = false;
	ZZX G;

	m = FindM(security,L,c,p, d, 0, 0);

	cout<<"m is: "<<m<<endl;

	cout<<endl<<"************************************************************"<<endl;

	FHEcontext context(m, p, r); 								// initialize context

	buildModChain(context, L, c);								// modify the context, adding primes to the modulus chain

	FHESecKey secretKey(context);							// construct a secret key structure

	const FHEPubKey& publicKey = secretKey; 	// an "upcast": FHESecKey is a subclass of FHEPubKey

	G = context.alMod.getFactorsOverZZ()[0];

	secretKey.GenSecKey(w);										// actually generate a secret key with Hamming weight w

	addSome1DMatrices(secretKey);

	EncryptedArray ea(context, G); 							// constuct an Encrypted array object ea that is  associated with the given context and the polynomial G

	long nslots = ea.size();

	cout<<"Keep Going"<<endl;

	//encryption Enc;
	zz_p::init(p);

	vector<long> plaintextVector;
	for(long i = 0;i < nslots; i++)
		plaintextVector.push_back(i);

 	Ctxt ciphertext(publicKey);

	double cpuTime = 0, gpuTime = 0;

	cpuTime = GetTime();
 	ea.encrypt(ciphertext, publicKey, plaintextVector);
 	cpuTime = GetTime() - cpuTime;
	
	//get the public key data
	Ctxt cipher(publicKey);
 	cipher = publicKey.pubEncrKey;

/*Precomputations for following FFTs*/
	//setting all the primes and invPrimes
	long mm = context.zMStar.getM();
	long numPrimes = context.numPrimes();
	uint64_t primes[numPrimes], invPrimes[numPrimes];
	for (int i = 0; i < numPrimes; i ++)
	{
		primes[i] = context.ithModulus(i).getQ();
	}
	
	uint64_t *powers_gpu[numPrimes];
	double2 *Rb_gpu[numPrimes];
	vector<zz_pX> powers;
	powers.resize(numPrimes);
	vector<Vec<mulmod_precon_t> >powers_aux(numPrimes);
	vector<zz_p> Root;
	Root.resize(numPrimes);
	for (int i = 0; i < numPrimes; i ++)
	{
		context.ithModulus(i).restoreModulus();
		long rt = context.ithModulus(i).getRoot();
		conv(Root[i], rt);
		gpuBlustnInit(mm, Root[i], powers[i], powers_aux[i], Rb_gpu[i], powers_gpu[i], invPrimes[i]);
	}

	//set up "in-zMStar" flags for FFT fucntion
	vector<long> ZmsIndex; int *dropFlags, *gpuDropFlags, *offsetFlags, *gpuOffsetFlags;//
	context.zMStar.getZmsIndex(ZmsIndex);
	dropFlags = (int*)malloc(sizeof(int) * ZmsIndex.size());
	offsetFlags = (int*)malloc(sizeof(int) * ZmsIndex.size());
	cudaMalloc((void**)&gpuDropFlags, ZmsIndex.size() * sizeof(int));
	cudaMalloc((void**)&gpuOffsetFlags, ZmsIndex.size() * sizeof(int));
	memset(dropFlags, 0, sizeof(int) * ZmsIndex.size());
	memset(offsetFlags, -1, sizeof(int) * ZmsIndex.size());

	int shift;
	for (int i = 0, k = 0; i < ZmsIndex.size(); i ++)
	{
		if (ZmsIndex[i] < 0)
		{
			dropFlags[i] = 1;
			offsetFlags[k++] = i;
		}
	}
	shift = offsetFlags[1] - offsetFlags[0];
	if (shift < 0)
		shift = 1;

	for (int i= 0; i < 10; i ++)
	{
		cout<<offsetFlags[i]<<", ";
		if (i == (10 - 1))
			cout<<endl;
	}

	cudaMemcpy(gpuDropFlags, dropFlags, ZmsIndex.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuOffsetFlags, offsetFlags, ZmsIndex.size() * sizeof(int), cudaMemcpyHostToDevice);

	//get the public key data
	int numOfParts = cipher.getSizeOfParts();
	vector<vector<Vec<long> > >ROW(numOfParts, vector<Vec<long> >(numPrimes));
	for (int j = 0; j < numOfParts; j ++)
	{
		int cnt = 0;
		for (long i = 0, ithModuli = 0; i < numPrimes; i ++)
		{
			ithModuli = cipher.getParts(0).getOneRow(ROW[j][i], i, true);
			if (ithModuli)
				cnt = cnt + 1;
		}
		ROW[j].resize(cnt);
		numPrimes = cnt;
	}
	long phiM = context.zMStar.getPhiM();

	uint64_t *CipherTxt[numOfParts * numPrimes];
	for (int i = 0; i < numOfParts * numPrimes; i ++)
		CipherTxt[i] = (uint64_t *)malloc(sizeof(uint64_t) * phiM);

	for (int k = 0; k < numOfParts; k++)
	{
		for (int i =0; i < numPrimes; i++)
		{
			for (int j = 0; j < phiM; j++)
				CipherTxt[k*numPrimes + i][j] = ROW[k][i][j];
		}
	}

	uint64_t *cipherOnGPU[numOfParts * numPrimes];
	for (int i = 0; i < numOfParts * numPrimes; i ++)
	{
		cudaMalloc((void**)&cipherOnGPU[i], phiM * sizeof(uint64_t));
		cudaMemcpy(cipherOnGPU[i], CipherTxt[i], phiM * sizeof(uint64_t), cudaMemcpyHostToDevice);
	}
	
	cudaDeviceSynchronize();

	cudaStream_t streamSmall[numPrimes * numOfParts];
	for (int i = 0; i < numPrimes * numOfParts; i ++)
	{
		cudaStreamCreate(&streamSmall[i]);
	}

	cudaStream_t streamGauss[numPrimes * numOfParts];
	for (int i = 0; i < numPrimes * numOfParts; i ++)
	{
		cudaStreamCreate(&streamGauss[i]);
	}

	long pPowerOfR = context.alMod.getPPowR();
	long ptxtSpace = publicKey.pubEncrKey.getPtxtSpace();
	long QmodP; ZZX input;
	QmodP = rem(context.productOfPrimes(ciphertext.getPrimeSet()), ptxtSpace);
	double stdev = to_double(context.stdev);
	double timeCPU;int testNum = 0;
	
	/////*************************verifying the encode function**********************************
	cout<<endl<<"*****************verifying bluesteinFFT functions******************"<<endl;
		gpuTime = GetTime();
		ZZX plaintext_ZZX;
		ea.encode(plaintext_ZZX, plaintextVector);
		// cpuTime = GetTime();
		// publicKey.Encrypt(ciphertext,plaintext_ZZX,ptxtSpace);
		// cpuTime = GetTime() - cpuTime;

		// gpuTime = GetTime();
		MulMod(input,plaintext_ZZX,QmodP,ptxtSpace);

									
		ZZX r_poly; zz_pX r_zzP[numPrimes];
		sampleSmall(r_poly, phiM);

		ZZX e_polyG, e_polyU; zz_pX e_zzPG[numPrimes] , e_zzPU[numPrimes]; 
		sampleGaussian(e_polyG, phiM, stdev);
		
		int *offset[numPrimes];
		int nx;


		uint64_t *r_smallSample[numPrimes];
		cufftDoubleReal *smallBuffer[numPrimes];
		cufftDoubleReal *cuSmallBuffer[numPrimes];
		cufftDoubleComplex *cuRaSmall[numPrimes]; 
		cufftDoubleComplex *resultSmall[numPrimes];
		cufftHandle planSmall[numPrimes];
		cufftHandle invPlanSmall[numPrimes];
		cufftDoubleReal *cuFFToutSmall[numPrimes];
#pragma omp parallel
{
	#pragma omp for
		for (int i = 0; i < numPrimes; i ++)
		{
			context.ithModulus(i).restoreModulus();
			conv(r_zzP[i], r_poly);
			memoryAllocations(r_zzP[i], mm, powers[i], powers_aux[i],nx, r_smallSample[i],smallBuffer[i], cuSmallBuffer[i], cuRaSmall[i],
				resultSmall[i], planSmall[i], cuFFToutSmall[i], invPlanSmall[i], offset[i], streamSmall[i]);
		}	
}
#pragma omp parallel
{
	#pragma omp for
		for (int i = 0; i < numPrimes; i ++)
		{
			deviceBlustn2(mm, smallBuffer[i], cuSmallBuffer[i],  cuRaSmall[i], Rb_gpu[i], resultSmall[i], cuFFToutSmall[i], 
				invPrimes[i], primes[i], nx, powers_gpu[i], r_smallSample[i], gpuOffsetFlags, shift, streamSmall[i], streamGauss[i], planSmall[i], invPlanSmall[i], 0);
		}
}
		// devBluestn2(mm, smallBuffer, cuSmallBuffer,  cuRaSmall, Rb_gpu, resultSmall, cuFFToutSmall, 
		// 	invPrimes, primes, nx, powers_gpu, r_smallSample,  gpuOffsetFlags, shift, streamSmall, streamGauss, planSmall, invPlanSmall,0,numPrimes);


		 // cout<<"check for smallSample"<<endl;
		 // for (int i = 0; i < numPrimes; i ++)
		 // 	readAndCheck(cuFFToutSmall[i], powers_gpu[i],  r_smallSample[i], nx, mm, streamSmall[i]);
		 // fftRep Rbsmall;
		 // context.ithModulus(testNum).restoreModulus();
		 // zz_pX productSmall;
		 // BluesteinInit(mm, Root[testNum], powers[testNum], powers_aux[testNum], Rbsmall);
		 // OriginalFFT(r_zzP[testNum], mm, Root[testNum], powers[testNum], powers_aux[testNum], Rbsmall, productSmall, timeCPU);
		 // for (int i = 0; i < 10; i++)
		 // 	cout<<productSmall[i]<<", ";
		 // cout<<endl;

		uint64_t *e_gaussianSample[numPrimes];
		cufftDoubleReal *gaussBuffer[numPrimes];
		cufftDoubleReal *cuGaussBuffer[numPrimes];
		cufftDoubleComplex *cuRaGauss[numPrimes]; 
		cufftDoubleComplex *resultGauss[numPrimes];
		cufftHandle planGauss[numPrimes];/////////////////////////////////////////
		cufftHandle invPlanGauss[numPrimes];
		cufftDoubleReal *cuFFToutGauss[numPrimes];
#pragma omp parallel
{
	#pragma omp for
		for (int i = 0; i < numPrimes; i ++)
		{
			context.ithModulus(i).restoreModulus();
			conv(e_zzPG[i], e_polyG);
			memoryAllocations(e_zzPG[i], mm, powers[i], powers_aux[i],nx, e_gaussianSample[i],gaussBuffer[i], cuGaussBuffer[i], cuRaGauss[i],
				resultGauss[i], planGauss[i], cuFFToutGauss[i], invPlanGauss[i], offset[i], streamGauss[i]);
		}
}
#pragma omp parallel
{
	#pragma omp for
		for (int i = 0; i < numPrimes; i ++)
		{
			deviceBlustn2(mm, gaussBuffer[i], cuGaussBuffer[i],  cuRaGauss[i], Rb_gpu[i], resultGauss[i], cuFFToutGauss[i], 
				invPrimes[i], primes[i], nx, powers_gpu[i], e_gaussianSample[i], gpuOffsetFlags, shift, streamGauss[i], streamSmall[i],planGauss[i],  invPlanGauss[i], 1);
		}
}
		// devBluestn2(mm, gaussBuffer, cuGaussBuffer,  cuRaGauss, Rb_gpu, resultGauss, cuFFToutGauss, 
		// 	invPrimes, primes, nx, powers_gpu, e_gaussianSample,  gpuOffsetFlags, shift, streamGauss, streamSmall, planGauss, invPlanGauss,1,numPrimes);

		// cout<<"check for e_gaussianSample"<<endl;
		// for (int i = 0; i < numPrimes; i ++)
		// 	readAndCheck(cuFFToutput[i], powers_gpu[i],  e_gaussianSample[i], nx, mm, stream[i]);
		// context.ithModulus(testNum).restoreModulus();
		// zz_pX productGauss; fftRep RbGauss;
		// BluesteinInit(mm, Root[testNum], powers[testNum], powers_aux[testNum], RbGauss);
		// OriginalFFT(e_zzPG[testNum], mm, Root[testNum], powers[testNum], powers_aux[testNum], RbGauss, productGauss, timeCPU);
		// for (int i = 0; i < 10; i++)
		// 	cout<<productGauss[i]<<", ";
		// cout<<endl;

		cufftDoubleReal *ptxtBuffer[numPrimes];
		cufftDoubleReal *cuPtxtBuffer[numPrimes];
		cufftDoubleComplex *cuRaPtxt[numPrimes]; 
		cufftDoubleComplex *resultPtxt[numPrimes];
		cufftHandle planPtxt[numPrimes];
		cufftHandle invPlanPtxt[numPrimes];
		cufftDoubleReal *cuFFToutPtxt[numPrimes];
		uint64_t *ptxt[numPrimes]; zz_pX ptxt_zzP[numPrimes];
#pragma omp parallel
{
	#pragma omp for
		for (int i = 0; i < numPrimes; i ++)
		{
			context.ithModulus(i).restoreModulus();
			conv(ptxt_zzP[i], input);
			memoryAllocations(ptxt_zzP[i], mm, powers[i], powers_aux[i],nx, ptxt[i],ptxtBuffer[i], cuPtxtBuffer[i], cuRaPtxt[i],
				resultPtxt[i], planPtxt[i], cuFFToutPtxt[i], invPlanPtxt[i], offset[i], streamSmall[i]);
		}
}
		// for (int i = 0; i < numPrimes; i ++)
		// {
		// 	deviceBlustn2(mm, ptxtBuffer[i], cuPtxtBuffer[i],  cuRaPtxt[i], Rb_gpu[i], resultPtxt[i], cuFFToutPtxt[i], 
		// 		invPrimes[i], primes[i], nx, powers_gpu[i], ptxt[i], gpuOffsetFlags, shift, streamSmall[i], streamGauss[i], planPtxt[i],  invPlanPtxt[i], 0);
		// }
		devBluestn2(mm, ptxtBuffer, cuPtxtBuffer,  cuRaPtxt, Rb_gpu, resultPtxt, cuFFToutPtxt, 
			invPrimes, primes, nx, powers_gpu, ptxt,  gpuOffsetFlags, shift, streamSmall, streamGauss, planPtxt, invPlanPtxt,0,numPrimes);
		 // cout<<"check for ptxt"<<endl;
		 // for (int i = 0; i < numPrimes; i ++)
		 // 	readAndCheck(cuFFToutPtxt[i], powers_gpu[i],  ptxt[i], nx, mm, streamSmall[i]);
		 // context.ithModulus(testNum).restoreModulus();
		 // zz_pX productPtxt; fftRep RbPtxt;
		 // BluesteinInit(mm, Root[testNum], powers[testNum], powers_aux[testNum], RbPtxt);
		 // OriginalFFT(ptxt_zzP[testNum], mm, Root[testNum], powers[testNum], powers_aux[testNum], RbPtxt, productPtxt, timeCPU);
		 // for (int i = 0; i < 10; i++)
		 // 	cout<<productPtxt[i]<<", ";
		 // cout<<endl;

		generalMulMod (phiM, numPrimes, numOfParts, streamSmall, cipherOnGPU, r_smallSample, primes, invPrimes);//correct
		
		mulModPtxtspace(phiM, numPrimes, streamSmall, e_gaussianSample, pPowerOfR, primes, invPrimes);// correct

		for (int  i= 0; i < numOfParts; i ++)
		{
			ithPartAddMod (phiM, numPrimes, streamSmall, cipherOnGPU, e_gaussianSample, primes, i);
		}	
	
	
		gpuTime = GetTime() - gpuTime;
		
		for (int i = 0; i < numPrimes; i ++)
		{
			cudaFree(r_smallSample[i]);
			cudaFree(e_gaussianSample[i]);
			cudaFree(ptxt[i]);
			cudaFree(Rb_gpu[i]);
			cudaFree(powers_gpu[i]);
		}


		for (int i = 0; i < numPrimes; i ++)
		{
			cudaFree(offset[i]); offset[i] = NULL;
			cufftDestroy(planSmall[i]);
			cufftDestroy(invPlanSmall[i]);
			cudaFree(cuRaSmall[i]); cuRaSmall[i] = NULL;
			cudaFree(resultSmall[i]); resultSmall[i] = NULL;
			cudaFree(cuFFToutSmall[i]); cuFFToutSmall[i] = NULL;
			free(smallBuffer[i]); smallBuffer[i] = NULL;
			cudaFree(cuSmallBuffer[i]); cuSmallBuffer[i] = NULL;
		}


		for (int i = 0; i < numPrimes; i ++)
		{
			cudaFree(offset[i]); offset[i] = NULL;
			cufftDestroy(planGauss[i]);
			cufftDestroy(invPlanGauss[i]);
			cudaFree(cuRaGauss[i]); cuRaGauss[i] = NULL;
			cudaFree(resultGauss[i]); resultGauss[i] = NULL;
			cudaFree(cuFFToutGauss[i]); cuFFToutGauss[i] = NULL;
			free(gaussBuffer[i]); gaussBuffer[i] = NULL;
			cudaFree(cuGaussBuffer[i]); cuGaussBuffer[i] = NULL;
		}

		for (int i = 0; i < numPrimes; i ++)
		{
			cudaFree(offset[i]); offset[i] = NULL;
			cufftDestroy(planPtxt[i]);
			cufftDestroy(invPlanPtxt[i]);
			cudaFree(cuRaPtxt[i]); cuRaPtxt[i] = NULL;
			cudaFree(resultPtxt[i]); resultPtxt[i] = NULL;
			cudaFree(cuFFToutPtxt[i]); cuFFToutPtxt[i] = NULL;
			free(ptxtBuffer[i]); ptxtBuffer[i] = NULL;
			cudaFree(cuPtxtBuffer[i]); cuPtxtBuffer[i] = NULL;
		}
		//cpu FFT methods and verification outputs
		//int testNum = 0;
		
		// vec_long output;
		// context.ithModulus(0).FFT(output, e_polyU);
		// context.ithModulus(testNum).restoreModulus();
		// fftRep Rb_fft; zz_pX productC;
		// //double timeCPU;
		// BluesteinInit(mm, Root[testNum], powers[testNum], powers_aux[testNum], Rb_fft);
		// OriginalFFT(e_zzPU[testNum], mm, Root[testNum], powers[testNum], powers_aux[testNum], Rb_fft, productC, timeCPU);
		
		// for (int i = 0; i < 10; i++)
		// 	cout<<productC[i]<<", ";
		// cout<<endl;

		// cout<<endl;
		// for (int i = 0; i < 10; i++)
		// 	cout<<output[i]<<", ";
		// cout<<endl;


		cudaFree(gpuDropFlags);
		cudaFree(gpuOffsetFlags);
		free(dropFlags);
		free(offsetFlags);
		for (int i = 0; i < numPrimes * numOfParts; i ++)
		{
			cudaStreamDestroy(streamSmall[i]);
			cudaStreamDestroy(streamGauss[i]);
		}

		for (int i = 0; i < numOfParts * numPrimes; i ++)
		{
			free(CipherTxt[i]);
			cudaFree(cipherOnGPU[i]);
		}

		cout<<"time is: "<<endl;
		cout<<cpuTime<<", "<<gpuTime<<endl;
		return 0;

	// /***************************verification Tool**************************/
	// uint64_t *inTemp = (uint64_t *)malloc(sizeof(uint64_t)*phiM);
	// cudaMemcpy(inTemp, cipherOnGPU[0], (sizeof(uint64_t)) * phiM, cudaMemcpyDeviceToHost);

	// uint64_t *rTemp = (uint64_t *)malloc(sizeof(uint64_t)*phiM);
	// cudaMemcpy(rTemp, e_gaussianSample[0], (sizeof(uint64_t)) * phiM, cudaMemcpyDeviceToHost);
	// uint64_t *outTemp = (uint64_t *)malloc(sizeof(uint64_t)*phiM);
	// cudaMemcpy(outTemp, cipherOnGPU[0], (sizeof(uint64_t)) * phiM, cudaMemcpyDeviceToHost);

	// for (int i = 0; i < 10; i ++)
	// {
	// 	cout<<inTemp[i]<<", "<<rTemp[i]<<", "<<outTemp[i]<<endl;
	// 	if ((inTemp[i] + rTemp[i]) % primes[0] == outTemp[i])
	// 		cout<<"yes"<<endl;
	// 	else
	// 		cout<<"no"<<endl;
	// }
	// free(inTemp);free(outTemp);free(rTemp);
}
