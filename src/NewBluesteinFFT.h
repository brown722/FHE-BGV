/*
 * NewBluesteinFFT.h
 *
 *  Created on: 2015年12月8日
 *      Author: xhuang
 */

#ifndef NEWBLUESTEINFFT_H_
#define NEWBLUESTEINFFT_H_

#define GPU 0
#include "stdint.h"
#include <cufft.h>
#include "cuda.h"
#include <cuda_runtime.h>
#include <cufftXt.h>


using namespace NTL;


void gpuBlustnInit(long n, const zz_p& root, zz_pX& powers,
	Vec<mulmod_precon_t>& powers_aux, double2 *&cuRbBuffer, //cudaFree();
	uint64_t *&powers_gpu/*cudaFree();*/, uint64_t &invP);


void BlustnOnGPU1(zz_pX& x, long n, double2 *&cuRbBuffer/*cudaFree(cuRbBuffer);*/,
	const zz_pX& powers, const Vec<mulmod_precon_t>& powers_aux, uint64_t *&powers_gpu/*cudaFree(powers_gpu);*/,
    uint64_t *&bluesteinOutput/*cudaFree();*/, uint64_t &invP, long p, int *&dropFlags, cudaStream_t &stream);

void memoryAllocations(zz_pX& x, long n, const zz_pX& powers, const Vec<mulmod_precon_t>& powers_aux, int &nx,
	uint64_t *&bluesteinOutput, cufftDoubleReal *&inputBuffer, cufftDoubleReal *&cuInputBuffer, cufftDoubleComplex *&cuRaBuffer, 
	cufftDoubleComplex *&result, cufftHandle &plan, cufftDoubleReal *&cuFFToutput, cufftHandle &planInverse, int *&offset, cudaStream_t &stream);

//void deviceBlustn2(long n, cufftDoubleReal *&inputBuffer,cufftDoubleReal *&cuInputBuffer,cufftDoubleComplex *&cuRaBuffer,cufftDoubleComplex *&cuRbBuffer,
//	cufftDoubleComplex *&result, cufftDoubleReal *&cuFFToutput, uint64_t &invP, long p, int &nx, uint64_t *&powers_gpu,  uint64_t *&bluesteinOutput,
//	int *&dropFlags,int *&offset, cudaStream_t &stream, cufftHandle &plan,cufftHandle &planInverse, int numGPU);
void deviceBlustn2(long n, cufftDoubleReal *&inputBuffer,cufftDoubleReal *&cuInputBuffer,cufftDoubleComplex *&cuRaBuffer,cufftDoubleComplex *&cuRbBuffer, 
	cufftDoubleComplex *&result, cufftDoubleReal *&cuFFToutput, uint64_t &invP, uint64_t p, int &nx, uint64_t *&powers_gpu,  uint64_t *&bluesteinOutput, 
	int *&dropFlags,int offset, cudaStream_t &stream, cudaStream_t &stream1, cufftHandle &plan,cufftHandle &planInverse, int numGPU);

// void devBluestn2(long mm, cufftDoubleReal *gaussBuffer[], cufftDoubleReal *cuGaussBuffer[],  cufftDoubleComplex *cuRaGauss[], cufftDoubleComplex *Rb_gpu[], cufftDoubleComplex *resultGauss[], cufftDoubleReal *cuFFToutGauss[], 
// 	uint64_t invPrimes[], uint64_t primes[], int &nx, uint64_t *powers_gpu[], uint64_t *e_gaussianSample[], int *gpuOffsetFlags, int offset, cudaStream_t streamGauss[], cudaStream_t streamSmall[], cufftHandle planGauss[], cufftHandle invPlanGauss[]);

void devBluestn2(long mm, cufftDoubleReal *gaussBuffer[], cufftDoubleReal *cuGaussBuffer[],  cufftDoubleComplex *cuRaGauss[], cufftDoubleComplex *Rb_gpu[], cufftDoubleComplex *resultGauss[], cufftDoubleReal *cuFFToutGauss[], 
	uint64_t invPrimes[], uint64_t primes[], int &nx, uint64_t *powers_gpu[], uint64_t *e_gaussianSample[], int *gpuOffsetFlags, int offset, cudaStream_t streamGauss[], cudaStream_t streamSmall[], cufftHandle planGauss[], cufftHandle invPlanGauss[],
	int numGPU, int numPrimes);

__global__ void dotProductionComplex(cufftDoubleComplex *x, cufftDoubleComplex *y, cufftDoubleComplex *result, int length);

__global__ void dotProductionReal(uint64_t *x, uint64_t *y, uint64_t *result, int length);

__global__ void dotProduct_32Complex(cufftComplex *x, cufftComplex *y, cufftDoubleComplex *result, int length);

__global__ void normalization (cufftDoubleReal *x, int length, uint64_t p);

// __global__ void barrettMul (double *a, uint64_t *b, uint64_t *c, int *offsetFlags, int *offset, uint64_t m, uint64_t inv_m, long length, long n);

__global__ void barrettMul (double *a, uint64_t *b, uint64_t *c, int *offsetFlags, int offset, uint64_t m, uint64_t inv_m, long length, long n);

__global__ void barrettMul_NTT (uint32_t *a, uint64_t *b, uint64_t *c, uint64_t m, uint64_t inv_m, long length, long n);

__global__ void generalBarrettMul (uint64_t *a, uint64_t *b, uint64_t *c, uint64_t m, uint64_t inv_m, long length);

__global__ void deviceAddMod (uint64_t *a, uint64_t *b, uint64_t *c, uint64_t m, long length);

__global__ void mulModWithConst (uint64_t *a, uint64_t b, uint64_t *c, uint64_t m, uint64_t inv_m, long length);


void mulModPtxtspace(long phiM, long numPrimes, cudaStream_t stream[], uint64_t *cipherOnGPU[], uint64_t ptxtSpace, uint64_t primes[], uint64_t invPrimes[]);


void ithPartAddMod (long phiM, long numPrimes, cudaStream_t stream[], uint64_t *cipherOnGPU[], uint64_t *bluesteinOutput2[], uint64_t primes[], int ithPart);


void readAndCheck(cufftDoubleReal *&cuFFToutput, uint64_t *&powers_gpu,  uint64_t *&bluesteinOutput, int &nx, int n, cudaStream_t &stream);


void generalMulMod (long phiM, long numPrimes, int numOfParts, cudaStream_t stream[], uint64_t *cipherOnGPU[], uint64_t *bluesteinOutput2[], uint64_t primes[], uint64_t invPrimes[]);


#endif /* NEWBLUESTEINFFT_H_ */
