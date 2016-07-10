/*
 * NewBluesteinFFT.cpp
 *
 *  Created on: 2015年12月8日
 *      Author: xhuang
 */

#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <stdint.h>
#include <NTL/mat_lzz_p.h>
#include <NTL/lzz_p.h>
#include <NTL/vector.h>
#include "NewBluesteinFFT.h"
#include <NTL/tools.h>
#include "ModP.h"
#include "testBluestein.h"
#include "NewBluesteinFFT.h"
#include "NTT.h"


#define N      0xFFFFFFFF00000001ull
#define ROOT   0xE9653C8DEFA860A9ull
#define INVERSE 1

using namespace std;
using namespace NTL;
using namespace cuHE;

//to compute the Bn (with length of 2N-1) in bluestein FFT

void gpuBlustnInit(long n, const zz_p& root, zz_pX& powers,
        Vec<mulmod_precon_t>& powers_aux, double2 *&cuRbBuffer, //cudaFree();
        uint64_t *&powers_gpu/*cudaFree();*/, uint64_t &invP)
{
	long p = zz_p::modulus();
	invP = 18446744073709551615/p;
	zz_p one;
	one = 1;
	
	
	long k = NextPowerOfTwo(2*n - 1);// Find the least k that was 2^k >= 2*n - 1;
	long k2 = 1L << k; //k2 = 2^k;

	powers.SetMaxLength(n);
	SetCoeff(powers, 0, one);
	

	uint64_t *powersTemp;//free(powersTemp)
	powersTemp = (uint64_t *)malloc(k2 * sizeof(uint64_t));
	memset(powersTemp, 0, sizeof(uint64_t) * k2);
	cout<<cudaMalloc((void**)&powers_gpu, k2 * sizeof(uint64_t))<<endl;
		

	for (long i = 1; i<n; i++)
	{
		long iSqr = MulMod(i, i, 2*n);// i^2 mod 2n
		SetCoeff(powers, i, power(root, iSqr));// powers[i] = root^(i^2)
		powersTemp[i] = rep(power(root, iSqr));
	}
	powersTemp[0] = 1;



	cout<<cudaMemcpy(powers_gpu, powersTemp, k2 * sizeof(uint64_t), cudaMemcpyHostToDevice)<<endl;
	free(powersTemp);

	powers_aux.SetLength(n);

	//uint64_t *powers_auxTemp;//free (powers_auxTemp);
	//powers_auxTemp = (uint64_t *)malloc(n * sizeof(uint64_t));
	//cudaMalloc((void**)&powers_aux_gpu, n * sizeof(uint64_t));
	for(long i = 0; i < n; i ++)
	{
		powers_aux[i] = PrepMulModPrecon(rep(powers[i]), p);
	}
	
	//cudaMemcpy(powers_aux_gpu, powers_auxTemp, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
	//free (powers_auxTemp);



	zz_pX b(INIT_SIZE, k2);
	zz_p rInv = inv(root);

	SetCoeff(b, n-1, one);

	for(long i = 1; i < n; i++)
	{
		long iSqr = MulMod(i, i, 2*n); // i^2 mod 2n
		zz_p bi = power(rInv,iSqr);
		SetCoeff(b,n-1+i, bi); // b[n-1+i] = b[n-1-i] = root^{-i^2}
		SetCoeff(b,n-1-i,bi);
	}

	ZZX b_ZZX;
	conv(b_ZZX, b);
	b_ZZX.SetLength(k2);

	int nx, batch;// set up parameters of cufft fucntions
	nx = k2/1; batch = 1;

	cufftDoubleReal *inputBuffer = (cufftDoubleReal *)malloc(nx * sizeof(cufftDoubleReal));

	cufftDoubleReal *cuInputBuffer;
	cudaMalloc((void **)&cuInputBuffer, nx * sizeof(cufftDoubleReal));
	cudaMalloc((void **)&cuRbBuffer, (nx/2+1) * sizeof(double2));


	for (int i = 0; i < k2; i ++)
	{
		uint64_t tempBuffer;
	 	conv(tempBuffer, b_ZZX[i]);
	 	inputBuffer[i] = tempBuffer;
	}
	


	cudaMemset(cuInputBuffer, 0, sizeof(cufftDoubleReal) * nx);
	cudaMemcpy(cuInputBuffer, inputBuffer, nx * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);

	
	cufftHandle plan;
	cufftPlan1d(&plan, nx, CUFFT_D2Z, batch);
	cufftExecD2Z(plan, cuInputBuffer, cuRbBuffer);
	
	cudaFree(cuInputBuffer);
	cufftDestroy(plan);
	free(inputBuffer);
}

void memoryAllocations(zz_pX& x, long n, const zz_pX& powers, const Vec<mulmod_precon_t>& powers_aux, int &nx,
	uint64_t *&bluesteinOutput, cufftDoubleReal *&inputBuffer, cufftDoubleReal *&cuInputBuffer, cufftDoubleComplex *&cuRaBuffer, 
	cufftDoubleComplex *&result, cufftHandle &plan, cufftDoubleReal *&cuFFToutput, cufftHandle &planInverse, int *&offset, cudaStream_t &stream)
{

	long p = zz_p::modulus();

	if (IsZero(x)) return;
	if (n<=0)
	{
		clear(x);
		return;
	}
	long k = NextPowerOfTwo(2*n-1);
	long k2 = 1L<<k;
	nx = k2/1;
	int batch = 1;

	inputBuffer = (cufftDoubleReal *)malloc(k2 * sizeof(cufftDoubleReal));
	memset(inputBuffer, 0, sizeof(cufftDoubleReal) * k2);
	long dx = deg(x);
	for (long i=0; i<=dx; i++){
		inputBuffer[i] = MulModPrecon(rep(x[i]), rep(powers[i]), p, powers_aux[i]);
	}
	x.normalize();

	cudaMalloc(&cuInputBuffer, nx * sizeof(cufftDoubleReal));

	cudaMalloc(&cuRaBuffer, sizeof(cufftDoubleComplex)*(nx/2+1) * batch);
	cudaMalloc(&result, sizeof(cufftDoubleComplex)*(nx/2+1) * batch);

	cufftPlan1d(&plan, nx, CUFFT_D2Z, batch);
	cufftSetStream(plan, stream);


	cudaMalloc(&cuFFToutput, nx * sizeof(cufftDoubleReal));
	

	cufftPlan1d(&planInverse, nx, CUFFT_Z2D, batch);
	cufftSetStream(planInverse, stream);
	
	cudaMalloc(&bluesteinOutput, nx * sizeof(uint64_t));
	
	// cudaMalloc(&offset, nx * sizeof(int));
	// cudaMemset (offset, 0, nx * sizeof(int));
}

void devBluestn2(long mm, cufftDoubleReal *gaussBuffer[], cufftDoubleReal *cuGaussBuffer[],  cufftDoubleComplex *cuRaGauss[], cufftDoubleComplex *Rb_gpu[], cufftDoubleComplex *resultGauss[], cufftDoubleReal *cuFFToutGauss[], 
	uint64_t invPrimes[], uint64_t primes[], int &nx, uint64_t *powers_gpu[], uint64_t *e_gaussianSample[], int *gpuOffsetFlags, int offset, cudaStream_t streamGauss[], cudaStream_t streamSmall[], cufftHandle planGauss[], cufftHandle invPlanGauss[],
	int numGPU, int numPrimes)
{
	for (int i = 0; i < numPrimes; i ++)
	{
		deviceBlustn2(mm, gaussBuffer[i], cuGaussBuffer[i],  cuRaGauss[i], Rb_gpu[i], resultGauss[i], cuFFToutGauss[i], 
			invPrimes[i], primes[i], nx, powers_gpu[i], e_gaussianSample[i], gpuOffsetFlags, offset, streamGauss[i], streamSmall[i],planGauss[i],  invPlanGauss[i], numGPU);
	}
}
void deviceBlustn2(long n, cufftDoubleReal *&inputBuffer,cufftDoubleReal *&cuInputBuffer,cufftDoubleComplex *&cuRaBuffer,cufftDoubleComplex *&cuRbBuffer, 
	cufftDoubleComplex *&result, cufftDoubleReal *&cuFFToutput, uint64_t &invP, uint64_t p, int &nx, uint64_t *&powers_gpu,  uint64_t *&bluesteinOutput, 
	int *&dropFlags,int offset, cudaStream_t &stream, cudaStream_t &stream1, cufftHandle &plan,cufftHandle &planInverse, int numGPU)
{
	cudaMemcpyAsync(cuInputBuffer, inputBuffer, nx * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice, stream1);

	
	cufftExecD2Z(plan, cuInputBuffer, cuRaBuffer);
	
	dotProductionComplex<<<((nx/2+1)/512) + 1, 512, numGPU, stream>>>(cuRaBuffer, cuRbBuffer, result, (nx/2+1));


	cufftExecZ2D(planInverse, result, cuFFToutput);


	normalization<<<(nx/512) + 1, 512, numGPU, stream>>> (cuFFToutput, nx, p);

	
	barrettMul<<<(nx/512) + 1, 512, numGPU, stream>>> (cuFFToutput, powers_gpu, bluesteinOutput, dropFlags, offset, p, invP, nx, n);

	cudaStreamSynchronize(stream);


}

void readAndCheck(cufftDoubleReal *&cuFFToutput, uint64_t *&powers_gpu,  uint64_t *&bluesteinOutput, int &nx, int n, cudaStream_t &stream)
{
	cudaDeviceSynchronize();
	uint64_t *bluesteinOutputTemp, *gpuPowersTemp;
	double  *cufftoutTemp;
	gpuPowersTemp = (uint64_t *)malloc(nx * sizeof(uint64_t));
	bluesteinOutputTemp = (uint64_t *)malloc(nx * sizeof(uint64_t));
	cufftoutTemp = (double *)malloc(nx * sizeof(double));
	cudaMemcpy(cufftoutTemp, cuFFToutput, nx * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(gpuPowersTemp, powers_gpu, nx * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(bluesteinOutputTemp, bluesteinOutput, nx * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaStreamSynchronize(stream);

	cout<<"in-fucntion output is: "<<endl<<endl;
	for (int i = 0; i < 13; i ++)
		cout<<bluesteinOutputTemp[i]<<", "<<(int64_t)cufftoutTemp[i + n - 1]<<", "<<gpuPowersTemp[i]<<endl;
	cout<<endl;

	free(bluesteinOutputTemp);
	free(cufftoutTemp);
	free(gpuPowersTemp);
}

// void BlustnOnGPU1(zz_pX& x, long n, double2 *&cuRbBuffer/*cudaFree(cuRbBuffer);*/,
// 	const zz_pX& powers, const Vec<mulmod_precon_t>& powers_aux, uint64_t *&powers_gpu/*cudaFree(powers_gpu);*/,
//     uint64_t *&bluesteinOutput/*cudaFree();*/, uint64_t &invP, long p, int *&dropFlags, cudaStream_t &stream)
// {

// 	cout<<endl<<"ON stram: " << stream <<" ,FFT for prime: "<<p<<endl;

// 	if (IsZero(x)) return;
// 	if (n<=0)
// 	{
// 		clear(x);
// 		return;
// 	}
// 	long k = NextPowerOfTwo(2*n-1);
// 	long k2 = 1L<<k;
// 	int nx = k2/1;
// 	int batch = 1;

// 	cufftDoubleReal * inputBuffer;
// 	inputBuffer = (cufftDoubleReal *)malloc(k2 * sizeof(cufftDoubleReal));
// 	memset(inputBuffer, 0, sizeof(cufftDoubleReal) * k2);
// 	long dx = deg(x);
// 	for (long i=0; i<=dx; i++){
// 		inputBuffer[i] = MulModPrecon(rep(x[i]), rep(powers[i]), p, powers_aux[i]);
// 	}
// 	x.normalize();


// 	cufftDoubleReal * cuInputBuffer; 
// 	cudaMalloc(&cuInputBuffer, nx * sizeof(cufftDoubleReal));

// 	cufftDoubleComplex *cuRaBuffer, *result;
// 	cudaMalloc(&cuRaBuffer, sizeof(cufftDoubleComplex)*(nx/2+1) * batch);
// 	cudaMalloc(&result, sizeof(cufftDoubleComplex)*(nx/2+1) * batch);

// 	cufftHandle plan;
// 	cufftSetStream(plan, stream);
// 	cufftPlan1d(&plan, nx, CUFFT_D2Z, batch);

// 	cufftDoubleReal *cuFFToutput;
// 	cudaMalloc(&cuFFToutput, nx * sizeof(cufftDoubleReal));
// 	cufftHandle planInverse;
// 	cufftSetStream(planInverse, stream);
// 	cufftPlan1d(&planInverse, nx, CUFFT_Z2D, batch);
	
// 	cudaMalloc(&bluesteinOutput, nx * sizeof(uint64_t));
// 	int *offset;
// 	cudaMalloc(&offset, nx * sizeof(int));
// 	cudaMemset (offset, 0, nx * sizeof(int));

// 	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 	cudaMemcpyAsync(cuInputBuffer, inputBuffer, nx * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice, stream);
	
// 	cufftExecD2Z(plan, cuInputBuffer, cuRaBuffer);
	
// 	dotProductionComplex<<<((nx/2+1)/512) + 1, 512, 0, stream>>>(cuRaBuffer, cuRbBuffer, result, (nx/2+1));

// 	cufftExecZ2D(planInverse, result, cuFFToutput);

// 	normalization<<<(nx/512) + 1, 512, 0, stream>>> (cuFFToutput, nx, p);
	
// 	barrettMul<<<(nx/512) + 1, 512, 0, stream>>> (cuFFToutput, powers_gpu, bluesteinOutput, dropFlags, offset, p, invP, nx, n);
	
// 	/*****************cout and check the correctness of the output******************/
// 	uint64_t *bluesteinOutputTemp, *gpuPowersTemp;
// 	double  *cufftoutTemp;
// 	gpuPowersTemp = (uint64_t *)malloc(nx * sizeof(uint64_t));
// 	bluesteinOutputTemp = (uint64_t *)malloc(nx * sizeof(uint64_t));
// 	cufftoutTemp = (double *)malloc(nx * sizeof(double));
// 	cudaMemcpyAsync(cufftoutTemp, cuFFToutput, nx * sizeof(double), cudaMemcpyDeviceToHost, stream);
// 	cudaMemcpyAsync(gpuPowersTemp, powers_gpu, nx * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
// 	cudaMemcpyAsync(bluesteinOutputTemp, bluesteinOutput, nx * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
	
// 	cout<<"in-fucntion output is: "<<endl<<endl;
// 	for (int i = 0; i < 10; i ++)
// 		cout<<bluesteinOutputTemp[i]<<", "<<(int64_t)cufftoutTemp[i + n - 1]<<", "<<gpuPowersTemp[i]<<endl;
// 	cout<<endl;

// 	free(bluesteinOutputTemp);
// 	free(cufftoutTemp);
// 	free(gpuPowersTemp);

// 	/*recycling the used memories*/
// 	cudaFree(offset);
// 	cufftDestroy(plan);cufftDestroy(planInverse);
// 	cudaFree(cuInputBuffer);cudaFree(cuRaBuffer);
// 	cudaFree(result);cudaFree(cuFFToutput);
// 	free(inputBuffer);
// 	cudaFree(cuRbBuffer);
// 	cudaFree(powers_gpu);
// 	cudaFree(bluesteinOutput);

// }

// void nttBlustnInit(long n, const zz_p& root, zz_pX& powers,
//         Vec<mulmod_precon_t>& powers_aux, uint64_t *&cuRbBuffer, //cudaFree();
//         uint64_t *&powers_gpu/*cudaFree();*/, uint64_t *&powers_aux_gpu/*cudaFree();*/,
//         uint64_t &invP)
// {
// 	long p = zz_p::modulus();
// 	invP = 18446744073709551615/p;
// 	zz_p one;
// 	one = 1;

// 	long k = NextPowerOfTwo(2*n - 1);// Find the least k that was 2^k >= 2*n - 1;
// 	long k2 = 1L << k; //k2 = 2^k;

// 	powers.SetMaxLength(n);
// 	SetCoeff(powers, 0, one);
	
// 	uint64_t *powersTemp;//free(powersTemp)
// 	powersTemp = (uint64_t *)malloc(k2 * sizeof(uint64_t));
// 	memset(powersTemp, 0, sizeof(uint64_t) * k2);
// 	cudaMalloc((void**)&powers_gpu, k2 * sizeof(uint64_t));
	
	

// 	for (long i = 1; i<n; i++)
// 	{
// 		long iSqr = MulMod(i, i, 2*n);// i^2 mod 2n
// 		SetCoeff(powers, i, power(root, iSqr));// powers[i] = root^(i^2)
// 		powersTemp[i] = rep(power(root, iSqr));
// 	}
// 	powersTemp[0] = 1;

// 	cudaMemcpy(powers_gpu, powersTemp, k2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
// 	free(powersTemp);

// 	powers_aux.SetLength(n);

// 	uint64_t *powers_auxTemp;//free (powers_auxTemp);
// 	powers_auxTemp = (uint64_t *)malloc(n * sizeof(uint64_t));
// 	cudaMalloc((void**)&powers_aux_gpu, n * sizeof(uint64_t));
// 	for(long i = 0; i < n; i ++)
// 	{
// 		powers_aux[i] = PrepMulModPrecon(rep(powers[i]), p);
// 		powers_auxTemp[i] = PrepMulModPrecon(rep(powers[i]), p);
// 	}
// 	cout<<endl<<"the original powers aux"<<endl;
// 	for (int i = 0; i < 10; i++)
// 	{
// 		cout<<powers_aux[i]<<", ";
// 	}
// 	cout<<endl;
// 	cout<<endl<<"the 0 powers aux"<<endl;
// 	for (int i = 0; i < 10; i++)
// 	{
// 		cout<<powers_auxTemp[i]<<", ";
// 	}
// 	cout<<endl;

// 	cudaMemcpy(powers_aux_gpu, powers_auxTemp, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
// 	free (powers_auxTemp);



// 	zz_pX b(INIT_SIZE, k2);
// 	zz_p rInv = inv(root);

// 	SetCoeff(b, n-1, one);

// 	for(long i = 1; i < n; i++)
// 	{
// 		long iSqr = MulMod(i, i, 2*n); // i^2 mod 2n
// 		zz_p bi = power(rInv,iSqr);
// 		SetCoeff(b,n-1+i, bi); // b[n-1+i] = b[n-1-i] = root^{-i^2}
// 		SetCoeff(b,n-1-i,bi);
// 	}

// 	ZZX b_ZZX;
// 	conv(b_ZZX, b);
// 	b_ZZX.SetLength(k2);

// 	int nx, batch;// set up parameters of cufft fucntions
// 	nx = k2/1; batch = 1;

// 	uint32_t *inputBuffer = (uint32_t *)malloc(nx * sizeof(uint32_t));

// 	uint32_t * cuInputBuffer;
// 	cudaMalloc((void**)&cuInputBuffer, nx * sizeof(uint32_t));
// 	cudaMalloc((void**)&cuRbBuffer, nx * sizeof(uint64_t));


// 	for (int i = 0; i < k2; i ++)
// 	{
// 		uint64_t tempBuffer;
// 	 	conv(tempBuffer, b_ZZX[i]);
// 	 	inputBuffer[i] = (uint32_t)tempBuffer;
// 	}
	
// 	cudaMemcpy(cuInputBuffer, inputBuffer, nx * sizeof(uint32_t), cudaMemcpyHostToDevice);

// 	initNtt(k2) ;
// 	_ntt(cuRbBuffer, cuInputBuffer, 0, 0, k2);
	
// 	cudaFree(cuInputBuffer);
// 	free(inputBuffer);
// }

// void BlustnWithNTT(zz_pX& x, long n, const zz_p& root,
// 	const zz_pX& powers, const Vec<mulmod_precon_t>& powers_aux,
// 	uint64_t *&cuRbBuffer/*cudaFree(cuRbBuffer);*/,double *&cuFFToutput/*cudaFree(cuFFToutput);*/,
// 	uint64_t *&powers_gpu/*cudaFree(powers_gpu);*/, uint64_t *&powers_aux_gpu/*cudaFree(powers_aux_gpu);*/,
// 	uint64_t *&bluesteinOutput/*cudaFree();*/, uint64_t &invP, double &Time)
// {
// 	Time = GetTime();
	
// 	if (IsZero(x)) return;
// 	if (n<=0)
// 	{
// 		clear(x);
// 		return;
// 	}
// 	long p = zz_p::modulus();

// 	long dx = deg(x);
// 	long k = NextPowerOfTwo(2*n-1);
// 	long k2 = 1L<<k;

// 	uint32_t * inputBuffer;
// 	inputBuffer = (uint32_t *)malloc(k2 * sizeof(uint32_t));//

// 	for (long i=0; i<=dx; i++)
// 		inputBuffer[i] = (uint32_t)MulModPrecon(rep(x[i]), rep(powers[i]), p, powers_aux[i]);

// 	x.SetLength(k2);
// 	int nx = k2/1;



// 	uint32_t *nttInputBuffer;
// 	cudaMalloc((void**)&nttInputBuffer, nx * sizeof(uint32_t));//
// 	cudaMemcpy(nttInputBuffer, inputBuffer, nx * sizeof(uint32_t), cudaMemcpyHostToDevice);
	
// 	uint64_t *nttRaBuffer;
// 	cudaMalloc((void**)&nttRaBuffer, nx * sizeof(uint64_t));//
	
// 	initNtt(nx) ;
// 	_ntt(nttRaBuffer, nttInputBuffer, 0, 0, nx);


// 	uint64_t *product;
// 	cudaMalloc((void**)&product, nx * sizeof(uint64_t));//
	
// 	dotProductionReal<<<nx/512, 512>>>(nttRaBuffer, cuRbBuffer, product, nx);

// 	cudaDeviceSynchronize();
	
// 	uint32_t *result;
// 	cudaMalloc((void**)&result, nx * sizeof(uint32_t));//

// 	_intt(result, product, p, 0, 0, nx);

// 	cudaDeviceSynchronize();

// 	cudaMalloc((void**)&bluesteinOutput, nx * sizeof(uint64_t));
	
// 	barrettMul_NTT<<<(nx/512),512>>> (result, powers_gpu, bluesteinOutput, p, invP, nx, n);



// 	Time = GetTime() - Time;

// 	/*****************cout and check the correctness of the output******************/
// 	// uint64_t *bluesteinOutputTemp;
// 	// bluesteinOutputTemp = (uint64_t *)malloc(nx * sizeof(uint64_t));
// 	// cout<<endl<<cudaMemcpy(bluesteinOutputTemp, bluesteinOutput, nx * sizeof(uint64_t), cudaMemcpyDeviceToHost)<<endl;
	
// 	// cout<<endl<<"in-fucntion output is: "<<endl;
// 	// for (int i = 0; i < 30; i ++)
// 	// 	cout<<bluesteinOutputTemp[i]<<", ";
// 	// cout<<endl;
// 	// free(bluesteinOutputTemp);


// free(inputBuffer);
// cudaFree(nttInputBuffer);
// cudaFree(nttRaBuffer);
// cudaFree(product);
// cudaFree(result);
// }

__global__ void dotProductionComplex(cufftDoubleComplex *x, cufftDoubleComplex *y, cufftDoubleComplex *result, int length)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	cufftDoubleComplex _x, _y, _result;

	if ((index < length)){
		_x = x[index];
		_y = y[index];
		// // cufftDoubleComplex _x, _y, _result;
		// _x.x = x[index].x;
		// _x.y = x[index].y;
		// _y.x = y[index].x;
		// _y.y = y[index].y;

		// _result.x = _x.x*_y.x - _x.y*_y.y;
		// _result.y = _x.x*_y.y + _x.y * _y.x;

		// result[index].x = _result.x;
		// result[index].y = _result.y;
		_result = cuCmul(_x, _y);

		result[index] = _result;
	}
}

__global__ void dotProductionReal(uint64_t *x, uint64_t *y, uint64_t *result, int length)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ uint64_t _x, _y, _result;

	if ((index < length)){
		_x = x[index];
		_y = y[index];

		_result = _mul_modP(_x, _y);

		result[index] = _result;
	}
}

__global__ void normalization (cufftDoubleReal *x, int length, uint64_t p){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ cufftDoubleReal _x;
	if (index < length)
	{
		_x = x[index];
		_x = (uint64_t)(_x/length)%p;
		x[index] = _x;
	}
}


__global__ void barrettMul (double *a, uint64_t *b, uint64_t *c, int *offsetFlags, int offset, uint64_t m, uint64_t inv_m, long length, long n)
{
	register int index = threadIdx.x + blockIdx.x * blockDim.x;
	register uint64_t _a, _b, _t,_y, _h, _z, _r, _x;
	register int _offset = 0;

	if (index < length)
	{
		// while(offsetFlags[_offset]<= index && offsetFlags[_offset] >= 0)
		// {
		// 	if (index == offsetFlags[_offset])
		// 		_offset = 0;
		// 	_offset++;
		// }

		if (!(index % offset))
			_offset = 0;
		else
			_offset = index/offset + 1;
		
		if (offset==1)
		{
			if (index == 0)
				_offset = 0;
			else
				_offset = 1;
		}

		_a = (uint64_t)a[index + n - 1];

		_b = b[index];
//
//		 _t = _mul_modP(_a, _b);//modular multiplication in finite field
//
//		 _h = _t >> 32;
//
//		 // _y = ((uint64_t)_h*(uint64_t)inv_m)>>32;
//		 _y = _mul_modP((uint64_t)_h, (uint64_t)inv_m)>>32;
//
//		 // _z = (uint64_t)_y*(uint64_t)m;
//		 _z = _mul_modP((uint64_t)_y, (uint64_t)m);
//
//		 // _r = _t - _z;
//		 _r = _sub_modP(_t, _z);
//
//		 //while(_r >= m) _r = _sub_modP(_r, m);
//		 _r = _r % m;
//
//		 _x = _r;

		_x = (_a * _b)% m;

		// c[index - _offset] = _x;

		atomicExch((unsigned long long int*)&c[index - _offset], (unsigned long long int)_x);

	}
}

__global__ void barrettMul_NTT (uint32_t *a, uint64_t *b, uint64_t *c, uint64_t m, uint64_t inv_m, long length, long n)
{
	register int index = threadIdx.x + blockIdx.x * blockDim.x;
	register uint64_t _a, _b, _t,_y, _h, _z, _r, _x;

	if (index < length)
	{
		_a = (uint64_t)a[index + n - 1];
		
		_b = b[index];
		
		_t = _mul_modP(_a, _b);//modular multiplication in finite field
		
		_h = _t >> 32;

		// _y = ((uint64_t)_h*(uint64_t)inv_m)>>32;
		_y = _mul_modP((uint64_t)_h, (uint64_t)inv_m)>>32;

		// _z = (uint64_t)_y*(uint64_t)m;
		_z = _mul_modP((uint64_t)_y, (uint64_t)m);

		// _r = _t - _z;
		_r = _sub_modP(_t, _z);
		
		while(_r >= m) _r = _sub_modP(_r, m);
		// _r = _r % m;
		
		_x = _r;
		
		c[index] = _x;
	}
}

__global__ void generalBarrettMul (uint64_t *a, uint64_t *b, uint64_t *c, uint64_t m, uint64_t inv_m, long length)
{
	register int index = threadIdx.x + blockIdx.x * blockDim.x;
	register uint64_t _a, _b, _t,_y, _h, _z, _r, _x;

	if (index < length)
	{
		_a = (uint64_t)a[index];

		_b = b[index];
		
//		 _t = _mul_modP(_a, _b);//modular multiplication in finite field
//
//		 _h = _t >> 32;
//
//		 // _y = ((uint64_t)_h*(uint64_t)inv_m)>>32;
//		 _y = _mul_modP((uint64_t)_h, (uint64_t)inv_m)>>32;
//
//		 // _z = (uint64_t)_y*(uint64_t)m;
//		 _z = _mul_modP((uint64_t)_y, (uint64_t)m);
//
//		 // _r = _t - _z;
//		 _r = _sub_modP(_t, _z);
//
//		 // while(_r >= m) _r = _sub_modP(_r, m);
//		 _r = _r % m;
//
//		 _x = _r;
//
		_x = (_a * _b)% m;

		c[index] = _x;

//		atomicExch((unsigned long long int*)&c[index], (unsigned long long int)_x);
	}
}

__global__ void deviceAddMod (uint64_t *a, uint64_t *b, uint64_t *c, uint64_t m, long length)
{
	register int index = threadIdx.x + blockIdx.x * blockDim.x;
	register uint64_t _a, _b, _c;

	if (index < length)
	{
		_a = (uint64_t)a[index];
		_b = b[index];
		_c = _add_modP( _a, _b) % m;
		c[index] = _c;
//		atomicExch((unsigned long long int*)&c[index], (unsigned long long int)_c);
	}
}

__global__ void mulModWithConst (uint64_t *a, uint64_t b, uint64_t *c, uint64_t m, uint64_t inv_m, long length)
{
	register int index = threadIdx.x + blockIdx.x * blockDim.x;
	register uint64_t _a, _b, _t,_y, _h, _z, _r, _x;

	if (index < length)
	{
		_a = (uint64_t)a[index];

		_b = b % m;

//
//		 _t = _mul_modP(_a, _b);//modular multiplication in finite field
//
//		 _h = _t >> 32;
//
//		 // _y = ((uint64_t)_h*(uint64_t)inv_m)>>32;
//		 _y = _mul_modP((uint64_t)_h, (uint64_t)inv_m)>>32;
//
//		 // _z = (uint64_t)_y*(uint64_t)m;
//		 _z = _mul_modP((uint64_t)_y, (uint64_t)m);
//
//		 // _r = _t - _z;
//		 _r = _sub_modP(_t, _z);
//
//		 // while(_r >= m) _r = _sub_modP(_r, m);
//		 _r = _r % m;
//
//		 _x = _r;

		_x = (_a * _b) % m;
		
		c[index] = _x;

//		atomicExch((unsigned long long int*)&c[index], (unsigned long long int)_x);
	}
}

void mulModPtxtspace(long phiM, long numPrimes, cudaStream_t stream[], uint64_t *cipherOnGPU[], uint64_t ptxtSpace, uint64_t primes[], uint64_t invPrimes[])
{
	for (int i = 0; i < numPrimes; i ++)
	{
		mulModWithConst <<<(phiM/512) + 1, 512, 0, stream[i]>>> (cipherOnGPU[i], ptxtSpace, cipherOnGPU[i], primes[i],invPrimes[i],phiM);
	}
}


void ithPartAddMod (long phiM, long numPrimes, cudaStream_t stream[], uint64_t *cipherOnGPU[], uint64_t *bluesteinOutput2[], uint64_t primes[], int ithPart)
{
	for (int i = 0; i < numPrimes; i ++)
	{
		deviceAddMod <<<(phiM/512) + 1, 512, 0, stream[i]>>> (cipherOnGPU[ithPart * numPrimes + i], bluesteinOutput2[i], cipherOnGPU[ithPart * numPrimes + i ], primes[i], phiM);
	}
}




void generalMulMod (long phiM, long numPrimes, int numOfParts, cudaStream_t stream[], uint64_t *cipherOnGPU[], uint64_t *bluesteinOutput2[], uint64_t primes[], uint64_t invPrimes[])
{

	for (int i = 0; i < numPrimes * numOfParts; i ++)
	{
		generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[i]>>> (cipherOnGPU[i], bluesteinOutput2[i % numPrimes], cipherOnGPU[i], primes[i % numPrimes],invPrimes[i % numPrimes],phiM);
	}

// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[0]>>> (cipherOnGPU[0], bluesteinOutput2[0], cipherOnGPU[0], primes[0], invPrimes[0], phiM, 0 * (numPrimes * phiM) + (0 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[1]>>> (cipherOnGPU[1], bluesteinOutput2[1], cipherOnGPU[1], primes[1], invPrimes[1], phiM, 0 * (numPrimes * phiM) + (1 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[2]>>> (cipherOnGPU[2], bluesteinOutput2[2], cipherOnGPU[2], primes[2], invPrimes[2], phiM, 0 * (numPrimes * phiM) + (2 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[3]>>> (cipherOnGPU[3], bluesteinOutput2[3], cipherOnGPU[3], primes[3], invPrimes[3], phiM, 0 * (numPrimes * phiM) + (3 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[4]>>> (cipherOnGPU[4], bluesteinOutput2[4], cipherOnGPU[4], primes[4], invPrimes[4], phiM, 0 * (numPrimes * phiM) + (4 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[5]>>> (cipherOnGPU[5], bluesteinOutput2[5], cipherOnGPU[5], primes[5], invPrimes[5], phiM, 0 * (numPrimes * phiM) + (5 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[6]>>> (cipherOnGPU[6], bluesteinOutput2[6], cipherOnGPU[6], primes[6], invPrimes[6], phiM, 0 * (numPrimes * phiM) + (6 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[7]>>> (cipherOnGPU[7], bluesteinOutput2[7], cipherOnGPU[7], primes[7], invPrimes[7], phiM, 0 * (numPrimes * phiM) + (7 * phiM));
	
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[8]>>> (cipherOnGPU[8], bluesteinOutput2[0], cipherOnGPU[8], primes[0], invPrimes[0], phiM, 1 * (numPrimes * phiM) + (0 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[9]>>> (cipherOnGPU[9], bluesteinOutput2[1], cipherOnGPU[9], primes[1], invPrimes[1], phiM, 1 * (numPrimes * phiM) + (1 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[10]>>> (cipherOnGPU[10], bluesteinOutput2[2], cipherOnGPU[10], primes[2], invPrimes[2], phiM, 1 * (numPrimes * phiM) + (2 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[11]>>> (cipherOnGPU[11], bluesteinOutput2[3], cipherOnGPU[11], primes[3], invPrimes[3], phiM, 1 * (numPrimes * phiM) + (3 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[12]>>> (cipherOnGPU[12], bluesteinOutput2[4], cipherOnGPU[12], primes[4], invPrimes[4], phiM, 1 * (numPrimes * phiM) + (4 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[13]>>> (cipherOnGPU[13], bluesteinOutput2[5], cipherOnGPU[13], primes[5], invPrimes[5], phiM, 1 * (numPrimes * phiM) + (5 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[14]>>> (cipherOnGPU[14], bluesteinOutput2[6], cipherOnGPU[14], primes[6], invPrimes[6], phiM, 1 * (numPrimes * phiM) + (6 * phiM));
// 	generalBarrettMul <<<(phiM/512) + 1, 512, 0, stream[15]>>> (cipherOnGPU[15], bluesteinOutput2[7], cipherOnGPU[15], primes[7], invPrimes[7], phiM, 1 * (numPrimes * phiM) + (7 * phiM));
}
