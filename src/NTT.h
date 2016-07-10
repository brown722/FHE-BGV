#ifndef _NTT_H_
#define _NTT_H_


#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
#include <cufft.h>
#include "cuda.h"
#include <cuda_runtime.h>
#include <cufftXt.h>

NTL_CLIENT


void initNtt(int length) ;

uint64_t **ptrNttSwap();
uint64_t *ptrNttSwap(int dev) ;

void _ntt(uint64_t *X, uint32_t *x, int dev, cudaStream_t st, uint64_t length);
void _intt(uint32_t *x, uint64_t *X, uint32_t crtidx, int dev, cudaStream_t st, uint64_t length);

#endif
