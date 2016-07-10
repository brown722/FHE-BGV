#include "Base.h"
#include "DeviceManager.h"
#include "Debug.h"
#include "ModP.h"
#include "Operations.h"
#include <stdint.h>

using namespace NTL;

static uint64_t** d_roots;
texture<uint32_t, 1> tex_roots_16k;
texture<uint32_t, 1> tex_roots_32k;
texture<uint32_t, 1> tex_roots_64k;
static uint64_t **d_swap; // conversion buffer
static uint32_t **d_hold; // intt result buffer

void initNtt(int length) 
{
	// twiddle factors
	const ZZ P = to_ZZ(0xffffffff00000001);
	const ZZ g = to_ZZ((uint64_t)15893793146607301539);
	int e0 = 65536/length;
	ZZ w0 =	PowerMod(g, e0, P);
	uint64_t *h_roots = new uint64_t[length];
	for (int i=0; i<length; i++)
		conv(h_roots[i], PowerMod(w0, i, P));
	preload_ntt(h_roots, length);
	delete [] h_roots;
	// temporary result allocation
	d_swap = new uint64_t *[1];
	d_hold = new uint32_t *[1];
	for (int dev=0; dev<1; dev++) {
		cudaSetDevice(dev);
		CSC(cudaMalloc(&d_swap[dev], length*sizeof(uint64_t)));
		CSC(cudaMalloc(&d_hold[dev], 1*length*sizeof(uint32_t)));
	}
}

uint64_t **ptrNttSwap() { return d_swap;}
uint64_t *ptrNttSwap(int dev) { return d_swap[dev];}


void _ntt(uint64_t *X, uint32_t *x, int dev, cudaStream_t st, uint64_t length) 
{
	if (length == 16384) {
		ntt_1_16k_ext<<<length/512, 64, 0>>>(ptrNttSwap(dev), x);
		CCE();
		ntt_2_16k<<<length/512, 64, 0>>>(ptrNttSwap(dev));
		CCE();
		ntt_3_16k<<<length/512, 64, 0>>>(X, ptrNttSwap(dev));
		CCE();
	}
	else if (length == 32768) {
		ntt_1_32k_ext<<<length/512, 64, 0>>>(ptrNttSwap(dev), x);
		CCE();
		ntt_2_32k<<<length/512, 64, 0>>>(ptrNttSwap(dev));
		CCE();
		ntt_3_32k<<<length/512, 64, 0>>>(X, ptrNttSwap(dev));
		CCE();
	}
	else if (length == 65536) {
		ntt_1_64k_ext<<<length/512, 64, 0>>>(ptrNttSwap(dev), x);
		CCE();
		ntt_2_64k<<<length/512, 64, 0>>>(ptrNttSwap(dev));
		CCE();
		ntt_3_64k<<<length/512, 64, 0>>>(X, ptrNttSwap(dev));
		CCE();
	}
}

void _intt(uint32_t *x, uint64_t *X, uint32_t crtidx, int dev, cudaStream_t st, uint64_t length) {
	if (length == 16384) {
		intt_1_16k<<<length/512, 64, 0, st>>>(ptrNttSwap(dev), X);
		CCE();
		ntt_2_16k<<<length/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		intt_3_16k_modcrt<<<length/512, 64, 0, st>>>(x, ptrNttSwap(dev), crtidx);
		CCE();
	}
	else if (length == 32768) {
		intt_1_32k<<<length/512, 64, 0, st>>>(ptrNttSwap(dev), X);
		CCE();
		ntt_2_32k<<<length/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		intt_3_32k_modcrt<<<length/512, 64, 0, st>>>(x, ptrNttSwap(dev), crtidx);
		CCE();
	}
	else if (length == 65536) {
		intt_1_64k<<<length/512, 64, 0, st>>>(ptrNttSwap(dev), X);
		CCE();
		ntt_2_64k<<<length/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		intt_3_64k_modcrt<<<length/512, 64, 0, st>>>(x, ptrNttSwap(dev), crtidx);
		CCE();
	}
}

