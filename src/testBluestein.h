#ifndef _TESTBLUSTN_H_
#define _TESTBLUSTN_H_

// #include "timing.h"
// #include "CModulus.h"
// #include <stdint.h>
// #include <NTL/tools.h>

using namespace NTL;
using namespace std;

uint64_t BarretMulMod(uint64_t t,uint64_t m/* uint32_t m*/, uint64_t inv_m);

uint64_t _add(uint64_t a, uint64_t b) ;

uint64_t _subtract(uint64_t a, uint64_t b) ;

uint64_t _multiply(uint64_t a, uint64_t b);

uint64_t _power(uint64_t a, uint64_t k);

uint64_t _normalize(uint64_t a);
///???????
void extendedGCD(uint64_t a, uint64_t b, int64_t *s, int64_t *t) ;

uint64_t _inverse(uint64_t a);

uint64_t _root(uint64_t size);

uint64_t _inverseRoot(uint64_t size);

void  transpose(uint64_t  *x, uint64_t  *X, uint32_t size, uint32_t xLength);

void smallFFT(uint64_t *x, uint64_t *X, uint32_t size, int inverse);

void transpose(uint64_t *x, uint64_t *X, uint32_t size, uint32_t xLength);

void largeFFT(uint64_t *x, uint64_t *X, uint32_t size, int inverse);


void OriginalFFT(zz_pX& x, long n, const zz_p& root,
		  const zz_pX& powers, const Vec<mulmod_precon_t>& powers_aux,
                 const fftRep& Rb, zz_pX& temp, double &Time);

#endif