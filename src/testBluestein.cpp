#include "timing.h"
#include "CModulus.h"
#include <stdint.h>
#include <NTL/tools.h>
#include "testBluestein.h"

#define _N      0xFFFFFFFF00000001ull
#define _ROOT   0xE9653C8DEFA860A9ull
#define INVERSE 1

using namespace NTL;
using namespace std;

uint64_t BarretMulMod(uint64_t t,uint64_t m/* uint32_t m*/, uint64_t inv_m) {
	//uint64_t       t;
	uint32_t       q=32;
	uint32_t       q2=64;
	uint32_t       t_high_half;
	uint64_t       r;
	uint32_t       x;
	//uint32_t       h;
	uint64_t	h;
	//uint32_t       y;
	uint64_t y;
	uint64_t       z;


//	t = (uint64_t)a*(uint64_t)b;
	//printf("a*b=%lld\n", t);
	h = t>>32;
	//printf("h=%lu\n",h);
	// y = ((uint64_t)h*(uint64_t)inv_m)>>32;
	uint64_t temp;
	temp = _multiply(h,inv_m);
	y = temp>>32;
	//printf("y=%lu\n", y);
	// z = (uint64_t)y*(uint64_t)m;
	z = _multiply(y, m);
	//printf("z=%lld\n", z);
	//r = t - z;
	r = _subtract(t,z);

	//printf("r=%lld\n", r);

	//while(r>=m)r = r - m;
	while(r>=m)r = _subtract(r, m);
//	 r = r % m;

	x = r;
	return x;
}

uint64_t _add(uint64_t a, uint64_t b) {
   uint64_t sum=a+b;

   if(sum<a) {
  if(sum>=_N)
   sum=sum-_N-_N;
  else
   sum=sum-_N;
   }
   return sum;
}

uint64_t _subtract(uint64_t a, uint64_t b) {
   uint64_t sum;

   if(b>=_N)
    b-=_N;
   if(a>=b)
    sum=a-b;
   else
    sum=a-b+_N;
   return sum;
}

uint64_t _multiply(uint64_t a, uint64_t b) {
   uint64_t al=(uint32_t)a, bl=(uint32_t)b, ah=a>>32, bh=b>>32;
   uint64_t albl=al*bl, albh=al*bh, ahbl=ah*bl, ahbh=ah*bh;
   uint64_t upper, carry=0;
   uint32_t uu, ul;

   upper=(albl>>32)+albh+ahbl;
   if(upper<ahbl)
    carry=0x100000000ull;
   upper=(upper>>32)+ahbh+carry;

   uu=upper>>32;
   ul=upper;

   if(ul==0)
    upper=_N-uu;
   else
    upper=(upper<<32)-uu-ul;

   return _add(a*b, upper);
}

uint64_t _power(uint64_t a, uint64_t k) {
   uint64_t current=1, square=a;

   while(k>0) {
  if((k&1)!=0)
   current=_multiply(current, square);
  square=_multiply(square, square);
  k=k>>1;
   }
   return current;
}

uint64_t _normalize(uint64_t a) {
   if(a>=_N)
    return a-_N;
   return a;
}

void extendedGCD(uint64_t a, uint64_t b, int64_t *s, int64_t *t) {
   int64_t x, y;

   if(a%b==0) {
    *s=0;
    *t=1;
   }
   else {
    extendedGCD(b, a%b, &x, &y);
    *s=y;
    *t=x-y*(a/b);
   }
}

uint64_t _inverse(uint64_t a) {
   int64_t s, t;

   extendedGCD(_N, a, &s, &t);
   if(t<0)
    return t+_N;
   return t;
}

uint64_t _root(uint64_t size) {
   uint64_t k=(3ull<<32)/size;

   return _normalize(_power(_ROOT, k));
}

uint64_t _inverseRoot(uint64_t size) {
   uint64_t k=(3ull<<32)/size;

   return _normalize(_power(_ROOT, (3ull<<32)-k));
}



void smallFFT(uint64_t *x, uint64_t *X, uint32_t size, int inverse) {
  uint32_t      i, j;
   uint64_t total, r, inv=1, value;

   if(inverse) {
    r=_inverseRoot(size);
    inv=_inverse(size);
   }
   else
    r=_root(size);
   for(i=0;i<size;i++) {
  total=0;
  for(j=0;j<size;j++) {
     value=_multiply(_power(r, i*j), x[j]);
   total=_add(total, value);
    }
  X[i]=_normalize(_multiply(total, inv));
   }
}


void transpose(uint64_t *x, uint64_t *X, uint32_t size, uint32_t xLength)
{
  uint32_t group, index;
  for(group =  0; group < size/xLength; group++)
  {
    for(index = 0; index < xLength; index++)
      X[group*xLength + index] = x[index*size/xLength + group];
  }
}


void largeFFT(uint64_t *x, uint64_t *X, uint32_t size, int inverse) {
  uint32_t count = 1, fft, i, j, k, length, m;
  uint64_t r, inv = 1, omega, currentOmega, c0, c1;
  uint64_t *buffer, *to, *from, *swap, *smallX, *smallY;

  if(size%3 == 0) {
    size = size/3;
    count = 3;
  }
  if(inverse) {
    r=_inverseRoot(size);
    inv=_inverse(size);
  }
  else
    r=_root(size);

  buffer=(uint64_t *)malloc(sizeof(uint64_t)*size);

  for(fft = 0; fft < count; fft++) {
    for(i = 0; i < size; i++)
      buffer[i] = x[count*i + fft];
    from = buffer;
    to = X + fft*size;
    length = size/2;
    m = 1;
    while(length >= 1)
    {
      omega = _power(r, size/(2*length));
      currentOmega = 1;
      for(j=0; j < length; j++) {
        for(k=0; k<m; k++) {
          c0=from[k + j*m];
          c1=from[k + j*m + length*m];
          to[k + 2*j*m]=_add(c0, c1);
          to[k + 2*j*m + m]=_multiply(currentOmega, _subtract(c0, c1));
        }
        currentOmega=_multiply(currentOmega, omega);
      }
      swap = from;
      from = to;
      to = swap;

      length = length >> 1; //length = length / 2
      m = m << 1;
    }
    if(from!=X+fft*size) {
      for(i=0;i<size;i++)
        X[fft*size+i]=from[i];
    }
  }

  if(count  > 1) {
    r = _root(size*count);
    smallX=(uint64_t *)malloc(sizeof(uint64_t)*count);
    smallY=(uint64_t *)malloc(sizeof(uint64_t)*count);
    for(fft=0;fft<size;fft++) {
      smallX[0]=X[fft];
      smallX[1]=_multiply(_power(r, fft), X[fft+size]);
      smallX[2]=_multiply(_power(r, fft+fft), X[fft+size+size]);
      smallFFT(smallX, smallY, 3, inverse);
      X[fft]=smallY[0];
      X[fft+size]=smallY[1];
      X[fft+size+size]=smallY[2];
    }
  free(smallX);
  free(smallY);
  }

  for(i = 0; i < size*count; i++)
  X[i] = _normalize(_multiply(X[i], inv));
  free(buffer);
}


void OriginalFFT(zz_pX& x, long n, const zz_p& root,
		  const zz_pX& powers, const Vec<mulmod_precon_t>& powers_aux,
                 const fftRep& Rb, zz_pX& temp, double &Time)
{
 // FHE_TIMER_START;
  // double time;
  Time = GetTime();
	if (IsZero(x)) return;
	if (n<=0) {
		clear(x);
		return;
	}

 	long p = zz_p::modulus();
  cout<<endl<<" original fun P is: "<<p<<endl;

 	long dx = deg(x);
 	for (long i=0; i<=dx; i++) {
   		x[i].LoopHole() = MulModPrecon(rep(x[i]), rep(powers[i]), p, powers_aux[i]);
 	}
 	x.normalize();

	long k = NextPowerOfTwo(2*n-1);

	fftRep& Ra = Cmodulus::getScratch_fftRep(k);
	TofftRep(Ra, x, k);

	mul(Ra,Ra,Rb);           // multiply in FFT representation

	FromfftRep(x, Ra, n-1, 2*(n-1)); // then convert back
	temp = x;

	// cout<<"the time of original fft is: "<<time<<endl;
 	dx = deg(x);
 	for (long i=0; i<=dx; i++) {
   		x[i].LoopHole() = MulModPrecon(rep(x[i]), rep(powers[i]), p, powers_aux[i]);
 	}
 	x.normalize();
  Time = GetTime() - Time;
}
