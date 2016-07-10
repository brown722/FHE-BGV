
#include <NTL/lzz_pXFactoring.h>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include "NumbTh.h"
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
#include "myprog.h"
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

/***********zz_pX***************/
void encryption::convZZxToZZ_pX(zz_pX& x, const ZZX& a)
{
   conv(x.rep, a.rep);
   x.normalize();
}
void encryption:: convLongToZZ_pX_vector(vector<zz_pX>& v1, vector<long>&v2 )
{
	   long n = v2.size();
	   v1.resize(n);
	   for (long i = 0; i < n; i++)
		   conv(v1[i], v2[i]);
}
void encryption::convLongToZZ_pX(zz_pX& x, long a)
{
	   if (a == 0) {
	      x.rep.SetLength(0);
	      return;
	   }

	   zz_p t;//zz_p

	   convLongToZZ_p(t, a);
	   convZZ_pToZZ_pX(x, t);
}

void encryption::convLongToZZ_p(zz_p& x, long a)
{
	x._zz_p__rep = rem(a, zz_pInfo->p, zz_pInfo->red_struct);
}

void encryption::convZZ_pToZZ_pX(zz_pX& x, zz_p a)
{
	if (IsZero(a))// see if zz_p__rep is zero;
	      x.rep.SetLength(0);
	else {
	      x.rep.SetLength(1);
	      x.rep[0] = a;
	}
}

/***************GF2X**********************/
 /*void encryption::set(GF2X& x)
{
	x.xrep.SetLength(1);
	x.xrep[0] = 1;
}
void encryption::setX(GF2X& x)
{
	x.xrep.SetLength(1);
   	x.xrep[0] = 2;
}

void encryption::clear(GF2X& x)
{
	 x.xrep.ZeroLength();
}*/

unsigned long encryption::exponentiate(const vector<unsigned long>& exps, bool onlySameOrd) const
{
 	/* if (isDryRun()) return 1;
	//Exsisting in original file;
 	*/
 	unsigned long t = 1;
 	unsigned long n = min(exps.size(),gens.size());
 	for(long i = 0; i<n; i ++)
 	{
// 		 if (onlySameOrd && !SameOrd(i)) continue;
// 		 //Exsisting in original file;

 		unsigned long g = PowerMod(gens[i] ,exps[i], m); // g = gens[i]^buffer[i] % m (m may be negative)  NTL
 		t = MulMod(t,g,m);//t = (t * g) %m  NTL
 	}
 	return t;
}
unsigned long encryption::ith_rep(unsigned long i) const
{
	return (i<nslots)? T[i] : 0;
}

void encryption::mappingToFt(zz_pX& w, const zz_pX& G, unsigned long t, const zz_pX* rF1)
{
	long i = (t>0 && t<m)? indexT[t]: -1;
	if (i < 0) { clear(w); return; }
	if (rF1 == NULL){
		if (G == factors[i])
		{
      			SetX(w);//set to the monomial X
      			return;
    		}


    		if (deg(G) == 1)
    		{
	     		w = -ConstTerm(G);// return constant term of G, x^0 term ?????
	      		return;
		}
/******seems like never used, don't know what is this ****************/
//	    assert(r == 1);  // the general case: currently only works when r == 1
//
//		zz_pEBak bak; //Q = "null", b = 0
//		bak.save();// Q = CurrentModulus, b = 1
//	  	zz_pE::init(factors[i]);// ZZ_pE::init(P) initializes the current modulus to P;  required: deg(P) >= 1.
//	  	zz_pEX Ga;
//	  	conv(Ga,G);//NTL;
//
//	  	vec_zz_pE roots;
//	  	FindRoots(roots, Ga); //NTL//// Performs square-free decomposition.
//
//	  	zz_pE* first = &roots[0];
//	  	zz_pE* last = first + roots.length();
//	  	//zz_pE* smallest = min_element(first, last);
//		// make a canonical choice
//		w=rep(*last);
//		return;
	}

  	zz_pXModulus Ft(factors[i]);
  	zz_pX X2t = PowerXMod(t,Ft); // X2t = X^t % Ft, t >= 0
  	w = CompMod(*rF1, X2t, Ft);   //w = rF1(X2t) mod Ft
}
void encryption::EDFa(vec_zz_pX& v, const zz_pX& f, long d)//(localFactors, phimxmod, zMStar.getORdP());
{
	EDF(v, f, PowerXMod(zz_p::modulus(), f), d);
}

void encryption::mapToSlots( const zz_pX& G)
{
/*	assert(deg(G) > 0 && zMStar.getOrdP() % deg(G) == 0);
 	assert(LeadCoeff(G) == 1);
 	//Exsisting in original file;
 */
 	mappingDataG = G;
 	mappingDataDegOfG = deg(G);
 	maps.resize(nslots);

/* 	if (isDryRun()){
 		mappingData.mas[0] = GF2X::zero();
 		return
 	}
 	//Exsisting in original file;
 */


 	indexT.assign(m, -1);//assign(m,-1);
 	vector<unsigned long> buffer(gens.size()); //???????????
 	T.resize(nslots);
 	long i,idx, ctr;
 	i = 0; idx = 0; ctr = 0;

 	do{
 	ctr++;
 	unsigned long t = exponentiate(buffer);
	for(long j = 0; j<buffer.size();j++)
	 		//dLogT[idx++] = buffer[j];//the original
	 		dLogT[j] = buffer[j];
/*	 	assert(GCD(t,mm) == 1); // sanity check for user-supplied gens
	   	assert(Tidx[t] == -1);
		//Exsisting in original file;
	*/
	   	T[i] = t;
	   	indexT[t] = i++;
  	}while(nextExpVector(buffer));

 	zz_pX phimxmod;

 	convZZxToZZ_pX (phimxmod, PhimX);//conv ZZX to zz_px;

  	vec_zz_pX localFactors;

  	//EDF(localFactors, phimxmod, PowerXMod(zz_p::modulus(), phimxmod),zMStar.getOrdP());
  	EDFa(localFactors, phimxmod, ordP);// equal-degree factorization//from NTL

  	factors = localFactors;

  	mappingToFt (maps[0], mappingDataG,1);
  	for (long i = 1; i<nslots; i++)
  		mappingToFt(maps[i] ,mappingDataG, ith_rep(i), &(maps[0]));//


  	zz_pEBak bak;
  	bak.save();
  	zz_pE::init(mappingDataG);
  	//mappingData.contextForG.save();//Exsisting in original file;

  	if ((G.rep.length() - 1) ==1 )
  		return;

  	rmaps.resize(nslots);

  	if (G == factors[0])
  	{
  		for (long i = 0; i<nslots; i++)
  		{
  			long t = ith_rep(i);
  			long tInv = InvMod(t,m);// computes t^{-1} mod m.
  			zz_pX ct_rep;
  			PowerXMod(ct_rep, tInv, G); //ct_rep ^ tInv mod G

  			zz_pE ct;
  			conv(ct, ct_rep);

  			zz_pEX Qi;
  			SetCoeff(Qi, 1, 1);
        			SetCoeff(Qi, 0, -ct);

        			rmaps[i] = Qi;
  		}
  	}
  	else //// the general case: currently only works when r == 1
  	{
  		assert(r == 1);

  		vec_zz_pEX FRts;
  		zz_pEX FrobeniusMap;
  		for (long i=0; i<nslots; i++)
  		{
	      		// We need to lift Fi from R[Y] to (R[X]/G(X))[Y]
	     		zz_pEX  Qi;
	      		long t, tInv = 0;

	      		if (i ==0)
	      		{
	      			conv(Qi, factors[i]);

	      			FrobeniusMap = PowerXMod(zz_pE::cardinality(), Qi);
	      			FRts = EDF(Qi, FrobeniusMap, deg(Qi)/deg(G)); // factor Fi over GF(p)[X]/G(X)
	      		}
	      		else
	      		{
	      			t = ith_rep(i);
	      			tInv =  InvMod(t, m);
	      		}

	      		long j;
	      		for (j = 0; j < FRts.length(); j++)
	      		{
	      			zz_pEX FRtsj;
	      			if (i == 0)
	      				FRtsj = FRts[j];
	      			else
	      			{
	      				zz_pEX X2tInv = PowerXMod(tInv, FRts[j]);
	            				IrredPolyMod(FRtsj, X2tInv, FRts[j]);
	      			}
	      			// FRtsj is the jth factor of factors[i] over the extension field.
	        			// For j > 0, we save some time by computing it from the jth factor
	        			// of factors[0] via a minimal polynomial computation.
	      			zz_pEX GRti;
	      			conv(GRti, maps[i]);
	      			GRti %= FRtsj;

	      			if(IsX(rep(ConstTerm(GRti)))) // is GRti == X?
	      			{
	      				Qi = FRtsj; // if so Qi is the right factor
	      				break;
	      			}	// If this does not happen then move to the next factor of Fi
	      		}
	      		assert(j < FRts.length());
	      		rmaps[i] = Qi;
  		}
	}
}

void encryption::embedInSlots(zz_pX& H, const vector<zz_pX>& alphas)
{
	 vector<zz_pX> crt(nslots);
	if (IsX(mappingDataG))
	{
		for (long i = 0; i < nslots; i++)
			crt[i] = ConstTerm(alphas[i]);
	}
	else
	{
		for(long i = 0; i<nslots; i++)
		{
			if (deg(alphas[i]) <= 0 )
				crt[i] = alphas[i];
			else
				CompMod(crt[i], alphas[i], maps[i], factors[i]);
		}
		CRT_reconstruct(H, crt);
	}

}
void encryption::CRT_reconstruct(zz_pX& H, vector<zz_pX>& crt) const
{
/*	if (isDryRun()) {
	    H = RX::zero();
	    return;
	  }
	  FHE_TIMER_START;
	  Existing in original code;
	  */

	const vector<zz_pX>& ctab = crtTable; //need generate crtTable first!!  not done
	H.rep.SetLength(0);// clear H;
	//zz_pX temp1, temp2;//existing in the original code
	bool easy = true;

	for (long i = 0; i < nslots; i++)
	    if (!IsZero(crt[i]) && !IsOne(crt[i]))
	    {
	    	easy = false;
	    	break;
	    }

	if (easy) {
		for (long i=0; i<nslots; i++)
			if (!IsZero(crt[i]))
				H += ctab[i];
	}
	else{
	    vector<zz_pX> crt1;
	    crt1.resize(nslots);
	    for (long i = 0; i < nslots; i++)//crt1 = (crt * crtCoeffs) % factors
	       MulMod(crt1[i], crt[i], crtCoeffs[i], factors[i]);
	    evalTree(H, crtTree, crt1, 0, nslots);//compute a H
	  }
}
//void encryption::SameOrd(unsigned long i)
//{
//	return (i < ords.size())? (ords[i]>0) : false;
//}
void encryption::evalTree(zz_pX& res, shared_ptr< TNode<zz_pX> > tree, const vector<zz_pX>& crt1, long offset, long extent) const
{
	 if (extent == 1)
		 res = crt1[offset];
	 else {
		long half = extent/2;
		zz_pX lres, rres;
		evalTree(lres, tree->left, crt1, offset, half);
		evalTree(rres, tree->right, crt1, offset+half, extent-half);
		zz_pX tmp1, tmp2;
		mul(tmp1, lres, tree->right->data);
		mul(tmp2, rres, tree->left->data);
		add(tmp1, tmp1, tmp2);
		res = tmp1;
	 }
}
bool encryption::nextExpVector(vector<unsigned long >& buffer) const
{
	long order;
	//if (!isDryRun())  //Exsisting in original file;
	for (long i=gens.size()-1; i>=0; i--)
	{
		if (i>=(long)buffer.size()) continue; // sanity check

		order = (i<ords.size())? abs(ords[i]) : 0;
		if (buffer[i] < order - 1)
		{
			buffer[i]++;
			for (unsigned long j=i+1; j<buffer.size(); j++)
				buffer[j] = 0;
			return true;
		}
	}
	return false;     // cannot increment the vector anymore
}
uint64_t encryption::ModuloAdd(uint64_t a, uint64_t b)
{
	uint64_t sum=a+b;
	if(sum<a)
	{
		if(sum>=N)
			sum=sum-N-N;
		else
			sum=sum-N;
	}
	else if(sum>=a && sum>=N)
		sum=sum-N;

	return sum;
}
uint64_t encryption::ModuloSubtract(uint64_t& a, uint64_t& b)
{
	uint64_t sum;
	if(b>=N)
		b-=N;
	if(a>=b)
		sum=a-b;
	else
		sum=a-b+N;

	return sum;
}
uint64_t encryption::ModuloMultiply(uint64_t& a, uint64_t& b){
	uint64_t al = (uint32_t)a, bl = (uint32_t)b, ah = a>>32, bh =b >>32;
	uint64_t albl = al * bl, albh = al * bh, ahbl = ah * bl, ahbh = ah * bh;
	uint64_t upper, carry = 0, ab;
	uint32_t uu, ul;

	upper = (albl>>32) + albh + ahbl;
	if(upper<ahbl)
		carry=0x100000000ull;
	upper=(upper>>32) + ahbh + carry;

	uu = upper>>32;
	ul = upper;

	if(ul==0)
	    upper = N - uu;
	else
	    upper = (upper<<32) - uu - ul;
	   //printf("upper=%Lx\n", upper>>32);
	   //printf("a*b=%Lx\n", a*b);
	ab = a*b;
	   return ModuloAdd(ab, upper);
}

uint64_t encryption::ModuloPower(uint64_t a, uint64_t k) {
	uint64_t current = 1, square = a;

   while(k > 0) {
	   if((k&1)!=0)
		   current=ModuloMultiply(current, square);
	   square=ModuloMultiply(square, square);
	   k = k>>1;
   }
   return current;
}
uint64_t encryption::ModuloNormalize(uint64_t a) {
   if(a >= N)
    return a - N;
   return a;
}

void encryption::extendedGCD(uint64_t a, uint64_t b, int64_t *s, int64_t *t) {
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

uint64_t  encryption::ModuloInverse(uint64_t  a) {
	int64_t  s, t;

	extendedGCD(N, a, &s, &t);
	if(t<0)
    return t+N;
   return t;
}

uint64_t encryption::root(uint64_t size) {
	uint64_t k=(3ull<<32)/size;
	return ModuloNormalize(ModuloPower(ROOT, k));
}

uint64_t encryption::InverseRoot(uint64_t size) {
	uint64_t k=(3ull<<32)/size;
	return ModuloNormalize(ModuloPower(ROOT, (3ull<<32)-k));
}

void encryption::transpose(uint64_t  *x, uint64_t  *X, uint32_t size, uint32_t xLength)
{
	int group, index;
	for(group = 0; group < size/xLength; group++)
		for(index=0; index < xLength; index++)
			X[group * xLength + index]=x[index * size/xLength + group];
}

void encryption::SmallFFT(ZZX &x, uint64_t *X, uint32_t size, int inverse, int *flag)
{
	int     i, j;
	uint64_t ith_x, jth_x;
	uint64_t total, r, value, power, inv = 1;

	if(inverse)
	{
		r = InverseRoot(size);
		inv = ModuloInverse(size);
	}
	else
		r = root(size);

	cout<<"the root of 64-point smallFFT is "<<r<<endl;
	for(i = 0; i < size; i++)
	{
		total = 0;
		conv(ith_x, x[i]);
		if (!inverse)
		{
			if (ith_x > 0xffffffff00000000)
				flag[i] = 1;
			else
				flag[i] = 0;
		}
		for(j = 0; j < size; j ++)
		{
			power = ModuloPower(r, i*j);
			conv(jth_x, x.rep[j]);
			value = ModuloMultiply(power,jth_x);
			total = ModuloAdd(total, value);
		}
		X[i] = ModuloNormalize(ModuloMultiply(total,inv));
		if (inverse && flag[i])
			X[i] = X[i] + 0xFFFFFFFF00000000;
	}
}

void encryption::largeFFT(ZZX &x, uint64_t *X, uint32_t size, int inverse, int *flags)
{
	int count = 1, fft, i, j, k, l, m;
	uint64_t r, inv = 1, omega, currentOmega, c0, c1, DifferenceOfC;
	uint64_t *buffer, *to, *from, *swap, *smallY;
	ZZX smallX;
	if(size % 3==0)  //size is multiple of 3
	{
		size = size/3;
		count = 3;
	}
	if(inverse) //IFFT
	{
		r =InverseRoot(size);
		inv = ModuloInverse(size);
	}
	else// FFT
		r = root (size);

	buffer = (uint64_t *)malloc(sizeof(uint64_t)*size);//assign a part of memory for buffer;

	for(fft = 0; fft < count; fft++)
	{
		for(i = 0; i < size; i++)
		{
			conv(buffer[i], x[count * i + fft]);//buffer[i] = x.rep[count * i + fft].LoopHole();  //x.rep[j].LoopHole()
			/*for negetive number*/
			if (!inverse)
			{
				if (buffer[i] > 0xffffffff00000000)
					flags[i] = 1;
				else
					flags[i] = 0;
			}
			/*for negetive number*/
		}
		from = buffer;
		to = X + fft * size;
		l = size / 2;
		m = 1;
		while(l >= 1)
		{
			omega = ModuloPower(r, size / (2 * l));
			currentOmega = 1;
				for(j=0;j<l;j++)
				{
					for(k=0;k<m;k++)
					{
						c0 = from [k + j * m];
						c1 = from [k + j * m + l * m];
						to[k + 2*j*m] = ModuloAdd(c0, c1);
						DifferenceOfC = ModuloSubtract(c0, c1);
						to[k + 2*j*m + m] = ModuloMultiply( currentOmega, DifferenceOfC);
					}
					currentOmega=ModuloMultiply(currentOmega, omega);
				}
				swap = from;
				from = to;
				to = swap;
				l = l>>1;
				m = m<<1;
			}
			if(from != X + fft*size)
			{
				for(i=0;i<size;i++)
				{
					X[fft*size + i]=from[i];
				}
			}
		}
		if(count>1)
		{
			r = root(size * count);
			//smallX = (unsigned long *)malloc(sizeof(unsigned long)*count);
			smallY = (uint64_t *)malloc(sizeof(uint64_t)*count);
			uint64_t PowerOfFFT1, PowerOfFFT2;
			int signs[3];//it is not real sign here, just for indicating the cases that is larger than 0XFFFFFFFF00000000
			for(fft=0;fft<size;fft++)
			{
				if (inverse){
					 signs[0] = flags[fft]; signs[1] = flags[fft + 1];  signs[2] = flags[fft + 2];
				}
				smallX[0] = X[fft];
				PowerOfFFT1 = ModuloPower(r, fft);
				PowerOfFFT2 = ModuloPower(r, fft+fft);
				smallX[1] = ModuloMultiply(PowerOfFFT1, X[fft + size]);
				smallX[2] = ModuloMultiply(PowerOfFFT2, X[fft + size + size]);
				SmallFFT(smallX, smallY, 3, inverse, signs);
				if (!inverse){
					 flags[fft] = signs[0]; flags[fft + 1] = signs[1];  flags[fft + 2] = signs[2];
				}
				X[fft] = smallY[0];
				X[fft+size] = smallY[1];
				X[fft+size+size] = smallY[2];
			}
			free(smallY);
		}

		for(i=0;i<size*count;i++)
		{
			X[i]=ModuloNormalize(ModuloMultiply(X[i], inv));
			/*for negative number*/
			if (inverse && flags[i])
				X[i]= X[i] + 0xFFFFFFFF00000000 +1;
			/*for negative number*/
		}
		free(buffer);
}

/*For expand 32-bits BIT to */
void split_by3(uint32_t *x, uint32_t xLength, uint64_t *X, uint32_t XLength)
{
   int byteCount = 0, current = 0, index;
   uint64_t value = 0;

   for(index=0;index<XLength;index++)
   {
	   if(byteCount<3)
	   {
		   if(current<xLength)
			   value+=((uint64_t)x[current++])<<byteCount*8;
		   byteCount+=4;
	   }
	   X[index]=value & 0xFFFFFF;
	   byteCount-=3;
	   value=value>>24;
   }
}
void resolve_by3(uint64_t *x, uint32_t length)
{
	uint64_t carry = 0, current;
	int      index;

	for(index = 0; index < length; index++)
	{
		current=x[index];
		carry += (current & 0xFFFFFF);
		x[index]=carry & 0xFFFFFF;
		carry=(carry>>24) + (current>>24);
	}
	if(carry>0)
	{
		printf("resolve_by3 failed! abort!\n");
		exit(1);
	}
}
void join_by3(uint64_t *x, uint32_t xLength, uint32_t *X, uint32_t XLength)
{
	int      byteCount=0, current=0, index;
	uint64_t value=0;

	value=0;
	byteCount=0;
	current=0;
	for(index=0;index<XLength;index++)
	{
		while(byteCount<4)
		{
			if(current<xLength)
				value+=(x[current++])<<byteCount*8;
			byteCount+=3;
		}
		X[index]=value;
		value=value>>32;
		byteCount-=4;
	}
}

void encryption::sampleSmall(ZZX &poly, long n)
{
  if (n<=0) n=deg(poly)+1; if (n<=0) return;
  poly.SetMaxLength(n); // allocate space for degree-(n-1) polynomial

  for (long i=0; i<n; i++) {    // Chosse coefficients, one by one
    long u = lrand48(); //random number ;
    if (u&1) {                 // with prob. 1/2 choose between -1 and +1
      u = (u & 2) -1;
      SetCoeff(poly, i, u);
    }
    else SetCoeff(poly, i, 0); // with ptob. 1/2 set to 0
  }
  poly.normalize(); // need to call this after we work on the coeffs
}

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
	long security = 128;
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

	encryption Enc;
	zz_p::init(p);

	vector<long> plaintextVector;
	for(long i = 0;i < nslots; i++)
		plaintextVector.push_back(i);

 	Ctxt ciphertext(publicKey);

	double cpuTime = 0, gpuTime = 0 ,gpuTime1 = 0;

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

	for (int i = 0, k = 0; i < ZmsIndex.size(); i ++)
	{
		if (ZmsIndex[i] < 0)
		{
			dropFlags[i] = 1;
			offsetFlags[k++] = i;
		}
	}

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
	double timeCPU;int testNum = 1;

	/////*************************verifying the encode function**********************************
	cout<<endl<<"*****************verifying bluesteinFFT functions******************"<<endl;
	gpuTime = GetTime();
	gpuTime1 = GetTime();
	ZZX plaintext_ZZX;
	ea.encode(plaintext_ZZX, plaintextVector);

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
	for (int i = 0; i < numPrimes; i ++)
	{
		context.ithModulus(i).restoreModulus();
		conv(r_zzP[i], r_poly);
		memoryAllocations(r_zzP[i], mm, powers[i], powers_aux[i],nx, r_smallSample[i],smallBuffer[i], cuSmallBuffer[i], cuRaSmall[i],
			resultSmall[i], planSmall[i], cuFFToutSmall[i], invPlanSmall[i], offset[i], streamSmall[i]);
	}
	for (int i = 0; i < numPrimes; i ++)
	{
		deviceBlustn2(mm, smallBuffer[i], cuSmallBuffer[i],  cuRaSmall[i], Rb_gpu[i], resultSmall[i], cuFFToutSmall[i], 
			invPrimes[i], primes[i], nx, powers_gpu[i], r_smallSample[i], gpuOffsetFlags, offset[i], streamSmall[i], planSmall[i], invPlanSmall[i], 0);
	}

//	 cout<<"check for smallSample"<<endl;
//	 for (int i = 0; i < numPrimes; i ++)
//	 	readAndCheck(cuFFToutSmall[i], powers_gpu[i],  r_smallSample[i], nx, mm, streamSmall[i]);
//	 fftRep Rbsmall;
//	 context.ithModulus(testNum).restoreModulus();
//	 zz_pX productSmall;
//	 BluesteinInit(mm, Root[testNum], powers[testNum], powers_aux[testNum], Rbsmall);
//	 OriginalFFT(r_zzP[testNum], mm, Root[testNum], powers[testNum], powers_aux[testNum], Rbsmall, productSmall, timeCPU);
//	 for (int i = 0; i < 10; i++)
//	 	cout<<productSmall[i]<<", ";
//	 cout<<endl;

	uint64_t *e_gaussianSample[numPrimes];
	cufftDoubleReal *gaussBuffer[numPrimes];
	cufftDoubleReal *cuGaussBuffer[numPrimes];
	cufftDoubleComplex *cuRaGauss[numPrimes]; 
	cufftDoubleComplex *resultGauss[numPrimes];
	cufftHandle planGauss[numPrimes];/////////////////////////////////////////
	cufftHandle invPlanGauss[numPrimes];

	cufftDoubleReal *cuFFToutGauss[numPrimes];

	for (int i = 0; i < numPrimes; i ++)
	{
		context.ithModulus(i).restoreModulus();
		conv(e_zzPG[i], e_polyG);
		memoryAllocations(e_zzPG[i], mm, powers[i], powers_aux[i],nx, e_gaussianSample[i],gaussBuffer[i], cuGaussBuffer[i], cuRaGauss[i],
			resultGauss[i], planGauss[i], cuFFToutGauss[i], invPlanGauss[i], offset[i], streamGauss[i]);
	}
	for (int i = 0; i < numPrimes; i ++)
	{
		deviceBlustn2(mm, gaussBuffer[i], cuGaussBuffer[i],  cuRaGauss[i], Rb_gpu[i], resultGauss[i], cuFFToutGauss[i], 
			invPrimes[i], primes[i], nx, powers_gpu[i], e_gaussianSample[i], gpuOffsetFlags, offset[i], streamGauss[i], planGauss[i],  invPlanGauss[i], 1);
	}
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
	for (int i = 0; i < numPrimes; i ++)
	{
		context.ithModulus(i).restoreModulus();
		conv(ptxt_zzP[i], input);
		memoryAllocations(ptxt_zzP[i], mm, powers[i], powers_aux[i],nx, ptxt[i],ptxtBuffer[i], cuPtxtBuffer[i], cuRaPtxt[i],
			resultPtxt[i], planPtxt[i], cuFFToutPtxt[i], invPlanPtxt[i], offset[i], streamSmall[i]);
	}
	for (int i = 0; i < numPrimes; i ++)
	{
		deviceBlustn2(mm, ptxtBuffer[i], cuPtxtBuffer[i],  cuRaPtxt[i], Rb_gpu[i], resultPtxt[i], cuFFToutPtxt[i], 
			invPrimes[i], primes[i], nx, powers_gpu[i], ptxt[i], gpuOffsetFlags, offset[i], streamSmall[i], planPtxt[i],  invPlanPtxt[i], 0);
	}
//	 cout<<"check for ptxt"<<endl;
//	 for (int i = 0; i < numPrimes; i ++)
//	 	readAndCheck(cuFFToutPtxt[i], powers_gpu[i],  ptxt[i], nx, mm, streamSmall[i]);
//	 context.ithModulus(testNum).restoreModulus();
//	 zz_pX productPtxt; fftRep RbPtxt;
//	 BluesteinInit(mm, Root[testNum], powers[testNum], powers_aux[testNum], RbPtxt);
//	 OriginalFFT(ptxt_zzP[testNum], mm, Root[testNum], powers[testNum], powers_aux[testNum], RbPtxt, productPtxt, timeCPU);
//	 for (int i = 0; i < 10; i++)
//	 	cout<<productPtxt[i]<<", ";
//	 cout<<endl;

	gpuTime1 = GetTime() - gpuTime1;

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
	cout<<cpuTime<<", "<<gpuTime1<<", "<<gpuTime<<endl;
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
