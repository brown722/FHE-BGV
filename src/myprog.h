
/*
 * myCode.h
 *
 *  Created on: 2015年10月19日
 *      Author: jiyang
 */

#ifndef MYCODE_H_
#define MYCODE_H_

#include <NTL/ZZX.h>
#include <NTL/GF2X.h>
#include <NTL/vec_long.h>
#include <stdint.h>
#include "NumbTh.h"
#include "cloned_ptr.h"
using namespace std;
using namespace NTL;

class encryption{

public:
	long m;
	long nslots;
	long r;
	long FFT_size;
	void convLongToZZ_pX(zz_pX& x, long a);
//	void set(zz_pX& x);
//	void clear(zz_pX& x) ;

	void convZZxToZZ_pX(zz_pX& x, const ZZX& a);
	void convLongToZZ_p(zz_p& x, long a);
	void convZZ_pToZZ_pX(zz_pX& x, zz_p a);
	void convLongToZZ_pX_vector(vector<zz_pX>& v1, vector<long>&v2 );
	void embedInSlots(zz_pX& H, const vector<zz_pX>& alphas);
	void mapToSlots(const zz_pX& G);
	void mappingToFt(zz_pX& w, const zz_pX& G, unsigned long t,const zz_pX* rF1=NULL);
	unsigned long exponentiate(const vector<unsigned long>& exps, bool onlySameOrd=false) const;
	bool nextExpVector(vector<unsigned long>& buffer) const;
	//void SameOrd(unsigned long i);
	void CRT_reconstruct(zz_pX& H, vector<zz_pX>& crt) const;
	void EDFa(vec_zz_pX& v, const zz_pX& f, long d);
	void evalTree(zz_pX& res, shared_ptr< TNode<zz_pX> > tree, const vector<zz_pX>& crt1, long offset, long extent) const;
	uint64_t ModuloAdd(uint64_t a, uint64_t b);
	uint64_t ModuloSubtract(uint64_t& a, uint64_t& b);
	uint64_t ModuloMultiply(uint64_t& a, uint64_t& b);
	uint64_t ModuloPower(uint64_t a, uint64_t k);
	uint64_t ModuloNormalize(uint64_t a) ;
	void extendedGCD(uint64_t a, uint64_t b, int64_t *s, int64_t *t);
	uint64_t  ModuloInverse(uint64_t  a);
	uint64_t root(uint64_t size) ;
	uint64_t InverseRoot(uint64_t size);
	void transpose(uint64_t  *x, uint64_t *X, uint32_t size, uint32_t xLength);
	void SmallFFT(ZZX &x, uint64_t *X, uint32_t size, int inverse, int *flag);
	void largeFFT(ZZX &x, uint64_t *X, uint32_t size, int inverse, int *flag);
	void OriginalFFT(vec_long &y, const ZZX& x)const;
	void sampleSmall(ZZX &poly, long n);//Small sample

	zz_pX  mappingDataG;
	long mappingDataDegOfG;
	vector<zz_pX> maps;
	vector<zz_pEX> rmaps;
	 vector<zz_pX> crtTable;
	vector<long> indexT;
	vector<long> ords;
	unsigned long ordP;
	ZZX PhimX;
	vector<long> gens;
	vector<long>dLogT;
	vector<unsigned long> T;
	vec_zz_pX factors;
	vec_zz_pX crtCoeffs;
	 shared_ptr< TNode<zz_pX> > crtTree;
	const bool SameOrd(unsigned long i);
	 unsigned long ith_rep(unsigned long i) const;
	 zz_pEContext contextForG;
};

#endif /* MYCODE_H_ */




