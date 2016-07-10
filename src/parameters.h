#ifndef GPU_PARAM_H_
#define GPU_PARAM_H_

#include "FHEContext.h"

class Parameters{
public:
	long m;
	long phiM;
	long numOfPrimes;
	long primes[];
	
	void setParam(FHEcontext &context);


};

#endif /* GPU_PARAM_H_ */