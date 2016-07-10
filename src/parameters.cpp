#include "parameters.h"

void Parameters::setParam(FHEcontext &context){
	m = context.zMStar.getM();
	phiM = context.zmStar.getPhiM();
	
}