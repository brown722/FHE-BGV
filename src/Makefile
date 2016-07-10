# Copyright (C) 2012,2013 IBM Corp.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
# 
CC = g++
#CC = gcc

#
#CFLAGS = -g -O2 -Wfatal-errors -Wshadow -Wall -I/usr/local/include 

# useful flags:
#   -std=c++11
#   -std=c++0x
#   -DUSE_ALT_CRT  tells helib to use an alterntive to the default DoubleCRT
#                  representation...experimental...can be faster or slower
#                  depending on the operation mixture
#
#   -DNO_HALF_SIZE_PRIME  tells helib to not use the half size prime
#                         in the prime chain
#
#   -DEVALMAP_CACHED=0  tells helib to cache certain constants as ZZX's
#   -DEVALMAP_CACHED=1  tells helib to cache certain constants as DoubleCRT's
#                       these flags only affect bootstrapping
#
#   -DFHE_THREADS  tells helib to enable generic multithreading capabilities;
#                  must be used with a thread-enabled NTL and the -pthread
#                  flag should be passed to gcc
#
#   -DFHE_DCRT_THREADS  tells helib to use a multithreading strategy at the
#                       DoubleCRT level; requires -DFHE_THREADS (see above)
#
#   -DFHE_BOOT_THREADS  tells helib to use a multithreading strategy for
#                       bootstrapping; requires -DFHE_THREADS (see above)

#  If you get compilation errors, you may need to add -std=c++11 or -std=c++0x

CFLAGS = -g -O2 -fopenmp -DNO_HALF_SIZE_PRIME

CUDA     := /usr/local/cuda
NVCC     := $(CUDA)/bin/nvcc
NVFLAGS  := -m64 --ptxas-options -v
# SM20     := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
SM20 = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
TARGET   := $(SM20)
INCLUDES := -I. -I$(CUDA)/include -I/home/jdong3/local/include -I/usr/local/cuda/inc
LIB      := -L$(CUDA)/lib64 -L/usr/local/cuda/lib -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcuda -lcudart -lntl -lgmp -lm -lcufft

$(info HElib requires NTL version 9.0.1 or higher, see http://shoup.net/ntl)
$(info If you get compilation errors, try to add/remove -std=c++11 in Makefile)
$(info )

LD = g++
AR = ar
ARFLAGS=ruv
GMP=-lgmp 

LDLIBS = -L/home/jdong3/local/lib -lm -lntl $(GMP) -L/home/jdong3/local/lib64 -L$(CUDA)/lib64 -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcuda -lcudart -lcufft

# LDLIBS = -L/home/jdong3/local/lib -lntl $(GMP) -lm

CCLDLIBS = -L/home/jdong3/local/lib -lntl $(GMP) -lm

HEADER = DeviceManager.h DeviceManager.h Base.h NTT.h Debug.h testBluestein.h ModP.h NewBluesteinFFT.h bluestein.h EncryptedArray.h FHE.h Ctxt.h CModulus.h FHEContext.h PAlgebra.h DoubleCRT.h NumbTh.h IndexSet.h timing.h IndexMap.h replicate.h hypercube.h matching.h powerful.h permutations.h polyEval.h multicore.h

SRC = testBluestein.cpp bluestein.cpp KeySwitching.cpp EncryptedArray.cpp FHE.cpp Ctxt.cpp CModulus.cpp FHEContext.cpp PAlgebra.cpp DoubleCRT.cpp NumbTh.cpp IndexSet.cpp timing.cpp replicate.cpp hypercube.cpp matching.cpp powerful.cpp BenesNetwork.cpp permutations.cpp PermNetwork.cpp OptimizePermutations.cpp eqtesting.cpp polyEval.cpp extractDigits.cpp EvalMap.cpp OldEvalMap.cpp recryption.cpp debugging.cpp

OBJ = DeviceManager.o DeviceManager.o Base.o NTT.o testBluestein.o NewBluesteinFFT.o bluestein.o NumbTh.o timing.o PAlgebra.o  CModulus.o FHEContext.o IndexSet.o DoubleCRT.o FHE.o KeySwitching.o Ctxt.o EncryptedArray.o replicate.o hypercube.o matching.o powerful.o BenesNetwork.o permutations.o PermNetwork.o OptimizePermutations.o eqtesting.o polyEval.o extractDigits.o EvalMap.o OldEvalMap.o recryption.o debugging.o

TESTPROGS = Test_General_x Test_PAlgebra_x Test_IO_x Test_Replicate_x Test_LinPoly_x Test_matmul_x Test_Powerful_x Test_Permutations_x Test_Timing_x Test_PolyEval_x Test_extractDigits_x Test_EvalMap_x Test_bootstrapping_x


all: fhe.a

check: myprog_x
	./myprog_x

test: $(TESTPROGS)

obj: $(OBJ)

DoubleCRT.o: DoubleCRT.cpp AltCRT.cpp $(HEADER)
	$(CC) $(CFLAGS) $(INCLUDES) -c DoubleCRT.cpp

# ModP.o: ModP.cu ModP.h
# 	$(NVCC) -c $(NVFLAGS) $(TARGET) $(INCLUDES) $(LIB) ModP.cu
	
NewBluesteinFFT.o: NewBluesteinFFT.cu
	$(NVCC) -c $(NVFLAGS) $(TARGET) $(INCLUDES) $(LIB) NewBluesteinFFT.cu

Base.o: Base.cu
	$(NVCC) -c $(NVFLAGS) $(TARGET) $(INCLUDES) $(LIB) Base.cu
	
NTT.o: NTT.cu
	$(NVCC) -c $(NVFLAGS) $(TARGET) $(INCLUDES) $(LIB) NTT.cu

Operations.o: Operations.cu
	$(NVCC) -c $(NVFLAGS) $(TARGET) $(INCLUDES) $(LIB) Operations.cu

DeviceManager.o: DeviceManager.cu
	$(NVCC) -c $(NVFLAGS) $(TARGET) $(INCLUDES) $(LIB) DeviceManager.cu


%.o: %.cpp $(HEADER)
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

fhe.a: $(OBJ)
	$(AR) $(ARFLAGS) fhe.a $(OBJ)

./%_x: %.cpp fhe.a
	$(NVCC) $(NVFLAGS) $(TARGET) $(INCLUDES) -o $@ $< fhe.a $(LDLIBS)
	# $(CC) $(CFLAGS) $(INCLUDES) -o $@ $< fhe.a $(CCLDLIBS)

clean:
	rm -f *.o *_x *_x.exe *.a core.*
	rm -rf *.dSYM

info:
	: HElib require NTL version 9.0.1 or higher
	: Compilation flags are 'CFLAGS=$(CFLAGS)'
	: If errors occur, try adding/removing '-std=c++11' in Makefile
	:
