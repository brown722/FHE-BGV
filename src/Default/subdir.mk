################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../AltCRT.cpp \
../BenesNetwork.cpp \
../CModulus.cpp \
../Ctxt.cpp \
../DoubleCRT.cpp \
../EncryptedArray.cpp \
../EvalMap.cpp \
../FHE.cpp \
../FHEContext.cpp \
../IndexSet.cpp \
../KeySwitching.cpp \
../NewBluesteinFFT.cpp \
../NumbTh.cpp \
../OldEvalMap.cpp \
../OptimizePermutations.cpp \
../PAlgebra.cpp \
../PermNetwork.cpp \
../Test_EvalMap.cpp \
../Test_General.cpp \
../Test_IO.cpp \
../Test_LinPoly.cpp \
../Test_OldEvalMap.cpp \
../Test_PAlgebra.cpp \
../Test_Permutations.cpp \
../Test_PolyEval.cpp \
../Test_Powerful.cpp \
../Test_Replicate.cpp \
../Test_Timing.cpp \
../Test_bootstrapping.cpp \
../Test_extractDigits.cpp \
../Test_matmul.cpp \
../bluestein.cpp \
../cgauss.cpp \
../debugging.cpp \
../eqtesting.cpp \
../extractDigits.cpp \
../hypercube.cpp \
../matching.cpp \
../myprog.cpp \
../params.cpp \
../permutations.cpp \
../polyEval.cpp \
../powerful.cpp \
../recryption.cpp \
../replicate.cpp \
../rotations.cpp \
../timing.cpp 

OBJS += \
./AltCRT.o \
./BenesNetwork.o \
./CModulus.o \
./Ctxt.o \
./DoubleCRT.o \
./EncryptedArray.o \
./EvalMap.o \
./FHE.o \
./FHEContext.o \
./IndexSet.o \
./KeySwitching.o \
./NewBluesteinFFT.o \
./NumbTh.o \
./OldEvalMap.o \
./OptimizePermutations.o \
./PAlgebra.o \
./PermNetwork.o \
./Test_EvalMap.o \
./Test_General.o \
./Test_IO.o \
./Test_LinPoly.o \
./Test_OldEvalMap.o \
./Test_PAlgebra.o \
./Test_Permutations.o \
./Test_PolyEval.o \
./Test_Powerful.o \
./Test_Replicate.o \
./Test_Timing.o \
./Test_bootstrapping.o \
./Test_extractDigits.o \
./Test_matmul.o \
./bluestein.o \
./cgauss.o \
./debugging.o \
./eqtesting.o \
./extractDigits.o \
./hypercube.o \
./matching.o \
./myprog.o \
./params.o \
./permutations.o \
./polyEval.o \
./powerful.o \
./recryption.o \
./replicate.o \
./rotations.o \
./timing.o 

CPP_DEPS += \
./AltCRT.d \
./BenesNetwork.d \
./CModulus.d \
./Ctxt.d \
./DoubleCRT.d \
./EncryptedArray.d \
./EvalMap.d \
./FHE.d \
./FHEContext.d \
./IndexSet.d \
./KeySwitching.d \
./NewBluesteinFFT.d \
./NumbTh.d \
./OldEvalMap.d \
./OptimizePermutations.d \
./PAlgebra.d \
./PermNetwork.d \
./Test_EvalMap.d \
./Test_General.d \
./Test_IO.d \
./Test_LinPoly.d \
./Test_OldEvalMap.d \
./Test_PAlgebra.d \
./Test_Permutations.d \
./Test_PolyEval.d \
./Test_Powerful.d \
./Test_Replicate.d \
./Test_Timing.d \
./Test_bootstrapping.d \
./Test_extractDigits.d \
./Test_matmul.d \
./bluestein.d \
./cgauss.d \
./debugging.d \
./eqtesting.d \
./extractDigits.d \
./hypercube.d \
./matching.d \
./myprog.d \
./params.d \
./permutations.d \
./polyEval.d \
./powerful.d \
./recryption.d \
./replicate.d \
./rotations.d \
./timing.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O2 -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


