################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../aes/Test_AES.cpp \
../aes/homAES.cpp \
../aes/simpleAES.cpp 

OBJS += \
./aes/Test_AES.o \
./aes/homAES.o \
./aes/simpleAES.o 

CPP_DEPS += \
./aes/Test_AES.d \
./aes/homAES.d \
./aes/simpleAES.d 


# Each subdirectory must supply rules for building sources it contributes
aes/%.o: ../aes/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O2 -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


