# This Makefile compiles the implementation in this directory along with
# the known answer tests generator located in the
# ../../../KAT/generator/ directory. The output is an executable program
# in the build/ subdirectory, whose name starts with 'kat', followed by
# the implementation name (e.g. 'kat512int' for the 'falcon512int'
# implementation). This program, when executed, generates the .req and
# .rsp files in the expected NIST format.

.POSIX:

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-12.8

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include 
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

# NVCC compiler options:
NVCC=$(CUDA_ROOT_DIR)/bin/nvcc
NVCC_FLAGS= -rdc=true -arch sm_80
NVCC_FLAGS += -use_fast_math 
NVCC_LIBS=

CC = gcc
CFLAGS = -W -Wall -O2
LD = c99
LDFLAGS = 
LIBS = 

OBJ1 = build/codec_gpu.o build/common_gpu.o build/fft_gpu.o build/fpr_gpu.o build/keygen_gpu.o build/nist_gpu.o build/rng_gpu.o build/shake_gpu.o build/sign_gpu.o build/vrfy_gpu.o

OBJ2 = build/PQCgenKAT_sign_gpu.o build/katrng_gpu.o

OBJ3 = build/codec.o build/common.o build/fft.o build/fpr.o build/keygen.o build/nist.o build/rng.o build/shake.o build/sign.o build/vrfy.o

OBJ4 = build/katrng.o 

OBJ5 = build/PQCgenKAT_sign.o

HEAD1 = GPU/api_gpu.cuh GPU/fpr_gpu.cuh GPU/inner_gpu.cuh CPU/typed.h
HEAD2 = GPU/api_gpu.cuh GPU/katrng_gpu.cuh GPU/inner_gpu.cuh CPU/typed.h
HEAD3 = CPU/api.h CPU/fpr.h CPU/inner.h CPU/typed.h
HEAD4 = CPU/api.h CPU/katrng.h CPU/inner.h CPU/typed.h

all: build build/kat512double_gpu

build:
	-mkdir build

clean:
	-rm -f build/kat512double_gpu $(OBJ3) $(OBJ4) $(OBJ1) $(OBJ2) 


# Link c and CUDA compiled object files to target executable:
build/kat512double_gpu: $(OBJ3) $(OBJ4) $(OBJ1) $(OBJ2) 
	$(NVCC) $(NVCC_FLAGS)  $(OBJ3) $(OBJ4) $(OBJ1) $(OBJ2) -o $@ $(CUDA_INC_DIR) -I ../CPU -I ../GPU $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) 

PTX: GPU/fft_gpu.cu $(HEAD1)
	$(NVCC) $(NVCC_FLAGS) -ptx -o build/fft_gpu.ptx GPU/fft_gpu.cu

build/codec_gpu.o: GPU/codec_gpu.cu $(HEAD1)
	$(NVCC) $(NVCC_FLAGS) -c -o build/codec_gpu.o GPU/codec_gpu.cu

build/common_gpu.o: GPU/common_gpu.cu $(HEAD1)
	$(NVCC) $(NVCC_FLAGS) -c -o build/common_gpu.o GPU/common_gpu.cu

# build/fft_gpu.o: GPU/fft_gpu.ptx $(HEAD1)
# 	$(NVCC) $(NVCC_FLAGS) -c -o build/fft_gpu.o GPU/fft_gpu.ptx
	
build/fft_gpu.o: GPU/fft_gpu.cu $(HEAD1)
	$(NVCC) $(NVCC_FLAGS) -c -o build/fft_gpu.o GPU/fft_gpu.cu

build/fpr_gpu.o: GPU/fpr_gpu.cu $(HEAD1)
	$(NVCC) $(NVCC_FLAGS) -c -o build/fpr_gpu.o GPU/fpr_gpu.cu

build/keygen_gpu.o: GPU/keygen_gpu.cu $(HEAD1)
	$(NVCC) $(NVCC_FLAGS) -c -o build/keygen_gpu.o GPU/keygen_gpu.cu

build/nist_gpu.o: GPU/nist_gpu.cu $(HEAD1)
	$(NVCC) $(NVCC_FLAGS) -c -o build/nist_gpu.o GPU/nist_gpu.cu

build/rng_gpu.o: GPU/rng_gpu.cu $(HEAD1)
	$(NVCC) $(NVCC_FLAGS) -c -o build/rng_gpu.o GPU/rng_gpu.cu

build/shake_gpu.o: GPU/shake_gpu.cu $(HEAD1)
	$(NVCC) $(NVCC_FLAGS) -c -o build/shake_gpu.o GPU/shake_gpu.cu

build/sign_gpu.o: GPU/sign_gpu.cu $(HEAD1)
	$(NVCC) $(NVCC_FLAGS) -c -o build/sign_gpu.o GPU/sign_gpu.cu

build/vrfy_gpu.o: GPU/vrfy_gpu.cu $(HEAD1)
	$(NVCC) $(NVCC_FLAGS) -c -o build/vrfy_gpu.o GPU/vrfy_gpu.cu

build/PQCgenKAT_sign_gpu.o: test/PQCgenKAT_sign_gpu.cu $(HEAD2) $(HEAD4) build/nist.o build/katrng.o
	$(NVCC) $(NVCC_FLAGS)  -I GPU -I CPU -DALGNAME=falcon512int -c -o build/PQCgenKAT_sign_gpu.o test/PQCgenKAT_sign_gpu.cu

build/katrng_gpu.o: GPU/katrng_gpu.cu $(HEAD2)
	$(NVCC) $(NVCC_FLAGS) -I GPU -c -o build/katrng_gpu.o GPU/katrng_gpu.cu	
	
build/codec.o: CPU/codec.cpp $(HEAD3)
	$(CC) $(CFLAGS)  -c -o build/codec.o CPU/codec.cpp

build/common.o: CPU/common.cpp $(HEAD3)
	$(CC) $(CFLAGS)  -c -o build/common.o CPU/common.cpp

build/fft.o: CPU/fft.cpp $(HEAD3)
	$(CC) $(CFLAGS)  -c -o build/fft.o CPU/fft.cpp

build/fpr.o: CPU/fpr.cpp $(HEAD3)
	$(CC) $(CFLAGS)  -c -o build/fpr.o CPU/fpr.cpp

build/keygen.o: CPU/keygen.cpp $(HEAD3)
	$(CC) $(CFLAGS)  -c -o build/keygen.o CPU/keygen.cpp

build/nist.o: CPU/nist.cpp $(HEAD3)
	$(CC) $(CFLAGS)  -c -o build/nist.o CPU/nist.cpp

build/rng.o: CPU/rng.cpp $(HEAD3)
	$(CC) $(CFLAGS)  -c -o build/rng.o CPU/rng.cpp

build/shake.o: CPU/shake.cpp $(HEAD3)
	$(CC) $(CFLAGS)  -c -o build/shake.o CPU/shake.cpp

build/sign.o: CPU/sign.cpp $(HEAD3)
	$(CC) $(CFLAGS)  -c -o build/sign.o CPU/sign.cpp

build/vrfy.o: CPU/vrfy.cpp $(HEAD3)
	$(CC) $(CFLAGS)  -c -o build/vrfy.o CPU/vrfy.cpp

build/katrng.o: CPU/katrng.cpp $(HEAD4)
	$(CC) $(CFLAGS)  -I CPU -c -o build/katrng.o CPU/katrng.cpp




