# FP-GOLF

## 1. Introduction
This project provides a GPU-accelerated implementation of Post-Quantum Cryptography (PQC) algorithms. The primary objective is to generate high-throughput Known Answer Tests (KAT) for the NIST standardization evaluation process. 
Through CPU-GPU heterogeneous co-processing, this system significantly accelerates the computation of cryptographic primitives (e.g., FFT transformations, sampling, key generation, and signature verification).

## 2. Prerequisites
Before compiling and running this system, please ensure your test platform meets the following hardware and software requirements:
* **Operating System**: Linux (Ubuntu 20.04 or higher recommended)
* **Compiler**: GCC (with C99 standard support)
* **CUDA Toolkit**: NVIDIA CUDA Toolkit 12.x (default configuration is 12.8)
* **Hardware**: CUDA-enabled NVIDIA GPU (Volta or Ampere architecture computing cards recommended, e.g., Titan V, A100, etc.)

## 3. Repository Structure
```text
.
├── CPU/            # Sequential logic and interface definitions on the CPU side
├── GPU/            # CUDA parallel operators on the GPU side (FFT, SHAKE, signing, etc.)
├── test/           # Testing framework and top-level KAT generation logic
├── build/          # Build output directory (stores .o files and executables)
└── Makefile        # Automated build script
```

## 4. Environment Configuration and Compilation Instructions
This project relies on the NVIDIA CUDA toolkit for GPU acceleration. Since hardware architectures and software environments vary across test platforms, you **must** adapt and modify the `Makefile` in the project root directory before compiling.

### 4.1 Configuring the CUDA Toolkit Path (`CUDA_ROOT_DIR`)
To ensure the compiler (`nvcc`) and associated linking libraries are correctly invoked, please modify the `CUDA_ROOT_DIR` variable in the `Makefile`.
Around line 12 of the `Makefile`, replace it with the actual CUDA root directory path on your system (which can be confirmed by running `which nvcc`):
```makefile
# Modify this to the actual CUDA path of your current test platform
CUDA_ROOT_DIR=/usr/local/cuda-12.8
```

### 4.2 Adapting to the Target GPU's Compute Capability
To fully utilize the underlying parallel computing resources of the hardware, the compute capability code of the target NVIDIA GPU must be accurately declared in the compilation options. Around line 21 of the `Makefile`, locate the `NVCC_FLAGS` variable and modify the `-arch` parameter:
* **`sm_80`**: Suitable for Ampere architecture data center cards such as NVIDIA A100 / A30 (Default).
* **`sm_70`**: Suitable for Volta architecture cards such as NVIDIA Titan V / V100.
* **`sm_86`**: Suitable for consumer-grade Ampere architecture cards such as RTX 3080/3090.

```makefile
# Modify the sm_80 below to match the compute capability of your target platform
NVCC_FLAGS= -rdc=true -arch sm_80
NVCC_FLAGS += -use_fast_math 
```

### 4.3 Switching Test Modes (tree vs. dyn)
By default, the codebase is configured to test the **tree** mode. If you want to evaluate the **dyn** (dynamic) mode, please modify the source code before compiling:
Open the `test/PQCgenKAT_sign_gpu.cu` file, locate the invocations of `crypto_expand_gpu()` and `crypto_sign_tree_gpu()`, and replace these two function calls with a single call to `crypto_sign_dyn_gpu()`.

### 4.4 Compilation
Once the above configurations and mode selections are complete, run the following commands in the project root directory to build the project:
```bash
make clean
make all
```

## 5. Execution and Result Generation
After successful compilation, the target executable will be output to the `build/` directory with the filename `kat512double_gpu`.

⚠️ **Execution Path Warning:**
To ensure the result files are written correctly, **you must execute the program from the project's root directory (i.e., the directory containing the `Makefile`)**. Do not run it from inside the `build/` folder.

**Execution Command:**
```bash
# Ensure you are currently in the project root directory
./build/kat512double_gpu
```
Upon normal execution, the terminal will output whether the KAT test has passed, along with the performance metrics of 3,000 sign/verify tests.
