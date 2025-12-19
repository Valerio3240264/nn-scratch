# CUDA Compilation Reference

## Prerequisites
- Install Ubuntu on your machine
- Install CUDA toolkit

## Project Folders
- **build**: holds intermediate object files produced during compilation (e.g., `matrix.o`, `test_cuda_weights.o`). Safe to delete; recreated on next build.
- **bin**: contains final executables you run (e.g., `test_cuda_weights` or `test_cuda_weights.exe`).
- **test**: contains test sources (e.g., `test/test_cuda_weights.cu`) that verify correctness and performance.

### Compilation Commands

#### For MNIST GPU Test (Recommended)

**Linux/WSL:**
```bash
# Compile all kernel files
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c Cuda_operations/activation.cu -o build/activation.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c Cuda_operations/loss.cu -o build/loss.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c Cuda_operations/matrix.cu -o build/matrix.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c Cuda_operations/softmax.cu -o build/softmax.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c Cuda_operations/utils.cu -o build/utils.o

# Compile test
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c test/test_mnist_gpu.cu -o build/test_mnist_gpu.o

# Link all together
nvcc -std=c++17 -O2 -use_fast_math -rdc=true build/test_mnist_gpu.o build/activation.o build/loss.o build/matrix.o build/softmax.o build/utils.o -lcurand -o bin/test_mnist_gpu
```