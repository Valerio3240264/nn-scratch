# CUDA Compilation Reference

## Prerequisites
- Install Ubuntu on your machine (or use WSL on Windows)
- Install CUDA toolkit (version 11.0 or higher recommended)
- Install g++ compiler (version 9.0 or higher for C++17 support)

## Project Folders
- **build**: holds intermediate object files produced during compilation (e.g., `matrix.o`, `test_cuda_weights.o`). Safe to delete; recreated on next build.
- **bin**: contains final executables you run (e.g., `test_cuda_weights` or `test_cuda_weights.exe`).

## Compilation Commands

### For CPU Tests

**Windows/Linux:**
```bash
# Compile CPU class source files
g++ -std=c++17 -O2 -I. -c classes/cpu/src/input.cpp -o build/input.o
g++ -std=c++17 -O2 -I. -c classes/cpu/src/weights.cpp -o build/weights.o
g++ -std=c++17 -O2 -I. -c classes/cpu/src/activation.cpp -o build/activation_cpu.o
g++ -std=c++17 -O2 -I. -c classes/cpu/src/softmax.cpp -o build/softmax_cpu.o
g++ -std=c++17 -O2 -I. -c classes/cpu/src/mse_loss.cpp -o build/mse_loss_cpu.o
g++ -std=c++17 -O2 -I. -c classes/cpu/src/cross_entropy_loss.cpp -o build/cross_entropy_loss_cpu.o

# Compile MLP source files (as C++)
g++ -std=c++17 -O2 -I. -x c++ -c classes/mlp/src/mlp.cu -o build/mlp_cpu.o
g++ -std=c++17 -O2 -I. -x c++ -c classes/mlp/src/layer.cu -o build/layer_cpu.o

# Compile test file
g++ -std=c++17 -O2 -I. -c test/test_mnist_cpu.cpp -o build/test_mnist_cpu.o

# Link everything together
g++ -std=c++17 -O2 -I. test/test_mnist_cpu.cpp build/mlp_cpu.o build/layer_cpu.o build/input.o build/weights.o build/activation_cpu.o build/softmax_cpu.o build/mse_loss_cpu.o build/cross_entropy_loss_cpu.o -o bin/test_mnist_cpu.exe
```


### For MNIST GPU Test (Recommended)

**Linux/WSL:**
```bash
# Compile CPU class implementation files (needed even for GPU builds)
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cpu/src/input.cpp -o build/input.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cpu/src/weights.cpp -o build/weights.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cpu/src/activation.cpp -o build/activation_cpu.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cpu/src/softmax.cpp -o build/softmax_cpu.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cpu/src/mse_loss.cpp -o build/mse_loss_cpu.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cpu/src/cross_entropy_loss.cpp -o build/cross_entropy_loss_cpu.o

# Compile CUDA class implementation files
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cuda/src/cuda_input.cu -o build/cuda_input.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cuda/src/cuda_weights.cu -o build/cuda_weights.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cuda/src/cuda_activation.cu -o build/cuda_activation.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cuda/src/cuda_softmax.cu -o build/cuda_softmax.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cuda/src/cuda_mse_loss.cu -o build/cuda_mse_loss.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/cuda/src/cuda_cross_entropy_loss.cu -o build/cuda_cross_entropy_loss.o

# Compile MLP implementation files with nvcc
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/mlp/src/mlp.cu -o build/mlp.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c classes/mlp/src/layer.cu -o build/layer.o

# Compile all kernel files
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c Cuda_operations/activation.cu -o build/activation.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c Cuda_operations/loss.cu -o build/loss.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c Cuda_operations/matrix.cu -o build/matrix.o
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c Cuda_operations/softmax.cu -o build/softmax.o

# Compile test
nvcc -std=c++17 -O2 -use_fast_math -rdc=true -I. -c test/test_mnist_gpu.cu -o build/test_mnist_gpu.o

# Link all together (including CPU object files)
nvcc -std=c++17 -O2 -use_fast_math -rdc=true \
  build/test_mnist_gpu.o \
  build/mlp.o \
  build/layer.o \
  build/input.o \
  build/weights.o \
  build/activation_cpu.o \
  build/softmax_cpu.o \
  build/mse_loss_cpu.o \
  build/cross_entropy_loss_cpu.o \
  build/cuda_input.o \
  build/cuda_weights.o \
  build/cuda_activation.o \
  build/cuda_softmax.o \
  build/cuda_mse_loss.o \
  build/cuda_cross_entropy_loss.o \
  build/activation.o \
  build/loss.o \
  build/matrix.o \
  build/softmax.o \
  -lcurand -o bin/test_mnist_gpu
```