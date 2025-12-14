# CUDA Compilation Reference

### File Organization

```
Cuda_operations/
├── activation.cu    - All activation function kernels (vectorized + non-vectorized)
├── matrix.cu        - SGEMV, vector_update, backward_W, backward_bias, loss kernels
├── softmax.cu       - Softmax forward pass kernel
├── utils.cu         - Warp and block reduction utilities
└── loss.cu          - (Currently empty)
```

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
nvcc -std=c++17 -O2 -use_fast_math -rdc=true build/test_mnist_gpu.o build/activation.o build/loss.o build/matrix.o build/softmax.o build/utils.o -lcurand -lcublas -o bin/test_mnist_gpu
```

#### One-liner (Quick Build)

**Linux/WSL:**
```bash
nvcc -std=c++17 -O2 -rdc=true -I. test/test_mnist_gpu.cu Cuda_operations/activation.cu Cuda_operations/matrix.cu Cuda_operations/softmax.cu -o bin/test_mnist_gpu
```

**Windows:**
```powershell
nvcc -std=c++17 -O2 -rdc=true -I. test\test_mnist_gpu.cu Cuda_operations\activation.cu Cuda_operations\matrix.cu Cuda_operations\softmax.cu -o bin\test_mnist_gpu.exe
```
