# CUDA Compilation Guide 🚀

## Prerequisites
- Install Ubuntu on your machine
- Install CUDA toolkit

## Project Folders
- **build**: holds intermediate object files produced during compilation (e.g., `matrix.o`, `test_cuda_weights.o`). Safe to delete; recreated on next build.
- **bin**: contains final executables you run (e.g., `test_cuda_weights` or `test_cuda_weights.exe`).
- **test**: contains test sources (e.g., `test/test_cuda_weights.cu`) that verify correctness and performance.

## Compilation Steps
1. Open Windows Command Prompt
2. Launch Ubuntu WSL:
   ```
   wsl -d Ubuntu
   ```

3. Navigate to your project directory

4. Compile CUDA file using NVIDIA CUDA Compiler (nvcc):
   ```
   nvcc -o output_file_name input_file_name.cu
   ```

## Recommended commands (separate compile + link)

### WSL (Ubuntu)
```bash
# 1) Compile CUDA kernel to object file
nvcc -std=c++17 -O2 -rdc=true -I. -c Cuda_operations/matrix.cu -o build/matrix.o

# 2) Compile test to object file
nvcc -std=c++17 -O2 -rdc=true -I. -c test/test_cuda_weights.cu -o build/test_cuda_weights.o

# 3) Link objects into final executable
nvcc -std=c++17 -O2 -rdc=true build/test_cuda_weights.o build/matrix.o -o bin/test_cuda_weights
```

### Native Windows (PowerShell, CUDA on PATH)
```powershell
# 1) Compile CUDA kernel to object file
nvcc -std=c++17 -O2 -rdc=true -I. -c Cuda_operations\matrix.cu -o build\matrix.obj

# 2) Compile test to object file
nvcc -std=c++17 -O2 -rdc=true -I. -c test\test_cuda_weights.cu -o build\test_cuda_weights.obj

# 3) Link objects into final executable
nvcc -std=c++17 -O2 -rdc=true build\test_cuda_weights.obj build\matrix.obj -o bin\test_cuda_weights.exe
```

### One-liner build (quick test)
```bash
nvcc -std=c++17 -O2 -rdc=true -I. test/test_cuda_weights.cu Cuda_operations/matrix.cu -o bin/test_cuda_weights
```

## What each command does
- **Compile kernel**: turns `Cuda_operations/matrix.cu` into an object file (`matrix.o`/`matrix.obj`) containing device code (`__global__` kernel) and any host code in that file.
- **Compile test**: turns `test/test_cuda_weights.cu` into an object file that references the kernel symbol and includes your library headers.
- **Link**: resolves all symbols (host and device) across the objects and produces the final executable in `bin`.

## Why these flags
- `-std=c++17`: use C++17 language standard for host code (and host portions inside `.cu` files).
- `-O2`: enable compiler optimizations for reasonable performance without long build times.
- `-rdc=true`: enable relocatable device code so device symbols (`__global__` kernels) defined in one `.cu` can be referenced from another; required when splitting device code across multiple translation units.
- `-I.`: add project root to the header search path so includes like `#include "classes/weights.h"` resolve.
- `-c`: compile-only; produce an object file (`.o`/`.obj`) without linking.
- `-o <file>`: set the output file name (object or executable, depending on whether `-c` is present).