# CUDA Weights Test Suite

This comprehensive test suite validates the CUDA-accelerated matrix-vector multiplication implementation in the `weights` class.

## 📋 What It Tests

### ✅ **Correctness Tests**
- **Small Matrix Verification**: Tests 2×3 matrices with known expected results
- **CPU vs GPU Comparison**: Ensures identical results between computation methods
- **Edge Cases**: Single element matrices, zero inputs, sequential patterns

### 🎯 **Input Pattern Tests**
- **Zeros**: All zero input values
- **Ones**: All one input values  
- **Sequential**: Values 1, 2, 3, ...
- **Alternating**: Pattern of 1, -1, 1, -1, ...
- **Random**: Random values between -1 and 1

### 🚀 **CUDA Functionality Tests**
- **Device Detection**: Verifies CUDA availability detection
- **Automatic Selection**: Tests smart GPU/CPU selection in `operator()`
- **Explicit CUDA**: Tests `operator_cuda()` method
- **Fallback Behavior**: Ensures CPU fallback when CUDA fails

### ⚡ **Performance Benchmarking**
- **Multiple Matrix Sizes**: 100×50, 500×200, 1000×500, 2000×1000
- **Speed Comparison**: CPU vs GPU execution times
- **Speedup Calculation**: Shows performance improvements
- **Iteration Average**: Multiple runs for accurate timing

### 🛡️ **Error Handling Tests**
- **Large Matrices**: Tests memory allocation limits
- **Invalid Inputs**: Zero-sized matrices and edge cases
- **Exception Safety**: Proper exception handling and cleanup

## 🔧 How to Build and Run

### **Option 1: Using the Makefile**
```bash
# Build and run all tests
make -f test_makefile.mk test

# Just build the executable
make -f test_makefile.mk

# Run manually
./test_cuda_weights
```

### **Option 2: Manual Compilation**
```bash
# 1. Compile CUDA kernel
nvcc -arch=sm_50 -std=c++17 -c Cuda_operations/matrix.cu -o matrix_kernel.o

# 2. Compile and link test program
nvcc -std=c++17 -O2 -I. test_cuda_weights.cpp matrix_kernel.o -o test_cuda_weights

# 3. Run tests
./test_cuda_weights
```

### **Option 3: CPU-Only Version**
```bash
# For systems without CUDA
make -f test_makefile.mk cpu-test
```

## 📊 Sample Output

```
CUDA Weights Test Suite
========================

=== Testing Basic Functionality ===
[PASS] Small Matrix CPU Correctness
[PASS] Small Matrix Auto Correctness  
[PASS] Small Matrix CPU vs Auto
[PASS] Single Element Test - Expected 5.0, got 5.0

=== Testing Different Input Patterns ===
[PASS] Pattern: zeros
[PASS] Pattern: ones
[PASS] Pattern: sequential
[PASS] Pattern: alternating
[PASS] Pattern: random

=== Testing CUDA Functionality ===
[PASS] CUDA Availability Detection - CUDA Available
[PASS] Explicit CUDA vs CPU

=== Performance Testing ===
Matrix size: 100x50
  CPU Time: 15 μs/iteration
  Auto Time: 12 μs/iteration
  Speedup: 1.25x
[PASS] Performance Test 100x50 - Speedup: 1.25x

Matrix size: 1000x500
  CPU Time: 1240 μs/iteration
  Auto Time: 89 μs/iteration
  Speedup: 13.93x
[PASS] Performance Test 1000x500 - Speedup: 13.93x

=== Test Summary ===
Total Tests: 15
Passed: 15
Failed: 0
Pass Rate: 100.0%
All tests passed! 🎉
```

## 🔍 What Each Test Validates

| Test Category | Purpose | Expected Behavior |
|---------------|---------|-------------------|
| **Basic Functionality** | Core matrix operations work correctly | CPU and GPU produce identical, mathematically correct results |
| **Input Patterns** | Different data patterns are handled | All input patterns processed consistently across CPU/GPU |
| **CUDA Functionality** | GPU acceleration works properly | Automatic device selection, proper fallback, explicit GPU calls |
| **Performance** | GPU provides speedup | Significant performance improvement for larger matrices |
| **Error Handling** | Graceful handling of edge cases | Proper exceptions, memory management, fallback behavior |

## 🎯 Expected Performance Characteristics

- **Small Matrices (< 100×100)**: CPU may be faster due to GPU overhead
- **Medium Matrices (100×100 to 1000×1000)**: GPU shows moderate speedup (2-5x)
- **Large Matrices (> 1000×1000)**: GPU shows significant speedup (10-50x)

## 🚨 Troubleshooting

### **CUDA Not Available**
```
[PASS] CUDA Availability Detection - CUDA Not Available
  Skipping CUDA-specific tests (CUDA not available)
```
**Solution**: Tests will automatically fall back to CPU-only mode.

### **Compilation Errors**
```
nvcc: command not found
```
**Solution**: Install NVIDIA CUDA Toolkit or use CPU-only version.

### **Memory Errors**
```
CUDA error: out of memory
```
**Solution**: Test automatically handles memory limitations and reports appropriately.

## 📈 Interpreting Results

- **✅ All Pass**: Implementation is working correctly
- **❌ Correctness Failures**: CPU and GPU results don't match - check kernel implementation
- **⚠️ Performance Issues**: GPU slower than expected - check matrix sizes and GPU specifications

The test suite provides comprehensive validation of the CUDA implementation, ensuring both correctness and performance of the matrix-vector multiplication operations.
