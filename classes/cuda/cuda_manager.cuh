#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <type_traits>
#include "../enums.h"
#include <random>

using namespace std;

// Forward declaration of CUDA kernels
/* MATRIX OPERATION KERNELS */
__global__ void SGEMV(float *w, float *in, float *res, float *bias, int K, int M);

// Vector update kernels
__global__ void vectorized_vector_update(float *V, float *U, float learning_rate, int size);
__global__ void non_vectorized_vector_update(float *V, float *U, float learning_rate, int size);

__global__ void backward_W(float *d_w, float *d_input_values, float *d_derivatives, float *d_grad_w, float *prevGrad, int output_size, int input_size);
__global__ void backward_bias(float *d_b, float *d_derivatives, float *d_grad_b, int output_size);

/* ACTIVATION FUNCTION KERNELS */
// Vectorized forward pass
__global__ void vectorized_activation_tanh(float *V, int size);
__global__ void vectorized_activation_relu(float *V, int size);

// Non-vectorized forward pass
__global__ void non_vectorized_activation_tanh(float *V, int size);
__global__ void non_vectorized_activation_relu(float *V, int size);

// Vectorized backward pass
__global__ void vectorized_backward_tanh(float *V, float *derivatives, float *grad, int size);
__global__ void vectorized_backward_relu(float *V, float *derivatives, float *grad, int size);

// Non-vectorized backward pass
__global__ void non_vectorized_backward_tanh(float *V, float *derivatives, float *grad, int size);
__global__ void non_vectorized_backward_relu(float *V, float *derivatives, float *grad, int size);

/* SOFTMAX KERNELS */
__global__ void vector_softmax_kernel(float *d_value, float temperature, int size);

// Backward pass kernels
__global__ void softmax_dot_product_kernel(float *d_value, float *d_derivatives, float *d_dot, int size);
__global__ void softmax_backward_kernel(float *d_value, float *d_derivatives, float *d_grad, float *d_dot, float temperature, int size);

/* LOSS KERNELS */
// One-hot encoding kernel
__global__ void one_hot_encoding_kernel(float *d_target, int target_index, int size);

// Reduction kernel
__global__ void reduce_sum_kernel(float *d_input, float *d_output, int size);

// MSE loss kernels
__global__ void mse_loss_kernel(float *d_predictions, float *d_target, float *d_grad, int size);
__global__ void backward_mse_loss_kernel(float *d_predictions, float *d_target, float *d_derivatives, float *d_grad, int size);
__global__ void backward_mse_loss_kernel_simple(float *d_predictions, float *d_target, float *d_grad, int size);

// Cross entropy loss kernels
__global__ void cross_entropy_loss_kernel(float *d_predictions, float *d_target, float *d_grad, int size);
__global__ void backward_cross_entropy_loss_kernel(float *d_predictions, float *d_target, float *d_derivatives, float *d_grad, int size);
__global__ void backward_cross_entropy_loss_kernel_simple(float *d_predictions, float *d_target, float *d_grad, int size);

// Forward declarations of utility functions
void check_cuda_error(cudaError_t error, const char* file, int line);
void check_cublas_error(cublasStatus_t status, const char* file, int line);

// CUDA error checking macro
#define CUDA_CHECK_MANAGER(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      check_cuda_error(error, __FILE__, __LINE__); \
    } \
  } while(0)

// CuBLAS error checking macro
#define CUBLAS_CHECK_MANAGER(call) \
  do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      check_cublas_error(status, __FILE__, __LINE__); \
    } \
  } while(0)

// File-scope inline variables (shared across translation units)
inline bool cuda_available_checked = false;
inline bool cuda_available = false;
inline cublasHandle_t cublas_handle;
inline bool cublas_initialized = false;

// Function declarations
// Device Management
bool is_cuda_available();

// Memory Management  
template<typename T>
void allocate_device_memory(T** device_ptr, size_t count);
template<typename T>
void allocate_device_memory_zeros(T** device_ptr, size_t count);
template<typename T>
void allocate_device_memory_random(T** device_ptr, size_t count);
template<typename T>
void zero_device_memory(T* device_ptr, size_t count);
void free_device_memory(void* device_ptr);

// Memory Transfers
template<typename T>
void copy_host_to_device(T* device_ptr, const T* host_ptr, size_t count);
template<typename T>
void copy_device_to_host(T* host_ptr, const T* device_ptr, size_t count);
template<typename T>
void copy_device_to_device(T* device_ptr, const T* device_ptr2, size_t count);

// Utility functions
template<typename T>
bool check_alignment(T* ptr, size_t size);

/* KERNEL LAUNCH UTILITIES */
// Weights kernels
void launch_SGEMV(float* d_w, float* d_input_values, float* d_b, float* d_result, int output_size, int input_size);
void launch_update(float* d_vector, float* d_update, float learning_rate, int size);
void launch_backward_W(float* d_w, float* d_input_values, float* d_derivatives, float* d_grad_w, float* prevGrad, int output_size, int input_size);
void launch_backward_bias(float* d_b, float* d_derivatives, float* d_grad_b, int output_size);

// Activation function kernel
void launch_activation_tanh(float* d_value, int size);
void launch_activation_relu(float* d_value, int size);
void launch_backward_tanh(float* d_value, float* d_derivatives, float* d_grad, int size);
void launch_backward_relu(float* d_value, float* d_derivatives, float* d_grad, int size);
void launch_backward_linear(float* d_value, float* d_derivatives, float* d_grad, int size);

// Softmax kernels
void launch_vector_softmax(float* d_value, float temperature, int size);
void launch_softmax_backward(float* d_value, float* d_derivatives, float* d_grad, float temperature, int size, float* d_dot);

// One-hot encoding kernel
void launch_one_hot_encoding(float* d_target, int target_index, int size);

// Reduction kernel
void launch_reduce_sum(float* d_input, float* d_output, int size);

// MSE loss kernels
void launch_mse_loss_kernel(float* d_predictions, float* d_target, float* d_grad, int size);
void launch_backward_mse_loss_kernel(float* d_predictions, float* d_target, float* d_derivatives, float* d_grad, int size);
void launch_backward_mse_loss_kernel_simple(float* d_predictions, float* d_target, float* d_grad, int size);

// Cross entropy loss kernels
void launch_cross_entropy_loss_kernel(float* d_predictions, float* d_target, float* d_grad, int size);
void launch_backward_cross_entropy_loss_kernel(float* d_predictions, float* d_target, float* d_derivatives, float* d_grad, int size);
void launch_backward_cross_entropy_loss_kernel_simple(float* d_predictions, float* d_target, float* d_grad, int size);

// CuBLAS Operations
void init_cublas();
void destroy_cublas();
void launch_gemm(float* d_A, float* d_B, float* d_C, 
               int m, int n, int k,
               bool transpose_A = false, bool transpose_B = false,
               float alpha = 1.0f, float beta = 0.0f);

// Error Checking
void check_cuda_error(cudaError_t error, const char* file, int line);
void check_cublas_error(cublasStatus_t status, const char* file, int line);

// Synchronization
void synchronize();

/* DEVICE MANAGEMENT */
// Function implementations
inline bool is_cuda_available() {
  if (!cuda_available_checked) {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    cuda_available = (error == cudaSuccess && device_count > 0);
    cuda_available_checked = true;
    if (cuda_available) {
      printf("CUDA detected: %d device(s) available\n", device_count);
    } else {
      printf("CUDA not available: %s\n", cudaGetErrorString(error));
    }
  }
  return cuda_available;
}

/* MEMORY MANAGEMENT */
// Allocate device memory
template<typename T>
void allocate_device_memory(T** device_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMalloc(device_ptr, count * sizeof(T)));
}

// Allocate device memory and set to zero
template<typename T>
void allocate_device_memory_zeros(T** device_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMalloc(device_ptr, count * sizeof(T)));
  CUDA_CHECK_MANAGER(cudaMemset(*device_ptr, 0, count * sizeof(T)));
}

// Allocate device memory and set to random values [0, 1]
template<typename T>
void allocate_device_memory_random(T** device_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMalloc(device_ptr, count * sizeof(T)));
  
  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
  
  if (std::is_same<T, double>::value) {
    curandGenerateUniformDouble(generator, reinterpret_cast<double*>(*device_ptr), count);
  } else if (std::is_same<T, float>::value) {
    curandGenerateUniform(generator, reinterpret_cast<float*>(*device_ptr), count);
  } else {
    curandDestroyGenerator(generator);
    throw std::runtime_error("Unsupported type for random number generation. Only float and double are supported.");
  }
  
  curandDestroyGenerator(generator);
}

// Allocate device memory with Xavier/Glorot initialization
// Weights are initialized uniformly in [-scale, scale] where scale = sqrt(1 / input_size)
template<typename T>
void allocate_device_memory_xavier(T** device_ptr, size_t count, size_t input_size) {
  CUDA_CHECK_MANAGER(cudaMalloc(device_ptr, count * sizeof(T)));

  float scale = sqrtf(1.0f / input_size);
  
  // Random number generator
  default_random_engine generator;
  uniform_real_distribution<float> distribution(-scale, scale);

  float *temp = new float[count];
  for (int i = 0; i < count; i++){
    temp[i] = distribution(generator);
  }

  copy_host_to_device(*device_ptr, temp, count);
  delete[] temp;
}

// Set device memory to zero
template<typename T>
void zero_device_memory(T* device_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMemset(device_ptr, 0, count * sizeof(T)));
}

// Free device memory
inline void free_device_memory(void* device_ptr) {
  if (device_ptr) {
    CUDA_CHECK_MANAGER(cudaFree(device_ptr));
  }
}

// Copy host memory to device memory
template<typename T>
void copy_host_to_device(T* device_ptr, const T* host_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMemcpy(device_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
}

// Copy device memory to host memory
template<typename T>
void copy_device_to_host(T* host_ptr, const T* device_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMemcpy(host_ptr, device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// Copy device memory to device memory
template<typename T>
void copy_device_to_device(T* device_ptr, const T* device_ptr2, size_t count) {
  CUDA_CHECK_MANAGER(cudaMemcpy(device_ptr, device_ptr2, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

/* UTILITY FUNCTIONS */
// Check if pointer is 16-byte aligned
template<typename T>
bool check_alignment(T* ptr, size_t size) {
  // Check 16-byte alignment (required for float4 operations)
  bool aligned16 = ( (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0 );
  if (aligned16 && size > 0) {
    return true;
  }
  else{
    return false;
  }
}

/* KERNEL LAUNCH UTILITIES */
/* WEIGHTS KERNELS */
// Launch SGEMV kernel (optimized vectorized matrix-vector multiplication)
inline void launch_SGEMV(float* d_w, float* d_input_values, float* d_b, float* d_result, int output_size, int input_size) {
  // SGEMV uses one block per output row (M blocks)
  // Each block processes one row of the matrix-vector multiplication
  const int THREADS_PER_BLOCK = 256;  // Threads per block for parallel reduction
  
  // Grid: one block per output row (M = output_size)
  int num_blocks = output_size;
  
  // Shared memory size for reduction: one float per thread
  size_t shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  // Launch SGEMV kernel with shared memory allocation
  // Parameters: w (weights MxK), in (input K), res (result M), bias (M), K (input_size), M (output_size)
  SGEMV<<<grid, block, shared_mem_size>>>(d_w, d_input_values, d_result, d_b, input_size, output_size);
  
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch update kernel (automatically chooses vectorized or non-vectorized based on alignment)
inline void launch_update(float* d_vector, float* d_update, float learning_rate, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  // Check alignment of both pointers used in vectorized operations
  if (check_alignment(d_vector, size) && check_alignment(d_update, size)) {
    vectorized_vector_update<<<grid, block>>>(d_vector, d_update, learning_rate, size);
  } else {
    non_vectorized_vector_update<<<grid, block>>>(d_vector, d_update, learning_rate, size);
  }
  
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch backward_W kernel
inline void launch_backward_W(float* d_w, float* d_input_values, float* d_derivatives, float* d_grad_w, float* prevGrad, int output_size, int input_size) {
  // Desired total threads per block (user-specified).
  const int THREADS_PER_BLOCK = 256;
  // Choose blockDim.x (threads per block in x) reasonably:
  // prefer a small, cache-friendly x dimension; clamp to input_size.
  int blockX = 16;                          // chosen default
  if (blockX > input_size) blockX = input_size;
  // Ensure blockX is not larger than THREADS_PER_BLOCK
  if (blockX > THREADS_PER_BLOCK) blockX = THREADS_PER_BLOCK;
  // Compute blockY so that blockX * blockY <= THREADS_PER_BLOCK
  int blockY = THREADS_PER_BLOCK / blockX;
  if (blockY < 1) blockY = 1;
  // Clamp blockY to output_size (no point having more threads than rows per block)
  if (blockY > output_size) blockY = output_size;
  // Final safety: ensure product <= device limit (commonly 1024).
  // If your device has a different max threads per block, adjust accordingly.
  const int DEVICE_MAX_THREADS_PER_BLOCK = 1024;
  if (blockX * blockY > DEVICE_MAX_THREADS_PER_BLOCK) {
    // reduce blockY to fit
    blockY = DEVICE_MAX_THREADS_PER_BLOCK / blockX;
    if (blockY < 1) blockY = 1;
  }

  // Compute grid sizes to cover all columns (input_size) and rows (output_size)
  int gridX = (input_size  + blockX - 1) / blockX;
  int gridY = (output_size + blockY - 1) / blockY;

  dim3 block(blockX, blockY);
  dim3 grid(gridX, gridY);
  
  backward_W<<<grid, block>>>(d_w, d_input_values, d_derivatives, d_grad_w, prevGrad, output_size, input_size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
  
}

// Launch bias_backward kernel
inline void launch_backward_bias(float* d_b, float* d_derivatives, float* d_grad_b, int output_size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  backward_bias<<<grid, block>>>(d_b, d_derivatives, d_grad_b, output_size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
  
}

/* ACTIVATION FUNCTION KERNELS */
// Launch activation_tanh kernel (automatically chooses vectorized or non-vectorized based on alignment)
inline void launch_activation_tanh(float* d_value, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  // Check alignment and choose appropriate kernel
  if (check_alignment(d_value, size)) {
    vectorized_activation_tanh<<<grid, block>>>(d_value, size);
  } else {
    non_vectorized_activation_tanh<<<grid, block>>>(d_value, size);
  }
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch activation_relu kernel (automatically chooses vectorized or non-vectorized based on alignment)
inline void launch_activation_relu(float* d_value, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  // Check alignment and choose appropriate kernel
  if (check_alignment(d_value, size)) {
    vectorized_activation_relu<<<grid, block>>>(d_value, size);
  } else {
    non_vectorized_activation_relu<<<grid, block>>>(d_value, size);
  }
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch backward_tanh kernel (automatically chooses vectorized or non-vectorized based on alignment)
inline void launch_backward_tanh(float* d_value, float* d_derivatives, float* d_grad, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  // Check alignment of all pointers used in vectorized operations
  if (check_alignment(d_value, size) && 
      check_alignment(d_derivatives, size) && 
      check_alignment(d_grad, size)) {
    vectorized_backward_tanh<<<grid, block>>>(d_value, d_derivatives, d_grad, size);
  } else {
    non_vectorized_backward_tanh<<<grid, block>>>(d_value, d_derivatives, d_grad, size);
  }
  CUDA_CHECK_MANAGER(cudaGetLastError());
  
}

// Launch backward_relu kernel (automatically chooses vectorized or non-vectorized based on alignment)
inline void launch_backward_relu(float* d_value, float* d_derivatives, float* d_grad, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  // Check alignment of all pointers used in vectorized operations
  if (check_alignment(d_value, size) && 
      check_alignment(d_derivatives, size) && 
      check_alignment(d_grad, size)) {
    vectorized_backward_relu<<<grid, block>>>(d_value, d_derivatives, d_grad, size);
  } else {
    non_vectorized_backward_relu<<<grid, block>>>(d_value, d_derivatives, d_grad, size);
  }
  CUDA_CHECK_MANAGER(cudaGetLastError());
  
}

// Launch backward_linear - optimized using memcpy (linear activation has identity derivative)
// For linear activation: grad = derivatives (no computation needed, just copy)
inline void launch_backward_linear(float* d_value, float* d_derivatives, float* d_grad, int size) {
  // Linear activation backward pass is just copying derivatives to grad
  // This is more efficient than launching a kernel for a simple memory copy
  copy_device_to_device(d_grad, d_derivatives, size);
}

/* SOFTMAX KERNELS */
// Launch vector softmax kernel
inline void launch_vector_softmax(float* d_value, float temperature, int size) {
  const int THREADS_PER_BLOCK = 1024;
  
  dim3 grid(1);  // Single block kernel
  dim3 block(THREADS_PER_BLOCK);
  
  vector_softmax_kernel<<<grid, block>>>(d_value, temperature, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch softmax backward pass
inline void launch_softmax_backward(float* d_value, float* d_derivatives, float* d_grad, float temperature, int size, float* d_dot) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  // Zero the dot product buffer
  zero_device_memory(d_dot, 1);
  
  // Step 1: Compute dot product
  softmax_dot_product_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_value, d_derivatives, d_dot, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
  CUDA_CHECK_MANAGER(cudaDeviceSynchronize());  // Ensure dot product is computed before use
  
  // Step 2: Compute gradient (pass device pointer directly - NO HOST COPY!)
  softmax_backward_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_value, d_derivatives, d_grad, d_dot, temperature, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
  
}

/* LOSS KERNELS */
/* ONE-HOT ENCODING KERNEL */
// Launch one-hot encoding kernel - writes directly to device memory
inline void launch_one_hot_encoding(float* d_target, int target_index, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  one_hot_encoding_kernel<<<grid, block>>>(d_target, target_index, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

/* REDUCTION KERNEL */
// Launch reduction kernel to sum array elements
inline void launch_reduce_sum(float* d_input, float* d_output, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  reduce_sum_kernel<<<grid, block>>>(d_input, d_output, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

/* MSE LOSS KERNELS */
// Launch MSE loss kernel
inline void launch_mse_loss_kernel(float* d_predictions, float* d_target, float* d_grad, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  mse_loss_kernel<<<grid, block>>>(d_predictions, d_target, d_grad, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch backward MSE loss kernel
inline void launch_backward_mse_loss_kernel(float* d_predictions, float* d_target, float* d_derivatives, float* d_grad, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  backward_mse_loss_kernel<<<grid, block>>>(d_predictions, d_target, d_derivatives, d_grad, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
  
}

// Launch backward MSE loss kernel (simplified version with derivatives = 1.0)
inline void launch_backward_mse_loss_kernel_simple(float* d_predictions, float* d_target, float* d_grad, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  backward_mse_loss_kernel_simple<<<grid, block>>>(d_predictions, d_target, d_grad, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
  
}

/* CROSS ENTROPY LOSS KERNELS */
// Launch cross entropy loss kernel
inline void launch_cross_entropy_loss_kernel(float* d_predictions, float* d_target, float* d_grad, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  cross_entropy_loss_kernel<<<grid, block>>>(d_predictions, d_target, d_grad, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch backward cross entropy loss kernel
inline void launch_backward_cross_entropy_loss_kernel(float* d_predictions, float* d_target, float* d_derivatives, float* d_grad, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  backward_cross_entropy_loss_kernel<<<grid, block>>>(d_predictions, d_target, d_derivatives, d_grad, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
  
}

// Launch backward cross entropy loss kernel (simplified version with derivatives = 1.0)
inline void launch_backward_cross_entropy_loss_kernel_simple(float* d_predictions, float* d_target, float* d_grad, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  backward_cross_entropy_loss_kernel_simple<<<grid, block>>>(d_predictions, d_target, d_grad, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

/* CUBLAS OPERATIONS */
// Initialize cuBLAS
inline void init_cublas() {
  if (!cublas_initialized) {
    CUBLAS_CHECK_MANAGER(cublasCreate(&cublas_handle));
    cublas_initialized = true;
  }
}

// Destroy cuBLAS
inline void destroy_cublas() {
  if (cublas_initialized) {
    CUBLAS_CHECK_MANAGER(cublasDestroy(cublas_handle));
    cublas_initialized = false;
  }
}

// Launch gemm kernel
inline void launch_gemm(float* d_A, float* d_B, float* d_C,
                        int m, int n, int k,
                        bool transpose_A, bool transpose_B,
                        float alpha, float beta) {  
  cublasOperation_t trans_A = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t trans_B = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;
  
  int lda = transpose_A ? k : m;
  int ldb = transpose_B ? n : k;
  int ldc = m;
  
  CUBLAS_CHECK_MANAGER(
    cublasSgemm(cublas_handle,
                trans_A, trans_B,
                m, n, k,
                &alpha,
                d_A, lda,
                d_B, ldb,
                &beta,
                d_C, ldc)
  );
}

/* ERROR CHECKING */
// Check CUDA error
inline void check_cuda_error(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
    throw std::runtime_error("CUDA operation failed");
  }
}

// Check cuBLAS error
inline void check_cublas_error(cublasStatus_t status, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    const char* error_string;
    switch(status) {
      case CUBLAS_STATUS_NOT_INITIALIZED:
        error_string = "CUBLAS_STATUS_NOT_INITIALIZED";
        break;
      case CUBLAS_STATUS_ALLOC_FAILED:
        error_string = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
      case CUBLAS_STATUS_INVALID_VALUE:
        error_string = "CUBLAS_STATUS_INVALID_VALUE";
        break;
      case CUBLAS_STATUS_ARCH_MISMATCH:
        error_string = "CUBLAS_STATUS_ARCH_MISMATCH";
        break;
      case CUBLAS_STATUS_MAPPING_ERROR:
        error_string = "CUBLAS_STATUS_MAPPING_ERROR";
        break;
      case CUBLAS_STATUS_EXECUTION_FAILED:
        error_string = "CUBLAS_STATUS_EXECUTION_FAILED";
        break;
      case CUBLAS_STATUS_INTERNAL_ERROR:
        error_string = "CUBLAS_STATUS_INTERNAL_ERROR";
        break;
      case CUBLAS_STATUS_NOT_SUPPORTED:
        error_string = "CUBLAS_STATUS_NOT_SUPPORTED";
        break;
      case CUBLAS_STATUS_LICENSE_ERROR:
        error_string = "CUBLAS_STATUS_LICENSE_ERROR";
        break;
      default:
        error_string = "UNKNOWN_CUBLAS_ERROR";
        break;
    }
    fprintf(stderr, "cuBLAS error at %s:%d: %s\n", file, line, error_string);
    throw std::runtime_error("cuBLAS operation failed");
  }
}

// Synchronize device
inline void synchronize() {
  CUDA_CHECK_MANAGER(cudaDeviceSynchronize());
}