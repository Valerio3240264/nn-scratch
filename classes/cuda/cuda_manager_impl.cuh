#ifndef CUDA_MANAGER_IMPL_CUH
#define CUDA_MANAGER_IMPL_CUH

#include "cuda_manager.cuh"
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <type_traits>
#include <random>
#include <cstdint>

using namespace std;

const int THREADS_PER_BLOCK = 512;

// File-scope inline variables (shared across translation units)
inline bool cuda_available_checked = false;
inline bool cuda_available = false;

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
inline void launch_SGEMV(float* d_w, float* d_input_values, float* d_b, float* d_result, int output_size, int input_size) {
  int num_blocks = output_size;
  
  size_t shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  SGEMV<<<grid, block, shared_mem_size>>>(d_w, d_input_values, d_result, d_b, input_size, output_size);
  
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch update kernel (automatically chooses vectorized or non-vectorized based on alignment)
inline void launch_update(float* d_vector, float* d_update, float learning_rate, int size) {
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

// Launch tiled_backward_Weights kernel (optimized combined backward pass)
inline void launch_tiled_backward_Weights(float* d_w, float* d_input_values, float* d_derivatives, 
                                         float* d_grad_w, float* d_InGrad, float* d_biasGrad, 
                                         int output_size, int input_size) {
  const int TILE_SIZE = 16;
  
  // Grid: one block per tile column
  int num_blocks = (input_size + TILE_SIZE - 1) / TILE_SIZE;
  
  dim3 grid(num_blocks);
  dim3 block(TILE_SIZE, TILE_SIZE);
  
  tiled_backward_Weights<<<grid, block>>>(d_w, d_input_values, d_derivatives, 
                                          d_grad_w, d_InGrad, d_biasGrad, 
                                          output_size, input_size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

/* ACTIVATION FUNCTION KERNELS */
// Launch activation_tanh kernel (automatically chooses vectorized or non-vectorized based on alignment)
inline void launch_activation_tanh(float* d_value, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  if (check_alignment(d_value, size)) {
    vectorized_activation_tanh<<<grid, block>>>(d_value, size);
  } else {
    non_vectorized_activation_tanh<<<grid, block>>>(d_value, size);
  }
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch activation_relu kernel (automatically chooses vectorized or non-vectorized based on alignment)
inline void launch_activation_relu(float* d_value, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  if (check_alignment(d_value, size)) {
    vectorized_activation_relu<<<grid, block>>>(d_value, size);
  } else {
    non_vectorized_activation_relu<<<grid, block>>>(d_value, size);
  }
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch backward_tanh kernel (automatically chooses vectorized or non-vectorized based on alignment)
inline void launch_backward_tanh(float* d_value, float* d_derivatives, float* d_grad, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
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
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
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
inline void launch_backward_linear(float* d_value, float* d_derivatives, float* d_grad, int size) {
  copy_device_to_device(d_grad, d_derivatives, size);
}

/* SOFTMAX KERNELS */
// Launch vector softmax kernel
inline void launch_vector_softmax(float* d_value, float temperature, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  size_t shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;

  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  vector_softmax_kernel<<<grid, block, shared_mem_size>>>(d_value, temperature, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch softmax backward pass
inline void launch_softmax_backward(float* d_value, float* d_derivatives, float* d_grad, float temperature, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  size_t shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  softmax_backward_kernel<<<grid, block, shared_mem_size>>>(d_value, d_derivatives, d_grad, nullptr, temperature, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

/* LOSS KERNELS */
/* ONE-HOT ENCODING KERNEL */
inline void launch_one_hot_encoding(float* d_target, int target_index, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  one_hot_encoding_kernel<<<grid, block>>>(d_target, target_index, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

/* MSE LOSS KERNELS */
inline void launch_mse_loss_kernel(float* d_predictions, float* d_target, float* d_grad, float* d_loss_sum, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  size_t shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;

  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  mse_loss_kernel<<<grid, block, shared_mem_size>>>(d_predictions, d_target, d_grad, d_loss_sum, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch backward MSE loss kernel
inline void launch_backward_mse_loss_kernel(float* d_predictions, float* d_target, float* d_derivatives, float* d_grad, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  backward_mse_loss_kernel<<<grid, block>>>(d_predictions, d_target, d_derivatives, d_grad, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch backward MSE loss kernel (simplified version with derivatives = 1.0)
inline void launch_backward_mse_loss_kernel_simple(float* d_predictions, float* d_target, float* d_grad, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  backward_mse_loss_kernel_simple<<<grid, block>>>(d_predictions, d_target, d_grad, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

/* CROSS ENTROPY LOSS KERNELS */
// Launch cross entropy loss kernel
inline void launch_softmax_cross_entropy_loss_kernel(float* d_predictions, float* d_target, float* d_grad, float* d_loss_sum, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  size_t shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  softmax_cross_entropy_loss_kernel<<<grid, block, shared_mem_size>>>(d_predictions, d_target, d_grad, d_loss_sum, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch backward cross entropy loss kernel
inline void launch_backward_cross_entropy_loss_kernel(float* d_predictions, float* d_target, float* d_derivatives, float* d_grad, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  backward_cross_entropy_loss_kernel<<<grid, block>>>(d_predictions, d_target, d_derivatives, d_grad, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

// Launch backward cross entropy loss kernel (simplified version with derivatives = 1.0)
inline void launch_backward_cross_entropy_loss_kernel_simple(float* d_predictions, float* d_target, float* d_grad, int size) {
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  backward_cross_entropy_loss_kernel_simple<<<grid, block>>>(d_predictions, d_target, d_grad, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

/* ERROR CHECKING */
// Check CUDA error
inline void check_cuda_error(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
    throw std::runtime_error("CUDA operation failed");
  }
}

// Synchronize device
inline void synchronize() {
  CUDA_CHECK_MANAGER(cudaDeviceSynchronize());
}

#endif