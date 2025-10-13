#ifndef CUDA_MANAGER_H
#define CUDA_MANAGER_H

#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <type_traits>

/*
TODO: 

Kernels:
1: vector update.
2: zeros matrix.
3: GeMM.
4: Update vector(learning rate).
*/

// Forward declaration of CUDA kernel
__global__ void matrix_vector_multiplication(double *M, double *V, double *R, int output_size, int input_size);

// CUDA error checking macro - centralized in CudaManager
#define CUDA_CHECK_MANAGER(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      CudaManager::check_cuda_error(error, __FILE__, __LINE__); \
    } \
  } while(0)

class CudaManager {
private:
  static bool cuda_available_checked;
  static bool cuda_available;
    
public:
  // Device Management
  static bool is_cuda_available();
  
  // Memory Management  
  template<typename T>
  static void allocate_device_memory(T** device_ptr, size_t count);

  template<typename T>
  static void allocate_device_memory_zeros(T** device_ptr, size_t count);

  template<typename T>
  static void allocate_device_memory_random(T** device_ptr, size_t count);

  template<typename T>
  static void zero_device_memory(T* device_ptr, size_t count);

  static void free_device_memory(void* device_ptr);
  
  // Memory Transfers
  template<typename T>
  static void copy_host_to_device(T* device_ptr, const T* host_ptr, size_t count);
  
  template<typename T>
  static void copy_device_to_host(T* host_ptr, const T* device_ptr, size_t count);
  
  // Kernel Launch Utilities
  static void launch_matrix_vector_multiply(double* d_matrix, double* d_vector, double* d_result, int output_size, int input_size);
    // TODO:
  static void launch_update(double* d_vector, double* d_update, double learning_rate, int size);
  static void launch_activation_function(double* d_value, Activation_name function_name, int size);

  // Error Checking
  static void check_cuda_error(cudaError_t error, const char* file, int line);
  
  // Synchronization
  static void synchronize();
};

// Static member initialization
bool CudaManager::cuda_available_checked = false;
bool CudaManager::cuda_available = false;

/* DEVICE MANAGEMENT */
// Implementation of static methods

bool CudaManager::is_cuda_available() {
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
template<typename T>
void CudaManager::allocate_device_memory(T** device_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMalloc(device_ptr, count * sizeof(T)));
}

template<typename T>
void CudaManager::allocate_device_memory_zeros(T** device_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMalloc(device_ptr, count * sizeof(T)));
  CUDA_CHECK_MANAGER(cudaMemset(*device_ptr, 0, count * sizeof(T)));
}

template<typename T>
void CudaManager::allocate_device_memory_random(T** device_ptr, size_t count) {
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

template<typename T>
void CudaManager::zero_device_memory(T* device_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMemset(device_ptr, 0, count * sizeof(T)));
}

void CudaManager::free_device_memory(void* device_ptr) {
  if (device_ptr) {
    CUDA_CHECK_MANAGER(cudaFree(device_ptr));
  }
}

template<typename T>
void CudaManager::copy_host_to_device(T* device_ptr, const T* host_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMemcpy(device_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void CudaManager::copy_device_to_host(T* host_ptr, const T* device_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMemcpy(host_ptr, device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

/* KERNEL LAUNCH UTILITIES */
void CudaManager::launch_matrix_vector_multiply(double* d_matrix, double* d_vector, double* d_result, int output_size, int input_size) {
  // Optimal threading configuration for the new kernel
  const int THREADS_PER_BLOCK = 256;  // Balanced for good occupancy
  int num_blocks = (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  // Ceiling division
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  matrix_vector_multiplication<<<grid, block>>>(d_matrix, d_vector, d_result, output_size, input_size);
  
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

void CudaManager::launch_update(double* d_vector, double* d_update, double learning_rate, int size) {
  /*TODO*/
}

void CudaManager::launch_activation_function(double* d_value, Activation_name function_name, int size) {
  /*TODO*/
}

/* ERROR CHECKING */
void CudaManager::check_cuda_error(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
    throw std::runtime_error("CUDA operation failed");
  }
}

void CudaManager::synchronize() {
  CUDA_CHECK_MANAGER(cudaDeviceSynchronize());
}

#endif