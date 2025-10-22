#ifndef CUDA_MANAGER_H
#define CUDA_MANAGER_H

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <type_traits>
#include "enums.h"

// Forward declaration of CUDA kernels
__global__ void matrix_vector_multiplication(double *M, double *V, double *R, int output_size, int input_size);
__global__ void vector_update(double *V, double *U, double learning_rate, int size);
__global__ void activation_tanh(double *V, int size);
__global__ void activation_relu(double *V, int size);

// CUDA error checking macro - centralized in CudaManager
#define CUDA_CHECK_MANAGER(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      CudaManager::check_cuda_error(error, __FILE__, __LINE__); \
    } \
  } while(0)

// CuBLAS error checking macro
#define CUBLAS_CHECK_MANAGER(call) \
  do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      CudaManager::check_cublas_error(status, __FILE__, __LINE__); \
    } \
  } while(0)

class CudaManager {
private:
  static bool cuda_available_checked;
  static bool cuda_available;
  static cublasHandle_t cublas_handle;
  static bool cublas_initialized;
    
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

  // CuBLAS Operations
  static void init_cublas();
  static void destroy_cublas();
  static void launch_gemm(double* d_A, double* d_B, double* d_C, 
                         int m, int n, int k,
                         bool transpose_A = false, bool transpose_B = false,
                         double alpha = 1.0, double beta = 0.0);

  // Error Checking
  static void check_cuda_error(cudaError_t error, const char* file, int line);
  static void check_cublas_error(cublasStatus_t status, const char* file, int line);
  
  // Synchronization
  static void synchronize();
};

// Static member initialization
bool CudaManager::cuda_available_checked = false;
bool CudaManager::cuda_available = false;
cublasHandle_t CudaManager::cublas_handle;
bool CudaManager::cublas_initialized = false;

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
  const int THREADS_PER_BLOCK = 256;
  int num_threads = (size + 1) / 2;
  int num_blocks = (num_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  vector_update<<<grid, block>>>(d_vector, d_update, learning_rate, size);
  
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

void CudaManager::launch_activation_function(double* d_value, Activation_name function_name, int size) {
  const int THREADS_PER_BLOCK = 256;
  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  dim3 grid(num_blocks);
  dim3 block(THREADS_PER_BLOCK);
  
  switch(function_name) {
    case TANH:
      activation_tanh<<<grid, block>>>(d_value, size);
      CUDA_CHECK_MANAGER(cudaGetLastError());
      break;
      
    case RELU:
      activation_relu<<<grid, block>>>(d_value, size);
      CUDA_CHECK_MANAGER(cudaGetLastError());
      break;
      
    case LINEAR:
      break;
      
    default:
      throw std::runtime_error("Invalid activation function");
  }
}

/* CUBLAS OPERATIONS */
void CudaManager::init_cublas() {
  if (!cublas_initialized) {
    CUBLAS_CHECK_MANAGER(cublasCreate(&cublas_handle));
    cublas_initialized = true;
  }
}

void CudaManager::destroy_cublas() {
  if (cublas_initialized) {
    CUBLAS_CHECK_MANAGER(cublasDestroy(cublas_handle));
    cublas_initialized = false;
  }
}

void CudaManager::launch_gemm(double* d_A, double* d_B, double* d_C,
                              int m, int n, int k,
                              bool transpose_A, bool transpose_B,
                              double alpha, double beta) {
  // Initialize cuBLAS if not already initialized
  if (!cublas_initialized) {
    init_cublas();
  }
  
  // cuBLAS uses column-major ordering, so we need to be careful about the operation
  // C = alpha * op(A) * op(B) + beta * C
  // where op(X) = X or X^T depending on the transpose flags
  
  cublasOperation_t trans_A = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t trans_B = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;
  
  // Leading dimensions
  int lda = transpose_A ? k : m;
  int ldb = transpose_B ? n : k;
  int ldc = m;
  
  // Call cuBLAS DGEMM (double precision general matrix multiply)
  CUBLAS_CHECK_MANAGER(
    cublasDgemm(cublas_handle,
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
void CudaManager::check_cuda_error(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
    throw std::runtime_error("CUDA operation failed");
  }
}

void CudaManager::check_cublas_error(cublasStatus_t status, const char* file, int line) {
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

void CudaManager::synchronize() {
  CUDA_CHECK_MANAGER(cudaDeviceSynchronize());
}

#endif