#ifndef CUDA_MANAGER_IMPL_CUH
#define CUDA_MANAGER_IMPL_CUH

#include "cuda_manager.cuh"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <type_traits>

namespace {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int MATRIX_TILE_SIZE = 16;

inline int block_count(size_t size) {
  return static_cast<int>((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
}

inline const char* curand_status_to_string(curandStatus_t status) {
  switch (status) {
    case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
    default: return "CURAND_STATUS_UNKNOWN";
  }
}

inline void check_curand_error(
    curandStatus_t status,
    const char* file,
    int line) {
  if (status != CURAND_STATUS_SUCCESS) {
    std::fprintf(
        stderr,
        "CURAND error at %s:%d: %s\n",
        file,
        line,
        curand_status_to_string(status));
    throw std::runtime_error("CURAND operation failed");
  }
}

#define CURAND_CHECK_MANAGER(call) \
  do { \
    curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
      check_curand_error(status, __FILE__, __LINE__); \
    } \
  } while (0)

template<typename T>
void allocate_uniform_scaled(
    T** device_ptr,
    size_t count,
    float scale) {
  static_assert(
      std::is_same_v<T, float>,
      "CUDA weight initialization currently supports float weights");
  CUDA_CHECK_MANAGER(cudaMalloc(device_ptr, count * sizeof(T)));

  curandGenerator_t generator = nullptr;
  CURAND_CHECK_MANAGER(
      curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK_MANAGER(
      curandSetPseudoRandomGeneratorSeed(
          generator,
          static_cast<unsigned long long>(std::time(nullptr))));

  CURAND_CHECK_MANAGER(
      curandGenerateUniform(
          generator,
          reinterpret_cast<float*>(*device_ptr),
          count));

  const dim3 block(THREADS_PER_BLOCK);
  const dim3 grid(block_count(count));
  scale_weights<<<grid, block>>>(
      reinterpret_cast<float*>(*device_ptr),
      count,
      scale);
  CUDA_CHECK_MANAGER(cudaGetLastError());
  CURAND_CHECK_MANAGER(curandDestroyGenerator(generator));
}

}  // namespace

inline bool is_cuda_available() {
  int device_count = 0;
  const cudaError_t error = cudaGetDeviceCount(&device_count);
  return error == cudaSuccess && device_count > 0;
}

template<typename T>
void allocate_device_memory(T** device_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMalloc(device_ptr, count * sizeof(T)));
}

template<typename T>
void allocate_device_memory_zeros(T** device_ptr, size_t count) {
  allocate_device_memory(device_ptr, count);
  zero_device_memory(*device_ptr, count);
}

template<typename T>
void allocate_device_memory_xavier(
    T** device_ptr,
    size_t input_size,
    size_t output_size) {
  const size_t count = input_size * output_size;
  const float scale =
      std::sqrt(6.0f / static_cast<float>(input_size + output_size));
  allocate_uniform_scaled(device_ptr, count, scale);
}

template<typename T>
void allocate_device_memory_he(
    T** device_ptr,
    size_t input_size,
    size_t output_size) {
  const size_t count = input_size * output_size;
  const float scale =
      std::sqrt(2.0f / static_cast<float>(input_size));
  allocate_uniform_scaled(device_ptr, count, scale);
}

template<typename T>
void zero_device_memory(T* device_ptr, size_t count) {
  CUDA_CHECK_MANAGER(cudaMemset(device_ptr, 0, count * sizeof(T)));
}

inline void free_device_memory(void* device_ptr) {
  if (device_ptr != nullptr) {
    CUDA_CHECK_MANAGER(cudaFree(device_ptr));
  }
}

template<typename T>
void copy_host_to_device(
    T* device_ptr,
    const T* host_ptr,
    size_t count) {
  CUDA_CHECK_MANAGER(
      cudaMemcpy(
          device_ptr,
          host_ptr,
          count * sizeof(T),
          cudaMemcpyHostToDevice));
}

template<typename T>
void copy_device_to_host(
    T* host_ptr,
    const T* device_ptr,
    size_t count) {
  CUDA_CHECK_MANAGER(
      cudaMemcpy(
          host_ptr,
          device_ptr,
          count * sizeof(T),
          cudaMemcpyDeviceToHost));
}

inline void launch_matmul_forward(
    const float* A,
    const float* B,
    const float* bias,
    float* result,
    int rows,
    int cols,
    int inner) {
  const dim3 block(MATRIX_TILE_SIZE, MATRIX_TILE_SIZE);
  const dim3 grid(
      (cols + MATRIX_TILE_SIZE - 1) / MATRIX_TILE_SIZE,
      (rows + MATRIX_TILE_SIZE - 1) / MATRIX_TILE_SIZE);
  tiled_matmul_add_bias<<<grid, block>>>(
      A, B, bias, result, rows, cols, inner);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_matmul_backward_weights(
    const float* A,
    const float* B,
    float* result,
    int rows,
    int cols,
    int inner) {
  const dim3 block(MATRIX_TILE_SIZE, MATRIX_TILE_SIZE);
  const dim3 grid(
      (cols + MATRIX_TILE_SIZE - 1) / MATRIX_TILE_SIZE,
      (rows + MATRIX_TILE_SIZE - 1) / MATRIX_TILE_SIZE);
  tiled_matmul_transpose_left_accumulate<<<grid, block>>>(
      A, B, result, rows, cols, inner);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_matmul_backward_input(
    const float* A,
    const float* B,
    float* result,
    int rows,
    int cols,
    int inner) {
  const dim3 block(MATRIX_TILE_SIZE, MATRIX_TILE_SIZE);
  const dim3 grid(
      (cols + MATRIX_TILE_SIZE - 1) / MATRIX_TILE_SIZE,
      (rows + MATRIX_TILE_SIZE - 1) / MATRIX_TILE_SIZE);
  tiled_matmul_transpose_right<<<grid, block>>>(
      A, B, result, rows, cols, inner);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_bias_gradient(
    float* result,
    const float* matrix,
    int rows,
    int cols) {
  const dim3 block(THREADS_PER_BLOCK);
  const dim3 grid(block_count(cols));
  accumulate_row_sum<<<grid, block>>>(result, matrix, rows, cols);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_update(
    float* values,
    const float* gradients,
    float learning_rate,
    size_t size) {
  const dim3 block(THREADS_PER_BLOCK);
  const dim3 grid(block_count(size));
  vector_update<<<grid, block>>>(
      values, gradients, learning_rate, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_activation_tanh(float* values, size_t size) {
  activation_tanh<<<block_count(size), THREADS_PER_BLOCK>>>(values, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_activation_relu(float* values, size_t size) {
  activation_relu<<<block_count(size), THREADS_PER_BLOCK>>>(values, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_backward_tanh(
    const float* values,
    float* gradients,
    size_t size) {
  backward_tanh<<<block_count(size), THREADS_PER_BLOCK>>>(
      values, gradients, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_backward_relu(
    const float* values,
    float* gradients,
    size_t size) {
  backward_relu<<<block_count(size), THREADS_PER_BLOCK>>>(
      values, gradients, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_softmax_forward(
    float* values,
    size_t cols,
    size_t rows) {
  softmax_forward_kernel<<<rows, THREADS_PER_BLOCK,
                           THREADS_PER_BLOCK * sizeof(float)>>>(
      values, cols, rows);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_softmax_backward(
    const float* values,
    float* gradients,
    size_t cols,
    size_t rows) {
  softmax_backward_kernel<<<rows, THREADS_PER_BLOCK,
                            THREADS_PER_BLOCK * sizeof(float)>>>(
      values, gradients, cols, rows);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_mse_loss_kernel(
    const float* predictions,
    const float* target,
    float* loss_sum,
    int size) {
  mse_loss_kernel<<<block_count(size), THREADS_PER_BLOCK,
                    THREADS_PER_BLOCK * sizeof(float)>>>(
      predictions, target, loss_sum, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_backward_mse_loss_kernel(
    const float* predictions,
    const float* target,
    float* gradients,
    int size) {
  backward_mse_loss_kernel<<<block_count(size), THREADS_PER_BLOCK>>>(
      predictions, target, gradients, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_cross_entropy_loss_kernel(
    const float* predictions,
    const float* target,
    float* loss_sum,
    int size) {
  cross_entropy_loss_kernel<<<block_count(size), THREADS_PER_BLOCK,
                              THREADS_PER_BLOCK * sizeof(float)>>>(
      predictions, target, loss_sum, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void launch_backward_cross_entropy_loss_kernel(
    const float* predictions,
    const float* target,
    float* gradients,
    float normalization,
    int size) {
  backward_cross_entropy_loss_kernel<<<
      block_count(size), THREADS_PER_BLOCK>>>(
      predictions, target, gradients, normalization, size);
  CUDA_CHECK_MANAGER(cudaGetLastError());
}

inline void check_cuda_error(
    cudaError_t error,
    const char* file,
    int line) {
  std::fprintf(
      stderr,
      "CUDA error at %s:%d: %s\n",
      file,
      line,
      cudaGetErrorString(error));
  throw std::runtime_error("CUDA operation failed");
}

#endif
