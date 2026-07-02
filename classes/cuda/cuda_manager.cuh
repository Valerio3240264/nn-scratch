#ifndef CUDA_MANAGER_CUH
#define CUDA_MANAGER_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include <curand.h>

/* MATRIX AND WEIGHT KERNELS */
__global__ void tiled_matmul_add_bias(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ result,
    int rows,
    int cols,
    int inner);

__global__ void tiled_matmul_transpose_left_accumulate(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ result,
    int rows,
    int cols,
    int inner);

__global__ void tiled_matmul_transpose_right(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ result,
    int rows,
    int cols,
    int inner);

__global__ void accumulate_row_sum(
    float* __restrict__ result,
    const float* __restrict__ matrix,
    int rows,
    int cols);

__global__ void vector_update(
    float* __restrict__ values,
    const float* __restrict__ gradients,
    float learning_rate,
    size_t size);

__global__ void scale_weights(
    float* __restrict__ data,
    size_t size,
    float scale);

/* ACTIVATION KERNELS */
__global__ void activation_tanh(float* __restrict__ values, size_t size);
__global__ void activation_relu(float* __restrict__ values, size_t size);
__global__ void backward_tanh(
    const float* __restrict__ values,
    float* __restrict__ gradients,
    size_t size);
__global__ void backward_relu(
    const float* __restrict__ values,
    float* __restrict__ gradients,
    size_t size);

/* SOFTMAX KERNELS */
__global__ void softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float temperature,
    size_t cols,
    size_t rows);

__global__ void softmax_backward_kernel(
    const float* __restrict__ values,
    const float* __restrict__ derivatives,
    float* __restrict__ gradients,
    float temperature,
    size_t cols,
    size_t rows);

/* LOSS KERNELS */
__global__ void mse_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ target,
    float* __restrict__ loss_sum,
    int size);

__global__ void backward_mse_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ target,
    float* __restrict__ gradients,
    int size);

__global__ void cross_entropy_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ target,
    float* __restrict__ loss_sum,
    int size);

__global__ void backward_cross_entropy_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ target,
    float* __restrict__ gradients,
    float normalization,
    int size);

void check_cuda_error(cudaError_t error, const char* file, int line);

#define CUDA_CHECK_MANAGER(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      check_cuda_error(error, __FILE__, __LINE__); \
    } \
  } while (0)

/* DEVICE AND MEMORY MANAGEMENT */
bool is_cuda_available();

template<typename T>
void allocate_device_memory(T** device_ptr, size_t count);

template<typename T>
void allocate_device_memory_zeros(T** device_ptr, size_t count);

template<typename T>
void allocate_device_memory_xavier(
    T** device_ptr,
    size_t input_size,
    size_t output_size);

template<typename T>
void allocate_device_memory_he(
    T** device_ptr,
    size_t input_size,
    size_t output_size);

template<typename T>
void zero_device_memory(T* device_ptr, size_t count);

void free_device_memory(void* device_ptr);

template<typename T>
void copy_host_to_device(T* device_ptr, const T* host_ptr, size_t count);

template<typename T>
void copy_device_to_host(T* host_ptr, const T* device_ptr, size_t count);

/* KERNEL LAUNCHERS */
void launch_matmul_forward(
    const float* A,
    const float* B,
    const float* bias,
    float* result,
    int rows,
    int cols,
    int inner);

void launch_matmul_backward_weights(
    const float* A,
    const float* B,
    float* result,
    int rows,
    int cols,
    int inner);

void launch_matmul_backward_input(
    const float* A,
    const float* B,
    float* result,
    int rows,
    int cols,
    int inner);

void launch_bias_gradient(
    float* result,
    const float* matrix,
    int rows,
    int cols);

void launch_update(
    float* values,
    const float* gradients,
    float learning_rate,
    size_t size);

void launch_activation_tanh(float* values, size_t size);
void launch_activation_relu(float* values, size_t size);
void launch_backward_tanh(
    const float* values,
    float* gradients,
    size_t size);
void launch_backward_relu(
    const float* values,
    float* gradients,
    size_t size);

void launch_softmax_forward(
    const float* input,
    float* output,
    float temperature,
    size_t cols,
    size_t rows);

void launch_softmax_backward(
    const float* values,
    const float* derivatives,
    float* gradients,
    float temperature,
    size_t cols,
    size_t rows);

void launch_mse_loss_kernel(
    const float* predictions,
    const float* target,
    float* loss_sum,
    int size);

void launch_backward_mse_loss_kernel(
    const float* predictions,
    const float* target,
    float* gradients,
    int size);

void launch_cross_entropy_loss_kernel(
    const float* predictions,
    const float* target,
    float* loss_sum,
    int size);

void launch_backward_cross_entropy_loss_kernel(
    const float* predictions,
    const float* target,
    float* gradients,
    float normalization,
    int size);

#endif
