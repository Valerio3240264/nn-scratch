#ifndef CUDA_MANAGER_CUH
#define CUDA_MANAGER_CUH

#include <cuda_runtime.h>
#include <curand.h>
#include "../enums.h"

// Forward declaration of CUDA kernels
/* MATRIX OPERATION KERNELS */
__global__ void SGEMV(float *__restrict__ w, float *__restrict__ in, float *__restrict__ res, float *__restrict__ bias, int K, int M);

// Vector update kernels
__global__ void vectorized_vector_update(float *__restrict__ V, float *__restrict__ U, float learning_rate, int size);
__global__ void non_vectorized_vector_update(float *__restrict__ V, float *__restrict__ U, float learning_rate, int size);
__global__ void tiled_backward_Weights(float *__restrict__ d_w, float *__restrict__ d_In, float *__restrict__ d_derivatives, 
                                       float *__restrict__ d_grad_w, float *__restrict__ d_InGrad, float *__restrict__ d_biasGrad, 
                                       int output_size, int input_size);

// Xavier initialization kernel
__global__ void scale_weights(float *__restrict__ d_data, int n, float scale);

/* ACTIVATION FUNCTION KERNELS */
// Vectorized forward pass
__global__ void vectorized_activation_tanh(float *__restrict__ V, int size);
__global__ void vectorized_activation_relu(float *__restrict__ V, int size);

// Non-vectorized forward pass
__global__ void non_vectorized_activation_tanh(float *__restrict__ V, int size);
__global__ void non_vectorized_activation_relu(float *__restrict__ V, int size);

// Vectorized backward pass
__global__ void vectorized_backward_tanh(float *__restrict__ V, float *__restrict__ derivatives, float *__restrict__ grad, int size);
__global__ void vectorized_backward_relu(float *__restrict__ V, float *__restrict__ derivatives, float *__restrict__ grad, int size);

// Non-vectorized backward pass
__global__ void non_vectorized_backward_tanh(float *__restrict__ V, float *__restrict__ derivatives, float *__restrict__ grad, int size);
__global__ void non_vectorized_backward_relu(float *__restrict__ V, float *__restrict__ derivatives, float *__restrict__ grad, int size);

/* SOFTMAX KERNELS */
__global__ void vector_softmax_kernel(float *__restrict__ d_value, float temperature, int size);

// Backward pass kernel
__global__ void softmax_backward_kernel(float *__restrict__ d_value, float *__restrict__ d_derivatives, float *__restrict__ d_grad, float *__restrict__ d_dot, float temperature, int size);

/* LOSS KERNELS */
// One-hot encoding kernel
__global__ void one_hot_encoding_kernel(float *__restrict__ d_target, int target_index, int size);

// MSE loss kernels
__global__ void mse_loss_kernel(float *__restrict__ d_predictions, float *__restrict__ d_target, float *__restrict__ d_grad, float *__restrict__ d_loss_sum, int size);
__global__ void backward_mse_loss_kernel(float *__restrict__ d_predictions, float *__restrict__ d_target, float *__restrict__ d_derivatives, float *__restrict__ d_grad, int size);
__global__ void backward_mse_loss_kernel_simple(float *__restrict__ d_predictions, float *__restrict__ d_target, float *__restrict__ d_grad, int size);

// Cross entropy loss kernels
__global__ void softmax_cross_entropy_loss_kernel(float *__restrict__ d_predictions, float *__restrict__ d_target, float *__restrict__ d_grad, float *__restrict__ d_loss_sum, int size);
__global__ void backward_cross_entropy_loss_kernel(float *__restrict__ d_predictions, float *__restrict__ d_target, float *__restrict__ d_derivatives, float *__restrict__ d_grad, int size);
__global__ void backward_cross_entropy_loss_kernel_simple(float *__restrict__ d_predictions, float *__restrict__ d_target, float *__restrict__ d_grad, int size);

// Forward declarations of utility functions
void check_cuda_error(cudaError_t error, const char* file, int line);

// CUDA error checking macro
#define CUDA_CHECK_MANAGER(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      check_cuda_error(error, __FILE__, __LINE__); \
    } \
  } while(0)

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
void allocate_device_memory_xavier(T** device_ptr, size_t count, size_t input_size);
template<typename T>
void allocate_device_memory_he(T** device_ptr, size_t count, size_t input_size);
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
void launch_tiled_backward_Weights(float* d_w, float* d_input_values, float* d_derivatives, float* d_grad_w, float* d_InGrad, float* d_biasGrad, int output_size, int input_size);

// Activation function kernel
void launch_activation_tanh(float* d_value, int size);
void launch_activation_relu(float* d_value, int size);
void launch_backward_tanh(float* d_value, float* d_derivatives, float* d_grad, int size);
void launch_backward_relu(float* d_value, float* d_derivatives, float* d_grad, int size);
void launch_backward_linear(float* d_value, float* d_derivatives, float* d_grad, int size);

// Softmax kernels
void launch_vector_softmax(float* d_value, float temperature, int size);
void launch_softmax_backward(float* d_value, float* d_derivatives, float* d_grad, float temperature, int size);

// One-hot encoding kernel
void launch_one_hot_encoding(float* d_target, int target_index, int size);

// MSE loss kernels
void launch_mse_loss_kernel(float* d_predictions, float* d_target, float* d_grad, float* d_loss_sum, int size);
void launch_backward_mse_loss_kernel(float* d_predictions, float* d_target, float* d_derivatives, float* d_grad, int size);
void launch_backward_mse_loss_kernel_simple(float* d_predictions, float* d_target, float* d_grad, int size);

// Cross entropy loss kernels
void launch_softmax_cross_entropy_loss_kernel(float* d_predictions, float* d_target, float* d_grad, float* d_loss_sum, int size);
void launch_backward_cross_entropy_loss_kernel(float* d_predictions, float* d_target, float* d_derivatives, float* d_grad, int size);
void launch_backward_cross_entropy_loss_kernel_simple(float* d_predictions, float* d_target, float* d_grad, int size);

// Synchronization
void synchronize();

#endif