#include "utils.cuh"

#include <cstddef>
#include <math_constants.h>

__global__ void softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float temperature,
    size_t cols,
    size_t rows) {
  extern __shared__ float shared[];

  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const size_t thread = threadIdx.x;
  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float local_max = -CUDART_INF_F;
  for (size_t col = thread; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, row_input[col]);
  }
  shared[thread] = local_max;
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < stride) {
      shared[thread] = fmaxf(shared[thread], shared[thread + stride]);
    }
    __syncthreads();
  }
  const float row_max = shared[0];

  float local_sum = 0.0f;
  for (size_t col = thread; col < cols; col += blockDim.x) {
    local_sum += expf((row_input[col] - row_max) / temperature);
  }
  shared[thread] = local_sum;
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < stride) {
      shared[thread] += shared[thread + stride];
    }
    __syncthreads();
  }
  const float normalization = shared[0];

  for (size_t col = thread; col < cols; col += blockDim.x) {
    row_output[col] =
        expf((row_input[col] - row_max) / temperature) / normalization;
  }
}

__global__ void softmax_backward_kernel(
    const float* __restrict__ values,
    const float* __restrict__ derivatives,
    float* __restrict__ gradients,
    float temperature,
    size_t cols,
    size_t rows) {
  extern __shared__ float shared[];

  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const size_t thread = threadIdx.x;
  const float* row_values = values + row * cols;
  const float* row_derivatives = derivatives + row * cols;
  float* row_gradients = gradients + row * cols;

  float local_dot = 0.0f;
  for (size_t col = thread; col < cols; col += blockDim.x) {
    local_dot += row_values[col] * row_derivatives[col];
  }
  shared[thread] = local_dot;
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < stride) {
      shared[thread] += shared[thread + stride];
    }
    __syncthreads();
  }
  const float dot = shared[0];

  for (size_t col = thread; col < cols; col += blockDim.x) {
    row_gradients[col] =
        row_values[col] * (row_derivatives[col] - dot) / temperature;
  }
}
