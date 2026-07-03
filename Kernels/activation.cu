#include <cstddef>
#include <math.h>
#include <math_constants.h>

__global__ void activation_tanh(
    float* __restrict__ values,
    size_t size) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    values[index] = tanhf(values[index]);
  }
}

__global__ void activation_relu(
    float* __restrict__ values,
    size_t size) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    values[index] = fmaxf(0.0f, values[index]);
  }
}

__global__ void backward_tanh(
    const float* __restrict__ values,
    float* __restrict__ gradients,
    size_t size) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    const float value = values[index];
    gradients[index] *= 1.0f - value * value;
  }
}

__global__ void backward_relu(
    const float* __restrict__ values,
    float* __restrict__ gradients,
    size_t size) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    gradients[index] = values[index] > 0.0f ? gradients[index] : 0.0f;
  }
}

__global__ void softmax_forward_kernel(
    float* __restrict__ values,
    size_t cols,
    size_t rows) {
  extern __shared__ float shared[];

  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const size_t thread = threadIdx.x;
  float* row_values = values + row * cols;

  float local_max = -CUDART_INF_F;
  for (size_t col = thread; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, row_values[col]);
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
    local_sum += expf(row_values[col] - row_max);
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
    row_values[col] = expf(row_values[col] - row_max) / normalization;
  }
}

__global__ void softmax_backward_kernel(
    const float* __restrict__ values,
    float* __restrict__ gradients,
    size_t cols,
    size_t rows) {
  extern __shared__ float shared[];

  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const size_t thread = threadIdx.x;
  const float* row_values = values + row * cols;
  float* row_gradients = gradients + row * cols;

  float local_dot = 0.0f;
  for (size_t col = thread; col < cols; col += blockDim.x) {
    local_dot += row_values[col] * row_gradients[col];
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
    row_gradients[col] = row_values[col] * (row_gradients[col] - dot);
  }
}
