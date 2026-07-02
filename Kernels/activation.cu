#include <cstddef>
#include <math.h>

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
