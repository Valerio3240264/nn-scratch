#include <math.h>

__global__ void mse_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ target,
    float* __restrict__ loss_sum,
    int size) {
  extern __shared__ float shared[];

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  float local_sum = 0.0f;
  if (index < size) {
    const float difference = predictions[index] - target[index];
    local_sum = difference * difference;
  }

  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(loss_sum, shared[0]);
  }
}

__global__ void backward_mse_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ target,
    float* __restrict__ gradients,
    int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    gradients[index] =
        (2.0f / static_cast<float>(size)) *
        (predictions[index] - target[index]);
  }
}

__global__ void cross_entropy_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ target,
    float* __restrict__ loss_sum,
    int size) {
  extern __shared__ float shared[];

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  float local_sum = 0.0f;
  if (index < size && target[index] != 0.0f) {
    local_sum = -target[index] * logf(predictions[index] + 1e-15f);
  }

  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(loss_sum, shared[0]);
  }
}

__global__ void backward_cross_entropy_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ target,
    float* __restrict__ gradients,
    float normalization,
    int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    const float probability = fmaxf(predictions[index], 1e-15f);
    gradients[index] =
        -(target[index] / probability) * normalization;
  }
}
