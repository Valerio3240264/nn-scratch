/* LOSS KERNELS */

/* MSE LOSS KERNELS */
// Forward pass: compute MSE loss contributions per element
__global__ void mse_loss_kernel(float *d_predictions, float *d_target, float *d_grad, float *d_loss_sum, int size) {
  extern __shared__ float shared_loss_sum[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float diff = 0.0f;
  float temp_loss_sum = 0.0f;

  for(int i = idx; i < size; i += stride) {
    diff = d_predictions[i] - d_target[i];
    d_grad[i] = diff * diff;
    temp_loss_sum += d_grad[i];
  }
  shared_loss_sum[threadIdx.x] = temp_loss_sum;
  __syncthreads();

  for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if(threadIdx.x < stride) {
      shared_loss_sum[threadIdx.x] += shared_loss_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if(threadIdx.x == 0) {
    atomicAdd(d_loss_sum, shared_loss_sum[0]);
  }
}

// Backward pass: compute gradient with respect to predictions (with incoming derivatives)
__global__ void backward_mse_loss_kernel(float *d_predictions, float *d_target, float *d_derivatives, float *d_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += stride) {
    d_grad[i] = (2.0f / size) * (d_predictions[i] - d_target[i]) * d_derivatives[i];
  }
}

// Backward pass: simplified version assuming derivatives = 1.0 (standard case)
__global__ void backward_mse_loss_kernel_simple(float *d_predictions, float *d_target, float *d_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += stride) {
    d_grad[i] = (2.0f / size) * (d_predictions[i] - d_target[i]);
  }
}

/* ONE-HOT ENCODING KERNEL */
// Kernel to write one-hot encoding
__global__ void one_hot_encoding_kernel(float *d_target, int target_index, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += stride) {
    d_target[i] = (i == target_index) ? 1.0f : 0.0f;
  }
}

/* CROSS ENTROPY LOSS KERNELS */
// Forward pass: compute cross entropy loss contributions per element
// This kernel can only be used if the previous layer is a softmax layer
__global__ void softmax_cross_entropy_loss_kernel(float *d_predictions, float *d_target, float *d_grad, float *d_loss_sum, int size) {
  extern __shared__ float shared_grad[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float epsilon = 1e-15f;
  float temp_grad = 0.0f;
  float temp_loss_sum = 0.0f;

  for(int i = idx; i < size; i += stride) {
    temp_grad = 0.0f;
    if (d_target[i] > 0) {
      temp_grad = -d_target[i] * logf(d_predictions[i] + epsilon);
    }
    d_grad[i] = temp_grad;
    temp_loss_sum += temp_grad;
  }

  shared_grad[threadIdx.x] = temp_grad;
  __syncthreads();

  for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if(threadIdx.x < stride) {
      shared_grad[threadIdx.x] += shared_grad[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if(threadIdx.x == 0) {
    atomicAdd(d_loss_sum, shared_grad[0]);
  }
}

// Backward pass: compute gradient with respect to predictions (with incoming derivatives)
__global__ void backward_cross_entropy_loss_kernel(float *d_predictions, float *d_target, float *d_derivatives, float *d_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += stride) {   
    d_grad[i] = (d_predictions[i] - d_target[i]) * d_derivatives[i];
  }
}

// Backward pass: simplified version assuming derivatives = 1.0 (standard case)
__global__ void backward_cross_entropy_loss_kernel_simple(float *d_predictions, float *d_target, float *d_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = idx; i < size; i += stride) {
    d_grad[i] = d_predictions[i] - d_target[i];
  }
}