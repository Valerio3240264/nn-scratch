/* LOSS KERNELS */
/* REDUCTION KERNEL FOR LOSS COMPUTATION */
// Parallel reduction to sum array elements
__global__ void reduce_sum_kernel(float *d_input, float *d_output, int size) {
  __shared__ float shared_sum[256];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  // Initialize shared memory with input values or 0
  shared_sum[tid] = (idx < size) ? d_input[idx] : 0.0f;
  __syncthreads();
  
  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }
  
  // Write block result to global memory
  if (tid == 0) {
    atomicAdd(d_output, shared_sum[0]);
  }
}

/* MSE LOSS KERNELS */
// Forward pass: compute MSE loss contributions per element
// Note: This stores the squared error in d_grad for now
// The actual loss value should be computed separately by summing and averaging
__global__ void mse_loss_kernel(float *d_predictions, float *d_target, float *d_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    float diff = d_predictions[idx] - d_target[idx];
    // Store squared error (will be used to compute loss value if needed)
    d_grad[idx] = diff * diff;
  }
}

// Backward pass: compute gradient with respect to predictions (with incoming derivatives)
__global__ void backward_mse_loss_kernel(float *d_predictions, float *d_target, float *d_derivatives, float *d_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    // MSE gradient: (2 / n) * (prediction - target)
    // Matches CPU implementation at line 172 in mse_loss.h
    d_grad[idx] = (2.0f / size) * (d_predictions[idx] - d_target[idx]) * d_derivatives[idx];
  }
}

// Backward pass: simplified version assuming derivatives = 1.0 (standard case)
__global__ void backward_mse_loss_kernel_simple(float *d_predictions, float *d_target, float *d_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    // When derivative of loss w.r.t. itself is 1.0, gradient is: (2 / n) * (prediction - target)
    d_grad[idx] = (2.0f / size) * (d_predictions[idx] - d_target[idx]);
  }
}

/* ONE-HOT ENCODING KERNEL */
// Kernel to write one-hot encoding directly to device memory
// Sets all elements to 0.0 except the target_index which is set to 1.0
__global__ void one_hot_encoding_kernel(float *d_target, int target_index, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    d_target[idx] = (idx == target_index) ? 1.0f : 0.0f;
  }
}

/* CROSS ENTROPY LOSS KERNELS */
// Forward pass: compute cross entropy loss contributions per element
// Note: This stores the loss contribution in d_grad
__global__ void cross_entropy_loss_kernel(float *d_predictions, float *d_target, float *d_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    // Cross entropy: -target * log(prediction + epsilon)
    // Add small epsilon to avoid log(0), matches CPU epsilon of 1e-15f
    float epsilon = 1e-15f;
    float loss_contrib = 0.0f;
    
    if (d_target[idx] > 0) {
      loss_contrib = -d_target[idx] * logf(d_predictions[idx] + epsilon);
    }
    
    // Store loss contribution
    d_grad[idx] = loss_contrib;
  }
}

// Backward pass: compute gradient with respect to predictions (with incoming derivatives)
__global__ void backward_cross_entropy_loss_kernel(float *d_predictions, float *d_target, float *d_derivatives, float *d_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    // Simplified cross entropy gradient (when combined with softmax): (predictions - targets)
    // This is the beautiful mathematical simplification that occurs when you combine
    // softmax activation with cross-entropy loss
    // Matches CPU implementation at line 186 in cross_entropy_loss.h
    d_grad[idx] = (d_predictions[idx] - d_target[idx]) * d_derivatives[idx];
  }
}

// Backward pass: simplified version assuming derivatives = 1.0 (standard case)
__global__ void backward_cross_entropy_loss_kernel_simple(float *d_predictions, float *d_target, float *d_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // When derivative of loss w.r.t. itself is 1.0, gradient is simply: predictions - targets
    d_grad[idx] = d_predictions[idx] - d_target[idx];
  }
}