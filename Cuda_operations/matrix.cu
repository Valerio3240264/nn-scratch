#include <stdio.h>
/* WEIGHTS KERNELS */

/* MATRIX-VECTOR MULTIPLICATION KERNEL */
__global__ void I_W_B_multiplication(float *d_w, float *d_input_values, float *d_b, float *d_result, int output_size, int input_size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < output_size) {
    float sum = d_b[row];
    for (int col = 0; col < input_size; col++) {
      sum += d_w[row * input_size + col] * d_input_values[col];
    }
    d_result[row] = sum;
  }

} 

/* VECTOR UPDATE KERNEL */
// Fixed: Direct memory access without problematic reinterpret_cast pattern
__global__ void vector_update(float *V, float *U, float learning_rate, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    V[idx] -= learning_rate * U[idx];
  }
}

__global__ void backward_W(float *d_w, float *d_input_values, float *d_derivatives, float *d_grad_w, float *d_prevGrad, int output_size, int input_size){
  // Map thread x -> col, y -> row
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < input_size && row < output_size) {
    int idx = row * input_size + col;
    float d_row = d_derivatives[row];
    float contrib_gradw = d_row * d_input_values[col];
    float contrib_prev = d_row * d_w[idx];

    atomicAdd(d_grad_w + idx, contrib_gradw);
    atomicAdd(d_prevGrad + col, contrib_prev);
  }
}

__global__ void backward_bias(float *d_b, float *d_derivatives, float *d_grad_b, int output_size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < output_size) {
    atomicAdd(d_grad_b + row, d_derivatives[row]);
  }
}

/* ACTIVATION FUNCTION KERNELS */
// Forward pass
__global__ void activation_tanh(float *V, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    V[idx] = tanh(V[idx]);
  }
}

__global__ void activation_relu(float *V, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    V[idx] = fmaxf(0.0f, V[idx]);
  }
}

// Backward pass
__global__ void backward_tanh(float *V, float *derivatives, float *grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    grad[idx] = derivatives[idx] * (1 - (V[idx]* V[idx]));
  }
}

__global__ void backward_relu(float *V, float *derivatives, float *grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    grad[idx] = derivatives[idx] * (V[idx] > 0 ? 1 : 0);
  }
} 

__global__ void backward_linear(float *V, float *derivatives, float *grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    grad[idx] = derivatives[idx];
  }
}

/* SOFTMAX KERNELS */
// Helper device function for atomic max with floats
__device__ static float atomicMaxFloat(float* address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

// Kernel to find max value (reduction)
__global__ void find_max_kernel(float *d_input, float *d_max, int size) {
  __shared__ float shared_max[256];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  // Initialize shared memory with input values or -infinity
  shared_max[tid] = (idx < size) ? d_input[idx] : -INFINITY;
  __syncthreads();
  
  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = fmax(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }
  
  // Write block result to global memory using atomic max for floats
  if (tid == 0) {
    atomicMaxFloat(d_max, shared_max[0]);
  }
}

// Kernel to compute exp and sum (in-place modification and reduction)
__global__ void softmax_exp_sum_kernel(float *d_value, float *d_max_val, float temperature, float *d_exp_sum, int size) {
  __shared__ float shared_sum[256];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  float max_val = *d_max_val;
  
  float exp_val = 0.0f;
  if (idx < size) {
    exp_val = expf((d_value[idx] - max_val) / temperature);
    d_value[idx] = exp_val;  // Store exp value temporarily
  }
  
  shared_sum[tid] = exp_val;
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
    atomicAdd(d_exp_sum, shared_sum[0]);
  }
}

// Kernel to normalize by sum
__global__ void softmax_normalize_kernel(float *d_value, float *d_exp_sum, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  float exp_sum = *d_exp_sum;  // Read sum value from device memory once per thread
  
  if (idx < size) {
    d_value[idx] = d_value[idx] / exp_sum;
  }
}

// Kernel to compute dot product for backward pass
__global__ void softmax_dot_product_kernel(float *d_value, float *d_derivatives, float *d_dot, int size) {
  __shared__ float shared_dot[256];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  float dot_val = 0.0f;
  if (idx < size) {
    dot_val = d_value[idx] * d_derivatives[idx];
  }
  
  shared_dot[tid] = dot_val;
  __syncthreads();
  
  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_dot[tid] += shared_dot[tid + stride];
    }
    __syncthreads();
  }
  
  // Write block result to global memory
  if (tid == 0) {
    atomicAdd(d_dot, shared_dot[0]);
  }
}

// Kernel to compute backward gradient
__global__ void softmax_backward_kernel(float *d_value, float *d_derivatives, float *d_grad, float *d_dot, float temperature, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  float dot = *d_dot;
  
  if (idx < size) {
    d_grad[idx] = d_value[idx] * (d_derivatives[idx] - dot) / temperature;
  }
}

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