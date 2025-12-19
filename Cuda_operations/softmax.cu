#include "utils.cu"

/* SOFTMAX KERNELS */
// FORWARD PASS
__global__ void vector_softmax_kernel(float *d_value, float temperature, int size){
  // SMEM declaration
  __shared__ float smem[1024];
  // Thread index
  int tid = threadIdx.x;

  // Local variables (register variables)
  float local_Z = 0.0f;
  float local_max = -INFINITY;
  float tot_Z;
  float tot_max;

  // Initialize shared memory
  if(tid < size){
    smem[tid] = d_value[tid];
    local_max = d_value[tid];
  } else {
    smem[tid] = -INFINITY;
  }

  // Evaluate local max and local Z
  for(int i = tid; i < size; i += blockDim.x){
    float x = d_value[i];
    if(x > local_max){
      local_Z *= expf((local_max - x) / temperature);
      local_max = x;
    }
    local_Z += expf((x - local_max) / temperature);
  }
  __syncthreads();

  // Reduce smem until we got the max value
  for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
    if(tid < stride && tid + stride < blockDim.x){
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  // Get the total max value
  tot_max = smem[0];

  // Compute the local Z
  smem[tid] = local_Z * expf((local_max - tot_max) / temperature);
  __syncthreads();

  // Reduce smem until we get the final Z
  for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
    if(tid < stride && tid + stride < blockDim.x){
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  
  // Get the final Z
  tot_Z = smem[0];

  // Normalize the values
  for(int i = tid; i < size; i += blockDim.x){
    d_value[i] = expf((d_value[i] - tot_max) / temperature) / tot_Z;
  }
}

__global__ void softmax_backward_kernel(float *d_value, float *d_derivatives, float *d_grad, float *d_dot, float temperature, int size){
  extern __shared__ float shared_dot[];
  
  int tid = threadIdx.x;
  float dot = 0.0f;
  for(int i = tid; i < size; i += blockDim.x){
    dot += d_value[i] * d_derivatives[i];
  }

  block_reduce_sum(dot, shared_dot, tid, blockDim.x);

  dot = shared_dot[0];

  for(int i = tid; i < size; i += blockDim.x){
    d_grad[i] = d_value[i] * (d_derivatives[i] - dot) / temperature;
  }
  
}