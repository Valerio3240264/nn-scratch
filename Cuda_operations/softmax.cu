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
    if(d_value[i] > local_max){
      local_Z *= expf((local_max - d_value[i]) / temperature);
      local_max = d_value[i];
    }
    local_Z += expf((d_value[i] - local_max) / temperature);
  }
  __syncthreads();

  // Reduce smem until we got the max value
  for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
    if(tid < stride && tid + stride < blockDim.x){
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  tot_max = smem[0];

  smem[tid] = local_Z * expf((local_max - tot_max) / temperature);
  __syncthreads();

  // Reduce smem until we get the final Z
  for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
    if(tid < stride && tid + stride < blockDim.x){
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  tot_Z = smem[0];

  // Normalize the values
  for(int i = tid; i < size; i += blockDim.x){
    d_value[i] = expf((d_value[i] - tot_max) / temperature) / tot_Z;
  }
}