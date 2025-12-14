#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))

__device__ __forceinline__ float warp_reduce_sum(float value){
  unsigned mask = __activemask();
  for(int d = 16; d > 0; d /= 2){
    value += __shfl_down_sync(mask, value, d);
  }
  return value;
}

__device__ __forceinline__ void block_reduce_sum(float value, float *smem, int tid, int blockDimX){
  value = warp_reduce_sum(value);

  if (blockDimX > warpSize && blockDimX < warpSize * warpSize) {
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    if (lane == 0) {
      smem[wid] = value;
    }
    __syncthreads();

    if (tid < warpSize) {
      value = tid < CEIL_DIV(blockDimX, warpSize) ? smem[tid] : 0.0f;
      value = warp_reduce_sum(value);
      if (tid == 0) smem[0] = value;
    }
  } 
  else if (blockDimX > warpSize) {
    smem[tid] = value;
    __syncthreads();
    for(int stride = blockDimX / 2; stride > 0; stride /= 2){
      if(tid < stride){
        smem[tid] += smem[tid + stride];
      }
      __syncthreads();
    }
  }
  else {
    if (tid == 0) smem[0] = value;
  }
}