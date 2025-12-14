#include "utils.cu"

#include <stdio.h>
#include <assert.h>
/* WEIGHTS KERNELS */

/* NAIVE
MATRIX-VECTOR MULTIPLICATION KERNEL
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
*/

/* VECTORIZED SGEMV KERNEL */
// Weights size = M x K
// Input size = K
__global__ void SGEMV(float *w, float *in, float *res, float *bias, int K, int M){
  extern __shared__ float smem[];

  int row = blockIdx.x;
  if(row >= M) return;

  int tid = threadIdx.x;

  int K4 = K / 4;
  int ceil_K4 = (K+3) / 4;
  
  float sum = 0.0f;
  float4* w4 = reinterpret_cast<float4*>(w + row * K);
  float4* in4 = reinterpret_cast<float4*>(in);

  for(int i = tid; i < ceil_K4; i += blockDim.x){
    if(i < K4){
      float4 w_val = w4[i];
      float4 in_val = in4[i];

      sum += w_val.x * in_val.x + w_val.y * in_val.y + w_val.z * in_val.z + w_val.w * in_val.w;
    }
    else if(i*4 < K){
      float wx = 0.0f, wy = 0.0f, wz = 0.0f;
      float ix = 0.0f, iy = 0.0f, iz = 0.0f;

      wx = w[(row*K) + i*4];
      ix = in[i*4];

      if(i*4+2 < K){
        wy = w[(row*K) + i*4+1];
        iy = in[i*4+1];
        wz = w[(row*K) + i*4+2];
        iz = in[i*4+2];
      }
      else if(i*4+1 < K){
        wy = w[(row*K) + i*4+1];
        iy = in[i*4+1];
      }

      sum += wx * ix + wy * iy + wz * iz;
    }
    else break;
  }

  block_reduce_sum(sum, smem, tid, blockDim.x);

  if(tid == 0) res[row] = smem[0] + bias[row];
}

/* VECTORIZED VECTOR UPDATE KERNEL */
__global__ void vectorized_vector_update(float *V, float *U, float learning_rate, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int size4 = size / 4;

  // Cast the pointer to float4
  float4 *V4 = reinterpret_cast<float4*>(V);
  float4 *U4 = reinterpret_cast<float4*>(U);

  // Process full float4 elements
  for (int i = tid; i < size4; i += stride) {
    float4 v = V4[i];
    float4 u = U4[i];
    v.x -= learning_rate * u.x;
    v.y -= learning_rate * u.y;
    v.z -= learning_rate * u.z;
    v.w -= learning_rate * u.w;
    V4[i] = v;
  }

  // Handle remaining tail elements
  int tail_start = size4 * 4;
  if(tid < (size - tail_start)) {
    float v = V[tid + tail_start];
    float u = U[tid + tail_start];
    v -= learning_rate * u;
    V[tid + tail_start] = v;
  }
}

__global__ void non_vectorized_vector_update(float *V, float *U, float learning_rate, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = idx; i < size; i += blockDim.x*gridDim.x){
    V[i] -= learning_rate * U[i];
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