/*
A special thanks to Claude for letting me be aware of the __restrict__ keyword.
https://stackoverflow.com/questions/43235899/cuda-restrict-tag-usage
https://en.wikipedia.org/wiki/Restrict
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__restrict__#restrict

What is the __restrict__ keyword?
The __restrict__ keyword is a keyword, used on pointers, to tell the compiler that the data which are being pointed to will not be accessed by any other pointer.
This avoids pointer aliasing and allows the compiler to make more optimized code.
*/

/* VECTORIZED FORWARD PASS */
// TANH
__global__ void vectorized_activation_tanh(float *__restrict__ V, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int size4 = size / 4;

  // Cast the pointer to float4
  float4 *V4 = reinterpret_cast<float4*>(V);

  // Process full float4 elements
  for (int i = tid; i < size4; i += stride) {
    float4 v = V4[i];
    v.x = tanhf(v.x);
    v.y = tanhf(v.y);
    v.z = tanhf(v.z);
    v.w = tanhf(v.w);
    V4[i] = v;
  }

  // Handle remaining tail elements
  int tail_start = size4 * 4;
  if(tid < (size - tail_start)) {
    float v = V[tid + tail_start];
    v = tanhf(v);
    V[tid + tail_start] = v;
  }
}

// RELU
__global__ void vectorized_activation_relu(float *__restrict__ V, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int size4 = size / 4;

  // Cast the pointer to float4
  float4 *V4 = reinterpret_cast<float4*>(V);

  // Process full float4 elements
  for (int i = tid; i < size4; i += stride) {
    float4 v = V4[i];
    v.x = fmaxf(0.0f, v.x);
    v.y = fmaxf(0.0f, v.y);
    v.z = fmaxf(0.0f, v.z);
    v.w = fmaxf(0.0f, v.w);
    V4[i] = v;
  }

  // Handle remaining tail elements
  int tail_start = size4 * 4;
  if(tid < (size - tail_start)) {
    float v = V[tid + tail_start];
    v = fmaxf(0.0f, v);
    V[tid + tail_start] = v;
  }
}

/* NON-VECTORIZED FORWARD PASS */
// TANH
__global__ void non_vectorized_activation_tanh(float *__restrict__ V, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < size; i += stride) {
    V[i] = tanhf(V[i]);
  }
}

// RELU
__global__ void non_vectorized_activation_relu(float *__restrict__ V, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < size; i += stride) {
    V[i] = fmaxf(0.0f, V[i]);
  }
}

/* VECTORIZED BACKWARD PASS */
// TANH
__global__ void vectorized_backward_tanh(float *__restrict__ V, float *__restrict__ derivatives, float *__restrict__ grad, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int size4 = size / 4;

  float4 *V4 = reinterpret_cast<float4*>(V);
  float4 *derivatives4 = reinterpret_cast<float4*>(derivatives);
  float4 *grad4 = reinterpret_cast<float4*>(grad);

  for (int i = tid; i < size4; i += stride) {
    float4 v = V4[i];
    float4 d = derivatives4[i];
    float4 g = grad4[i];
    g.x = d.x * (1 - (v.x* v.x));
    g.y = d.y * (1 - (v.y* v.y));
    g.z = d.z * (1 - (v.z* v.z));
    g.w = d.w * (1 - (v.w* v.w));
    grad4[i] = g;
  }

  int tail_start = size4 * 4;
  if(tid < (size - tail_start)) {
    float v = V[tid + tail_start];
    float d = derivatives[tid + tail_start];
    float g = grad[tid + tail_start];
    g = d * (1 - (v* v));
    grad[tid + tail_start] = g;
  }
}

// RELU
__global__ void vectorized_backward_relu(float *__restrict__ V, float *__restrict__ derivatives, float *__restrict__ grad, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int size4 = size / 4;
  
  float4 *V4 = reinterpret_cast<float4*>(V);
  float4 *derivatives4 = reinterpret_cast<float4*>(derivatives);
  float4 *grad4 = reinterpret_cast<float4*>(grad);

  for (int i = tid; i < size4; i += stride) {
    float4 v = V4[i];
    float4 d = derivatives4[i];
    float4 g = grad4[i];
    g.x = v.x > 0.0f ? d.x : 0.0f;
    g.y = v.y > 0.0f ? d.y : 0.0f;
    g.z = v.z > 0.0f ? d.z : 0.0f;
    g.w = v.w > 0.0f ? d.w : 0.0f;
    grad4[i] = g;
  }

  int tail_start = size4 * 4;
  if(tid < (size - tail_start)) {
    float v = V[tid + tail_start];
    float d = derivatives[tid + tail_start];
    float g = grad[tid + tail_start]; 
    g = v > 0.0f ? d : 0.0f;
    grad[tid + tail_start] = g;
  }
}

/* NON-VECTORIZED BACKWARD PASS */
// TANH
__global__ void non_vectorized_backward_tanh(float *__restrict__ V, float *__restrict__ derivatives, float *__restrict__ grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < size; i += stride) {
    grad[i] = derivatives[i] * (1 - (V[i]* V[i]));
  }
}

// RELU
__global__ void non_vectorized_backward_relu(float *__restrict__ V, float *__restrict__ derivatives, float *__restrict__ grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < size; i += stride) {
    grad[i] = V[i] > 0.0f ? derivatives[i] : 0.0f;
  } 
}