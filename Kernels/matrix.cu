#include <cstddef>

namespace {

constexpr int TILE_SIZE = 16;

}  // namespace

// A: rows x inner
// B: inner x cols
// bias: cols
// result: rows x cols
__global__ void tiled_matmul_add_bias(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ result,
    int rows,
    int cols,
    int inner) {
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

  const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float sum = 0.0f;

  for (int tile = 0; tile < inner; tile += TILE_SIZE) {
    const int A_col = tile + threadIdx.x;
    const int B_row = tile + threadIdx.y;

    tile_A[threadIdx.y][threadIdx.x] =
        row < rows && A_col < inner
            ? A[row * inner + A_col]
            : 0.0f;
    tile_B[threadIdx.y][threadIdx.x] =
        B_row < inner && col < cols
            ? B[B_row * cols + col]
            : 0.0f;

    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < rows && col < cols) {
    result[row * cols + col] = sum + bias[col];
  }
}

// A: inner x rows
// B: inner x cols
// result: rows x cols
// The result accumulates to match CPU parameter-gradient semantics.
__global__ void tiled_matmul_transpose_left_accumulate(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ result,
    int rows,
    int cols,
    int inner) {
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

  const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float sum = 0.0f;

  for (int tile = 0; tile < inner; tile += TILE_SIZE) {
    const int A_row = tile + threadIdx.x;
    const int B_row = tile + threadIdx.y;

    tile_A[threadIdx.y][threadIdx.x] =
        row < rows && A_row < inner
            ? A[A_row * rows + row]
            : 0.0f;
    tile_B[threadIdx.y][threadIdx.x] =
        B_row < inner && col < cols
            ? B[B_row * cols + col]
            : 0.0f;

    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < rows && col < cols) {
    result[row * cols + col] += sum;
  }
}

// A: rows x inner
// B: cols x inner
// result: rows x cols
__global__ void tiled_matmul_transpose_right(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ result,
    int rows,
    int cols,
    int inner) {
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

  const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float sum = 0.0f;

  for (int tile = 0; tile < inner; tile += TILE_SIZE) {
    const int A_col = tile + threadIdx.x;
    const int B_col = tile + threadIdx.y;

    tile_A[threadIdx.y][threadIdx.x] =
        row < rows && A_col < inner
            ? A[row * inner + A_col]
            : 0.0f;
    tile_B[threadIdx.y][threadIdx.x] =
        col < cols && B_col < inner
            ? B[col * inner + B_col]
            : 0.0f;

    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < rows && col < cols) {
    result[row * cols + col] = sum;
  }
}

// result += row_sum(matrix)
__global__ void accumulate_row_sum(
    float* __restrict__ result,
    const float* __restrict__ matrix,
    int rows,
    int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }

  float sum = 0.0f;
  for (int row = 0; row < rows; ++row) {
    sum += matrix[row * cols + col];
  }
  result[col] += sum;
}

__global__ void vector_update(
    float* __restrict__ values,
    const float* __restrict__ gradients,
    float learning_rate,
    size_t size) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    values[index] -= learning_rate * gradients[index];
  }
}

__global__ void scale_weights(
    float* __restrict__ data,
    size_t size,
    float scale) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    data[index] = (2.0f * data[index] - 1.0f) * scale;
  }
}
