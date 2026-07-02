#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include "../../utils/MatricesOp.h"

#define CUDA_CHECK(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      std::cerr << cudaGetErrorString(error) << std::endl; \
      return 1; \
    } \
  } while (0)

#include "../../Kernels/matrix.cu"

namespace {

int run_case(int rows, int inner, int cols, std::mt19937& random) {
  const size_t A_size = static_cast<size_t>(inner) * rows;
  const size_t B_size = static_cast<size_t>(inner) * cols;
  const size_t result_size = static_cast<size_t>(rows) * cols;

  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> A(A_size);
  std::vector<float> B(B_size);
  for (float& value : A) value = distribution(random);
  for (float& value : B) value = distribution(random);

  std::vector<float> expected(result_size);
  Multiply_Transpose1(
      A.data(), B.data(), expected.data(), rows, inner, cols);

  float* device_A = nullptr;
  float* device_B = nullptr;
  float* device_result = nullptr;
  CUDA_CHECK(cudaMalloc(&device_A, A_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_B, B_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_result, result_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(
      device_A, A.data(), A_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      device_B, B.data(), B_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_result, 0, result_size * sizeof(float)));

  const dim3 block(16, 16);
  const dim3 grid((cols + 15) / 16, (rows + 15) / 16);
  tiled_matmul_transpose_left_accumulate<<<grid, block>>>(
      device_A,
      device_B,
      device_result,
      rows,
      cols,
      inner);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> actual(result_size);
  CUDA_CHECK(cudaMemcpy(
      actual.data(),
      device_result,
      result_size * sizeof(float),
      cudaMemcpyDeviceToHost));

  float max_error = 0.0f;
  for (size_t i = 0; i < result_size; ++i) {
    max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
  }

  CUDA_CHECK(cudaFree(device_A));
  CUDA_CHECK(cudaFree(device_B));
  CUDA_CHECK(cudaFree(device_result));

  std::cout << "transpose-left max error: " << max_error << std::endl;
  return max_error < 1e-3f ? 0 : 1;
}

}  // namespace

int main() {
  std::mt19937 random(42);
  if (run_case(256, 128, 192, random) != 0) return 1;
  if (run_case(253, 125, 189, random) != 0) return 1;
  return 0;
}
