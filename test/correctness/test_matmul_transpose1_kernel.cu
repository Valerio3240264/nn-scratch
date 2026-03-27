#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include "../utils/MatricesOp.h"

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t err__ = (call);                                                   \
    if (err__ != cudaSuccess) {                                                   \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "     \
                << cudaGetErrorString(err__) << std::endl;                        \
      return 1;                                                                    \
    }                                                                             \
  } while (0)

// Adapt local aliases used by Kernels/matrix.cu.
#define blockidx blockIdx
#define threadidx threadIdx
#define blockdimx blockDim
#include "../Kernels/matrix.cu"
#undef blockidx
#undef threadidx
#undef blockdimx

int main() {
  // User-requested matrix sizes.
  constexpr int N = 4096;  // A^T rows / C rows
  constexpr int K = 1024;  // A rows / B rows
  constexpr int M = 4096;  // B cols / C cols

  // Parameters adjusted to fit 48KB shared memory/block.
  constexpr int BN = 128;
  constexpr int BK = 32;
  constexpr int BM = 128;
  constexpr int RN = 8;
  constexpr int RM = 8;

  // Consistent launch for RN/RM micro-tiling with BN/BM above.
  constexpr int blocks = 1024;
  constexpr int threads_per_block = 256;

  // A is stored as K x N so kernel computes A^T (N x K) * B (K x M).
  const size_t sizeA = static_cast<size_t>(K) * N;
  const size_t sizeB = static_cast<size_t>(K) * M;
  const size_t sizeC = static_cast<size_t>(N) * M;

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::cerr << "No CUDA devices found." << std::endl;
    return 1;
  }
  CUDA_CHECK(cudaSetDevice(0));

  std::vector<float> hA(sizeA);
  std::vector<float> hB(sizeB);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < sizeA; ++i) hA[i] = dist(rng);
  for (size_t i = 0; i < sizeB; ++i) hB[i] = dist(rng);

  float* dA = nullptr;
  float* dB = nullptr;
  float* dC = nullptr;

  CUDA_CHECK(cudaMalloc(&dA, sizeA * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, sizeB * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, sizeC * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));

  std::cout << "Launching matmul_transpose1 with requested parameters..." << std::endl;
  matmul_transpose1<BN, BK, BM, RN, RM><<<blocks, threads_per_block>>>(dA, dB, dC, N, M, K);

  const cudaError_t launch_err = cudaGetLastError();
  std::cout << "Kernel launch status: " << cudaGetErrorString(launch_err) << std::endl;

  const cudaError_t sync_err = cudaDeviceSynchronize();
  std::cout << "Kernel sync status: " << cudaGetErrorString(sync_err) << std::endl;

  // Copy A, B, and kernel result C back to host for CPU reference check.
  std::vector<float> hA_from_device(sizeA);
  std::vector<float> hB_from_device(sizeB);
  std::vector<float> hC_from_device(sizeC);
  std::vector<float> hC_cpu(sizeC, 0.0f);

  CUDA_CHECK(cudaMemcpy(
      hA_from_device.data(), dA, sizeA * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      hB_from_device.data(), dB, sizeB * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      hC_from_device.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU reference: R = A^T * B (row-major), using utils/MatricesOp.cpp
  Multiply_Transpose1(hA_from_device.data(), hB_from_device.data(), hC_cpu.data(), N, K, M);

  double max_abs_err = 0.0;
  for (size_t i = 0; i < sizeC; ++i) {
    const double gpu = static_cast<double>(hC_from_device[i]);
    const double cpu = static_cast<double>(hC_cpu[i]);
    const double abs_err = std::abs(gpu - cpu);
    max_abs_err = std::max(max_abs_err, abs_err);
  }

  std::cout << "max_abs_err=" << max_abs_err << std::endl;

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}
