#ifndef MATRICES_OP_H
#define MATRICES_OP_H

// Basic dense matrix operations for float arrays stored in row-major order.
// Matrices are represented as contiguous 1D arrays:
// element (i, j) of an N x M matrix is at index i * M + j.

// STANDARD MATRIX MULTIPLICATION FUNCTIONS

// R = A * B
// A: N x K, B: K x M, R: N x M
void Multiply(float* A, float* B, float* R, int N, int K, int M);

// R = A * B + C
// A: N x K, B: K x M, C: N x M, R: N x M
void MultiplyAndAdd(float* A, float* B, float* C, float* R, int N, int K, int M);

// R += A * B
// A: N x K, B: K x M, R: N x M
void InPlaceMultiplyAndAdd(float* A, float* B, float* R, int N, int K, int M);

// TRANSPOSE A FUNCTIONS

// R = A^T * B
// A: K x N, B: K x M, R: N x M
void Multiply_Transpose1(float* A, float* B, float* R, int N, int K, int M);

// R = A^T * B + C
// A: K x N, B: K x M, C: N x M, R: N x M
void MultiplyAndAdd_Transpose1(float* A, float* B, float* C, float* R, int N, int K, int M);

// R += A^T * B
// A: K x N, B: K x M, R: N x M
void InPlaceMultiplyAndAdd_Transpose1(float* A, float* B, float* R, int N, int K, int M);

// TRANSPOSE B FUNCTIONS

// R = A * B^T
// A: N x K, B: M x K, R: N x M
void Multiply_transpose2(float* A, float* B, float* R, int N, int K, int M);

// R = A * B^T + C
// A: N x K, B: M x K, C: N x M, R: N x M
void MultiplyAndAdd_transpose2(float* A, float* B, float* C, float* R, int N, int K, int M);

// R += A * B^T
// A: N x K, B: M x K, R: N x M
void InPlaceMultiplyAndAdd_transpose2(float* A, float* B, float* R, int N, int K, int M);

// POINTWISE OPERATIONS

// R = A + B
// A: N x M, B: N x M, R: N x M
void Matrix_Add(float* A, float* B, float* R, int N, int M);

// A += B
// A: N x M, B: N x M
void InPlaceMatrix_Add(float* A, float* B, int N, int M);

#endif  // MATRICES_OP_H
