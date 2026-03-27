#include "MatricesOp.h"


/* STANDAR MATRIX MULTIPLICATION FUNCTIONS*/

/* MULTIPLY R = A*B */
// Dim(A) = N x K
// Dim(B) = K x M
// Dim(R) = N x M
void Multiply(float *A, float *B, float *R, int N, int K, int M){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      R[i*M + j] = 0.0f;
      for(int k = 0; k < K; k++){
        R[i*M + j] += A[i*K + k] * B[k*M + j];
      }
    }
  }
}

/* MULTIPLY AND ADD R = A*B + C */
// Dim(A) = N x K
// Dim(B) = K x M
// Dim(C) = N x M
// Dim(R) = N x M
void MultiplyAndAdd(float *A, float *B, float *C, float *R, int N, int K, int M){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      R[i*M + j] = 0.0f;
      for(int k = 0; k < K; k++){
        R[i*M + j] += A[i*K + k] * B[k*M + j];
      }
      R[i*M + j] += C[i*M + j];
    }
  }
}

/* IN-PLACE MULTIPLY AND ADD R += A*B */
// Dim(A) = N x K
// Dim(B) = K x M
// Dim(R) = N x M
// !! IMPORTANT: The matrix R must have values in it or the result will not be registered correctly
void InPlaceMultiplyAndAdd(float *A, float *B, float *R, int N, int K, int M){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      for(int k = 0; k < K; k++){
        R[i*M + j] += A[i*K + k] * B[k*M + j];
      }
    }
  }
}

/* TRANSPOSE A */
// These functions are used to perform matrix multiplication with the first matrix transposed

/* MULTIPLY R = A^T * B */
// Dim(A) = K x N
// Dim(B) = K x M
// Dim(R) = N x M
void Multiply_Transpose1(float *A, float *B, float *R, int N, int K, int M){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      R[i*M + j] = 0.0f;
      for(int k = 0; k < K; k++){
        R[i*M + j] += A[k*N + i] * B[k*M + j];
      }
    }
  }
}


/* MULTIPLY AND ADD R = A^T * B + C */
// Dim(A) = K x N
// Dim(B) = K x M
// Dim(C) = N x M
// Dim(R) = N x M
void MultiplyAndAdd_Transpose1(float *A, float *B, float *C, float *R, int N, int K, int M){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      R[i*M + j] = 0.0f;
      for(int k = 0; k < K; k++){
        R[i*M + j] += A[k*N + i] * B[k*M + j];
      }
      R[i*M + j] += C[i*M + j];
    }
  }
}


/* IN-PLACE MULTIPLY AND ADD R += A^T * B + C */
// Dim(A) = K x N
// Dim(B) = K x M
// Dim(R) = N x M
// !! IMPORTANT: The matrix R must have values in it or the result will not be registered correctly
void InPlaceMultiplyAndAdd_Transpose1(float *A, float *B, float *R, int N, int K, int M){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      for(int k = 0; k < K; k++){
        R[i*M + j] += A[k*N + i] * B[k*M + j];
      }
    }
  }
}


/* TRANSPOSE B */
// These functions are used to perform matrix multiplication with the second matrix transposed

/* MULTIPLY R = A * B^T */
// Dim(A) = N x K
// Dim(B) = K x M (transposed dim(B^T) = M x K)
// Dim(R) = N x M
void Multiply_transpose2(float *A, float *B, float *R, int N, int K, int M){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      R[i*M + j] = 0.0f;
      for(int k = 0; k < K; k++){
        R[i*M + j] += A[i*K + k] * B[j*K + k];
      }
    }
  }
}

/* MULTIPLY AND ADD R = A * B^T + C */
// Dim(A) = N x K
// Dim(B) = M x K (transposed dim(B^T) = K x M)
// Dim(C) = N x M
void MultiplyAndAdd_transpose2(float *A, float *B, float *C, float *R, int N, int K, int M){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      R[i*M + j] = 0.0f;
      for(int k = 0; k < K; k++){
        R[i*M + j] += A[i*K + k] * B[j*K + k];
      }
      R[i*M + j] += C[i*M + j];
    }
  }
}

/* IN-PLACE MULTIPLY AND ADD R += A * B^T */
// Dim(A) = N x K
// Dim(B) = M x K (transposed dim(B^T) = K x M)
// Dim(R) = N x M
// !! IMPORTANT: The matrix R must have values in it or the result will not be registered correctly
void InPlaceMultiplyAndAdd_transpose2(float *A, float *B, float *R, int N, int K, int M){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      for(int k = 0; k < K; k++){
        R[i*M + j] += A[i*K + k] * B[j*K + k];
      }
    }
  }
}

/* POINTWISE OPERATIONS */
/* MATRIX ADD R = A + B */
// Dim(A) = N x M
// Dim(B) = N x M
// Dim(R) = N x M
void Matrix_Add(float *A, float *B, float *R, int N, int M){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < M; j++){
      R[i*M + j] = A[i*M + j] + B[i*M + j];
    }
  }
}

/* IN-PLACE MATRIX ADD R += A + B */
// Dim(A) = N x M
// Dim(B) = N x M
// !! IMPORTANT: The matrix A must have values in it or the result will not be registered correctly
void InPlaceMatrix_Add(float *A, float *B, int N, int M){
  for(int i = 0; i < N*M; i++){
    A[i] += B[i];
  }
}