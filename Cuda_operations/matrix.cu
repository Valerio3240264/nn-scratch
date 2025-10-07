
__global__ void matrix_vector_multiplication(double *M, double *V, double *R, int output_size, int input_size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < output_size) {
    double sum = 0.0;
    for (int col = 0; col < input_size; col++) {
      sum += M[row * input_size + col] * V[col];
    }
    R[row] = sum;
  }
}