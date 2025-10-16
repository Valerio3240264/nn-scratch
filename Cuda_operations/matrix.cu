/* MATRIX-VECTOR MULTIPLICATION KERNEL */
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

/* VECTOR UPDATE KERNEL */
/*----TO DOCUMENT AND CHECK IF IT IS CORRECT----*/
__global__ void vector_update(double *V, double *U, double learning_rate, int size) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  
  if (idx + 1 < size) {
    double2 v_vec = reinterpret_cast<double2*>(V)[idx / 2];
    double2 u_vec = reinterpret_cast<double2*>(U)[idx / 2];
    
    v_vec.x -= learning_rate * u_vec.x;
    v_vec.y -= learning_rate * u_vec.y;
    
    reinterpret_cast<double2*>(V)[idx / 2] = v_vec;
  } else if (idx < size) {
    V[idx] -= learning_rate * U[idx];
  }
}

/* ACTIVATION FUNCTION KERNELS */
/*----TO DOCUMENT AND CHECK IF IT IS CORRECT----*/
__global__ void activation_tanh(double *V, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    V[idx] = tanh(V[idx]);
  }
}

/*----TO DOCUMENT AND CHECK IF IT IS CORRECT----*/
__global__ void activation_relu(double *V, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < size) {
    V[idx] = fmax(0.0, V[idx]);
  }
}