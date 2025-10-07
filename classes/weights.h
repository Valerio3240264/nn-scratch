#ifndef WEIGHTS_H
#define WEIGHTS_H
#include "input.h"
#include "virtual_classes.h"
#include "cuda_manager.cuh"
#include <iostream>

/* TODO 
1: Optimize the matrix multiplication using a personalized CUDA kernel.
2: Optimize the gradient computation using a personalized CUDA kernel.
3: Create a function to evaluate to process a whole batch of data (not only one single data point).
4: Create a function to evaluate the gradient of a whole batch.
5: Optimize the batch operations using personalized cuda kernels.
*/

/*
WEIGHTS CLASS DOCUMENTATION:
PURPOSE:
This class is used to store the weights of a layer and perform the matrix multiplication between the weights and the input values.
It also stores the gradients of the weights and the input values to perform the backward pass on the whole neural network.

Attributes:
- w: pointer to the weights array
- grad_w: pointer to the gradients array
- input_size: size of the input values
- output_size: size of the output values
- input_values: pointer to the input values
- pred: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)

Constructors:
- weights(int input_size, int output_size): creates a new array for the weights and gradients arrays and sets the predecessor to nullptr.

Methods:
- values_pointer(): returns the pointer to the weights array.
- grad_pointer(): returns the pointer to the gradients array.
- operator()(BackwardClass *in): performs the matrix multiplication between the weights and the input values.
- zero_grad(): sets all the gradients to 0.
- backward(double *derivatives): accumulates the gradients and propagates them to the predecessor.
- update(double learning_rate): updates the weights using the computed gradients.
*/

using namespace std;

class weights: public BackwardClass{
  private:
    double *w;
    double *grad_w;
    int input_size;
    int output_size;
    double *input_values;
    BackwardClass *pred;
    
    // CUDA-related members
    double *d_w;         // Device weights
    double *d_input;     // Device input vector
    double *d_output;    // Device output vector
    bool cuda_initialized;

  public:
    // Constructors
    weights(int input_size, int output_size);
    // Destructor
    ~weights();
    // Getters
    double *values_pointer() override;
    double *grad_pointer() override;
    // Methods
    double *operator()(BackwardClass * in);
    double *operator_cpu(BackwardClass * in);
    // Backpropagation functions
    void zero_grad() override;
    void backward(double *derivatives) override;
    void update(double learning_rate);
    // CUDA Methods
    void init_cuda();
    void cleanup_cuda();
    double *operator_cuda(BackwardClass * in);
    // Testing functions
    void print_weights();
    void print_grad_weights();
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
weights::weights(int input_size, int output_size)
{
  this->input_size = input_size;
  this->output_size = output_size;
  this->w = new double[input_size * output_size];
  this->grad_w = new double[input_size * output_size];
  this->input_values = nullptr;
  this->pred = nullptr;
  
  // Initialize CUDA-related members
  this->d_w = nullptr;
  this->d_input = nullptr;
  this->d_output = nullptr;
  this->cuda_initialized = false;
  
  for (int i = 0; i < input_size * output_size; i++)
  {
    this->w[i] = (rand() % 100)/100.0;
    this->grad_w[i] = 0;
  }
}

// Destructor
weights::~weights(){
  delete[] this->w;
  delete[] this->grad_w;
  cleanup_cuda();  // Clean up CUDA resources
}

/* GETTERS */
// Get the values pointer
double *weights::values_pointer(){
  return this->w;
}

// Get the gradient pointer
double *weights::grad_pointer(){
  return this->grad_w;
}

/* METHODS */
// Operator to evaluate the output
// CPU-only implementation
double *weights::operator_cpu(BackwardClass *in){
  this->input_values = in->values_pointer();
  this->pred = in;
  double *output = new double[output_size];
  for (int row = 0; row < this->output_size; row++){
    output[row] = 0;
    for(int col = 0; col< this->input_size; col++){
      output[row] += this->w[row * this->input_size + col] * this->input_values[col];
    }
  }
  return output;
}

// Smart operator - automatically chooses GPU or CPU
double *weights::operator()(BackwardClass *in){
  // Check if CUDA is available and beneficial
  if (CudaManager::is_cuda_available()) {
    // Initialize CUDA if not already done
    if (!cuda_initialized) {
      init_cuda();
    }
    
    // If CUDA initialization succeeded, use GPU
    if (cuda_initialized) {
      this->input_values = in->values_pointer();
      this->pred = in;
      
      try {
        // Transfer data to GPU using CudaManager
        CudaManager::copy_host_to_device(d_input, input_values, input_size);
        CudaManager::copy_host_to_device(d_w, w, input_size * output_size);
        
        // Launch kernel using CudaManager
        CudaManager::launch_matrix_vector_multiply(d_w, d_input, d_output, output_size, input_size);
        CudaManager::synchronize();
        
        // Get results from GPU
        double *output = new double[output_size];
        CudaManager::copy_device_to_host(output, d_output, output_size);
        
        return output;
        
      } catch (const std::exception& e) {
        fprintf(stderr, "CUDA operation failed: %s. Falling back to CPU.\n", e.what());
      }
    }
  }
  
  // Fall back to CPU computation
  return operator_cpu(in);
}

/* BACKPROPAGATION FUNCTIONS */
// Zero the gradient
void weights::zero_grad(){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    this->grad_w[i] = 0.0;
  }
}

// Backward pass
void weights::backward(double *derivatives){
  double *prevGrad = new double[this->input_size];
  for(int col = 0; col < this->input_size; col++){
    prevGrad[col] = 0;
    for(int row = 0; row< this->output_size; row++){
      prevGrad[col] += derivatives[col] * this->w[row * this->input_size + col];
      this->grad_w[row * this->input_size + col] += derivatives[row] * this->input_values[col];
    }
  }
  this->pred->backward(prevGrad);
  delete[] prevGrad;
}

// Update the weights
void weights::update(double learning_rate){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    this->w[i] -= learning_rate * this->grad_w[i];  // Fixed: subtract gradient for descent
  }
}

/* TESTING FUNCTIONS */
// Print the weights
void weights::print_weights(){
  for (int i = 0; i < this->input_size * this->output_size; i++)
  {
    cout << this->w[i] << " ";
  }
  cout << endl;
}

// Print the gradient of the weights
void weights::print_grad_weights(){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    cout << this->grad_w[i] << " ";
  }
  cout << endl;
}

/* CUDA IMPLEMENTATION */

// Initialize CUDA memory using CudaManager
void weights::init_cuda() {
  if (cuda_initialized) {
    return;
  }
  
  if (!CudaManager::is_cuda_available()) {
    fprintf(stderr, "Warning: CUDA not available. Using CPU computation.\n");
    return;
  }
  
  try {
    // Allocate GPU memory using CudaManager
    CudaManager::allocate_device_memory(&d_w, input_size * output_size);
    CudaManager::allocate_device_memory(&d_input, input_size);
    CudaManager::allocate_device_memory(&d_output, output_size);
    
    // Copy weights to GPU
    CudaManager::copy_host_to_device(d_w, w, input_size * output_size);
    
    cuda_initialized = true;
    printf("CUDA initialized for weights layer (%dx%d)\n", input_size, output_size);
      
  } catch (const std::exception& e) {
    fprintf(stderr, "CUDA initialization failed: %s\n", e.what());
    cleanup_cuda();
  }
}

// Cleanup CUDA memory using CudaManager
void weights::cleanup_cuda() {
    if (cuda_initialized) {
        CudaManager::free_device_memory(d_w);
        CudaManager::free_device_memory(d_input);
        CudaManager::free_device_memory(d_output);
        
        d_w = d_input = d_output = nullptr;
        cuda_initialized = false;
    }
}

// CUDA-accelerated operator using CudaManager
double *weights::operator_cuda(BackwardClass *in) {
  // Initialize CUDA if not already done
  if (!cuda_initialized) {
    init_cuda();
    if (!cuda_initialized) {
      printf("Falling back to CPU implementation\n");
      return operator()(in);  // Fallback to CPU
    }
  }
  
  this->input_values = in->values_pointer();
  this->pred = in;
  
  try {
    // Transfer data to GPU using CudaManager
    CudaManager::copy_host_to_device(d_input, input_values, input_size);
    CudaManager::copy_host_to_device(d_w, w, input_size * output_size);
    
    // Launch kernel using CudaManager
    CudaManager::launch_matrix_vector_multiply(d_w, d_input, d_output, output_size, input_size);
    CudaManager::synchronize();
    
    // Get results from GPU
    double *output = new double[output_size];
    CudaManager::copy_device_to_host(output, d_output, output_size);
    
    return output;
    
  } catch (const std::exception& e) {
    fprintf(stderr, "CUDA operation failed: %s. Falling back to CPU.\n", e.what());
    return operator()(in);  // Fallback to CPU on any error
  }
}

#endif