#include "../headers/cuda_weights.cuh"

#include "../cuda_manager.cuh"
#include "../cuda_manager_impl.cuh"
#include "../../enums.h"
#include <iostream>

using namespace std;

// Initialize weights based on the activation function used
void cuda_weights::init_weights(Activation_name function_name){
  if(function_name == TANH){
    allocate_device_memory_xavier<float>(
        &this->d_w,
        this->input_size,
        this->output_size);
  }
  else if(function_name == RELU){
    allocate_device_memory_he<float>(
        &this->d_w,
        this->input_size,
        this->output_size);
  }
  else if(function_name == LINEAR){
    allocate_device_memory_xavier<float>(
        &this->d_w,
        this->input_size,
        this->output_size);
  }
  else{
    throw invalid_argument("Invalid activation function");
  }
}

// Check pred component matches dimensions
void cuda_weights::check_pred(){
  if(this->pred == nullptr)
    return;
  else if(this->pred->get_output_size() != this->input_size){
    throw invalid_argument("Pred component doesn't matches weights input_size");
    exit(1);
  }
  else if(this->pred->get_batch_size() != this->batch_size){
    throw invalid_argument("Pred component doesn't matches weights batch_size");
    exit(1);
  }
  return;
}

// Check next component matches dimensions
void cuda_weights::check_next(){
  if(this->next == nullptr)
    return;
  else if(this->next->get_output_size() != this->output_size){
    throw invalid_argument("Next component doesn't matches weights output_size");
    exit(1);
  }
  else if(this->next->get_batch_size() != this->batch_size){
    throw invalid_argument("Next component doesn't matches weights batch_size");
    exit(1);
  }
  return;
}

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
cuda_weights::cuda_weights( size_t input_size,
                            size_t output_size,
                            size_t batch_size,
                            Activation_name function_name,
                            BackwardClass *pred,
                            BackwardClass *next){
  if(input_size <= 0 || output_size <= 0){
    throw invalid_argument("Input and output size must be greater than 0");
    exit(1);
  }

  this->input_size = input_size;
  this->output_size = output_size;
  this->batch_size = batch_size;
  this->d_w = nullptr;
  this->d_grad_w = nullptr;
  this->d_b = nullptr;
  this->d_grad_b = nullptr;

  this->pred = pred;
  this->next = next;
  this->check_pred();
  this->check_next();

  // Initialize weights and biases
  init_weights(function_name);
  allocate_device_memory_zeros<float>(
      &this->d_grad_w,
      input_size * output_size);
  allocate_device_memory_zeros<float>(&this->d_b, output_size);
  allocate_device_memory_zeros<float>(&this->d_grad_b, output_size);
}

// Destructor
cuda_weights::~cuda_weights(){
  free_device_memory(this->d_w);
  free_device_memory(this->d_grad_w);
  free_device_memory(this->d_b);
  free_device_memory(this->d_grad_b);
}

/* GETTERS */
// Get the weights pointer
float *cuda_weights::values_pointer(){
  return this->d_w;
}

// Get the gradient pointer
float *cuda_weights::grad_pointer(){
  return this->d_grad_w;
}

// Get the bias pointer
float *cuda_weights::bias_pointer(){
  return this->d_b;
}

// Get the bias gradient pointer
float *cuda_weights::grad_bias_pointer(){
  return this->d_grad_b;
}

size_t cuda_weights::get_output_size(){
  return this->output_size;
}

size_t cuda_weights::get_batch_size(){
  return this->batch_size;
}

/* SETTERS */
// Set pred pointer
void cuda_weights::set_pred(BackwardClass *pred){
  this->pred = pred;
  this->check_pred();
}

// Set next pointer
void cuda_weights::set_next(BackwardClass *next){
  this->next = next;
  this->check_next();
}

/* METHODS */
// Forward pass
// x*W + b
void cuda_weights::operator()(){
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute forward pass without weights pred component.");
    exit(1);
  }
  if(this->next == nullptr){
    throw invalid_argument("Error: can't compute forward pass without weights next component.");
    exit(1);
  }

  float *d_input_values = this->pred->values_pointer();
  float *out_values = this->next->values_pointer();

  // Matmul x * W
  launch_matmul_forward(
    d_input_values,
    this->d_w,
    this->d_b,
    out_values,
    this->batch_size,
    this->output_size,
    this->input_size);

  return;
}

// Zero the gradient
void cuda_weights::zero_grad(){
  zero_device_memory(
      this->d_grad_w,
      this->input_size * this->output_size);
  zero_device_memory(this->d_grad_b, this->output_size);
}

// Backward pass
void cuda_weights::backward(){
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute backward pass without weights pred component.");
    exit(1);
  }
  if(this->next == nullptr){
    throw invalid_argument("Error: can't compute backward pass without weights next component.");
    exit(1);
  }

  float *d_prevGrad = this->pred->grad_pointer();
  float *d_input_values = this->pred->values_pointer();
  float *d_nextGrad = this->next->grad_pointer();

  // Weights gradient:
  // dL/dW = X^T * dL/dY where X is (batch_size x input_size)
  launch_matmul_backward_weights(
    d_input_values,
    d_nextGrad,
    this->d_grad_w,
    this->input_size,
    this->output_size,
    this->batch_size
  );

  // Input gradient:
  // dL/dX = dL/dY * W^T where dL/dY is (batch_size x output_size)
  launch_matmul_backward_input(
      d_nextGrad,
      this->d_w,
      d_prevGrad,
      this->batch_size,
      this->input_size,
      this->output_size
  );

  // Bias gradient
  launch_bias_gradient(
      this->d_grad_b,
      d_nextGrad,
      this->batch_size,
      this->output_size
  );

  this->pred->backward();
}

// Update the weights
void cuda_weights::update(float learning_rate){
  launch_update(
      this->d_w,
      this->d_grad_w,
      learning_rate,
      this->input_size * this->output_size);
  launch_update(
      this->d_b,
      this->d_grad_b,
      learning_rate,
      this->output_size);
}

// Print weights (copies from device to host)
void cuda_weights::print_weights(){
  float *h_w = new float[this->input_size * this->output_size];
  copy_device_to_host(
      h_w,
      this->d_w,
      this->input_size * this->output_size);
  std::cout << "Weights: ";
  for(size_t i = 0; i < this->input_size; i++){
    for(size_t j = 0; j < this->output_size; j++){
      std::cout << h_w[i * this->output_size + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  delete[] h_w;
}

// Print gradient weights (copies from device to host)
void cuda_weights::print_grad_weights(){
  float *h_grad_w = new float[this->input_size * this->output_size];
  copy_device_to_host(
      h_grad_w,
      this->d_grad_w,
      this->input_size * this->output_size);
  std::cout << "Gradient Weights: ";
  for(size_t i = 0; i < this->input_size; i++){
    for(size_t j = 0; j < this->output_size; j++){
      std::cout << h_grad_w[i * this->output_size + j] << " ";
    }
  }
  std::cout << std::endl;
  delete[] h_grad_w;
}
