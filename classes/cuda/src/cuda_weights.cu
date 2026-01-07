#include "../headers/cuda_weights.cuh"

#include "../cuda_manager.cuh"
#include "../cuda_manager_impl.cuh"
#include "../../enums.h"
#include <iostream>

using namespace std;

void cuda_weights::init_weights(Activation_name function_name){
  if(function_name == TANH){
    allocate_device_memory_xavier<float>(&this->d_w, input_size * output_size, this->input_size, this->output_size);
  }
  else if(function_name == RELU){
    allocate_device_memory_he<float>(&this->d_w, input_size * output_size, this->input_size);
  }
  else if(function_name == LINEAR){
    allocate_device_memory_xavier<float>(&this->d_w, input_size * output_size, this->input_size, this->output_size);
  }
  else{
    throw invalid_argument("Invalid activation function");
  }
}

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
cuda_weights::cuda_weights(int input_size, int output_size, Activation_name function_name){
  if(input_size <= 0 || output_size <= 0){
    throw invalid_argument("Input and output size must be greater than 0");
    exit(1);
  }
  
  this->input_size = input_size;
  this->output_size = output_size;
  this->d_w = nullptr;
  this->d_grad_w = nullptr;
  this->d_b = nullptr;
  this->d_grad_b = nullptr;
  this->d_input_grad_buffer = nullptr;
  this->d_input_values = nullptr;
  this->pred = nullptr;

  // Initialize weights and biases
  init_weights(function_name);
  allocate_device_memory_zeros<float>(&this->d_grad_w, input_size * output_size);
  allocate_device_memory_zeros<float>(&this->d_b, output_size);
  allocate_device_memory_zeros<float>(&this->d_grad_b, output_size);
  allocate_device_memory_zeros<float>(&this->d_input_grad_buffer, input_size);
}

// Destructor
cuda_weights::~cuda_weights(){
  free_device_memory(this->d_w);
  free_device_memory(this->d_grad_w);
  free_device_memory(this->d_b);
  free_device_memory(this->d_grad_b);
  free_device_memory(this->d_input_grad_buffer);
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

/* METHODS */
// Forward pass
// W x Input + b
void cuda_weights::operator()(BackwardClass *in, float *output_pointer){
  this->d_input_values = in->values_pointer();
  this->pred = in;
  launch_SGEMV(this->d_w, this->d_input_values, this->d_b, output_pointer, this->output_size, this->input_size);
  return;
}

// Zero the gradient
void cuda_weights::zero_grad(){
  zero_device_memory(this->d_grad_w, this->input_size * this->output_size);
  zero_device_memory(this->d_grad_b, this->output_size);
}

// Backward pass
void cuda_weights::backward(float *derivatives){
  zero_device_memory(this->d_input_grad_buffer, this->input_size);
  
  // Use optimized tiled kernel that computes weight gradients, input gradients, and bias gradients in one pass
  launch_tiled_backward_Weights(this->d_w, this->d_input_values, derivatives, 
                                this->d_grad_w, this->d_input_grad_buffer, this->d_grad_b, 
                                this->output_size, this->input_size);
  
  this->pred->backward(this->d_input_grad_buffer);
}

// Update the weights
void cuda_weights::update(float learning_rate){
  launch_update(this->d_w, this->d_grad_w, learning_rate, this->input_size * this->output_size);
  launch_update(this->d_b, this->d_grad_b, learning_rate, this->output_size);
}

// Print weights (copies from device to host)
void cuda_weights::print_weights(){
  float *h_w = new float[this->input_size * this->output_size];
  copy_device_to_host(h_w, this->d_w, this->input_size * this->output_size);
  std::cout << "Weights: ";
  for(int i = 0; i < this->input_size * this->output_size; i++){
    std::cout << h_w[i] << " ";
  }
  std::cout << std::endl;
  delete[] h_w;
}

// Print gradient weights (copies from device to host)
void cuda_weights::print_grad_weights(){
  float *h_grad_w = new float[this->input_size * this->output_size];
  copy_device_to_host(h_grad_w, this->d_grad_w, this->input_size * this->output_size);
  std::cout << "Gradient Weights: ";
  for(int i = 0; i < this->input_size * this->output_size; i++){
    std::cout << h_grad_w[i] << " ";
  }
  std::cout << std::endl;
  delete[] h_grad_w;
}