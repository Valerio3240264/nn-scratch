#ifndef CUDA_WEIGHTS_CUH
#define CUDA_WEIGHTS_CUH
#include "cuda_input.cuh"
#include "../virtual_classes.h"
#include "cuda_manager.cuh"
#include <iostream>

using namespace std;

/*
CUDA WEIGHTS CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the weights class but it is used to store the weights and gradients in device memory.

Note: When using this class, we assume that each class that interacts with this class (in the raw layer) has memory allocated in device memory.
*/

class cuda_weights: public WeightsClass{
  private:
    float *d_w;
    float *d_grad_w;
    float *d_b;
    float *d_grad_b;
    float *d_input_grad_buffer;
    int input_size;
    int output_size;
    float *d_input_values;
    BackwardClass *pred;

  public:

    // Constructor
    cuda_weights(int input_size, int output_size);
  
    // Destructor
    ~cuda_weights();
  
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float *bias_pointer();
    float *grad_bias_pointer();
  
    // Methods
    void backward(float *derivatives) override;
    void zero_grad() override;
    void operator()(BackwardClass *in, float *output_pointer) override;
    void update(float learning_rate);
    void print_weights() override;
    void print_grad_weights() override;
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
cuda_weights::cuda_weights(int input_size, int output_size){
  this->input_size = input_size;
  this->output_size = output_size;
  this->d_w = nullptr;
  this->d_grad_w = nullptr;
  this->d_b = nullptr;
  this->d_grad_b = nullptr;
  this->d_input_grad_buffer = nullptr;
  this->d_input_values = nullptr;
  this->pred = nullptr;

  // Initialize weights with Xavier/Glorot initialization
  allocate_device_memory_xavier<float>(&this->d_w, input_size * output_size, input_size);
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
  launch_I_W_B_multiplication(this->d_w, this->d_input_values, this->d_b, output_pointer, this->output_size, this->input_size);
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
  
  launch_backward_W(this->d_w, this->d_input_values, derivatives, this->d_grad_w, this->d_input_grad_buffer, this->output_size, this->input_size);
  launch_backward_bias(this->d_b, derivatives, this->d_grad_b, this->output_size);
  
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

#endif