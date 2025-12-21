#include "../headers/cuda_activation.cuh"

#include "../cuda_manager.cuh"
#include "../cuda_manager_impl.cuh"
#include <iostream>

using namespace std;

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
cuda_activation::cuda_activation(int size, float *value, Activation_name function_name, BackwardClass *pred){
  this->size = size;
  this->d_value = value;
  this->d_grad = nullptr;
  this->pred = pred;
  this->function_name = function_name;

  allocate_device_memory_zeros<float>(&this->d_grad, size);
}

// Destructor
cuda_activation::~cuda_activation(){
  free_device_memory(this->d_grad);
}

/* GETTERS */
// Get the values pointer
float *cuda_activation::values_pointer(){
  return this->d_value;
}

// Get the gradient pointer
float *cuda_activation::grad_pointer(){
  return this->d_grad;
}

/* METHODS */
// Forward pass
void cuda_activation::operator()(){
  if(this->function_name == TANH){
    launch_activation_tanh(this->d_value, this->size);
  }
  else if(this->function_name == RELU){
    launch_activation_relu(this->d_value, this->size);
  }
  else if(this->function_name == LINEAR){
    return;
  }
  else{
    throw invalid_argument("Invalid activation function");
  }
}

// Zero the gradient
void cuda_activation::zero_grad(){
  zero_device_memory(this->d_grad, this->size);
}

// Backward pass
void cuda_activation::backward(float *derivatives){
  if(this->function_name == TANH){
    launch_backward_tanh(this->d_value, derivatives, this->d_grad, this->size);
  }
  else if(this->function_name == RELU){
    launch_backward_relu(this->d_value, derivatives, this->d_grad, this->size);
  }
  else if(this->function_name == LINEAR){
    launch_backward_linear(this->d_value, derivatives, this->d_grad, this->size);
  }
  else{
    throw invalid_argument("Invalid activation function");
  }
  
  this->pred->backward(this->d_grad);
}
