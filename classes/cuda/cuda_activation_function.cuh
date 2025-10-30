#ifndef CUDA_ACTIVATION_FUNCTION_CUH
#define CUDA_ACTIVATION_FUNCTION_CUH
#include "cuda_manager.cuh"
#include "../virtual_classes.h"
#include "../enums.h"
#include <iostream>

/*
CUDA ACTIVATION FUNCTION CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the activation function class but it is used to store the values and gradients in device memory.

Note: When using this class, we assume that each class that interacts with this class (in the raw layer) has memory allocated in device memory.
*/

using namespace std;

class cuda_activation_function: public ActivationClass{
  private:
    float *d_value;
    float *d_grad;
    int size;
    BackwardClass *pred;
    Activation_name function_name;
  public:

    // Constructor
    cuda_activation_function(int size, float *value, Activation_name function_name, BackwardClass *pred);
  
    // Destructor
    ~cuda_activation_function();
  
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
  
    // Methods
    void operator()();
    void zero_grad() override;
    void backward(float *derivatives) override;
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
cuda_activation_function::cuda_activation_function(int size, float *value, Activation_name function_name, BackwardClass *pred){
  this->size = size;
  this->d_value = value;
  this->d_grad = nullptr;
  this->pred = pred;
  this->function_name = function_name;

  allocate_device_memory_zeros<float>(&this->d_grad, size);
}

// Destructor
cuda_activation_function::~cuda_activation_function(){
  free_device_memory(this->d_grad);
}

/* GETTERS */
// Get the values pointer
float *cuda_activation_function::values_pointer(){
  return this->d_value;
}

// Get the gradient pointer
float *cuda_activation_function::grad_pointer(){
  return this->d_grad;
}

/* METHODS */
// Forward pass
void cuda_activation_function::operator()(){
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
void cuda_activation_function::zero_grad(){
  zero_device_memory(this->d_grad, this->size);
}

// Backward pass
void cuda_activation_function::backward(float *derivatives){
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

#endif