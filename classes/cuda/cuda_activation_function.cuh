#ifndef CUDA_ACTIVATION_FUNCTION_CUH
#define CUDA_ACTIVATION_FUNCTION_CUH
#include "cuda_manager.cuh"
#include "virtual_classes.h"
#include <iostream>

/*
CUDA ACTIVATION FUNCTION CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the activation function class but it is used to store the values and gradients in device memory.
*/

using namespace std;

class cuda_activation_function: public BackwardClass{
  private:
    double *d_value;
    double *d_grad;
    int size;
    BackwardClass *pred;
    Activation_name function_name;
  public:
    cuda_activation_function(int size, double *value, Activation_name function_name, BackwardClass *pred);
    ~cuda_activation_function();
    double *values_pointer() override;
    double *grad_pointer() override;
    void zero_grad() override;
    void backward(double *derivatives) override;
    void operator()();
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
cuda_activation_function::cuda_activation_function(int size, double *value, Activation_name function_name, BackwardClass *pred){
  this->size = size;
  this->d_value = value; // We assume that the value is already in device memory
  this->d_grad = nullptr;
  this->pred = pred;
  this->function_name = function_name;

  cuda_manager::allocate_device_memory_zeros(this->d_grad, size);
}

// Destructor
cuda_activation_function::~cuda_activation_function(){
  cuda_manager::free_device_memory(this->d_value);
  cuda_manager::free_device_memory(this->d_grad);
}

/* GETTERS */
// Get the values pointer
double *cuda_activation_function::values_pointer(){
  return this->d_value;
}

// Get the gradient pointer
double *cuda_activation_function::grad_pointer(){
  return this->d_grad;
}

/* METHODS */
// Operator to apply the activation function
void cuda_activation_function::operator()(){
  if(this->function_name != TANH && this->function_name != RELU && this->function_name != LINEAR){
    throw invalid_argument("Invalid activation function");
  }
  cuda_manager::launch_activation_function(this->d_value, this->function_name, this->size);
}

/* BACKPROPAGATION FUNCTIONS */
// Zero the gradient
void cuda_activation_function::zero_grad(){
  cuda_manager::zero_device_memory(this->d_grad, this->size);
}

// Backward pass
void cuda_activation_function::backward(double *derivatives){
  /*TODO*/
}

#endif