#include "../headers/cuda_activation.cuh"

#include "../cuda_manager.cuh"
#include "../cuda_manager_impl.cuh"
#include <iostream>

using namespace std;

// Check pred component matches dimensions
void cuda_activation::check_pred(){
  if(this->pred == nullptr)
    return;
  else if(this->pred->get_output_size() != this->size){
    throw invalid_argument("Pred component doesn't matches activation size");
    exit(1);
  }
  else if(this->pred->get_batch_size() != this->batch_size){
    throw invalid_argument("Pred component doesn't matches activation batch_size");
    exit(1);
  }
  return;
}

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
cuda_activation::cuda_activation( size_t size,
                                  size_t batch_size,
                                  Activation_name function_name,
                                  BackwardClass *pred){
  this->size = size;
  this->batch_size = batch_size;
  this->d_values = nullptr;
  this->d_grad = nullptr;
  this->function_name = function_name;

  const size_t elements = this->size * this->batch_size;
  allocate_device_memory_zeros<float>(&this->d_values, elements);
  allocate_device_memory_zeros<float>(&this->d_grad, elements);

  this->pred = pred;
  this->check_pred();
}

// Destructor
cuda_activation::~cuda_activation(){
  free_device_memory(this->d_values);
  free_device_memory(this->d_grad);
}

/* GETTERS */
// Get activation function
Activation_name cuda_activation::get_activation_fun(){
  return this->function_name;
}

// Get the values pointer
float *cuda_activation::values_pointer(){
  return this->d_values;
}

// Get the gradient pointer
float *cuda_activation::grad_pointer(){
  return this->d_grad;
}

size_t cuda_activation::get_output_size(){
  return this->size;
}

size_t cuda_activation::get_batch_size(){
  return this->batch_size;
}

/* SETTERS */
// Set pred pointer
void cuda_activation::set_pred(BackwardClass *pred){
  this->pred = pred;
  this->check_pred();
}

/* METHODS */
// Forward pass
void cuda_activation::operator()(){
  if(this->pred == nullptr){
    throw invalid_argument("cuda_activation::operator() pred is null");
  }
  this->check_pred();
  const size_t elements = this->batch_size * this->size;
  if(this->function_name == TANH){
    launch_activation_tanh(this->d_values, elements);
  }
  else if(this->function_name == RELU){
    launch_activation_relu(this->d_values, elements);
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
  zero_device_memory(this->d_grad, this->size * this->batch_size);
}

// Backward pass
void cuda_activation::backward(){
  if(this->pred == nullptr){
    throw invalid_argument("cuda_activation::backward pred is null");
  }
  this->check_pred();
  const size_t elements = this->size * this->batch_size;
  if(this->function_name == TANH){
    launch_backward_tanh(this->d_values, this->d_grad, elements);
  }
  else if(this->function_name == RELU){
    launch_backward_relu(this->d_values, this->d_grad, elements);
  }
  else if(this->function_name == LINEAR){
    // The incoming gradient already is the linear activation gradient.
  }
  else{
    throw invalid_argument("Invalid activation function");
  }

  this->pred->backward();
}
