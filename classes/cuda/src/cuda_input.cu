#include "../headers/cuda_input.cuh"

#include "../cuda_manager.cuh"
#include "../cuda_manager_impl.cuh"
#include <iostream>

using namespace std;

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor - allocates its own memory
cuda_input::cuda_input(size_t size, size_t batch_size){
  this->size = size;
  this->batch_size = batch_size;
  this->d_values = nullptr;
  this->d_grad = nullptr;

  allocate_device_memory_zeros<float>(
      &this->d_grad,
      this->size * this->batch_size);
}

// Destructor
cuda_input::~cuda_input(){
  free_device_memory(this->d_grad);
}

/* GETTERS */
// Get the value pointer
float *cuda_input::values_pointer(){
  return this->d_values;
}

// Get the gradient pointer
float *cuda_input::grad_pointer(){
  return this->d_grad;
}

// Get size
size_t cuda_input::get_output_size(){
  return this->size;
}

// Get batch size
size_t cuda_input::get_batch_size(){
  return this->batch_size;
}

/* SETTERS */
// Changes values pointer
void cuda_input::set_values(float *new_values){
  this->d_values = new_values;
}

// Zero the gradient
void cuda_input::zero_grad(){
  zero_device_memory(this->d_grad, this->size * this->batch_size);
}

// Backward pass
// Leaf node: no component to propagate
void cuda_input::backward(){
  return;
}
