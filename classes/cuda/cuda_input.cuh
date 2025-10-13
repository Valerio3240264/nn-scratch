#ifndef CUDA_INPUT_CUH
#define CUDA_INPUT_CUH
#include "cuda_manager.cuh"
#include "virtual_classes.h"
#include <iostream>

/*
CUDA INPUT CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the input class but it is used to store the values and gradients in device memory.
*/

using namespace std;

class cuda_input: public BackwardClass{
  private:
    double *d_value;
    double *d_grad;
    int size;
    BackwardClass *pred;

  public:
  // Constructors
    cuda_input(int size);
    cuda_input(int size, double *value);
    cuda_input(int size, BackwardClass *pred);
  // Destructor
    ~cuda_input();
  // Getters
    double *values_pointer() override;
    double *grad_pointer() override;
  // Methods
    void zero_grad() override;
    void backward(double *derivatives) override;
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
cuda_input::cuda_input(int size){
  this->size = size;
  this->d_value = nullptr;
  this->d_grad = nullptr;
  this->pred = nullptr;

  cuda_manager::allocate_device_memory_random(this->d_value, size);
  cuda_manager::allocate_device_memory_zeros(this->d_grad, size);
}

// Constructor for input with value
cuda_input::cuda_input(int size, double *value){
  this->size = size;
  this->d_value = nullptr;
  this->d_grad = nullptr;
  this->pred = nullptr;

  cuda_manager::allocate_device_memory(this->d_value, size);
  cuda_manager::copy_host_to_device(this->d_value, value, size);
  cuda_manager::allocate_device_memory_zeros(this->d_grad, size);
}

// Constructor for input with predecessor
cuda_input::cuda_input(int size, BackwardClass *pred){
  this->size = size;
  this->d_value = pred->values_pointer(); // We assume that the predecessor is already in device memory
  this->d_grad = nullptr;
  this->pred = pred;

  cuda_manager::allocate_device_memory_zeros(this->d_grad, size);
}

// Destructor
cuda_input::~cuda_input(){
  if(this->pred == nullptr){
    cuda_manager::free_device_memory(this->d_value);
  }
  cuda_manager::free_device_memory(this->d_grad);
}

/* GETTERS */
// Get the value pointer
double *cuda_input::values_pointer(){
  return this->d_value;
}

// Get the gradient pointer
double *cuda_input::grad_pointer(){
  return this->d_grad;
}

/* METHODS */
// Zero the gradient
void cuda_input::zero_grad(){
  cuda_manager::zero_device_memory(this->d_grad, this->size);
}

// Backward pass
void cuda_input::backward(double *derivatives){
  /*TODO*/
}
#endif