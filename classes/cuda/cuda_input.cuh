#ifndef CUDA_INPUT_CUH
#define CUDA_INPUT_CUH
#include "cuda_manager.cuh"
#include "../virtual_classes.h"
#include <iostream>

/*
CUDA INPUT CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the input class but it is used to store the values and gradients in device memory.

Note: When using this class, we assume that each class that interacts with this class (in the raw layer) has memory allocated in device memory.
*/

using namespace std;

class cuda_input: public BackwardClass{
  private:
    float *d_value;
    float *d_grad;
    int size;
    BackwardClass *pred;

  public:

    // Constructors
    cuda_input(int size);
    cuda_input(int size, float *value);
    cuda_input(int size, BackwardClass *pred);

    // Destructor
    ~cuda_input();

    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
  
    // Methods
    void zero_grad() override;
    void backward(float *derivatives) override;
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor - allocates its own memory
cuda_input::cuda_input(int size){
  this->size = size;
  this->d_value = nullptr;
  this->d_grad = nullptr;
  this->pred = nullptr;

  allocate_device_memory_random<float>(&this->d_value, size);
  allocate_device_memory_zeros<float>(&this->d_grad, size);
}

// Constructor - allocates its own memory and copies values from an external array
cuda_input::cuda_input(int size, float *value){
  this->size = size;
  this->d_value = nullptr;
  this->d_grad = nullptr;
  this->pred = nullptr;

  allocate_device_memory<float>(&this->d_value, size);
  copy_host_to_device<float>(this->d_value, value, size);
  allocate_device_memory_zeros<float>(&this->d_grad, size);
}

// Constructor - sets the predecessor pointer
cuda_input::cuda_input(int size, BackwardClass *pred){
  // We assume that the predecessor values are already in device memory
  this->size = size;
  this->d_value = pred->values_pointer();
  this->d_grad = nullptr;
  this->pred = pred;

  allocate_device_memory_zeros<float>(&this->d_grad, size);
}

// Destructor
cuda_input::~cuda_input(){
  if(this->pred == nullptr){
    free_device_memory(this->d_value);
  }
  free_device_memory(this->d_grad);
}

/* GETTERS */
// Get the value pointer
float *cuda_input::values_pointer(){
  return this->d_value;
}

// Get the gradient pointer
float *cuda_input::grad_pointer(){
  return this->d_grad;
}

/* METHODS */
// Zero the gradient
void cuda_input::zero_grad(){
  zero_device_memory(this->d_grad, this->size);
}

// Backward pass
void cuda_input::backward(float *derivatives){
  if(this->pred != nullptr){
    this->pred->backward(derivatives);
  }
}

#endif