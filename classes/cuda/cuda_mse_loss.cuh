#ifndef CUDA_MSE_LOSS_H
#define CUDA_MSE_LOSS_H

#include "../virtual_classes.h"
#include "cuda_manager.cuh"

/*
MSE LOSS CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the mse_loss class but it is used to store the values and gradients in device memory.

Note: When using this class, we assume that each class that interacts with this class (in the raw layer) has memory allocated in device memory.
*/

using namespace std;

class cuda_mse_loss : public LossClass {
  private:
    BackwardClass *pred;
    float *target;
    float *grad;
    float loss_value;
    float *d_loss_sum;
    int size;
    bool has_target;  // Track if we have a target
    bool owns_target; // Track if we own the target memory

  public:
    // Constructors
    cuda_mse_loss(BackwardClass *pred, int size);
    cuda_mse_loss(BackwardClass *pred, int size, float *target);

    // Destructor
    ~cuda_mse_loss();

    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float get_loss() override;

    // Methods
    void operator()(float *target) override;
    void operator()(int target_index) override;
    void operator()() override;
    void zero_grad() override;
    void backward(float *derivatives) override;
    void backward() override;
} ;

/* CONSTRUCTORS */
// Constructor - sets the predecessor pointer without target
cuda_mse_loss::cuda_mse_loss(BackwardClass *pred, int size) {
  this->pred = pred;
  this->size = size;
  this->grad = nullptr;
  this->target = nullptr;
  this->loss_value = 0.0f;
  
  this->has_target = false;
  this->owns_target = true;
  allocate_device_memory<float>(&this->target, size);
  allocate_device_memory_zeros<float>(&this->grad, size);
  allocate_device_memory_zeros<float>(&this->d_loss_sum, 1);
}

// Constructor - sets the predecessor pointer and target
cuda_mse_loss::cuda_mse_loss(BackwardClass *pred, int size, float *target) {
  this->pred = pred;
  this->size = size;
  this->grad = nullptr;
  this->target = target;
  this->loss_value = 0.0f;

  this->has_target = true;
  this->owns_target = false;  
  allocate_device_memory_zeros<float>(&this->grad, size);
  allocate_device_memory_zeros<float>(&this->d_loss_sum, 1);
}

/* DESTRUCTOR */
cuda_mse_loss::~cuda_mse_loss() {
  free_device_memory(this->grad);
  if (this->owns_target) {
    free_device_memory(this->target);
  }
  free_device_memory(this->d_loss_sum);
}

/* GETTERS */
// Get the values pointer of the current predecessor
float *cuda_mse_loss::values_pointer() {
  return this->pred->values_pointer();
}

// Get the gradient pointer
float *cuda_mse_loss::grad_pointer() {
  return this->grad;
}

// Get the loss value
float cuda_mse_loss::get_loss() {
  return this->loss_value;
}

/* METHODS */
// Forward pass with target array
void cuda_mse_loss::operator()(float *target) {
  copy_host_to_device<float>(this->target, target, this->size);
  this->has_target = true;
  this->operator()();
}

// Forward pass with target index
void cuda_mse_loss::operator()(int target_index) {
  if(target_index < 0 || target_index >= this->size) {
    throw std::invalid_argument("Target index is out of bounds");
  }
  
  launch_one_hot_encoding(this->target, target_index, this->size);

  this->has_target = true;
  this->operator()();
}

// Forward pass with stored target
void cuda_mse_loss::operator()(){
  if(!this->has_target) {
    throw std::invalid_argument("No target set for forward pass");
  }
  
  zero_device_memory(this->d_loss_sum, 1);
  launch_mse_loss_kernel(this->pred->values_pointer(), this->target, this->grad, this->d_loss_sum, this->size);
  copy_device_to_host<float>(&this->loss_value, this->d_loss_sum, 1);
  this->loss_value /= this->size;
}

// Zero the gradient
void cuda_mse_loss::zero_grad() {
  zero_device_memory(this->grad, this->size);
}

// Backward pass with incoming derivatives (assumes that the derivatives are already in device memory)
void cuda_mse_loss::backward(float *derivatives) {
  if(!this->has_target) {
    throw std::invalid_argument("No target set for backward pass");
  }
  
  launch_backward_mse_loss_kernel(this->pred->values_pointer(), this->target, derivatives, this->grad, this->size);
  this->pred->backward(this->grad);
}

// Backward pass with simplified gradient (assumes derivative of loss w.r.t. itself is 1)
void cuda_mse_loss::backward(){
  if(!this->has_target) {
    throw std::invalid_argument("No target set for backward pass");
  }
  
  launch_backward_mse_loss_kernel_simple(this->pred->values_pointer(), this->target, this->grad, this->size);
  this->pred->backward(this->grad);
}
#endif