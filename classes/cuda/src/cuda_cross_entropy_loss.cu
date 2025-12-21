#include "../headers/cuda_cross_entropy_loss.cuh"

#include <iostream>
#include "../cuda_manager.cuh"
#include "../cuda_manager_impl.cuh"

using namespace std;

/* CONSTRUCTORS */
// Constructor - sets the predecessor pointer without target
cuda_cross_entropy_loss::cuda_cross_entropy_loss(BackwardClass *pred, int size) {
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
cuda_cross_entropy_loss::cuda_cross_entropy_loss(BackwardClass *pred, int size, float *target) {
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
cuda_cross_entropy_loss::~cuda_cross_entropy_loss() {
  free_device_memory(this->grad);
  if (this->owns_target) {
    free_device_memory(this->target);
  }
  free_device_memory(this->d_loss_sum);
}

/* GETTERS */
// Get the values pointer of the current predecessor
float *cuda_cross_entropy_loss::values_pointer() {
  return this->pred->values_pointer();
}

// Get the gradient pointer
float *cuda_cross_entropy_loss::grad_pointer() {
  return this->grad;
}

// Get the loss value
float cuda_cross_entropy_loss::get_loss() {
  return this->loss_value;
}

/* METHODS */
// Forward pass with target array
void cuda_cross_entropy_loss::operator()(float *target) {
  copy_host_to_device(this->target, target, this->size);
  this->has_target = true;
  this->operator()();
}

// Forward pass with target index
void cuda_cross_entropy_loss::operator()(int target_index) {
  if(target_index < 0 || target_index >= this->size) {
    throw std::invalid_argument("Target index is out of bounds");
  }
  
  // Convert target_index to one-hot encoding directly on device (no host allocation!)
  launch_one_hot_encoding(this->target, target_index, this->size);

  this->has_target = true;
  this->operator()();
}

// Forward pass with stored target
void cuda_cross_entropy_loss::operator()(){
  if(!this->has_target) {
    throw std::invalid_argument("No target set for forward pass");
  }

  zero_device_memory(this->d_loss_sum, 1);
  launch_softmax_cross_entropy_loss_kernel(this->pred->values_pointer(), this->target, this->grad, this->d_loss_sum, this->size);
  copy_device_to_host<float>(&this->loss_value, this->d_loss_sum, 1);
}

// Zero the gradient
void cuda_cross_entropy_loss::zero_grad() {
  zero_device_memory(this->grad, this->size);
}

// Backward pass with incoming derivatives (assumes that the derivatives are already in device memory)
void cuda_cross_entropy_loss::backward(float *derivatives) {
  if(!this->has_target) {
    throw std::invalid_argument("No target set for backward pass");
  }
  
  launch_backward_cross_entropy_loss_kernel(this->pred->values_pointer(), this->target, derivatives, this->grad, this->size);  
  
  this->pred->backward(this->grad);
}

// Backward pass with simplified gradient (assumes derivative of loss w.r.t. itself is 1)
void cuda_cross_entropy_loss::backward(){
  if(!this->has_target) {
    throw std::invalid_argument("No target set for backward pass");
  }
  launch_backward_cross_entropy_loss_kernel_simple(this->pred->values_pointer(), this->target, this->grad, this->size);
  
  this->pred->backward(this->grad);
}