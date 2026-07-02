#include "../headers/cuda_mse_loss.cuh"

#include "../cuda_manager.cuh"
#include "../cuda_manager_impl.cuh"
#include <iostream>

using namespace std;

/* CONSTRUCTORS */
// Constructor - sets the predecessor pointer without target
cuda_mse_loss::cuda_mse_loss(size_t size, size_t batch_size, BackwardClass *pred) {
  this->pred = pred;
  this->size = size;
  this->target = nullptr;
  this->loss_value = 0.0f;
  this->batch_size = batch_size;

  this->has_target = false;
  this->owns_target = true;
  allocate_device_memory<float>(&this->target, size * batch_size);
  allocate_device_memory_zeros<float>(&this->d_loss_sum, 1);
}

// Constructor - sets the predecessor pointer and target
cuda_mse_loss::cuda_mse_loss(size_t size, size_t batch_size, BackwardClass *pred, float *target) {
  this->pred = pred;
  this->size = size;
  this->target = target;
  this->loss_value = 0.0f;
  this->batch_size = batch_size;

  this->has_target = true;
  this->owns_target = false;
  allocate_device_memory_zeros<float>(&this->d_loss_sum, 1);
}

/* DESTRUCTOR */
cuda_mse_loss::~cuda_mse_loss() {
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

// Get the loss value
float cuda_mse_loss::get_loss() {
  return this->loss_value;
}

/* METHODS */
// Forward pass with target array
void cuda_mse_loss::operator()(float *target) {
  if(target == nullptr){
    throw std::invalid_argument("cuda_mse_loss::operator() null input");
  }
  if(this->pred == nullptr){
    throw std::invalid_argument("cuda_mse_loss::operator() pred is null");
  }
  copy_host_to_device<float>(this->target, target, this->size * this->batch_size);
  this->has_target = true;
  this->loss_value = 0.0f;
  for(size_t row = 0; row < this->batch_size; row++){
    zero_device_memory(this->d_loss_sum, 1);
    size_t row_offset = row * this->size;
    launch_mse_loss_kernel(
        this->pred->values_pointer() + row_offset,
        this->target + row_offset,
        this->d_loss_sum,
        static_cast<int>(this->size));
    float row_loss = 0.0f;
    copy_device_to_host<float>(&row_loss, this->d_loss_sum, 1);
    this->loss_value += row_loss;
  }
  this->loss_value /= static_cast<float>(this->size);
}

// Forward pass with target indices
void cuda_mse_loss::operator()(size_t* target_indices) {
  if(target_indices == nullptr){
    throw std::invalid_argument("Target indices pointer is null");
  }
  if(this->pred == nullptr){
    throw std::invalid_argument("cuda_mse_loss::operator() pred is null");
  }
  float *h_target = new float[this->size * this->batch_size];
  for(size_t row = 0; row < this->batch_size; row++){
    int target_index = static_cast<int>(target_indices[row]);
    if(target_index < 0 || target_index >= static_cast<int>(this->size)) {
      delete[] h_target;
      throw std::invalid_argument("Target index is out of bounds");
    }
    for(size_t col = 0; col < this->size; col++){
      h_target[row * this->size + col] = (static_cast<int>(col) == target_index) ? 1.0f : 0.0f;
    }
  }
  copy_host_to_device<float>(this->target, h_target, this->size * this->batch_size);
  delete[] h_target;

  this->has_target = true;
  this->loss_value = 0.0f;
  for(size_t row = 0; row < this->batch_size; row++){
    zero_device_memory(this->d_loss_sum, 1);
    size_t row_offset = row * this->size;
    launch_mse_loss_kernel(
        this->pred->values_pointer() + row_offset,
        this->target + row_offset,
        this->d_loss_sum,
        static_cast<int>(this->size));
    float row_loss = 0.0f;
    copy_device_to_host<float>(&row_loss, this->d_loss_sum, 1);
    this->loss_value += row_loss;
  }
  this->loss_value /= static_cast<float>(this->size);
}

// Backward pass for L = sum((prediction - target)^2) / output_size
void cuda_mse_loss::backward(){
  if(!this->has_target) {
    throw std::invalid_argument("No target set for backward pass");
  }
  if(this->pred == nullptr){
    throw std::invalid_argument("cuda_mse_loss::backward pred is null");
  }
  for(size_t row = 0; row < this->batch_size; row++){
    size_t row_offset = row * this->size;
    launch_backward_mse_loss_kernel(
        this->pred->values_pointer() + row_offset,
        this->target + row_offset,
        this->pred->grad_pointer() + row_offset,
        static_cast<int>(this->size));
  }
  this->pred->backward();
}
