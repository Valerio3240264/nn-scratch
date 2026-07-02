#include "../headers/cuda_softmax.cuh"

#include "../cuda_manager.cuh"
#include "../cuda_manager_impl.cuh"
#include <iostream>

using namespace std;

// Check pred component matches dimensions
void cuda_softmax::check_pred(){
  if(this->pred == nullptr){
    return;
  }
  else if(this->pred->get_output_size() != this->size){
    throw invalid_argument("Pred component doesn't matches softmax size");
  }
  else if(this->pred->get_batch_size() != this->batch_size){
    throw invalid_argument("Pred component doesn't matches softmax batch_size");
  }
  return;
}

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor - allocates its own memory
cuda_softmax::cuda_softmax( size_t size,
                            size_t batch_size,
                            float temperature,
                            BackwardClass *pred){
  this->size = size;
  this->batch_size = batch_size;
  this->d_value = nullptr;
  this->d_grad = nullptr;
  this->temperature = temperature;
  this->pred = pred;
  this->check_pred();
  
  const size_t elements = this->size * this->batch_size;
  allocate_device_memory_zeros<float>(&this->d_value, elements);
  allocate_device_memory_zeros<float>(&this->d_grad, elements);
}


cuda_softmax::~cuda_softmax(){
  free_device_memory(this->d_value);
  free_device_memory(this->d_grad);
}

/* GETTERS */
// Get activation function
Activation_name cuda_softmax::get_activation_fun(){
  return SOFTMAX;
}
// Get the values pointer
float *cuda_softmax::values_pointer(){
  return this->d_value;
}

// Get the gradient pointer
float *cuda_softmax::grad_pointer(){
  return this->d_grad;
}

size_t cuda_softmax::get_output_size(){
  return this->size;
}

size_t cuda_softmax::get_batch_size(){
  return this->batch_size;
}

float cuda_softmax::get_temperature(){
  return this->temperature;
}

void cuda_softmax::set_pred(BackwardClass *pred){
  this->pred = pred;
  this->check_pred();
}

void cuda_softmax::get_predictions(size_t *predictions){
  float *h_values = new float[this->size * this->batch_size];
  copy_device_to_host(
      h_values,
      this->d_value,
      this->size * this->batch_size);
  for(size_t row = 0; row < this->batch_size; row++){
    size_t max_idx = 0;
    float *row_values = h_values + row * this->size;
    for(size_t col = 1; col < this->size; col++){
      if(row_values[col] > row_values[max_idx]){
        max_idx = col;
      }
    }
    predictions[row] = max_idx;
  }
  delete[] h_values;
}

/* METHODS */
// Forward pass
void cuda_softmax::operator()(){
  if(this->pred == nullptr){
    throw invalid_argument("cuda_softmax::operator() pred is null");
  }
  this->check_pred();
  launch_softmax_forward( this->pred->values_pointer(),
                          this->d_value,
                          this->temperature,
                          this->size,
                          this->batch_size);
}

// Backward pass
void cuda_softmax::backward(){
  if(this->pred == nullptr){
    throw invalid_argument("cuda_softmax::backward pred is null");
  }
  this->check_pred();
  launch_softmax_backward(
      this->d_value,
      this->d_grad,
      this->pred->grad_pointer(),
      this->temperature,
      this->size,
      this->batch_size);
  this->pred->backward();
}

// Zero the gradient
void cuda_softmax::zero_grad(){
  zero_device_memory(this->d_grad, this->size * this->batch_size);
}
