#ifndef CUDA_SOFTMAX_CUH
#define CUDA_SOFTMAX_CUH
#include "cuda_manager.cuh"
#include "../virtual_classes.h"
#include <iostream>

/*
CUDA SOFTMAX CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the softmax class but it is used to store the values and gradients in device memory.

Note: When using this class, we assume that each class that interacts with this class (in the raw layer) has memory allocated in device memory.
*/

using namespace std;

class cuda_softmax: public SoftmaxClass{
  private:
    float *d_value;
    float *d_grad;
    float *d_max;        // Persistent buffer for max value in forward pass
    float *d_exp_sum;    // Persistent buffer for exp sum in forward pass
    float *d_dot;        // Persistent buffer for dot product in backward pass
    int size;
    float temperature;
    BackwardClass *pred;
  public:

    // Constructor
    cuda_softmax(int size, BackwardClass *pred) ;
    cuda_softmax(int size, float temperature, BackwardClass *pred) ;
    cuda_softmax(int size, float *value, BackwardClass *pred) ;
    cuda_softmax(int size, float *value, float temperature, BackwardClass *pred) ;
    
    // Destructor
    ~cuda_softmax() override;
    
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    int get_prediction() override;
    float get_prediction_probability(int index) override;
   
    // Setters
    void set_value(float *value) override;
    void copy_values(float *value) override;

    // Methods
    void backward(float *derivatives) override;
    void zero_grad() override;
    void operator()() override;
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor - allocates its own memory
cuda_softmax::cuda_softmax(int size, BackwardClass *pred){
  this->size = size;
  this->d_value = nullptr;
  this->d_grad = nullptr;
  this->d_max = nullptr;
  this->d_exp_sum = nullptr;
  this->d_dot = nullptr;
  this->temperature = 1.0f;
  this->pred = pred;
  
  allocate_device_memory_zeros<float>(&this->d_value, size);
  allocate_device_memory_zeros<float>(&this->d_grad, size);
  allocate_device_memory<float>(&this->d_max, 1);
  allocate_device_memory_zeros<float>(&this->d_exp_sum, 1);
  allocate_device_memory_zeros<float>(&this->d_dot, 1);
}

// Constructor - allocates its own memory and sets the temperature
cuda_softmax::cuda_softmax(int size, float temperature, BackwardClass *pred){
  this->size = size;
  this->d_value = nullptr;
  this->d_grad = nullptr;
  this->d_max = nullptr;
  this->d_exp_sum = nullptr;
  this->d_dot = nullptr;
  this->temperature = temperature;
  this->pred = pred;
  
  allocate_device_memory_zeros<float>(&this->d_value, size);
  allocate_device_memory_zeros<float>(&this->d_grad, size);
  allocate_device_memory<float>(&this->d_max, 1);
  allocate_device_memory_zeros<float>(&this->d_exp_sum, 1);
  allocate_device_memory_zeros<float>(&this->d_dot, 1);
}

// Constructor - allocates its own memory and copies values from an external array
cuda_softmax::cuda_softmax(int size, float *value, BackwardClass *pred){
  this->size = size;
  this->d_value = value;
  this->d_grad = nullptr;
  this->d_max = nullptr;
  this->d_exp_sum = nullptr;
  this->d_dot = nullptr;
  this->temperature = 1.0f;
  this->pred = pred;
  
  allocate_device_memory_zeros<float>(&this->d_grad, size);
  allocate_device_memory<float>(&this->d_max, 1);
  allocate_device_memory_zeros<float>(&this->d_exp_sum, 1);
  allocate_device_memory_zeros<float>(&this->d_dot, 1);
}

// Constructor - allocates its own memory and copies values from an external array
cuda_softmax::cuda_softmax(int size, float *value, float temperature, BackwardClass *pred){
  this->size = size;
  this->d_value = value;
  this->d_grad = nullptr;
  this->d_max = nullptr;
  this->d_exp_sum = nullptr;
  this->d_dot = nullptr;
  this->temperature = temperature;
  this->pred = pred;
  
  allocate_device_memory_zeros<float>(&this->d_grad, size);
  allocate_device_memory<float>(&this->d_max, 1);
  allocate_device_memory_zeros<float>(&this->d_exp_sum, 1);
  allocate_device_memory_zeros<float>(&this->d_dot, 1);
}


cuda_softmax::~cuda_softmax(){
  free_device_memory(this->d_value);
  free_device_memory(this->d_grad);
  free_device_memory(this->d_max);
  free_device_memory(this->d_exp_sum);
  free_device_memory(this->d_dot);
}

/* GETTERS */
// Get the values pointer
float *cuda_softmax::values_pointer(){
  return this->d_value;
}

// Get the gradient pointer
float *cuda_softmax::grad_pointer(){
  return this->d_grad;
}

// Get the prediction
int cuda_softmax::get_prediction(){
  float *h_values = new float[this->size];
  copy_device_to_host(h_values, this->d_value, this->size);
  
  int max_idx = 0;
  float max_val = h_values[0];
  for(int i = 1; i < this->size; i++){
    if(h_values[i] > max_val){
      max_val = h_values[i];
      max_idx = i;
    }
  }
  
  delete[] h_values;
  return max_idx;
}

float cuda_softmax::get_prediction_probability(int index){
  if(index < 0 || index >= this->size){
    return 0.0f;
  }
  
  float h_value;
  copy_device_to_host(&h_value, &this->d_value[index], 1);
  return h_value;
}

/* SETTERS */
// Set the value
void cuda_softmax::set_value(float *value){
  if(this->d_value != nullptr){
  free_device_memory(this->d_value);
  }
  this->d_value = value;
}

// Copy values
void cuda_softmax::copy_values(float *value){
  copy_device_to_device(this->d_value, value, this->size);
}

/* METHODS */
// Backward pass
void cuda_softmax::backward(float *derivatives){
  launch_softmax_backward(this->d_value, derivatives, this->d_grad, this->temperature, this->size, this->d_dot);
  
  if(this->pred){
    this->pred->backward(this->d_grad);
  }
}

// Zero the gradient
void cuda_softmax::zero_grad(){
  zero_device_memory(this->d_grad, this->size);
}

// Operator to apply the softmax function
void cuda_softmax::operator()(){
  launch_softmax_forward(this->d_value, this->temperature, this->size, this->d_max, this->d_exp_sum);
}
#endif