#ifndef CUDA_WEIGHTS_CUH
#define CUDA_WEIGHTS_CUH
#include "cuda_input.cuh"
#include "virtual_classes.h"
#include "cuda_manager.cuh"
#include <iostream>

using namespace std;

/*
CUDA WEIGHTS CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the weights class but it is used to store the weights and gradients in device memory.
*/

class cuda_weights: public BackwardClass{
  private:
    double *d_w;
    double *d_grad_w;
    int input_size;
    int output_size;
    double *d_input_values;
    BackwardClass *pred;

  public:
  // Constructor
    cuda_weights(int input_size, int output_size);
  // Destructor
    ~cuda_weights();
  // Getters
    double *values_pointer() override;
    double *grad_pointer() override;
  // Methods
    void backward(double *derivatives) override;
    void zero_grad() override;
    double *operator()(BackwardClass *in);
    void update(double learning_rate);
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
cuda_weights::cuda_weights(int input_size, int output_size){
  this->input_size = input_size;
  this->output_size = output_size;
  this->d_w = nullptr;
  this->d_grad_w = nullptr;
  this->d_input_values = nullptr;
  this->pred = nullptr;

  cuda_manager::allocate_device_memory_random(this->d_w, input_size * output_size);
  cuda_manager::allocate_device_memory_zeros(this->d_grad_w, input_size * output_size);
}

// Destructor
cuda_weights::~cuda_weights(){
  cuda_manager::free_device_memory(this->d_w);
  cuda_manager::free_device_memory(this->d_grad_w);
}

/* GETTERS */
// Get the values pointer
double *cuda_weights::values_pointer(){
  return this->d_w;
}

// Get the gradient pointer
double *cuda_weights::grad_pointer(){
  return this->d_grad_w;
}

/* METHODS */
// Operator to evaluate the output
// W x Inputt
double *cuda_weights::operator()(BackwardClass *in){
  this->d_input_values = in->values_pointer(); // We assume that the predecessor is already in device memory
  this->pred = in;
  double *output = nullptr;
  cuda_manager::allocate_device_memory(output, this->output_size);
  cuda_manager::launch_matrix_vector_multiply(this->d_w, this->d_input_values, output, this->output_size, this->input_size);
  return output;
}

/* BACKPROPAGATION FUNCTIONS */
// Zero the gradient
void cuda_weights::zero_grad(){
  cuda_manager::zero_device_memory(this->d_grad_w, this->input_size * this->output_size);
}

// Backward pass
void cuda_weights::backward(double *derivatives){
  /*TODO*/
}

// Update the weights
void cuda_weights::update(double learning_rate){
  cuda_manager::launch_update(this->d_w, this->d_grad_w, learning_rate, this->input_size * this->output_size);
}

#endif