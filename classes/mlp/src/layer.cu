#include "../headers/layer.h"

#include <iostream>

#include "../../cpu/headers/weights.h"
#include "../../cpu/headers/input.h"
#include "../../cpu/headers/activation.h"

#ifdef __CUDACC__
#include "../../cuda/cuda_manager.cuh"
#include "../../cuda/cuda_manager_impl.cuh"
#include "../../cuda/headers/cuda_weights.cuh"
#include "../../cuda/headers/cuda_input.cuh"
#include "../../cuda/headers/cuda_activation.cuh"
#endif

using namespace std;

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
layer::layer(int input_size, int output_size, Activation_name function_name, bool use_cuda){
  if(use_cuda){
  #ifdef __CUDACC__
    this->input_size = input_size;
    this->output_size = output_size;
    this->W = new cuda_weights(input_size, output_size);
    this->function_name = function_name;
    this->in = nullptr;
    float *output_buffer;
    allocate_device_memory<float>(&output_buffer, output_size);
    this->out = new cuda_activation(output_size, output_buffer, function_name, this->W);
    this->use_cuda = use_cuda;
  #else
    cout<<"Error: CUDA not available. Compile with nvcc."<<endl;
    exit(1);
  #endif
  }
  else{
    this->input_size = input_size;
    this->output_size = output_size;
    this->W = new weights(input_size, output_size);
    this->function_name = function_name;
    this->in = nullptr;
    float *output_buffer = new float[output_size];
    this->out = new activation(output_size, output_buffer, function_name, this->W);
    this->use_cuda = use_cuda;
  }
}

// Destructor
layer::~layer(){
  delete this->W;
  if(this->out != nullptr){
    delete this->out;
  }
}

/* METHODS */
// Operator to evaluate the output
void layer::operator()(BackwardClass *in){
  this->in = in;
  float *out_buffer = this->out->values_pointer();
  (*this->W)(in, out_buffer);
  this->out->operator()();
}

// BACKPROPAGATION FUNCTIONS
void layer::zero_grad(){
  this->W->zero_grad();
  if(this->out != nullptr){
    this->out->zero_grad();
  }
}

void layer::update(float learning_rate){
  this->W->update(learning_rate);
}

/* GETTERS */
BackwardClass *layer::get_output(){
  return this->out;
}

/* PRINT FUNCTIONS */
void layer::print_weights(){
  this->W->print_weights();
}

void layer::print_grad_weights(){
  this->W->print_grad_weights();
}
