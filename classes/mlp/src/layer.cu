#include "../headers/layer.h"

#include <iostream>
#include <cstddef>

#include "../../cpu/headers/weights.h"
#include "../../cpu/headers/input.h"
#include "../../cpu/headers/activation.h"
#include "../../virtual_classes.h"
#include "../../enums.h"

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
layer::layer( size_t input_size,
              size_t output_size,
              size_t batch_size,
              Activation_name function_name,
              BackwardClass *input,
              bool use_cuda){
  if(use_cuda){
  #ifdef __CUDACC__
    this->input_size = input_size;
    this->output_size = output_size;
    this->batch_size = batch_size;
    this->W = new cuda_weights(input_size, output_size, batch_size, function_name, input, nullptr);
    this->out = new cuda_activation(output_size, batch_size, function_name, this->W);
    this->W->set_next(this->out);
    this->use_cuda = use_cuda;
  #else
    throw invalid_argument("__CUDACC__ not defined.");
    exit(1);
  #endif
  }
  else{
    this->input_size = input_size;
    this->output_size = output_size;
    this->batch_size = batch_size;
    this->W = new weights(input_size, output_size, batch_size, function_name, input, nullptr);
    this->out = new activation(output_size, batch_size, function_name, this->W);
    this->W->set_next(this->out);
    this->use_cuda = use_cuda;
  }
}

// Destructor
layer::~layer(){
  delete this->W;
  delete this->out;
}

/* METHODS */
// Operator to evaluate the output
void layer::operator()(){
  this->W->operator()();
  this->out->operator()();
}

// BACKPROPAGATION FUNCTIONS
void layer::zero_grad(){
  this->W->zero_grad();
}

void layer::update(float learning_rate){
  this->W->update(learning_rate);
}

/* GETTERS */
BackwardClass *layer::get_output(){
  return this->out;
}

Activation_name layer::get_function(){
  return this->out->get_activation_fun();
}

/* SETTERS */
void layer::set_input(BackwardClass *in){
  this->W->set_pred(in);
}

/* PRINT FUNCTIONS */
void layer::print_weights(){
  this->W->print_weights();
}

void layer::print_grad_weights(){
  this->W->print_grad_weights();
}
