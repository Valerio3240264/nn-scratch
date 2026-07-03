#include "../headers/mlp.h"
#include "../../enums.h"
#include "../../virtual_classes.h"

#include <iostream>
#include <cstddef>
#include "../headers/layer.h"
#include "../../cpu/headers/input.h"
#include "../../cpu/headers/activation.h"
#include "../../cpu/headers/mse_loss.h"
#include "../../cpu/headers/cross_entropy_loss.h"

#ifdef __CUDACC__
#include "../../cuda/headers/cuda_input.cuh"
#include "../../cuda/headers/cuda_mse_loss.cuh"
#include "../../cuda/headers/cuda_cross_entropy_loss.cuh"
#include "../../cuda/cuda_manager.cuh"
#include "../../cuda/cuda_manager_impl.cuh"
#endif

using namespace std;

/* INITIALIZATION FUNCTIONS */
void mlp::cuda_init(size_t *hidden_sizes){
#ifdef __CUDACC__

  BackwardClass *in = nullptr;
  // Create layers (layer class handles CUDA internally)
  if(this->num_layers > 1){
    this->layers[0]
      = new layer(this->input_size,
                  hidden_sizes[0],
                  this->batch_size,
                  this->activation_functions[0],
                  nullptr,
                  true);
    in = this->layers[0]->get_output();

    for(size_t i = 1; i < this->num_layers - 1; i++){
      this->layers[i]
        = new layer(hidden_sizes[i-1], 
                    hidden_sizes[i],
                    this->batch_size,
                    this->activation_functions[i],
                    in,
                    true);
      in = this->layers[i]->get_output();
    }
    this->layers[this->num_layers - 1]
      = new layer(hidden_sizes[this->num_layers - 2],
                  this->output_size,
                  this->batch_size,
                  this->activation_functions[this->num_layers - 1],
                  in,
                  true);
  }
  else if(this->num_layers == 1){
    this->layers[0]
      = new layer(this->input_size,
                  this->output_size,
                  this->batch_size,
                  this->activation_functions[0],
                  nullptr,
                  true);
  }
  else{
    cout<<"Error: num_layers must be greater than 0"<<endl;
    exit(1);
  }

  // Loss predecessor is always the final layer output.
  in = this->layers[this->num_layers - 1]->get_output();
  
  // Create loss layer
  if(this->loss_function == MSE){
    this->loss_layer
      = new cuda_mse_loss(this->output_size,
                          this->batch_size,
                          in);
  } else {
    this->loss_layer
      = new cuda_cross_entropy_loss(this->output_size,
                                    this->batch_size,
                                    in);
  }
#else
  throw invalid_argument("__CUDACC__ not defined.");
  exit(1);
#endif
}

void mlp::cpu_init(size_t *hidden_sizes){
  
  BackwardClass *in = nullptr;
  // Create layers
  if(this->num_layers > 1){
    this->layers[0]
      = new layer(this->input_size,
                  hidden_sizes[0],
                  this->batch_size,
                  this->activation_functions[0],
                  nullptr,
                  false);
    in = this->layers[0]->get_output();

    for(size_t i = 1; i < this->num_layers - 1; i++){
      this->layers[i]
        = new layer(hidden_sizes[i-1],
                    hidden_sizes[i],
                    this->batch_size,
                    this->activation_functions[i],
                    in,
                    false);
      in = this->layers[i]->get_output();
    }
    this->layers[this->num_layers - 1] 
      = new layer(hidden_sizes[this->num_layers - 2],
                  this->output_size,
                  this->batch_size,
                  this->activation_functions[this->num_layers - 1],
                  in,
                  false);
  }
  else if(this->num_layers == 1){
    this->layers[0]
      = new layer(this->input_size,
                  this->output_size,
                  this->batch_size,
                  this->activation_functions[0],
                  nullptr,
                  false);
  }
  else{
    throw invalid_argument("Error: num_layers must be greater than 0");
    exit(1);
  }
  
  in = this->layers[this->num_layers - 1]->get_output();
  
  if(this->loss_function == MSE){
    this->loss_layer
      = new mse_loss( this->output_size,
                      this->batch_size,
                      in);
  } else {
    this->loss_layer
      = new cross_entropy_loss( this->output_size,
                                this->batch_size,
                                in);
  }
}

/* CONSTRUCTOR AND DESTRUCTOR */
mlp::mlp( size_t input_size, 
          size_t output_size,
          size_t batch_size,
          size_t num_layers,
          size_t *hidden_sizes,
          Activation_name *activation_functions,
          Loss_name loss_function,
          bool use_cuda){
  this->input_size = input_size;
  this->output_size = output_size;
  this->batch_size = batch_size;
  this->num_layers = num_layers;
  this->activation_functions = new Activation_name[num_layers];
  for(int i = 0; i < num_layers; i++){
    this->activation_functions[i] = activation_functions[i];
  }
  this->loss_function = loss_function;
  this->current_loss = 0.0f;
  this->layers = new layer*[num_layers];
  this->use_cuda = use_cuda;

  if(use_cuda){
    this->cuda_init(hidden_sizes);
  }
  else{
    this->cpu_init(hidden_sizes);
  }
}

// Legacy constructor (all layers use same activation and loss = MSE)
mlp::mlp( size_t input_size,
          size_t output_size,
          size_t batch_size,
          size_t num_layers,
          size_t *hidden_sizes,
          Activation_name activation_function,
          bool use_cuda){
  this->input_size = input_size;
  this->output_size = output_size;
  this->batch_size = batch_size;
  this->num_layers = num_layers;
  this->activation_functions = new Activation_name[num_layers];
  for(int i = 0; i < num_layers; i++){
    this->activation_functions[i] = activation_function;
  }
  this->loss_function = MSE;
  this->current_loss = 0.0f;
  this->layers = new layer*[num_layers]; 
  this->use_cuda = use_cuda;

  if(use_cuda){
    this->cuda_init(hidden_sizes);
  }
  else{
    this->cpu_init(hidden_sizes);
  }
}

mlp::~mlp(){
  for(int i = 0; i < this->num_layers; i++){
    delete layers[i];
  }
  delete[] layers;
  delete[] activation_functions;

  if(this->loss_layer != nullptr){
    delete this->loss_layer;
  }
}

/* GETTERS */
void mlp::get_predictions(size_t *predictions){
  if(this->activation_functions[this->num_layers - 1] != SOFTMAX){
    throw invalid_argument("Error: can't call get_predictions if final activation is not SOFTMAX.");
    exit(1);
  }

  BackwardClass *output = this->layers[this->num_layers - 1]->get_output();

  if(this->use_cuda){
#ifdef __CUDACC__
    float *host_values = new float[this->output_size * this->batch_size];
    copy_device_to_host(
        host_values,
        output->values_pointer(),
        this->output_size * this->batch_size);

    for(size_t row = 0; row < this->batch_size; row++){
      size_t max_idx = 0;
      float *row_values = host_values + row * this->output_size;
      for(size_t col = 1; col < this->output_size; col++){
        if(row_values[col] > row_values[max_idx]){
          max_idx = col;
        }
      }
      predictions[row] = max_idx;
    }

    delete[] host_values;
#else
    throw invalid_argument("CUDA mode requested but __CUDACC__ is not defined.");
#endif
  }
  else{
    float *values = output->values_pointer();
    for(size_t row = 0; row < this->batch_size; row++){
      size_t max_idx = 0;
      float *row_values = values + row * this->output_size;
      for(size_t col = 1; col < this->output_size; col++){
        if(row_values[col] > row_values[max_idx]){
          max_idx = col;
        }
      }
      predictions[row] = max_idx;
    }
  }
}

/* METHODS */
void mlp::operator()(BackwardClass *in){
  this->layers[0]->set_input(in);

  // Forward pass through all layers
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->operator()();
  }
}

// Compute loss with target array
void mlp::compute_loss(float *target){
  this->loss_layer->operator()(target);
  this->current_loss += this->loss_layer->get_loss();
}

void mlp::compute_loss(size_t *target_indeces){
  this->loss_layer->operator()(target_indeces);
  this->current_loss += this->loss_layer->get_loss();
}

// Get the loss value
float mlp::get_loss(){
  return this->current_loss;
}

// Zero the loss value
void mlp::zero_loss(){
  this->current_loss = 0.0f;
}

/* BACKPROPAGATION FUNCTIONS */
void mlp::update(float learning_rate){
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->update(learning_rate);
  }
}

void mlp::backward(){
  this->loss_layer->backward();
}

void mlp::zero_grad(){
  for(size_t i = 0; i < this->num_layers; i++){
    layers[i]->zero_grad();
  }
}

/* PRINT FUNCTIONS */
void mlp::print_weights(){
  for(size_t i = 0; i < this->num_layers; i++){
    cout<<"Layer "<< i+1<<endl;
    layers[i]->print_weights();
  }
}

void mlp::print_grad_weights(){
  for(size_t i = 0; i < this->num_layers; i++){
    cout<<"Layer "<< i+1<<endl;
    layers[i]->print_grad_weights();
  }
}

void mlp::print_loss(){
  cout << "Loss: " << this->current_loss << endl;
}
