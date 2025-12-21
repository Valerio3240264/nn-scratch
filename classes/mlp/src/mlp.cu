#include "../headers/mlp.h"

#include <iostream>
#include "../headers/layer.h"
#include "../../cpu/headers/input.h"
#include "../../cpu/headers/activation.h"
#include "../../cpu/headers/softmax.h"
#include "../../cpu/headers/mse_loss.h"
#include "../../cpu/headers/cross_entropy_loss.h"

#ifdef __CUDACC__
#include "../../cuda/headers/cuda_input.cuh"
#include "../../cuda/headers/cuda_softmax.cuh"
#include "../../cuda/headers/cuda_mse_loss.cuh"
#include "../../cuda/headers/cuda_cross_entropy_loss.cuh"
#include "../../cuda/cuda_manager.cuh"
#include "../../cuda/cuda_manager_impl.cuh"
#endif

using namespace std;

/* INITIALIZATION FUNCTIONS */
void mlp::cuda_init(int *hidden_sizes){
#ifdef __CUDACC__
  // Create layers (layer class handles CUDA internally)
  if(this->num_layers > 1){
    this->layers[0] = new layer(this->input_size, hidden_sizes[0], this->activation_functions[0], true);
    for(int i = 1; i < this->num_layers - 1; i++){
      this->layers[i] = new layer(hidden_sizes[i-1], hidden_sizes[i], this->activation_functions[i], true);
    }
    this->layers[this->num_layers - 1] = new layer(hidden_sizes[this->num_layers - 2], this->output_size, this->activation_functions[this->num_layers - 1], true);
  }
  else if(this->num_layers == 1){
    this->layers[0] = new layer(this->input_size, this->output_size, this->activation_functions[0], true);
  }
  else{
    cout<<"Error: num_layers must be greater than 0"<<endl;
    exit(1);
  }
  
  // Create softmax layer if needed
  if(this->has_softmax){
    BackwardClass *last_layer_act = this->layers[this->num_layers - 1]->get_output();
    this->softmax_layer = new cuda_softmax(this->output_size, last_layer_act);
  } else {
    this->softmax_layer = nullptr;
  }
  
  // Determine loss predecessor (softmax or last layer)
  BackwardClass *loss_predecessor = this->has_softmax ? 
    (BackwardClass*)this->softmax_layer : 
    (BackwardClass*)this->layers[this->num_layers - 1]->get_output();
  
  // Create loss layer
  if(this->loss_function == MSE){
    this->loss_layer = new cuda_mse_loss(loss_predecessor, this->output_size);
  } else {
    this->loss_layer = new cuda_cross_entropy_loss(loss_predecessor, this->output_size);
  }
#else
  cout<<"Error: CUDA not available. Compile with nvcc."<<endl;
  exit(1);
#endif
}

void mlp::cpu_init(int *hidden_sizes){
  // Create layers
  if(this->num_layers > 1){
    this->layers[0] = new layer(this->input_size, hidden_sizes[0], this->activation_functions[0], false);
    for(int i = 1; i < this->num_layers - 1; i++){
      this->layers[i] = new layer(hidden_sizes[i-1], hidden_sizes[i], this->activation_functions[i], false);
    }
    this->layers[this->num_layers - 1] = new layer(hidden_sizes[this->num_layers - 2], this->output_size, this->activation_functions[this->num_layers - 1], false);
  }
  else if(this->num_layers == 1){
    this->layers[0] = new layer(this->input_size, this->output_size, this->activation_functions[0], false);
  }
  else{
    cout<<"Error: num_layers must be greater than 0"<<endl;
    exit(1);
  }
  
  if(this->has_softmax){
    BackwardClass *last_layer_act = this->layers[this->num_layers - 1]->get_output();
    this->softmax_layer = new softmax(this->output_size, last_layer_act);
  } else {
    this->softmax_layer = nullptr;
  }
  
  BackwardClass *loss_predecessor = this->has_softmax ? 
    (BackwardClass*)this->softmax_layer : 
    (BackwardClass*)this->layers[this->num_layers - 1]->get_output();
  
  if(this->loss_function == MSE){
    this->loss_layer = new mse_loss(loss_predecessor, this->output_size);
  } else {
    this->loss_layer = new cross_entropy_loss(loss_predecessor, this->output_size);
  }
}

/* CONSTRUCTOR AND DESTRUCTOR */
mlp::mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, 
          Activation_name *activation_functions, Loss_name loss_function, bool use_softmax, bool use_cuda){
  this->input_size = input_size;
  this->output_size = output_size;
  this->num_layers = num_layers;
  this->activation_functions = new Activation_name[num_layers];
  for(int i = 0; i < num_layers; i++){
    this->activation_functions[i] = activation_functions[i];
  }
  this->loss_function = loss_function;
  this->has_softmax = use_softmax;
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

// Legacy constructor (all layers use same activation)
mlp::mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, Activation_name function_name, bool use_cuda){
  this->input_size = input_size;
  this->output_size = output_size;
  this->num_layers = num_layers;
  this->activation_functions = new Activation_name[num_layers];
  for(int i = 0; i < num_layers; i++){
    this->activation_functions[i] = function_name;
  }
  this->loss_function = MSE;
  this->has_softmax = false;
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

  if(this->softmax_layer != nullptr){
    delete this->softmax_layer;
  }
  if(this->loss_layer != nullptr){
    delete this->loss_layer;
  }
}

/* GETTERS */
int mlp::get_prediction(){
  if(this->has_softmax){
    return this->softmax_layer->get_prediction();
  }
  else{
    return -1;
  }
}

float mlp::get_prediction_probability(int index){
  if(this->has_softmax){
    return this->softmax_layer->get_prediction_probability(index);
  }
  else{
    return 0.0f;
  }
}

/* METHODS */
BackwardClass* mlp::operator()(BackwardClass *in){
  BackwardClass *out = in;
  
  // Forward pass through all layers
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->operator()(out);
    out = layers[i]->get_output();
  }

  // Use softmax if needed
  if(this->has_softmax){
    float *last_values = out->values_pointer();
    this->softmax_layer->copy_values(last_values);
    this->softmax_layer->operator()();
    return this->softmax_layer;
  }

  return out;
}

// Compute loss with target array
void mlp::compute_loss(float *target){
  this->loss_layer->operator()(target);
  this->current_loss += this->loss_layer->get_loss();
  this->loss_layer->backward();
}

// Compute loss with target index (for classification)
void mlp::compute_loss(int target_index){
  if(this->loss_function == MSE){
    // Convert target index to one-hot for MSE
    float *target = new float[this->output_size];
    for(int i = 0; i < this->output_size; i++){
      target[i] = (i == target_index) ? 1.0f : 0.0f;
    }
    this->loss_layer->operator()(target);
    this->current_loss += this->loss_layer->get_loss();
    this->loss_layer->backward();
    delete[] target;
  } else if(this->loss_function == CROSS_ENTROPY){
    this->loss_layer->operator()(target_index);
    this->current_loss += this->loss_layer->get_loss();
    this->loss_layer->backward();
  }
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

void mlp::zero_grad(){
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->zero_grad();
  }
}

/* PRINT FUNCTIONS */
void mlp::print_weights(){
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->print_weights();
  }
}

void mlp::print_grad_weights(){
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->print_grad_weights();
  }
}

void mlp::print_loss(){
  cout << "Loss: " << this->current_loss << endl;
}

void mlp::print_last_layer_weights(){
  layers[this->num_layers - 1]->print_weights();
}

void mlp::print_last_layer_grad_weights(){
  std::cout << "=== LAST LAYER GRADIENTS ===" << std::endl;
  layers[this->num_layers - 1]->print_grad_weights();
}
  
void mlp::print_loss_gradients(){
  if(this->use_cuda){
  #ifdef __CUDACC__
    float *d_grad = this->loss_layer->grad_pointer();
    float *h_grad = new float[this->output_size];
    copy_device_to_host(h_grad, d_grad, this->output_size);
    std::cout << "=== LOSS LAYER GRADIENTS ===" << std::endl;
    for(int i = 0; i < this->output_size; i++){
      std::cout << h_grad[i] << " ";
    }
    std::cout << std::endl;
    delete[] h_grad;
  #else
    cout<<"Error: CUDA not available. Compile with nvcc."<<endl;
    exit(1);
  #endif
  }
  else{
    float *grad = this->loss_layer->grad_pointer();
    std::cout << "=== LOSS LAYER GRADIENTS ===" << std::endl;
    for(int i = 0; i < this->output_size; i++){
      std::cout << grad[i] << " ";
    }
    std::cout << std::endl;
  }
}