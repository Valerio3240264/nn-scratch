#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "input.h"
#include "activation_function.h"
#include "loss.h"  // Add this include

using namespace std;

class mlp{
  private:
    layer **layers;  // Change to pointer to pointers
    int num_layers;
    int input_size;
    int output_size;

  public:
    mlp(int input_size, int output_size, int num_layers, int *hidden_sizes);
    ~mlp();

    // Methods
    input* operator()(input *in);  // Change return type
    void compute_loss(input *target);  // Rename to avoid confusion
    
    // Backpropagation functions
    void update(double learning_rate);
    void zero_grad();

    // Print functions
    void print_weights();
    void print_grad_weights();
};

/* CONSTRUCTOR AND DESTRUCTOR */
mlp::mlp(int input_size, int output_size, int num_layers, int *hidden_sizes){
  this->input_size = input_size;
  this->output_size = output_size;
  this->num_layers = num_layers;

  this->layers = new layer*[num_layers];  // Array of pointers

  this->layers[0] = new layer(input_size, hidden_sizes[0]);
  for(int i = 1; i < num_layers - 1; i++){
    this->layers[i] = new layer(hidden_sizes[i-1], hidden_sizes[i]);
  }
  this->layers[num_layers - 1] = new layer(hidden_sizes[num_layers - 2], output_size);
}

mlp::~mlp(){
  for(int i = 0; i < num_layers; i++){
    delete layers[i];
  }
  delete[] layers;
}

/* METHODS */
input* mlp::operator()(input *in){
  input *out = in;
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->operator()(out);
    out = layers[i]->get_output();
  }
  return out;
}

void mlp::compute_loss(input *target){
  input *final_output = layers[this->num_layers - 1]->get_output();
  loss *loss_fn = new loss(final_output, this->output_size, target->values_pointer());
  loss_fn->operator()();
  
  // Start backward pass
  double *ones = new double[this->output_size];
  for(int i = 0; i < this->output_size; i++){
    ones[i] = 1.0;
  }
  loss_fn->backward(ones);
  
  delete[] ones;
  delete loss_fn;
}

/* BACKPROPAGATION FUNCTIONS */
void mlp::update(double learning_rate){
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

#endif