#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "input.h"
#include "activation_function.h"
#include "loss.h" 
#include "enums.h"

/*TODO
1: Create functions to evaluate the output and gradient of a whole batch.
2: Optimize the batch operations using personalized cuda kernels.
*/

/*
MLP CLASS DOCUMENTATION:
PURPOSE:
This class is used to store the layers, input size, output size and activation function name of a multi-layer perceptron.

Architecture:
layer_0 -> layer_1 -> ... -> layer_n-1 -> layer_n

Attributes:
- layers: pointer to the layers (Layer class)
- num_layers: number of layers
- input_size: size of the input
- output_size: size of the output
- function_name: name of the activation function

Constructors:
- mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, Activation_name activation_function): creates a new mlp with the passed input size, output size, number of layers and hidden sizes.
- ~mlp(): destructor to delete the layers.

Methods:
- operator()(input *in): evaluates the output of the mlp.
- compute_loss(input *target): computes the loss of the mlp. 
  It also computes the gradients of the whole neural network calling the backward pass on the loss that will be linked with the last layer and so on until the input layer.
- update(double learning_rate): updates the weights using the computed gradients.
- zero_grad(): sets all the gradients to 0.
- print_weights(): prints the weights.
- print_grad_weights(): prints the gradients of the weights.

*/

using namespace std;

class mlp{
  private:
    layer **layers;  // Change to pointer to pointers
    int num_layers;
    int input_size;
    int output_size;
    Activation_name function_name;

  public:
    mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, Activation_name activation_function);
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
mlp::mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, Activation_name function_name){
  this->input_size = input_size;
  this->output_size = output_size;
  this->num_layers = num_layers;
  this->function_name = function_name;
  this->layers = new layer*[num_layers]; 

  if(num_layers > 1){
    this->layers[0] = new layer(input_size, hidden_sizes[0], function_name);
    for(int i = 1; i < num_layers - 1; i++){
      cout<<"Creating layer: "<<i<<endl;
      this->layers[i] = new layer(hidden_sizes[i-1], hidden_sizes[i], function_name);
      cout<<"Layer "<<i<<" created"<<endl;
    }
    this->layers[num_layers - 1] = new layer(hidden_sizes[num_layers - 2], output_size, function_name);
  }
  else if(num_layers == 1){
    this->layers[0] = new layer(input_size, output_size, function_name);
  }
  else{
    cout<<"Error: num_layers must be greater than 0"<<endl;
    exit(1);
  }
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