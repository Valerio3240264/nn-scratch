#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include "cpu/weights.h"
#include "cpu/input.h"
#include "cpu/activation_function.h"
#include "enums.h"

/*TODO
1: Create functions to evaluate the output and gradient of a whole batch.
*/

/*
LAYER CLASS DOCUMENTATION:
PURPOSE:
This class is used to store the input(Input class), output(Activation_function class), weights(Weights class), input size, output size and activation function name of a layer.

Architecture:
Input -> Weights -> Activation_function -> Output

Attributes:
- in: pointer to the input (Input class)
- out: pointer to the output (Activation_function class)
- W: pointer to the weights (Weights class)
- input_size: size of the input
- output_size: size of the output
- function_name: name of the activation function

Constructors:
- layer(double input_size, double output_size, Activation_name activation_function): creates a new layer with the passed input size, output size and activation function name.

Methods:
- operator()(input *in): evaluates the output of the layer.
- zero_grad(): sets all the gradients to 0.
- update(double learning_rate): updates the weights using the computed gradients.
- get_output(): returns the output of the layer.
- print_weights(): prints the weights.
- print_grad_weights(): prints the gradients of the weights.

*/

using namespace std;

class layer
{
  private:
    BackwardClass *in;
    activation_function *out;
    weights* W;
    int input_size;
    int output_size;
    Activation_name function_name;

  public:
    layer(int input_size, int output_size, Activation_name activation_function);
    ~layer();

    // Methods
    void operator()(BackwardClass *in);

    // Backpropagation functions
    void zero_grad();
    void update(double learning_rate);

    // Getters
    BackwardClass *get_output();

    // Print functions
    void print_weights();
    void print_grad_weights();
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
layer::layer(int input_size, int output_size, Activation_name function_name){
  this->input_size = input_size;
  this->output_size = output_size;
  this->W = new weights(input_size, output_size);
  this->function_name = function_name;
  this->in = nullptr;
  double *output_buffer = new double[output_size];
  this->out = new activation_function(output_size, output_buffer, function_name, this->W);
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
  double *weights_output = (*this->W)(in);
  
  double *out_buffer = this->out->values_pointer();
  for(int i = 0; i < this->output_size; i++){
    out_buffer[i] = weights_output[i];
  }
  
  this->out->operator()();
}

// BACKPROPAGATION FUNCTIONS
void layer::zero_grad(){
  this->W->zero_grad();
  if(this->out != nullptr){
    this->out->zero_grad();
  }
  if(this->in != nullptr){
    this->in->zero_grad();
  }
}

void layer::update(double learning_rate){
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

#endif