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
    input *in;
    activation_function *out;
    weights* W;
    double input_size;
    double output_size;
    Activation_name function_name;

  public:
    layer(double input_size, double output_size, Activation_name activation_function);
    ~layer();

    // Methods
    void operator()(input *in);

    // Backpropagation functions
    void zero_grad();
    void update(double learning_rate);

    // Getters
    input *get_output();

    // Print functions
    void print_weights();
    void print_grad_weights();
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
layer::layer(double input_size, double output_size, Activation_name function_name){
  this->input_size = input_size;
  this->output_size = output_size;
  this->W = new weights(input_size, output_size);
  this->out = nullptr;
  this->in = nullptr;
  this->function_name = function_name;
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
void layer::operator()(input *in){
  this->in = in;
  double *weights_output = new double[this->output_size];
  weights_output = (*this->W)(in);
  this->out = new activation_function(this->output_size, weights_output, this->function_name, this->W);
  this->out->operator()();
}

// BACKPROPAGATION FUNCTIONS
void layer::zero_grad(){
  this->W->zero_grad();
  this->out->zero_grad();
  this->in->zero_grad();
}

void layer::update(double learning_rate){
  this->W->update(learning_rate);
}

/* GETTERS */
input *layer::get_output(){
  return new input(this->output_size, this->out);
}

/* PRINT FUNCTIONS */
void layer::print_weights(){
  this->W->print_weights();
}

void layer::print_grad_weights(){
  this->W->print_grad_weights();
}

#endif