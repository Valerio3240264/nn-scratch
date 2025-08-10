#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include "weights.h"
#include "input.h"
#include "activation_function.h"

using namespace std;

class layer
{
  private:
    input *in;
    activation_function *out;
    weights* W;
    double input_size;
    double output_size;

  public:
    layer(double input_size, double output_size);
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
layer::layer(double input_size, double output_size){
  this->input_size = input_size;
  this->output_size = output_size;
  this->W = new weights(input_size, output_size);
  this->out = nullptr;
  this->in = nullptr;
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
  double *weights_output = (*this->W)(in);
  this->out = new activation_function(this->output_size, weights_output, TANH, this->W);
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