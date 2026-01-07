#include "../headers/weights.h"
#include "../../enums.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>

using namespace std;

void weights::init_weights(Activation_name function_name){
  float scale;
  if(function_name == TANH){
    // Xavier initialization
    scale = sqrtf(6.0f / (this->input_size + this->output_size));
  }
  else if(function_name == RELU){
    // He initialization
    scale = sqrtf(2.0f / this->input_size);
  }
  else if(function_name == LINEAR){
    // Xavier initialization
    scale = sqrtf(6.0f / (this->input_size + this->output_size));
  }
  else{
    throw invalid_argument("Invalid activation function");
    exit(1);
  }

  default_random_engine generator;
  uniform_real_distribution<float> distribution(-scale, scale);

  for (int i = 0; i < this->input_size * this->output_size; i++){
    this->w[i] = distribution(generator);
    this->grad_w[i] = 0.0f;
  }
  for (int i = 0; i < this->output_size; i++){
    this->b[i] = 0.0f;
    this->grad_b[i] = 0.0f;
  }
}

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
weights::weights(int input_size, int output_size, Activation_name function_name){
  if(input_size <= 0 || output_size <= 0){
    throw invalid_argument("Input and output size must be greater than 0");
    exit(1);
  }

  this->input_size = input_size;
  this->output_size = output_size;
  this->w = new float[input_size * output_size];
  this->grad_w = new float[input_size * output_size];
  this->b = new float[output_size];
  this->grad_b = new float[output_size];
  this->input_values = nullptr;
  this->pred = nullptr;

  init_weights(function_name);
}

// Destructor
weights::~weights(){
  delete[] this->w;
  delete[] this->grad_w;
  delete[] this->b;
  delete[] this->grad_b;
}

/* GETTERS */
// Get the weights pointer
float *weights::values_pointer(){
  return this->w;
}

// Get the gradient pointer
float *weights::grad_pointer(){
  return this->grad_w;
}

// Get the bias pointer
float *weights::bias_pointer(){
  return this->b;
}

// Get the bias gradient pointer
float *weights::grad_bias_pointer(){
  return this->grad_b;
}

/* METHODS */
// Forward pass
// W x Input + b
void weights::operator()(BackwardClass *in, float *output_pointer){
  this->input_values = in->values_pointer();
  this->pred = in;
  for (int row = 0; row < this->output_size; row++){
    output_pointer[row] = this->b[row];
    for(int col = 0; col< this->input_size; col++){
      output_pointer[row] += this->w[row * this->input_size + col] * this->input_values[col];
    }
  }
  return;
}

/* BACKPROPAGATION FUNCTIONS */
// Zero the gradient
void weights::zero_grad(){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    this->grad_w[i] = 0.0f;
  }
  for (int i = 0; i < this->output_size; i++){
    this->grad_b[i] = 0.0f;
  }
}

// Backward pass
void weights::backward(float *derivatives){
  float *prevGrad = new float[this->input_size];
  for(int col = 0; col < this->input_size; col++){
    prevGrad[col] = 0;
    for(int row = 0; row< this->output_size; row++){
      prevGrad[col] += derivatives[row] * this->w[row * this->input_size + col];
      this->grad_w[row * this->input_size + col] += derivatives[row] * this->input_values[col];
    }
  }

  for(int row = 0; row < this->output_size; row++){
    this->grad_b[row] += derivatives[row];
  }
  this->pred->backward(prevGrad);
  delete[] prevGrad;
}

// Update the weights
void weights::update(float learning_rate){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    this->w[i] -= learning_rate * this->grad_w[i];
  }
  for (int i = 0; i < this->output_size; i++){
    this->b[i] -= learning_rate * this->grad_b[i];
  }
}

/* TESTING FUNCTIONS */
// Print the weights
void weights::print_weights(){
  for (int i = 0; i < this->input_size * this->output_size; i++)
  {
    cout << this->w[i] << " ";
  }
  cout << endl;
}

// Print the gradient of the weights
void weights::print_grad_weights(){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    cout << this->grad_w[i] << " ";
  }
  cout << endl;
}

// Print the bias
void weights::print_bias(){
  for (int i = 0; i < this->output_size; i++){
    cout << this->b[i] << " ";
  }
  cout << endl;
}

// Print the gradient of the bias
void weights::print_grad_bias(){
  for (int i = 0; i < this->output_size; i++){
    cout << this->grad_b[i] << " ";
  }
  cout << endl;
}