#include "../headers/weights.h"
#include "../../enums.h"
#include "../../../utils/MatricesOp.h"

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
// x*W + b
void weights::operator()(BackwardClass *in, float *output_pointer){
  this->input_values = in->values_pointer();
  this->pred = in;

  // X*W + b
  // X dimension: input_size (no batch dimension)
  MultiplyAndAdd(this->input_values, this->w, this->b, output_pointer, 1, this->input_size, this->output_size);

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

  // Weights gradient:
  // dL/dW = X^T * dL/dY
  // X dimension: 1 x input_size (no batch dimension)
  InPlaceMultiplyAndAdd(this->input_values, derivatives, this->grad_w, this->input_size, 1, this->output_size);
  
  // Input gradient:
  // dL/dX = dL/dY * W^T
  // W dimension: output_size x input_size
  // dL/dY dimension: output_size (no batch dimension)
  Multiply_Transposed(derivatives, this->w, prevGrad, 1, this->output_size, this->input_size);

  // Bias gradient:
  // dL/db = dL/dY
  // dL/dY dimension: output_size (no batch dimension)
  InPlaceMatrix_Add(this->grad_b, derivatives, this->output_size, 1);

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
  for (int i = 0; i < this->input_size * this->output_size; i++){
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