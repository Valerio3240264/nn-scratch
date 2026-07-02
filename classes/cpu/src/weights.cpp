#include "../headers/weights.h"
#include "../../virtual_classes.h"
#include "../../enums.h"
#include "../../../utils/MatricesOp.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>

using namespace std;

// Initialize weights based on the activation function used
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

  for (size_t i = 0; i < this->input_size * this->output_size; i++){
    this->w[i] = distribution(generator);
    this->grad_w[i] = 0.0f;
  }
  for (size_t i = 0; i < this->output_size; i++){
    this->b[i] = 0.0f;
    this->grad_b[i] = 0.0f;
  }
}

// Check pred component matches dimensions
void weights::check_pred(){
  if(this->pred == nullptr)
    return;
  else if(this->pred->get_output_size() != this->input_size){
    throw invalid_argument("Pred component doesn't matches weights input_size");
    exit(1); 
  }
  else if(this->pred->get_batch_size() != this->batch_size){
    throw invalid_argument("Pred component doesn't matches weights batch_size");
    exit(1); 
  }
  return;
}

// Check next component matches dimensions
void weights::check_next(){
  if(this->next == nullptr)
    return;
  else if(this->next->get_output_size() != this->output_size){
    throw invalid_argument("Next component doesn't matches weights output_size");
    exit(1); 
  }
  else if(this->next->get_batch_size() != this->batch_size){
    throw invalid_argument("Next component doesn't matches weights batch_size");
    exit(1);
  }
  return;
}

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
weights::weights( size_t input_size, 
                  size_t output_size, 
                  size_t batch_size, 
                  Activation_name function_name,
                  BackwardClass *pred,
                  BackwardClass *next){
  if(input_size <= 0 || output_size <= 0){
    throw invalid_argument("Input and output size must be greater than 0");
    exit(1);
  }

  this->input_size = input_size;
  this->output_size = output_size;
  this->batch_size = batch_size;
  this->w = new float[input_size * output_size];
  this->grad_w = new float[input_size * output_size];
  this->b = new float[output_size];
  this->grad_b = new float[output_size];

  // Assing pointers and checks
  this->pred = pred;
  this->next = next;
  this->check_pred();
  this->check_next();

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

// Get input size
size_t weights::get_input_size(){
  return this->input_size;
}

// Get output size
size_t weights::get_output_size(){
  return this->output_size;
}

// Get batch size
size_t weights::get_batch_size(){
  return this->batch_size;
}

/* SETTERS */
// Set pred pointer
void weights::set_pred(BackwardClass *pred){
  this->pred = pred;
  this->check_pred();
}

// Set next pointer
void weights::set_next(BackwardClass *next){
  this->next = next;
  this->check_next();
}

/* METHODS */
// Forward pass
// x*W + b
void weights::operator()(){
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute forward pass without weights pred component.");
    exit(1);
  }
  if(this->next == nullptr){
    throw invalid_argument("Error: can't compute forward pass without weights next component.");
    exit(1);
  }

  float *input_values = this->pred->values_pointer();
  float *out_values = this->next->values_pointer();

  Multiply(input_values, this->w, out_values, this->batch_size, this->input_size, this->output_size);

  // Add bias to each row of the batch output.
  for (size_t row = 0; row < this->batch_size; row++){
    for (size_t col = 0; col < this->output_size; col++){
      out_values[row * this->output_size + col] += this->b[col];
    }
  }

  return;
}

/* BACKPROPAGATION FUNCTIONS */
// Zero the gradient
void weights::zero_grad(){
  for (size_t i = 0; i < this->input_size * this->output_size; i++){
    this->grad_w[i] = 0.0f;
  }
  for (size_t i = 0; i < this->output_size; i++){
    this->grad_b[i] = 0.0f;
  }
}

void weights::backward(){
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute backward pass without weights pred component.");
    exit(1);
  }
  if(this->next == nullptr){
    throw invalid_argument("Error: can't compute backward pass without weights next component.");
    exit(1);
  }

  float *prevGrad = this->pred->grad_pointer();
  float *input_values = this->pred->values_pointer();
  float *nextGrad = this->next->grad_pointer();

  // Weights gradient:
  // dL/dW = X^T * dL/dY where X is (batch_size x input_size)
  InPlaceMultiplyAndAdd_Transpose1(
    input_values,
    nextGrad,
    this->grad_w,
    this->input_size,
    this->batch_size,
    this->output_size
  );
  
  // Input gradient:
  // dL/dX = dL/dY * W^T where dL/dY is (batch_size x output_size)
  Multiply_transpose2(
    nextGrad,
    this->w,
    prevGrad,
    this->batch_size,
    this->output_size,
    this->input_size
  );

  // Bias gradient:
  // dL/db = sum over batch rows of dL/dY
  InPlaceVector_Add_MatrixT(
    this->grad_b,
    nextGrad,
    this->output_size,
    this->batch_size
  );

  this->pred->backward();
}

// Update the weights
void weights::update(float learning_rate){
  for (size_t i = 0; i < this->input_size * this->output_size; i++){
    this->w[i] -= learning_rate * this->grad_w[i];
  }
  for (size_t i = 0; i < this->output_size; i++){
    this->b[i] -= learning_rate * this->grad_b[i];
  }
}

/* TESTING FUNCTIONS */
// Print the weights
void weights::print_weights(){
  for (size_t i = 0; i < this->input_size * this->output_size; i++){
    if(i % this->input_size == 0)
      cout << endl;
    cout << this->w[i] << " ";
  }
  cout << endl;
}

// Print the gradient of the weights
void weights::print_grad_weights(){
  for (size_t i = 0; i < this->input_size * this->output_size; i++){
    if(i % this->input_size == 0)
      cout << endl;
    cout << this->grad_w[i] << " ";
  }
  cout << endl;
}

// Print the bias
void weights::print_bias(){
  for (size_t i = 0; i < this->output_size; i++){
    cout << this->b[i] << " ";
  }
  cout << endl;
}

// Print the gradient of the bias
void weights::print_grad_bias(){
  for (size_t i = 0; i < this->output_size; i++){
    cout << this->grad_b[i] << " ";
  }
  cout << endl;
}
