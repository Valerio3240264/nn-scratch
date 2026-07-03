#include "../headers/activation.h"

#include <iostream>
#include <cmath>

using namespace std;

// Check pred component matches dimensions
void activation::check_pred(){
  if(this->pred == nullptr)
    return;
  else if(this->pred->get_output_size() != this->size){
    throw invalid_argument("Pred component doesn't matches activation size");
    exit(1); 
  }
  else if(this->pred->get_batch_size() != this->batch_size){
    throw invalid_argument("Pred component doesn't matches activation batch_size");
    exit(1); 
  }
  return;
}

/* CONSTRUCTORS AND DESTRUCTOR */
// Constructor
activation::activation( size_t size, 
                        size_t batch_size, 
                        Activation_name function_name, 
                        BackwardClass *pred){
  this->size = size;
  this->batch_size = batch_size;
  this->value = new float[size * batch_size];
  this->grad = new float[size * batch_size];
  this->function_name = function_name;
  for(size_t i = 0; i < size * batch_size; ++i){
    grad[i] = 0.0f;
  }

  // Assing pointer and check
  this->pred = pred;
  this->check_pred();
}

// Destructor
activation::~activation(){
  delete[] this->grad;
  delete[] this->value;
}

/* GETTERS */
// Get the activation name
Activation_name activation::get_activation_fun(){
  return this->function_name;
}

// Get the values pointer
float *activation::values_pointer(){
  return this->value;
}

// Get the gradient pointer
float *activation::grad_pointer(){
  return this->grad;
}

// Get the value at a specific index
float activation::get_value(size_t index){
  return this->value[index];
}

// Get the gradient at a specific index
float activation::get_grad(size_t index){
  return this->grad[index];
}

// Get size
size_t activation::get_output_size(){
  return this->size;
}

// Get batch size
size_t activation::get_batch_size(){
  return this->batch_size;
}

/* SETTERS */
// Set pred pointer
void activation::set_pred(BackwardClass *pred){
  this->pred = pred;
  this->check_pred();
}

/* OPERATORS */
// Operator to apply the activation function
void activation::operator()(){
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute forward pass without activation pred component.");
    exit(1);
  }
  this->check_pred();

  if(this->function_name == SOFTMAX){
    for(size_t row = 0; row < this->batch_size; row++){
      float *row_values = this->value + row * this->size;
      float max_val = row_values[0];
      for(size_t col = 1; col < this->size; col++){
        if(row_values[col] > max_val){
          max_val = row_values[col];
        }
      }

      float normalization = 0.0f;
      for(size_t col = 0; col < this->size; col++){
        normalization += expf(row_values[col] - max_val);
      }

      for(size_t col = 0; col < this->size; col++){
        row_values[col] = expf(row_values[col] - max_val) / normalization;
      }
    }
    return;
  }

  size_t elements = this->size * this->batch_size;
  for(size_t i = 0; i < elements; i++){
    if(this->function_name == TANH){
      this->value[i] = tanhf(this->value[i]);
    }
    else if(this->function_name == RELU){
      this->value[i] = max(0.0f, this->value[i]);
    }
    else if(this->function_name == LINEAR){
      break;
    }
    else{
      throw invalid_argument("Invalid activation function");
    }
  }
}

/* BACKPROPAGATION FUNCTIONS */
// Zero the gradient
void activation::zero_grad(){
  size_t elements = this->size * this->batch_size;
  for(size_t i = 0; i < elements; i++){
    this->grad[i] = 0.0f;
  }
}

// Backward pass with explicit batch size
// Derivatives(size x batch_size)
void activation::backward(){
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute backward pass without activation pred component.");
    exit(1);
  }
  this->check_pred();

  if(this->function_name == SOFTMAX){
    for(size_t row = 0; row < this->batch_size; row++){
      float *row_values = this->value + row * this->size;
      float *row_grad = this->grad + row * this->size;

      float dot = 0.0f;
      for(size_t col = 0; col < this->size; col++){
        dot += row_values[col] * row_grad[col];
      }

      for(size_t col = 0; col < this->size; col++){
        row_grad[col] = row_values[col] * (row_grad[col] - dot);
      }
    }
    this->pred->backward();
    return;
  }

  size_t elements = this->size * batch_size;
  for(size_t i = 0; i < elements; i++){
    if(this->function_name == TANH){
      this->grad[i] = this->grad[i] * (1 - (this->value[i]* this->value[i]));
    }
    else if(this->function_name == RELU){
      this->grad[i] = this->grad[i] * (this->value[i] > 0 ? 1 : 0);
    }
    else if(this->function_name == LINEAR){
      this->grad[i] = this->grad[i];
    }
    else{
      throw invalid_argument("Invalid activation function");
    }
  }
  this->pred->backward();
}

/* TESTING FUNCTIONS */
// Print the value
void activation::print_value(){
  size_t elements = this->size * this->batch_size;
  for (size_t i = 0; i < elements; i++){
    if(i % this->size == 0){
      cout << endl;
    }
    cout << this->value[i] << " ";
  }
  cout << endl;
}

// Print the gradient
void activation::print_grad(){
  size_t elements = this->size * this->batch_size;
  for (size_t i = 0; i < elements; i++){
    if(i % this->size == 0){
      cout << endl;
    }
    cout << this->grad[i] << " ";
  }
  cout << endl;
}
