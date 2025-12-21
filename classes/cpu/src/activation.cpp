#include "../headers/activation.h"

#include <iostream>
#include <cmath>

using namespace std;

/* CONSTRUCTORS AND DESTRUCTOR */
// Constructor
activation::activation(
  int size, float *value, Activation_name function_name, BackwardClass *pred){
  this->size = size;
  this->value = value;
  this->grad = new float[size];
  for(int i = 0; i < size; i++){
    this->grad[i] = 0;
  }
  this->function_name = function_name;
  this->pred = pred;
}

// Destructor
activation::~activation(){
  delete[] this->grad;
}

/* GETTERS */

// Get the values pointer
float *activation::values_pointer(){
  return this->value;
}

// Get the gradient pointer
float *activation::grad_pointer(){
  return this->grad;
}

// Get the value at a specific index
float activation::get_value(int index){
  return this->value[index];
}

// Get the gradient at a specific index
float activation::get_grad(int index){
  return this->grad[index];
}

/* OPERATORS */
// Operator to apply the activation function
void activation::operator()(){
  for(int i = 0; i < this->size; i++){
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
  for(int i = 0; i < this->size; i++){
    this->grad[i] = 0.0f;
  }
}

// Backward pass
void activation::backward(float *derivatives){
  for(int i = 0; i < this->size; i++){
    if(this->function_name == TANH){
      this->grad[i] = derivatives[i] * (1 - (this->value[i]* this->value[i]));
    }
    else if(this->function_name == RELU){
      this->grad[i] = derivatives[i] * (this->value[i] > 0 ? 1 : 0);
    }
    else if(this->function_name == LINEAR){
      this->grad[i] = derivatives[i];
    }
    else{
      throw invalid_argument("Invalid activation function");
    }
  }
  this->pred->backward(this->grad);
}

/* TESTING FUNCTIONS */
// Print the value
void activation::print_value(){
  for (int i = 0; i < size; i++){
    cout << this->value[i] << " ";
  }
  cout << endl;
}

// Print the gradient
void activation::print_grad(){
  for (int i = 0; i < size; i++){
    cout << this->grad[i] << " ";
  }
  cout << endl;
}
