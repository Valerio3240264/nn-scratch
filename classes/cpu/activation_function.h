#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H
#include <iostream>
#include <cmath>
#include "../enums.h"
#include "../virtual_classes.h"

/*TODO
1: Create a function to evaluate the output of a whole batch.
2: Create a function to evaluate the gradient of a whole batch.
*/

/*
ACTIVATION FUNCTION CLASS DOCUMENTATION:
PURPOSE:
This class is used to store the values and gradients of the activation_function performed on the weighted sum of the previous layer.
It also stores the name of the activation function and the predecessor pointer to perform the backward pass on the whole neural network.
This class stores the values and when it is called it will apply the activation function to the values array.

Attributes:
- size: size of the values and gradients arrays
- value: pointer to the values array
- grad: pointer to the gradients array
- pred: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)
- function_name: name of the activation function

Constructors:
- activation_function(int size, double *value, Activation_name function_name, BackwardClass *pred): creates a new array for the values and gradients arrays and sets the predecessor to the passed pointer.

Methods:
- values_pointer(): returns the pointer to the values array.
- grad_pointer(): returns the pointer to the gradients array.
- operator()(): applies the activation function to the values array.
- zero_grad(): sets all the gradients to 0.
- backward(double *derivatives): accumulates the gradients and propagates them to the predecessor.
- print_value(): prints the values array.
- print_grad(): prints the gradients array.

*/

using namespace std;

class activation_function : public ActivationClass {
  private:
    int size;
    float *value;
    float *grad;
    BackwardClass *pred;
    Activation_name function_name;

  public:
    // Constructors
    activation_function(int size, float *value, Activation_name function_name, BackwardClass *pred);
    
    // Destructor
    ~activation_function();
    
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float get_value(int index);
    float get_grad(int index);

    // Methods
    void operator()();
    
    // Backpropagation functions
    void zero_grad() override;
    void backward(float * derivatives) override;
    
    // Testing functions
    void print_value();
    void print_grad();
}; 

/* CONSTRUCTORS AND DESTRUCTOR */
// Constructor
activation_function::activation_function(
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
activation_function::~activation_function(){
  delete[] this->grad;
}

/* GETTERS */

// Get the values pointer
float *activation_function::values_pointer(){
  return this->value;
}

// Get the gradient pointer
float *activation_function::grad_pointer(){
  return this->grad;
}

// Get the value at a specific index
float activation_function::get_value(int index){
  return this->value[index];
}

// Get the gradient at a specific index
float activation_function::get_grad(int index){
  return this->grad[index];
}

/* OPERATORS */
// Operator to apply the activation function
void activation_function::operator()(){
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
void activation_function::zero_grad(){
  for(int i = 0; i < this->size; i++){
    this->grad[i] = 0.0f;
  }
}

// Backward pass
void activation_function::backward(float *derivatives){
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
void activation_function::print_value(){
  for (int i = 0; i < size; i++){
    cout << this->value[i] << " ";
  }
  cout << endl;
}

// Print the gradient
void activation_function::print_grad(){
  for (int i = 0; i < size; i++){
    cout << this->grad[i] << " ";
  }
  cout << endl;
}

#endif