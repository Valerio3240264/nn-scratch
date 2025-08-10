#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H
#include <iostream>
#include <cmath>
#include "enums.h"
#include "virtual_classes.h"

using namespace std;

class activation_function : public BackwardClass {
  private:
    int size;
    double *value;
    double *grad;
    BackwardClass *pred;
    Activation_name function_name;

  public:
    // Constructors
    activation_function(int size, double *value, Activation_name function_name, BackwardClass *pred);
    
    // Destructor
    ~activation_function();
    
    // Getters
    double *values_pointer() override;
    double *grad_pointer() override;
    double get_value(int index);
    double get_grad(int index);

    // Methods
    void operator()();
    
    // Backpropagation functions
    void zero_grad() override;
    void backward(double * derivatives) override;
    
    // Testing functions
    void print_value();
    void print_grad();
}; 

/* CONSTRUCTORS AND DESTRUCTOR */
// Constructor
activation_function::activation_function(
  int size, double *value, Activation_name function_name, BackwardClass *pred){
  this->size = size;
  this->value = value;
  this->grad = new double[size];
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
double *activation_function::values_pointer(){
  return this->value;
}

// Get the gradient pointer
double *activation_function::grad_pointer(){
  return this->grad;
}

// Get the value at a specific index
double activation_function::get_value(int index){
  return this->value[index];
}

// Get the gradient at a specific index
double activation_function::get_grad(int index){
  return this->grad[index];
}

/* OPERATORS */
// Operator to apply the activation function
void activation_function::operator()(){
  for(int i = 0; i < this->size; i++){
    if(this->function_name == TANH){
      this->value[i] = tanh(this->value[i]);
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
    this->grad[i] = 0.0;
  }
}

// Backward pass
void activation_function::backward(double *derivatives){
  double *prevGrad = new double[this->size];
  for(int i = 0; i < this->size; i++){
    if(this->function_name == TANH){
      prevGrad[i] = derivatives[i] * (1 - (this->value[i]* this->value[i]));
      this->grad[i] += prevGrad[i];
    }
    else{
      throw invalid_argument("Invalid activation function");
    }
  }
  this->pred->backward(prevGrad);
  delete[] prevGrad;
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