#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <iostream>
#include <cmath>
#include "../enums.h"
#include "../virtual_classes.h"

class softmax : public BackwardClass {
  private:
    double *value;
    double *grad;
    int size;
    double temperature;
    BackwardClass *pred;

  public:
    // Constructors
    softmax(int size, double *value, BackwardClass *pred);
    softmax(int size, double *value, double temperature, BackwardClass *pred);
    // Destructor
    ~softmax();
    // Getters
    double *values_pointer() override;
    double *grad_pointer() override;
    // Methods
    void backward(double *derivatives) override;
    void zero_grad() override;
    void operator()();
    // Testing functions
    void print_value();
    void print_grad();
};

/* CONSTRUCTORS AND DESTRUCTOR */
// Constructor
softmax::softmax(int size, double *value, BackwardClass *pred){
  this->size = size;
  this->value = value;
  this->pred = pred;
  this->temperature = 1.0;
  this->grad = new double[size];
  for(int i = 0; i < size; i++){
    this->grad[i] = 0.0;
  }
}

// Constructor with temperature
softmax::softmax(int size, double *value, double temperature, BackwardClass *pred){
  this->size = size;
  this->value = value;
  this->temperature = temperature;
  this->pred = pred;
  this->grad = new double[size];
  for(int i = 0; i < size; i++){
    this->grad[i] = 0.0;
  }
}

// Destructor
softmax::~softmax(){
  delete[] this->grad;
  delete[] this->value;
}

/* GETTERS */
// Get the values pointer
double *softmax::values_pointer(){
  return this->value;
}

// Get the gradient pointer
double *softmax::grad_pointer(){
  return this->grad;
}

/* METHODS */
// Operator to apply the softmax function
void softmax::operator()(){
  double Z = 0;
  for(int i = 0; i < this->size; i++){
    Z += exp(this->value[i] / this->temperature);
  }
  for(int i = 0; i < this->size; i++){
    this->value[i] = exp(this->value[i] / this->temperature) / Z;
  }
}

// Zero the gradient
void softmax::zero_grad(){
  return;
}

// Backward pass
void softmax::backward(double *derivatives){
  double dot = 0.0;
  for (int k = 0; k < this->size; ++k) {
    dot += this->value[k] * derivatives[k];
  }

  double *prevGrad = new double[this->size];
  for (int j = 0; j < this->size; ++j) {
    prevGrad[j] = this->value[j] * (derivatives[j] - dot) / this->temperature;
  }

  if (this->pred) {
    this->pred->backward(prevGrad);
  }

  delete[] prevGrad;
}

/* TESTING FUNCTIONS */
// Print the values
void softmax::print_value(){
  for(int i = 0; i < this->size; i++){
    std::cout << this->value[i] << " ";
  }
  std::cout << std::endl;
}

// Print the gradient
void softmax::print_grad(){
  for(int i = 0; i < this->size; i++){
    std::cout << this->grad[i] << " ";
  }
  std::cout << std::endl;
}

#endif 