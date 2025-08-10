#ifndef LOSS_H
#define LOSS_H

#include <iostream>
#include <vector>
#include <cmath>
#include "virtual_classes.h"

class loss: public BackwardClass{
  private:
    BackwardClass *pred;
    double *target;
    double *grad;
    double *loss_value;
    int size;

  public:
    // Constructors
    loss(BackwardClass *pred, int size);
    loss(BackwardClass *pred, int size, double *target);
    // Destructor
    ~loss();
    // Methods
    void operator()(double *target);
    void operator()();
    // Backpropagation functions
    void zero_grad() override;
    void backward(double *derivatives) override;
    // Getters
    double *values_pointer() override;
    double *grad_pointer() override;
    // Testing functions
    void print_loss();
    void print_grad();
};

/* CONSTRUCTORS */
// Constructor for loss function without target
loss::loss(BackwardClass *pred, int size){
  this->pred = pred;
  this->grad = new double[size];
  this->loss_value = new double[size];
  this->size = size;
  this->target = nullptr;
  for(int i = 0; i < size; i++){
    this->grad[i] = 0;
  }
}

// Constructor for loss function with target
loss::loss(BackwardClass *pred, int size, double *target){
  this->pred = pred;
  this->grad = new double[size];
  for(int i = 0; i < size; i++){
    this->grad[i] = 0;
  }
  this->loss_value = new double[size];
  this->size = size;
  this->target = target;
}

// Destructor
loss::~loss(){
  delete[] this->grad;
  delete[] this->loss_value;
}

/* GETTERS */
// Get the values pointer
double *loss::values_pointer(){
  return this->loss_value;
}

// Get the gradient pointer
double *loss::grad_pointer(){
  return this->grad;
}

/* METHODS */
// Operator to set the target
void loss::operator()(double *target){
  double *predictions = this->pred->values_pointer();
  for(int i = 0; i < this->size; i++){
    this->loss_value[i] = (predictions[i] - target[i]) * (predictions[i] - target[i]);
  }
}

// Operator to calculate the loss
void loss::operator()(){
  if(this->target == nullptr){
    throw std::invalid_argument("Target is not set");
  }
  else{
    double *predictions = this->pred->values_pointer();
    for(int i = 0; i < this->size; i++){
      this->loss_value[i] = (predictions[i] - this->target[i]) * (predictions[i] - this->target[i]);
    }
  }
}

/* BACKPROPAGATION FUNCTIONS */
// Zero the gradient
void loss::zero_grad(){
  for(int i = 0; i < this->size; i++){
    this->grad[i] = 0.0;
  }
}

// Backward pass
void loss::backward(double *derivatives){
  double *predictions = this->pred->values_pointer();
  double *prevGrad = new double[this->size];
  for(int i = 0; i < this->size; i++){
    prevGrad[i] = 2 * (predictions[i] - this->target[i]) * derivatives[i];
    this->grad[i] += prevGrad[i];
  }
  this->pred->backward(prevGrad);
  delete[] prevGrad;
}

/* TESTING FUNCTIONS */
// Print the loss
void loss::print_loss(){
  for(int i = 0; i < this->size; i++){
    std::cout << this->loss_value[i] << " ";
  }
  std::cout << std::endl;
}

// Print the gradient
void loss::print_grad(){
  for(int i = 0; i < this->size; i++){
    std::cout << this->grad[i] << " ";
  }
  std::cout << std::endl;
}

#endif