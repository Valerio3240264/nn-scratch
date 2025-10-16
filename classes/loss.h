#ifndef LOSS_H
#define LOSS_H

#include <iostream>
#include <vector>
#include <cmath>
#include "virtual_classes.h"

/*TODO
1: Write a backward function that does not need to know the derivatives value since it is the first step of the backward pass and the derivatives are known.
1: Create a function to evaluate the loss of a whole batch.
2: Create a function to evaluate the gradient of a whole batch.
3: Optimize the batch operations using personalized cuda kernels.
*/

/*
LOSS CLASS DOCUMENTATION:
PURPOSE:
This class is used to store the loss value and the gradients of the loss function.
It also stores a pointer to the target values and the predecessor pointer to perform the backward pass on the whole neural network.

Attributes:
- pred: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)
- target: pointer to the target values
- grad: pointer to the gradients
- loss_value: pointer to the loss values
- size: size of the loss_value and grad arrays

Constructors:
- loss(BackwardClass *pred, int size): creates a new array for the gradients and loss value arrays and sets the predecessor to the passed pointer (The target pointer will be set when the operator() is called).
- loss(BackwardClass *pred, int size, double *target): creates a new array for the gradients and loss value arrays and sets the predecessor to the passed pointer and the target to the passed pointer.

Methods:
- operator()(double *target): sets the target values and calculates the loss value.
- operator()(): calculates the loss value.
- zero_grad(): sets all the gradients to 0.
- backward(double *derivatives): accumulates the gradients and propagates them to the predecessor.
- print_loss(): prints the loss value.
- print_grad(): prints the gradients.

*/

using namespace std;  

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
    double *get_total_loss();
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

// Get the total loss
double *loss::get_total_loss(){
  double total_loss = 0;
  for(int i = 0; i < this->size; i++){
    total_loss += this->loss_value[i];
  }
  return total_loss;
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

// Operator to calculate the loss for a classification task
void loss::operator()(int target_index){
  if(target_index < 0 || target_index >= this->size){
    throw std::invalid_argument("Target index is out of bounds");
  }
  double *predictions = this->pred->values_pointer();
  int prediction_index = 0;
  for(int i = 1; i < this->size; i++){
    if(predictions[i] > predictions[prediction_index]){
      prediction_index = i;
    }
  }
  if(prediction_index == target_index){
    for(int i = 0; i < this->size; i++){
      this->loss_value[i] = 0;
    }
  }
  else{
    for(int i = 0; i < this->size; i++){
      this->loss_value[i] = 1;
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