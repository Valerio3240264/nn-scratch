#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include <iostream>
#include <vector>
#include <cmath>
#include "../virtual_classes.h"

/*
MSE LOSS CLASS DOCUMENTATION:
PURPOSE:
Mean Squared Error loss function for regression tasks.
Formula: L = (1/n) * sum((prediction - target)^2)
Gradient: dL/dprediction = 2 * (prediction - target)

Attributes:
- pred: pointer to the predecessor (output layer)
- target: pointer to the target values
- grad: pointer to the gradients
- loss_value: scalar loss value
- size: size of the output vector

Methods:
- operator()(double *target): forward pass with target array
- operator()(): forward pass with stored target
- backward(): backward pass (simplified, assumes derivative = 1)
- backward(double *derivatives): backward pass with incoming derivatives
*/

using namespace std;

class mse_loss : public BackwardClass {
  private:
    BackwardClass *pred;
    double *target;
    double *grad;
    double loss_value;
    int size;

  public:
    // Constructors
    mse_loss(BackwardClass *pred, int size);
    mse_loss(BackwardClass *pred, int size, double *target);
    
    // Destructor
    ~mse_loss();
    
    // Forward pass methods
    void operator()(double *target);
    void operator()();
    
    // Backpropagation functions
    void zero_grad() override;
    void backward(double *derivatives) override;
    void backward();
    
    // Getters
    double *values_pointer() override;
    double *grad_pointer() override;
    double get_loss();
    
    // Testing functions
    void print_loss();
    void print_grad();
};

/* CONSTRUCTORS */
mse_loss::mse_loss(BackwardClass *pred, int size) {
  this->pred = pred;
  this->size = size;
  this->grad = new double[size];
  this->target = nullptr;
  this->loss_value = 0.0;
  
  for(int i = 0; i < size; i++) {
    this->grad[i] = 0.0;
  }
}

mse_loss::mse_loss(BackwardClass *pred, int size, double *target) {
  this->pred = pred;
  this->size = size;
  this->grad = new double[size];
  this->target = target;
  this->loss_value = 0.0;
  
  for(int i = 0; i < size; i++) {
    this->grad[i] = 0.0;
  }
}

/* DESTRUCTOR */
mse_loss::~mse_loss() {
  delete[] this->grad;
}

/* METHODS */
void mse_loss::operator()(double *target) {
  if(this->target != nullptr){
    delete[] this->target;
    this->target = nullptr;
  }
  this->target = target;
  this->operator()();
}

// Forward with stored target
void mse_loss::operator()() {
  double *predictions = this->pred->values_pointer();
  this->loss_value = 0.0;

  for(int i = 0; i < this->size; i++) {
    double diff = predictions[i] - this->target[i];
    this->loss_value += diff * diff;
  }
  this->loss_value /= this->size;
}

/* BACKPROPAGATION */
void mse_loss::zero_grad() {
  for(int i = 0; i < this->size; i++) {
    this->grad[i] = 0.0;
  }
}

void mse_loss::backward(double *derivatives) {
  if(this->target == nullptr) {
    throw std::invalid_argument("No target set for backward pass");
  }
  
  double *predictions = this->pred->values_pointer();
  double *prevGrad = new double[this->size];
  
  for(int i = 0; i < this->size; i++) {
    prevGrad[i] = (2.0 / this->size) * (predictions[i] - this->target[i]) * derivatives[i];
    this->grad[i] += prevGrad[i];
  }
  
  this->pred->backward(prevGrad);
  delete[] prevGrad;
}

void mse_loss::backward() {
  if(this->target == nullptr) {
    throw std::invalid_argument("No target set for backward pass");
  }
  
  double *predictions = this->pred->values_pointer();
  double *prevGrad = new double[this->size];
  
  for(int i = 0; i < this->size; i++) {
    prevGrad[i] = (2.0 / this->size) * (predictions[i] - this->target[i]);
    this->grad[i] += prevGrad[i];
  }
  
  this->pred->backward(prevGrad);
  delete[] prevGrad;
}

/* GETTERS */
double *mse_loss::values_pointer() {
  return &this->loss_value;
}

double *mse_loss::grad_pointer() {
  return this->grad;
}

double mse_loss::get_loss() {
  return this->loss_value;
}

/* TESTING FUNCTIONS */
void mse_loss::print_loss() {
  std::cout << "MSE Loss: " << this->loss_value << std::endl;
}

void mse_loss::print_grad() {
  std::cout << "Gradients: ";
  for(int i = 0; i < this->size; i++) {
    std::cout << this->grad[i] << " ";
  }
  std::cout << std::endl;
}

#endif

