#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include <iostream>
#include <vector>
#include <cmath>
#include "../virtual_classes.h"

/*
CROSS-ENTROPY LOSS CLASS DOCUMENTATION:
PURPOSE:
Cross-Entropy loss function for multi-class classification tasks (after Softmax).
This is the standard loss function for classification problems.

Formula: L = -log(prediction[correct_class])
Gradient (when combined with Softmax): prediction - one_hot_target

This loss function is numerically stable and provides strong gradients for training.
It measures the KL divergence between the predicted and true probability distributions.

Attributes:
- pred: pointer to the predecessor (softmax layer)
- target: pointer to the target values (one-hot encoded)
- grad: pointer to the gradients
- loss_value: scalar loss value
- size: number of classes

Methods:
- operator()(double *target): forward pass with one-hot encoded target
- operator()(int target_index): forward pass with class index (converts to one-hot)
- operator()(): forward pass with stored target
- backward(): backward pass (simplified, assumes derivative = 1)
- backward(double *derivatives): backward pass with incoming derivatives
*/

using namespace std;

class cross_entropy_loss : public BackwardClass {
  private:
    BackwardClass *pred;
    double *target;
    double *grad;
    double loss_value;
    int size;

  public:
    // Constructors
    cross_entropy_loss(BackwardClass *pred, int size);
    cross_entropy_loss(BackwardClass *pred, int size, double *target);
    
    // Destructor
    ~cross_entropy_loss();
    
    // Forward pass methods
    void operator()(double *target);
    void operator()(int target_index);
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
cross_entropy_loss::cross_entropy_loss(BackwardClass *pred, int size) {
  this->pred = pred;
  this->size = size;
  this->grad = new double[size];
  this->target = nullptr;
  this->loss_value = 0.0;
  
  for(int i = 0; i < size; i++) {
    this->grad[i] = 0.0;
  }
}

cross_entropy_loss::cross_entropy_loss(BackwardClass *pred, int size, double *target) {
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
cross_entropy_loss::~cross_entropy_loss() {
  delete[] this->grad;
  if(this->target != nullptr) {
    delete[] this->target;
  }
}

/* FORWARD PASS METHODS */
// Forward with one-hot encoded target
void cross_entropy_loss::operator()(double *target) {
  if(this->target != nullptr) {
    delete[] this->target;
    this->target = nullptr;
  }
  this->target = target;
  this->operator()();
}

// Forward with class index (converts to one-hot encoding)
void cross_entropy_loss::operator()(int target_index) {
  if(target_index < 0 || target_index >= this->size) {
    throw std::invalid_argument("Target index is out of bounds");
  }
  
  // Convert target_index to one-hot encoding
  if(this->target != nullptr) {
    delete[] this->target;
  }
  this->target = new double[this->size];
  for(int i = 0; i < this->size; i++) {
    this->target[i] = (i == target_index) ? 1.0 : 0.0;
  }
  
  this->operator()();
}

// Forward with stored target
void cross_entropy_loss::operator()() {
  if(this->target == nullptr) {
    throw std::invalid_argument("No target set");
  }
  
  double *predictions = this->pred->values_pointer();
  this->loss_value = 0.0;
  
  // Cross-entropy: -sum(target * log(prediction))
  for(int i = 0; i < this->size; i++) {
    if(this->target[i] > 0) {
      // Add small epsilon to avoid log(0)
      this->loss_value -= this->target[i] * log(predictions[i] + 1e-15);
    }
  }
}

/* BACKPROPAGATION */
void cross_entropy_loss::zero_grad() {
  for(int i = 0; i < this->size; i++) {
    this->grad[i] = 0.0;
  }
}

// Standard backward with incoming derivatives
void cross_entropy_loss::backward(double *derivatives) {
  if(this->target == nullptr) {
      throw std::invalid_argument("No target set for backward pass");
  }
  
  double *predictions = this->pred->values_pointer();
  double *prevGrad = new double[this->size];
  
  // Gradient: (predictions - targets) * derivatives
  for(int i = 0; i < this->size; i++) {
    prevGrad[i] = (predictions[i] - this->target[i]) * derivatives[i];
    this->grad[i] += prevGrad[i];
  }
  
  this->pred->backward(prevGrad);
  delete[] prevGrad;
}

// Simplified backward (assumes derivative of loss w.r.t. itself is 1)
void cross_entropy_loss::backward() {
  if(this->target == nullptr) {
    throw std::invalid_argument("No target set for backward pass");
  }
  
  double *predictions = this->pred->values_pointer();
  double *prevGrad = new double[this->size];
  
  // Beautiful simplification: gradient = predictions - one_hot_target
  for(int i = 0; i < this->size; i++) {
    prevGrad[i] = predictions[i] - this->target[i];
    this->grad[i] += prevGrad[i];
  }
  
  this->pred->backward(prevGrad);
  delete[] prevGrad;
}

/* GETTERS */
double *cross_entropy_loss::values_pointer() {
    return &this->loss_value;
}

double *cross_entropy_loss::grad_pointer() {
    return this->grad;
}

double cross_entropy_loss::get_loss() {
    return this->loss_value;
}

/* TESTING FUNCTIONS */
void cross_entropy_loss::print_loss() {
    std::cout << "Cross-Entropy Loss: " << this->loss_value << std::endl;
}

void cross_entropy_loss::print_grad() {
  std::cout << "Gradients: ";
  for(int i = 0; i < this->size; i++) {
    std::cout << this->grad[i] << " ";
  }
  std::cout << std::endl;
}

#endif