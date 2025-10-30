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

class mse_loss : public LossClass {
  private:
    BackwardClass *pred;
    float *target;
    float *grad;
    float loss_value;
    int size;

  public:
    // Constructors
    mse_loss(BackwardClass *pred, int size);
    mse_loss(BackwardClass *pred, int size, float *target);
    
    // Destructor
    ~mse_loss() override;
    
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float get_loss() override;
    
    // Methods
    void operator()(float *target) override;
    void operator()(int target_index) override;
    void operator()() override;
    void zero_grad() override;
    void backward(float *derivatives) override;
    void backward() override;
  
    // Testing functions
    void print_loss();
    void print_grad();
};

/* CONSTRUCTORS */
// Constructor - sets the predecessor pointer without target
mse_loss::mse_loss(BackwardClass *pred, int size) {
  this->pred = pred;
  this->size = size;
  this->grad = new float[size];
  this->target = nullptr;
  this->loss_value = 0.0f;
  
  for(int i = 0; i < size; i++) {
    this->grad[i] = 0.0f;
  }
}

// Constructor - sets the predecessor pointer and target
mse_loss::mse_loss(BackwardClass *pred, int size, float *target) {
  this->pred = pred;
  this->size = size;
  this->grad = new float[size];
  this->target = target;
  this->loss_value = 0.0f;
  
  for(int i = 0; i < size; i++) {
    this->grad[i] = 0.0f;
  }
}

/* DESTRUCTOR */
mse_loss::~mse_loss() {
  delete[] this->grad;
}

/* GETTERS */
// Get the values pointer of the current predecessor
float *mse_loss::values_pointer() {
  return this->pred->values_pointer();
}

// Get the gradient pointer
float *mse_loss::grad_pointer() {
  return this->grad;
}

// Get the loss value
float mse_loss::get_loss() {
  return this->loss_value;
}

/* METHODS */
// Forward pass with target array
void mse_loss::operator()(float *target) {
  if(this->target != nullptr){
    delete[] this->target;
    this->target = nullptr;
  }
  this->target = target;
  this->operator()();
}

// Forward with class index (converts to one-hot encoding for MSE)
void mse_loss::operator()(int target_index) {
  if(target_index < 0 || target_index >= this->size) {
    throw std::invalid_argument("Target index is out of bounds");
  }
  
  // Convert target_index to one-hot encoding
  if(this->target != nullptr) {
    delete[] this->target;
  }
  this->target = new float[this->size];
  for(int i = 0; i < this->size; i++) {
    this->target[i] = (i == target_index) ? 1.0f : 0.0f;
  }
  
  this->operator()();
}

// Forward with stored target
void mse_loss::operator()() {
  float *predictions = this->pred->values_pointer();
  this->loss_value = 0.0f;

  for(int i = 0; i < this->size; i++) {
    float diff = predictions[i] - this->target[i];
    this->loss_value += diff * diff;
  }
  this->loss_value /= this->size;
}

// Zero the gradient
void mse_loss::zero_grad() {
  for(int i = 0; i < this->size; i++) {
    this->grad[i] = 0.0f;
  }
}

// Backward pass with incoming derivatives
void mse_loss::backward(float *derivatives) {
  if(this->target == nullptr) {
    throw std::invalid_argument("No target set for backward pass");
  }
  
  float *predictions = this->pred->values_pointer();
  
  for(int i = 0; i < this->size; i++) {
    this->grad[i] = (2.0f / this->size) * (predictions[i] - this->target[i]) * derivatives[i];
  }
  
  this->pred->backward(this->grad);
}

// Backward pass with simplified gradient (assumes derivative of loss w.r.t. itself is 1)
void mse_loss::backward() {
  if(this->target == nullptr) {
    throw std::invalid_argument("No target set for backward pass");
  }
  
  float *predictions = this->pred->values_pointer();
  
  for(int i = 0; i < this->size; i++) {
    this->grad[i] = (2.0f / this->size) * (predictions[i] - this->target[i]);
  }
  
  this->pred->backward(this->grad);
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

