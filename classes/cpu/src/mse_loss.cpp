#include "../headers/mse_loss.h"

#include <iostream>
#include <cmath>

using namespace std;

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