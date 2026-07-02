#include "../headers/cross_entropy_loss.h"

#include <iostream>
#include <cmath>
#include <cstddef>

using namespace std;

// Check pred component matches dimensions
void cross_entropy_loss::check_pred(){
  if(this->pred == nullptr)
    return;
  else if(this->pred->get_output_size() != this->size){
    throw invalid_argument("Pred component doesn't matches cross_entropy_loss size");
    exit(1); 
  }
  else if(this->pred->get_batch_size() != this->batch_size){
    throw invalid_argument("Pred component doesn't matches cross_entropy_loss batch_size");
    exit(1); 
  }
  return;
}

/* CONSTRUCTOR */
cross_entropy_loss::cross_entropy_loss( size_t size, 
                                        size_t batch_size, 
                                        BackwardClass *pred) {
  this->size = size;
  this->target = new float[size * batch_size];
  this->loss_value = 0.0f;
  this->batch_size = batch_size;

  // Assign pointer and check
  this->pred = pred;
  this->check_pred();
}

/* DESTRUCTOR */
cross_entropy_loss::~cross_entropy_loss() {
  delete[] this->target;
}

/* GETTERS */
// Get the of the current predecessor
float *cross_entropy_loss::values_pointer() {
  return nullptr;
}

// Get the loss value
float cross_entropy_loss::get_loss() {
  return this->loss_value;
}

// Get size
size_t cross_entropy_loss::get_output_size(){
  return this->size;
}

// Get batch size
size_t cross_entropy_loss::get_batch_size(){
  return this->batch_size;
}

/* SETTERS */
// Set pred pointer
void cross_entropy_loss::set_pred(BackwardClass *pred){
  this->pred = pred;
  this->check_pred();
}

/* METHODS */
void cross_entropy_loss::operator()(float *target) {
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute cross entropy loss without pred component.");
    exit(1);
  }
  this->check_pred();

  // Evaluate the loss
  float *predictions = this->pred->values_pointer();
  this->loss_value = 0.0f;
  for(size_t i = 0; i < this->size * this->batch_size; i++) {
    this->target[i] = target[i];
    this->loss_value -= this->target[i] * logf(predictions[i] + 1e-15f);
  }
  this->loss_value /= static_cast<float>(this->batch_size);
}

// Forward with class indices (converts to one-hot encoding)
void cross_entropy_loss::operator()(size_t* target_indices) {
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute cross entropy loss without pred component.");
    exit(1);
  }
  this->check_pred();
  
  // One hot encoding
  for(size_t row = 0; row < this->batch_size; row++){
    int idx = target_indices[row];
    if(idx < 0 || idx >= static_cast<int>(this->size)) {
      throw std::invalid_argument("Target index is out of bounds");
    }
    for(size_t col = 0; col < this->size; col++) {
      this->target[row * this->size + col] = (static_cast<int>(col) == idx) ? 1.0f : 0.0f;
    }
  }
  
  // Evaluate the loss
  float *predictions = this->pred->values_pointer();
  this->loss_value = 0.0f;
  for(size_t i = 0; i < this->size * this->batch_size; i++) {
    this->loss_value -= this->target[i] * logf(predictions[i] + 1e-15f);
  }
  this->loss_value /= static_cast<float>(this->batch_size);
}

// Backward function 
// Writes the partial derivatives in the grad of the previous component and calls it
void cross_entropy_loss::backward() {
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute cross entropy loss backward without pred component.");
    exit(1);
  }
  this->check_pred();
  size_t elements = batch_size * this->size;
  float *predictions = this->pred->values_pointer();
  float *pred_grad = this->pred->grad_pointer();

  // dL/ds_i = -y_i / s_i (averaged over batch).
  const float eps = 1e-15f;
  for(size_t i = 0; i < elements; i++) {
    float p = predictions[i] > eps ? predictions[i] : eps;
    pred_grad[i] = -(this->target[i] / p) / static_cast<float>(this->batch_size);
  }
  
  this->pred->backward();
}

/* TESTING FUNCTIONS */
void cross_entropy_loss::print_loss() {
    std::cout << "Cross-Entropy Loss: " << this->loss_value << std::endl;
}