#include "../headers/mse_loss.h"

#include <iostream>
#include <cmath>

using namespace std;

// Check pred component matches dimensions
void mse_loss::check_pred(){
  if(this->pred == nullptr)
    return;
  else if(this->pred->get_output_size() != this->size){
    throw invalid_argument("Pred component doesn't matches mse_loss size");
    exit(1); 
  }
  else if(this->pred->get_batch_size() != this->batch_size){
    throw invalid_argument("Pred component doesn't matches mse_loss batch_size");
    exit(1); 
  }
  return;
}

/* CONSTRUCTOR */
mse_loss::mse_loss( size_t size, 
                    size_t batch_size, 
                    BackwardClass *pred) {
  this->size = size;
  this->batch_size = batch_size;
  this->target = new float[size * batch_size];
  this->loss_value = 0.0f;

  // Assign pointer and check
  this->pred = pred;
  this->check_pred();
}

/* DESTRUCTOR */
mse_loss::~mse_loss() {
  delete[] this->target;
}

/* GETTERS */
// Get the values pointer of the current predecessor
float *mse_loss::values_pointer() {
  return nullptr;
}

// Get the loss value
float mse_loss::get_loss() {
  return this->loss_value;
}

// Get size
size_t mse_loss::get_size(){
  return this->size;
}

// Get batch size
size_t mse_loss::get_batch_size(){
  return this->batch_size;
}

/* SETTERS */
// Set pred pointer
void mse_loss::set_pred(BackwardClass *pred){
  this->pred = pred;
  this->check_pred();
}

/* METHODS */
void mse_loss::operator()(float *target) {

  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute mse loss without pred component.");
    exit(1);
  }
  this->check_pred();

  // Evaluate the loss
  float *predictions = this->pred->values_pointer();
  this->loss_value = 0.0f;
  for(size_t i = 0; i < this->size * this->batch_size; i++) {
    this->target[i] = target[i];
    float diff = predictions[i] - this->target[i];
    this->loss_value += diff * diff;
  }
  this->loss_value /= static_cast<float>(this->size);
}

// Forward with class indices (converts to one-hot encoding for MSE)
void mse_loss::operator()(size_t* target_indices) {

  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute mse loss without pred component.");
    exit(1);
  }
  this->check_pred();

  // One hot encoding
  for(size_t row = 0; row < this->batch_size; row++){
    int idx = target_indices[row];
    if(idx < 0 || idx >= static_cast<int>(this->size)) {
      throw std::invalid_argument("Target index is out of bounds");
    }
    for(size_t col = 0; col < this->size; col++){
      this->target[row * this->size + col] = (static_cast<int>(col) == idx) ? 1.0f : 0.0f;
    }
  }

  float *predictions = this->pred->values_pointer();
  this->loss_value = 0.0f;
  for(size_t i = 0; i < this->size * this->batch_size; i++) {
    float diff = predictions[i] - this->target[i];
    this->loss_value += diff * diff;
  }
  this->loss_value /= static_cast<float>(this->size);
}

//Backward function
// Writes the partial derivatives in the grad of the previous component and calls it
void mse_loss::backward() {
  
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute mse loss backward without pred component.");
    exit(1);
  }
  this->check_pred();
  
  size_t elements = batch_size * this->size;

  // dL/dy for L = (1/size) * sum_{batch,dim} (y - t)^2
  float norm = 2.0f / static_cast<float>(this->size);
  float *predictions = this->pred->values_pointer();
  float *pred_grad = this->pred->grad_pointer();
  for(size_t i = 0; i < elements; i++) {
    pred_grad[i] = norm * (predictions[i] - this->target[i]);
  }
  
  this->pred->backward();
}

/* TESTING FUNCTIONS */
void mse_loss::print_loss() {
  std::cout << "MSE Loss: " << this->loss_value << std::endl;
}