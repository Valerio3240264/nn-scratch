#include "../headers/softmax.h"

#include <iostream>
#include <cmath>

using namespace std;

// Check pred component matches dimensions
void softmax::check_pred(){
  if(this->pred == nullptr)
    return;
  else if(this->pred->get_output_size() != this->size){
    throw invalid_argument("Pred component doesn't matches softmax size");
    exit(1); 
  }
  else if(this->pred->get_batch_size() != this->batch_size){
    throw invalid_argument("Pred component doesn't matches softmax batch_size");
    exit(1); 
  }
  return;
}

/* CONSTRUCTORS AND DESTRUCTOR */
// Constructor with temperature
softmax::softmax( size_t size, 
                  size_t batch_size, 
                  float temperature, 
                  BackwardClass *pred){
  this->size = size;
  this->batch_size = batch_size;
  this->values = new float[size * batch_size];
  this->temperature = temperature;
  this->grad = new float[size * batch_size];
  for(size_t i = 0; i < size*batch_size; i++){
    this->values[i] = 0.0f;
    this->grad[i] = 0.0f;
  }

  // Assign pointer and check
  this->pred = pred;
  this->check_pred();
}

// Destructor
softmax::~softmax(){
  delete[] this->grad;
  delete[] this->values;
}

/* GETTERS */
Activation_name softmax::get_activation_fun(){
  return SOFTMAX;
}

// Get the values pointer
float *softmax::values_pointer(){
  return this->values;
}

// Get the gradient pointer
float *softmax::grad_pointer(){
  return this->grad;
}

// Get temperature
float softmax::get_temperature(){
  return this->temperature;
}

// Recive and array and populates it with the guesses
void softmax::get_predictions(size_t *predictions){
  for(size_t row = 0; row < this->batch_size; row++){
    size_t max_idx = 0;
    float *row_values = this->values + row * this->size;
    for(size_t col = 1; col < this->size; col++){
      if(row_values[col] > row_values[max_idx]){
        max_idx = col;
      }
    }
    predictions[row] = max_idx;
  }
}

// Get size
size_t softmax::get_output_size(){
  return this->size;
}

// Get batch_size
size_t softmax::get_batch_size(){
  return this->batch_size;
}

/* SETTERS */
// Set pred pointer
void softmax::set_pred(BackwardClass *pred){
  this->pred = pred;
  this->check_pred();
}

/* METHODS */
void softmax::operator()(){
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute forward pass without softmax pred component.");
    exit(1);
  }
  this->check_pred();
  float *pred_values = this->pred->values_pointer();

  for(size_t row = 0; row < batch_size; row++){
    pred_values += row != 0 ? this->size : 0;
    float *Crow_values = this->values + row * this->size;
    float max_val = pred_values[0];
    float Z = 0.f;

    // Calculate max value and Z in the same cycle
    for(size_t i = 0; i < this->size; i++){
      if(pred_values[i] > max_val){
        Z *= expf((max_val - pred_values[i]) / this->temperature);
        max_val = pred_values[i];
      }
      Z += expf((pred_values[i] - max_val) / this->temperature);
    }

    // Normalize each row
    for(size_t i = 0; i < this->size; i++){
      Crow_values[i] = expf((pred_values[i] - max_val) / this->temperature) / Z;
    }
  }
}

// Zero the gradient
void softmax::zero_grad(){
  for(size_t i = 0; i < this->size * this->batch_size; i++){
    this->grad[i] = 0;
  }
}

void softmax::backward(){
  if(this->pred == nullptr){
    throw invalid_argument("Error: can't compute forward pass without softmax pred component.");
    exit(1);
  }
  this->check_pred();
  float *pred_grad = this->pred->grad_pointer();
  
  for(size_t row = 0; row < batch_size; row++){
    float *row_values = this->values + row * this->size;
    float *row_derivatives = this->grad + row * this->size;
    pred_grad += row != 0 ? this->size : 0;

    float dot = 0.0f;
    for (size_t k = 0; k < this->size; ++k) {
      dot += row_values[k] * row_derivatives[k];
    }

    for (size_t j = 0; j < this->size; ++j) {
      pred_grad[j] = row_values[j] * (row_derivatives[j] - dot) / this->temperature;
    }
  }

  this->pred->backward();
}

/* TESTING FUNCTIONS */
// Print the values
void softmax::print_value(){
  if(this->values == nullptr){
    cout<<"Error: value is not set"<<endl;
    exit(1);
    return;
  }
  for(size_t i = 0; i < this->size * this->batch_size; i++){
    if(i % this->size == 0)
      std::cout<<endl;
    std::cout << this->values[i] << " ";
  }
  std::cout << std::endl;
}

// Print the gradient
void softmax::print_grad(){
  if(this->values == nullptr){
    cout<<"Error: value is not set"<<endl;
    exit(1);
    return;
  }
  for(size_t i = 0; i < this->size * this->batch_size; i++){
    if(i % this->size == 0)
      std::cout<<endl;
    std::cout << this->grad[i] << " ";
  }
  std::cout << std::endl;
}
