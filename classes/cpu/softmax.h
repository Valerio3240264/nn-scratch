#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <iostream>
#include <cmath>
#include "../enums.h"
#include "../virtual_classes.h"

class softmax : public BackwardClass {
  private:
    float *value;
    float *grad;
    int size;
    float temperature;
    BackwardClass *pred;

  public:
    // Constructors
    softmax(int size, float *value, BackwardClass *pred);
    softmax(int size, float *value, float temperature, BackwardClass *pred);
    // Destructor
    ~softmax();
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    int get_prediction();
    float get_prediction_probability(int index);
    // Methods
    void backward(float *derivatives) override;
    void zero_grad() override;
    void operator()();
    // Testing functions
    void print_value();
    void print_grad();
};

/* CONSTRUCTORS AND DESTRUCTOR */
// Constructor
softmax::softmax(int size, float *value, BackwardClass *pred){
  this->size = size;
  this->value = value;
  this->pred = pred;
  this->temperature = 1.0f;
  this->grad = new float[size];
  for(int i = 0; i < size; i++){
    this->grad[i] = 0.0f;
  }
}

// Constructor with temperature
softmax::softmax(int size, float *value, float temperature, BackwardClass *pred){
  this->size = size;
  this->value = value;
  this->temperature = temperature;
  this->pred = pred;
  this->grad = new float[size];
  for(int i = 0; i < size; i++){
    this->grad[i] = 0.0f;
  }
}

// Destructor
softmax::~softmax(){
  delete[] this->grad;
  delete[] this->value;
}

/* GETTERS */
// Get the values pointer
float *softmax::values_pointer(){
  return this->value;
}

// Get the gradient pointer
float *softmax::grad_pointer(){
  return this->grad;
}

int softmax::get_prediction(){
  int max_val_indx = 0;
  for(int i = 1; i < this->size; i++){
    if(this->value[i] > this->value[max_val_indx]){
      max_val_indx = i;
    }
  }
  return max_val_indx;
}

float softmax::get_prediction_probability(int index){
  return this->value[index];
}

/* METHODS */
// Operator to apply the softmax function
void softmax::operator()(){
  // Find max value for numerical stability
  float max_val = this->value[0];
  for(int i = 1; i < this->size; i++){
    if(this->value[i] > max_val){
      max_val = this->value[i];
    }
  }
  
  // Compute exp(x - max) and sum for numerical stability
  // This is mathematically equivalent to the softmax function, but it is more numerically stable.
  // This is just the softmax function multiplied by e^(-max_val)/e^(-max_val) = 1
  float Z = 0.f;
  for(int i = 0; i < this->size; i++){
    Z += expf((this->value[i] - max_val) / this->temperature);
  }
  
  // Normalize
  for(int i = 0; i < this->size; i++){
    this->value[i] = expf((this->value[i] - max_val) / this->temperature) / Z;
  }
}

// Zero the gradient
void softmax::zero_grad(){
  return;
}

// Backward pass
void softmax::backward(float *derivatives){
  float dot = 0.0f;
  for (int k = 0; k < this->size; ++k) {
    dot += this->value[k] * derivatives[k];
  }

  float *prevGrad = new float[this->size];
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