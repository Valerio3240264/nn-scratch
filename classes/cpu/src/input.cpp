#include "../headers/input.h"

#include <iostream>
#include <cmath>

using namespace std;

/* CONSTRUCTORS AND DESTRUCTOR */
input::input(size_t size, size_t batch_size){
  this->values = nullptr;
  this->grad = new float[size * batch_size];
  this->size = size;
  this->batch_size = batch_size;
}

// Destructor
input::~input(){
  delete[] this->grad;
}

/* GETTERS */
// Get the value pointer
float *input::values_pointer(){
  return this->values;
}

// Get the gradient pointer
float *input::grad_pointer(){
  return this->grad;
}

// Get size
size_t input::get_output_size(){
  return this->size;
}

// Get batch size
size_t input::get_batch_size(){
  return this->batch_size;
}

/* SETTERS */
// Changes values pointer
void input::set_values(float *new_values){
  this->values = new_values; 
}

// Zero the gradient
void input::zero_grad(){
  for (size_t i = 0; i < this->size * this->batch_size; i++){
    this->grad[i] = 0.0f;
  }
}

/* BACKWARD */
// Leaf node: no component to propagate
void input::backward(){
  return;
}

/* TESTING FUNCTIONS */
// Print the value
void input::print_values(){
  for (size_t i = 0; i < this->size * this->batch_size; i++){
    if(i % this->size == 0){
      cout << endl;
    }
    cout << this->values[i] << " ";
  }
  cout << endl;
}

// Print the gradient
void input::print_grad(){
  for (size_t i = 0; i < this->size * this->batch_size; i++){
    if(i % this->size == 0){
      cout << endl;
    }
    cout << this->grad[i] << " ";
  }
  cout << endl;
}
