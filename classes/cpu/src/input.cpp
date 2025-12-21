#include "../headers/input.h"

#include <iostream>
#include <cmath>

using namespace std;

/* CONSTRUCTORS AND DESTRUCTOR */
// Constructor - allocates its own memory
input::input(int size){
  this->size = size;
  this->value = new float[size];
  this->grad = new float[size];
  this->pred = nullptr;
  
  for (int i = 0; i < size; i++) {
    this->value[i] = 0.0f;
    this->grad[i] = 0.0f;
  }
}

// Constructor - allocates its own memory and copies values from an external array
input::input(int size, float *value) {
  this->size = size;
  this->value = value;
  this->grad = new float[size];
  this->pred = nullptr;

  for (int i = 0; i < size; i++) {
    this->grad[i] = 0.0f;
  }
}

// Constructor - sets the predecessor pointer
input::input(int size, BackwardClass *pred) {
  this->size = size;
  this->value = pred->values_pointer();
  this->grad = new float[size];
  this->pred = pred;
  
  for (int i = 0; i < size; i++) {
    this->grad[i] = 0.0f;
  }
}

// Destructor
input::~input(){
  if(this->pred == nullptr){
    delete[] this->value;
  }
  delete[] this->grad;
}

/* GETTERS */
// Get the value pointer
float *input::values_pointer(){
  return this->value;
}

// Get the gradient pointer
float *input::grad_pointer(){
  return this->grad;
}

// Get the value at a specific index
float input::get_value(int index){
  return this->value[index];
}

// Get the gradient at a specific index
float input::get_grad(int index){
  return this->grad[index];
}

/* BACKPROPAGATION FUNCTIONS */
// Zero the gradient
void input::zero_grad(){
  for (int i = 0; i < size; i++){
    this->grad[i] = 0.0f;
  }
}

// Backward pass
void input::backward(float *derivatives){
  for (int i = 0; i < size; i++){
    this->grad[i] = derivatives[i];
  }
  if (this->pred != nullptr) {
    this->pred->backward(this->grad);
  }
}

/* TESTING FUNCTIONS */
// Print the value
void input::print_value(){
  for (int i = 0; i < size; i++){
    cout << this->value[i] << " ";
  }
  cout << endl;
}

// Print the gradient
void input::print_grad(){
  for (int i = 0; i < size; i++){
    cout << this->grad[i] << " ";
  }
  cout << endl;
}
