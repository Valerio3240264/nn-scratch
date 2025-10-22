#ifndef INPUT_H
#define INPUT_H

#include <iostream>
#include "../virtual_classes.h"

/*TODO
1: Add batch representation.
*/

/*
INPUT CLASS DOCUMENTATION:
PURPOSE:
This class is used to store an array of values and gradients that needs to be stored temporarily for the gradient evaluation on the neural network.
The attribute pred will store the predecessor pointer and, in this way, call the backward method of the predecessor.

In the neural network it is used to link the layers together.
Layer_i will call Layer_i-1 activation_function and so on until the input layer.

Attributes:
- value: pointer to the values array (this can be copied or create a new array, depends on the constructor used)
- grad: pointer to the gradients array
- size: size of the values and gradients arrays
- pred: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)

Constructors:
- input(int size): creates a new array for the values and gradients arrays and sets the predecessor to nullptr (usefull when you want to store the values of the input layer).
- input(int size, double *value): creates a new array for the gradients array and sets the predecessor to nullptr (usefull when you want to copy values from an external array).
- input(int size, BackwardClass *pred): creates a new array for the values and gradients arrays and sets the predecessor to the passed pointer (usefull when you want to pass values between layers).

Methods:
- values_pointer(): returns the pointer to the values array.
- grad_pointer(): returns the pointer to the gradients array
- zero_grad(): sets all the gradients to 0
- backward(double *derivatives): accumulates the gradients and propagates them to the predecessor
- print_value(): prints the values array
- print_grad(): prints the gradients array
*/

using namespace std;

class input : public BackwardClass {
  private:
    float *value;
    float *grad;
    int size;
    BackwardClass *pred;

  public:
    
    // Constructors
    input(int size);
    input(int size, float *value);
    input(int size, BackwardClass *pred);

    // Destructor
    ~input();

    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float get_value(int index);
    float get_grad(int index);
    
    // Backpropagation functions
    void zero_grad() override;
    void backward(float *derivatives) override;

    // Testing functions
    void print_value();
    void print_grad();
};

/* CONSTRUCTORS AND DESTRUCTOR */
// Constructor for base input
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

// Constructor for input with value
input::input(int size, float *value) {
  this->size = size;
  this->value = value;
  this->grad = new float[size];
  this->pred = nullptr;

  for (int i = 0; i < size; i++) {
    this->grad[i] = 0.0f;
  }
}

// Constructor for input with predecessor
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
    this->grad[i] += derivatives[i];
  }
  if (this->pred != nullptr) {
    this->pred->backward(derivatives);
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

#endif