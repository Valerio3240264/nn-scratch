#ifndef WEIGHTS_H
#define WEIGHTS_H
#include "input.h"
#include "../virtual_classes.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

/* TODO 
1: Create a function to evaluate to process a whole batch of data.
2: Create a function to evaluate the gradient of a whole batch.
*/

/*
WEIGHTS CLASS DOCUMENTATION:
PURPOSE:
This class is used to store the weights and biases of a layer and perform the affine transformation (Wx + b) between the weights and the input values.
It also stores the gradients of the weights, biases, and input values to perform the backward pass on the whole neural network.

Attributes:
- w: pointer to the weights array
- grad_w: pointer to the weight gradients array
- b: pointer to the bias array
- grad_b: pointer to the bias gradients array
- input_size: size of the input values
- output_size: size of the output values
- input_values: pointer to the input values
- pred: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)

Constructors:
- weights(int input_size, int output_size): creates arrays for weights, biases, and their gradients. Initializes weights with Xavier/Glorot initialization and biases to zero.

Methods:
- values_pointer(): returns the pointer to the weights array.
- grad_pointer(): returns the pointer to the weight gradients array.
- bias_pointer(): returns the pointer to the bias array.
- grad_bias_pointer(): returns the pointer to the bias gradients array.
- operator()(BackwardClass *in): performs the affine transformation (Wx + b) between the weights, input values, and biases.
- zero_grad(): sets all the gradients (weights and biases) to 0.
- backward(double *derivatives): accumulates the gradients for both weights and biases, and propagates them to the predecessor.
- update(double learning_rate): updates the weights and biases using the computed gradients.
- print_weights(), print_grad_weights(), print_bias(), print_grad_bias(): testing/debugging functions.
*/

using namespace std;

class weights: public BackwardClass{
  private:
    float *w;
    float *grad_w;
    float *b;
    float *grad_b;
    int input_size;
    int output_size;
    float *input_values;
    BackwardClass *pred;

  public:
    // Constructors
    weights(int input_size, int output_size);
    // Destructor
    ~weights();
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float *bias_pointer();
    float *grad_bias_pointer();
    // Methods
    float *operator()(BackwardClass * in);
    // Backpropagation functions
    void zero_grad() override;
    void backward(float *derivatives) override;
    void update(float learning_rate);
    // Testing functions
    void print_weights();
    void print_grad_weights();
    void print_bias();
    void print_grad_bias();
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
weights::weights(int input_size, int output_size){
  this->input_size = input_size;
  this->output_size = output_size;
  this->w = new float[input_size * output_size];
  this->grad_w = new float[input_size * output_size];
  this->b = new float[output_size];
  this->grad_b = new float[output_size];
  this->input_values = nullptr;
  this->pred = nullptr;
  
  // Seed random number generator once (static ensures it's only done once)
  static bool seeded = false;
  if(!seeded){
    srand(time(NULL));
    seeded = true;
  }
  
  // Xavier/Glorot initialization: scale by sqrt(1/input_size) for better convergence
  /*
    Layer saturation occurs when the latest layer of a neural network doesnt change its output much as the network trains. 
    This is because the weights are not being updated much.
    This brings to the vanishing gradient problem, where the gradients become too small to update the weights effectively.
    This issue is related to the variance of the weights and the input values for each layer.
    In Understanding the difficulty of training deep feedforward neural networks, Glorot and Bengio (2010) proposed a way to initialize the weights of a layer to achive a good balance between the variance of the weights and the input values.
  */
  /*
    Experiments on the MNIST dataset with a 3 layer neural network (784 -> 128 -> 64 -> 10 -> softmax):
    With the same network configuration, using sqrt(6.0 / (input_size + output_size)) resulted in approximately 50% validation accuracy after 5 epochs.
    In contrast, using scale = sqrt(1.0 / input_size) improved performance significantly, achieving around 95% validation accuracy in the same number of epochs.
    This suggests that for this particular problem and network, the simpler sqrt(1.0 / input_size) initialization works better, highlighting the importance of empirical validation of initialization strategies.
  */
  float scale = sqrtf(1.0f / input_size);
  
  for (int i = 0; i < input_size * output_size; i++)
  {
    // Random value between -1 and 1, then scale
    this->w[i] = ((rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
    this->grad_w[i] = 0;
  }
  
  // Initialize biases to zero (common practice)
  for (int i = 0; i < output_size; i++)
  {
    this->b[i] = 0.0f;
    this->grad_b[i] = 0.0f;
  }
}

// Destructor
weights::~weights(){
  delete[] this->w;
  delete[] this->grad_w;
  delete[] this->b;
  delete[] this->grad_b;
}

/* GETTERS */
// Get the values pointer
float *weights::values_pointer(){
  return this->w;
}

// Get the gradient pointer
float *weights::grad_pointer(){
  return this->grad_w;
}

// Get the bias pointer
float *weights::bias_pointer(){
  return this->b;
}

// Get the bias gradient pointer
float *weights::grad_bias_pointer(){
  return this->grad_b;
}

/* METHODS */
// Operator to evaluate the output
// W x Input + b
float *weights::operator()(BackwardClass *in){
  this->input_values = in->values_pointer();
  this->pred = in;
  float *output = new float[output_size];
  for (int row = 0; row < this->output_size; row++){
    output[row] = this->b[row];  // Initialize with bias
    for(int col = 0; col< this->input_size; col++){
      output[row] += this->w[row * this->input_size + col] * this->input_values[col];
    }
  }
  return output;
}

/* BACKPROPAGATION FUNCTIONS */
// Zero the gradient
void weights::zero_grad(){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    this->grad_w[i] = 0.0f;
  }
  for (int i = 0; i < this->output_size; i++){
    this->grad_b[i] = 0.0f;
  }
}

// Backward pass
void weights::backward(float *derivatives){
  float *prevGrad = new float[this->input_size];
  for(int col = 0; col < this->input_size; col++){
    prevGrad[col] = 0;
    for(int row = 0; row< this->output_size; row++){
      prevGrad[col] += derivatives[row] * this->w[row * this->input_size + col];
      this->grad_w[row * this->input_size + col] += derivatives[row] * this->input_values[col];
    }
  }
 
  for(int row = 0; row < this->output_size; row++){
    this->grad_b[row] += derivatives[row];
  }
  this->pred->backward(prevGrad);
  delete[] prevGrad;
}

// Update the weights
void weights::update(float learning_rate){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    this->w[i] -= learning_rate * this->grad_w[i];
  }
  for (int i = 0; i < this->output_size; i++){
    this->b[i] -= learning_rate * this->grad_b[i];
  }
}

/* TESTING FUNCTIONS */
// Print the weights
void weights::print_weights(){
  for (int i = 0; i < this->input_size * this->output_size; i++)
  {
    cout << this->w[i] << " ";
  }
  cout << endl;
}

// Print the gradient of the weights
void weights::print_grad_weights(){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    cout << this->grad_w[i] << " ";
  }
  cout << endl;
}

// Print the bias
void weights::print_bias(){
  for (int i = 0; i < this->output_size; i++){
    cout << this->b[i] << " ";
  }
  cout << endl;
}

// Print the gradient of the bias
void weights::print_grad_bias(){
  for (int i = 0; i < this->output_size; i++){
    cout << this->grad_b[i] << " ";
  }
  cout << endl;
}

#endif