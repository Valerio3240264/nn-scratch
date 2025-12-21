#include "../headers/weights.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>

using namespace std;

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
  
  // Random number generator
  default_random_engine generator;
  uniform_real_distribution<float> distribution(-scale, scale);

  for (int i = 0; i < input_size * output_size; i++){
    this->w[i] = distribution(generator);
    this->grad_w[i] = 0.0f;
  }
  for (int i = 0; i < output_size; i++){
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
// Get the weights pointer
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
// Forward pass
// W x Input + b
void weights::operator()(BackwardClass *in, float *output_pointer){
  this->input_values = in->values_pointer();
  this->pred = in;
  for (int row = 0; row < this->output_size; row++){
    output_pointer[row] = this->b[row];
    for(int col = 0; col< this->input_size; col++){
      output_pointer[row] += this->w[row * this->input_size + col] * this->input_values[col];
    }
  }
  return;
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