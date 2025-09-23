#ifndef WEIGHTS_H
#define WEIGHTS_H
#include "input.h"
#include "virtual_classes.h"

/* TODO 
1: Optimize the matrix multiplication using a personalized CUDA kernel.
2: Optimize the gradient computation using a personalized CUDA kernel.
3: Create a function to evaluate to process a whole batch of data (not only one single data point).
4: Create a function to evaluate the gradient of a whole batch.
5: Optimize the batch operations using personalized cuda kernels.
*/

/*
WEIGHTS CLASS DOCUMENTATION:
PURPOSE:
This class is used to store the weights of a layer and perform the matrix multiplication between the weights and the input values.
It also stores the gradients of the weights and the input values to perform the backward pass on the whole neural network.

Attributes:
- w: pointer to the weights array
- grad_w: pointer to the gradients array
- input_size: size of the input values
- output_size: size of the output values
- input_values: pointer to the input values
- pred: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)

Constructors:
- weights(int input_size, int output_size): creates a new array for the weights and gradients arrays and sets the predecessor to nullptr.

Methods:
- values_pointer(): returns the pointer to the weights array.
- grad_pointer(): returns the pointer to the gradients array.
- operator()(BackwardClass *in): performs the matrix multiplication between the weights and the input values.
- zero_grad(): sets all the gradients to 0.
- backward(double *derivatives): accumulates the gradients and propagates them to the predecessor.
- update(double learning_rate): updates the weights using the computed gradients.
*/

using namespace std;

class weights: public BackwardClass{
  private:
    double *w;
    double *grad_w;
    int input_size;
    int output_size;
    double *input_values;
    BackwardClass *pred;

  public:
    // Constructors
    weights(int input_size, int output_size);
    // Destructor
    ~weights();
    // Getters
    double *values_pointer() override;
    double *grad_pointer() override;
    // Methods
    double *operator()(BackwardClass * in);
    // Backpropagation functions
    void zero_grad() override;
    void backward(double *derivatives) override;
    void update(double learning_rate);
    // Testing functions
    void print_weights();
    void print_grad_weights();
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
weights::weights(int input_size, int output_size)
{
  this->input_size = input_size;
  this->output_size = output_size;
  this->w = new double[input_size * output_size];
  this->grad_w = new double[input_size * output_size];
  this->input_values = nullptr;
  this->pred = nullptr;
  for (int i = 0; i < input_size * output_size; i++)
  {
    this->w[i] = (rand() % 100)/100.0;
    this->grad_w[i] = 0;
  }
}

// Destructor
weights::~weights(){
  delete[] this->w;
  delete[] this->grad_w;
}

/* GETTERS */
// Get the values pointer
double *weights::values_pointer(){
  return this->w;
}

// Get the gradient pointer
double *weights::grad_pointer(){
  return this->grad_w;
}

/* METHODS */
// Operator to evaluate the output
double *weights::operator()(BackwardClass *in){
  this->input_values = in->values_pointer();
  this->pred = in;
  double *output = new double[output_size];
  for (int i = 0; i < this->output_size; i++){
    output[i] = 0;
    for (int j = 0; j < this->input_size; j++){
      output[i] += this->w[j * this->output_size + i] * this->input_values[j];
    }
  }
  return output;
}

/* BACKPROPAGATION FUNCTIONS */
// Zero the gradient
void weights::zero_grad(){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    this->grad_w[i] = 0.0;
  }
}

// Backward pass
void weights::backward(double *derivatives){
  double *prevGrad = new double[this->input_size];
  for (int i = 0; i <  this->input_size; i++){
    prevGrad[i] = 0;
    for (int j = 0; j < this->output_size; j++){
      prevGrad[i] += derivatives[j] * this->w[i * this->output_size + j];
      this->grad_w[i * this->output_size + j] += this->input_values[i] * derivatives[j];
    }
  }
  this->pred->backward(prevGrad);
  delete[] prevGrad;
}

// Update the weights
void weights::update(double learning_rate){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    this->w[i] -= learning_rate * this->grad_w[i];  // Fixed: subtract gradient for descent
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
  for (int i = 0; i < this->input_size * this->output_size; i++)
  {
    cout << this->grad_w[i] << " ";
  }
  cout << endl;
}

#endif