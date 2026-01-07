#ifndef WEIGHTS_H
#define WEIGHTS_H
#include "input.h"
#include "../../virtual_classes.h"
#include "../../enums.h"


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

class weights: public WeightsClass{
  private:
    float *w;
    float *grad_w;
    float *b;
    float *grad_b;
    int input_size;
    int output_size;
    float *input_values;
    BackwardClass *pred;

    // Initialization based on the activation function name
    void init_weights(Activation_name function_name) override;

  public:
    // Constructors
    weights(int input_size, int output_size, Activation_name function_name);
    // Destructor
    ~weights();
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float *bias_pointer();
    float *grad_bias_pointer();
    // Methods
    void operator()(BackwardClass * in, float *output_pointer) override;
    // Backpropagation functions
    void zero_grad() override;
    void backward(float *derivatives) override;
    void update(float learning_rate);
    // Testing functions
    void print_weights() override;
    void print_grad_weights() override;
    void print_bias();
    void print_grad_bias();
};

#endif