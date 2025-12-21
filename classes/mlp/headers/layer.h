#ifndef LAYER_H
#define LAYER_H

#include "../../virtual_classes.h"
#include "../../enums.h"

/*TODO
1: Create functions to evaluate the output and gradient of a whole batch.
*/

/*
LAYER CLASS DOCUMENTATION:
PURPOSE:
This class is used to store the input(Input class), output(Activation_function class), weights(Weights class), input size, output size and activation function name of a layer.

Architecture:
Input -> Weights -> Activation_function -> Output

Attributes:
- in: pointer to the input (Input class)
- out: pointer to the output (Activation_function class)
- W: pointer to the weights (Weights class)
- input_size: size of the input
- output_size: size of the output
- function_name: name of the activation function

Constructors:
- layer(double input_size, double output_size, Activation_name activation_function): creates a new layer with the passed input size, output size and activation function name.

Methods:
- operator()(input *in): evaluates the output of the layer.
- zero_grad(): sets all the gradients to 0.
- update(double learning_rate): updates the weights using the computed gradients.
- get_output(): returns the output of the layer.
- print_weights(): prints the weights.
- print_grad_weights(): prints the gradients of the weights.

*/

using namespace std;

class layer{
  private:
    BackwardClass *in;
    ActivationClass *out;
    WeightsClass *W;
    int input_size;
    int output_size;
    Activation_name function_name;
    bool use_cuda;

  public:
    layer(int input_size, int output_size, Activation_name function_name, bool use_cuda = false);
    ~layer();

    // Methods
    void operator()(BackwardClass *in);

    // Backpropagation functions
    void zero_grad();
    void update(float learning_rate);

    // Getters
    BackwardClass *get_output();

    // Print functions
    void print_weights();
    void print_grad_weights();
};

#endif