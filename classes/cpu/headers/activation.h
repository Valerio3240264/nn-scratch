#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../../enums.h"
#include "../../virtual_classes.h"

/*TODO
1: Create a function to evaluate the output of a whole batch.
2: Create a function to evaluate the gradient of a whole batch.
*/

/*
ACTIVATION FUNCTION CLASS DOCUMENTATION:
PURPOSE:
This class is used to store the values and gradients of the activation_function performed on the weighted sum of the previous layer.
It also stores the name of the activation function and the predecessor pointer to perform the backward pass on the whole neural network.
This class stores the values and when it is called it will apply the activation function to the values array.

Attributes:
- size: size of the values and gradients arrays
- value: pointer to the values array
- grad: pointer to the gradients array
- pred: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)
- function_name: name of the activation function

Constructors:
- activation_function(int size, float *value, Activation_name function_name, BackwardClass *pred): creates a new array for the values and gradients arrays and sets the predecessor to the passed pointer.

Methods:
- values_pointer(): returns the pointer to the values array.
- grad_pointer(): returns the pointer to the gradients array.
- operator()(): applies the activation function to the values array.
- zero_grad(): sets all the gradients to 0.
- backward(float *derivatives): accumulates the gradients and propagates them to the predecessor.
- print_value(): prints the values array.
- print_grad(): prints the gradients array.

*/

class activation : public ActivationClass {
  private:
    int size;
    float *value;
    float *grad;
    BackwardClass *pred;
    Activation_name function_name;

  public:
    // Constructors
    activation(int size, float *value, Activation_name function_name, BackwardClass *pred);
    
    // Destructor
    ~activation();
    
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float get_value(int index);
    float get_grad(int index);

    // Methods
    void operator()() override;
    
    // Backpropagation functions
    void zero_grad() override;
    void backward(float * derivatives) override;
    
    // Testing functions
    void print_value();
    void print_grad();
}; 

#endif