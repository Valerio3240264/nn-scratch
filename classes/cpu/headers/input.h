#ifndef INPUT_H
#define INPUT_H

#include "../../virtual_classes.h"

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
- input(int size, float *value): creates a new array for the gradients array and sets the predecessor to nullptr (usefull when you want to copy values from an external array).
- input(int size, BackwardClass *pred): creates a new array for the values and gradients arrays and sets the predecessor to the passed pointer (usefull when you want to pass values between layers).

Methods:
- values_pointer(): returns the pointer to the values array.
- grad_pointer(): returns the pointer to the gradients array
- zero_grad(): sets all the gradients to 0
- backward(float *derivatives): accumulates the gradients and propagates them to the predecessor
- print_value(): prints the values array
- print_grad(): prints the gradients array
*/

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

#endif