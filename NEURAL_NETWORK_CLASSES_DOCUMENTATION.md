# Neural Network Classes Documentation

This documentation provides a comprehensive overview of all classes in the neural network implementation, following the structure and specifications detailed in each class header file.

## Architecture Overview

The neural network follows this computational flow:
```
Input -> Weights -> Activation_function -> Layer -> MLP -> Loss
```

The backpropagation uses a graph-based approach where each component has a predecessor pointer for gradient flow.

## Table of Contents

1. [Core Virtual Interface](#core-virtual-interface)
2. [Enumeration Types](#enumeration-types)
3. [Input Management](#input-management)
4. [Weight Management](#weight-management)
5. [Activation Functions](#activation-functions)
6. [Layer Implementation](#layer-implementation)
7. [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
8. [Loss Function](#loss-function)
9. [TODO Items](#todo-items)

---

## Core Virtual Interface

### BackwardClass (virtual_classes.h)

**PURPOSE:**
Base interface for all layers that participate in the backpropagation algorithm. Provides a common interface for gradient computation and value management across all neural network components.

**Methods:**
- `virtual double* values_pointer() = 0`: Returns pointer to values array
- `virtual double* grad_pointer() = 0`: Returns pointer to gradients array  
- `virtual void backward(double *derivatives) = 0`: Performs backward pass with given derivatives
- `virtual void zero_grad() = 0`: Sets all gradients to zero

---

## Enumeration Types

### Activation_name (enums.h)

**PURPOSE:**
Defines available activation function types for the neural network.

**Values:**
- `RELU`: Rectified Linear Unit activation
- `SIGMOID`: Sigmoid activation function
- `TANH`: Hyperbolic tangent activation
- `SOFTMAX`: Softmax activation (for classification)
- `LINEAR`: Linear activation (no transformation)

---

## Input Management

### input (input.h)

**PURPOSE:**
This class is used to store an array of values and gradients that needs to be stored temporarily for the gradient evaluation on the neural network. The attribute pred will store the predecessor pointer and, in this way, call the backward method of the predecessor.

In the neural network it is used to link the layers together. Layer_i will call Layer_i-1 activation_function and so on until the input layer.

**Attributes:**
- `value`: pointer to the values array (this can be copied or create a new array, depends on the constructor used)
- `grad`: pointer to the gradients array
- `size`: size of the values and gradients arrays
- `pred`: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)

**Constructors:**
- `input(int size)`: creates a new array for the values and gradients arrays and sets the predecessor to nullptr (useful when you want to store the values of the input layer)
- `input(int size, double *value)`: creates a new array for the gradients array and sets the predecessor to nullptr (useful when you want to copy values from an external array)
- `input(int size, BackwardClass *pred)`: creates a new array for the values and gradients arrays and sets the predecessor to the passed pointer (useful when you want to pass values between layers)

**Methods:**
- `values_pointer()`: returns the pointer to the values array
- `grad_pointer()`: returns the pointer to the gradients array
- `zero_grad()`: sets all the gradients to 0
- `backward(double *derivatives)`: accumulates the gradients and propagates them to the predecessor
- `print_value()`: prints the values array
- `print_grad()`: prints the gradients array

---

## Weight Management

### weights (weights.h)

**PURPOSE:**
This class is used to store the weights of a layer and perform the matrix multiplication between the weights and the input values. It also stores the gradients of the weights and the input values to perform the backward pass on the whole neural network.

**Attributes:**
- `w`: pointer to the weights array
- `grad_w`: pointer to the gradients array
- `input_size`: size of the input values
- `output_size`: size of the output values
- `input_values`: pointer to the input values
- `pred`: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)

**Constructors:**
- `weights(int input_size, int output_size)`: creates a new array for the weights and gradients arrays and sets the predecessor to nullptr

**Methods:**
- `values_pointer()`: returns the pointer to the weights array
- `grad_pointer()`: returns the pointer to the gradients array
- `operator()(BackwardClass *in)`: performs the matrix multiplication between the weights and the input values
- `zero_grad()`: sets all the gradients to 0
- `backward(double *derivatives)`: accumulates the gradients and propagates them to the predecessor
- `update(double learning_rate)`: updates the weights using the computed gradients
- `print_weights()`: prints the weights array
- `print_grad_weights()`: prints the gradients array

---

## Activation Functions

### activation_function (activation_function.h)

**PURPOSE:**
This class is used to store the values and gradients of the activation_function performed on the weighted sum of the previous layer. It also stores the name of the activation function and the predecessor pointer to perform the backward pass on the whole neural network. This class stores the values and when it is called it will apply the activation function to the values array.

**Attributes:**
- `size`: size of the values and gradients arrays
- `value`: pointer to the values array
- `grad`: pointer to the gradients array
- `pred`: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)
- `function_name`: name of the activation function

**Constructors:**
- `activation_function(int size, double *value, Activation_name function_name, BackwardClass *pred)`: creates a new array for the values and gradients arrays and sets the predecessor to the passed pointer

**Methods:**
- `values_pointer()`: returns the pointer to the values array
- `grad_pointer()`: returns the pointer to the gradients array
- `operator()()`: applies the activation function to the values array
- `zero_grad()`: sets all the gradients to 0
- `backward(double *derivatives)`: accumulates the gradients and propagates them to the predecessor
- `print_value()`: prints the values array
- `print_grad()`: prints the gradients array

**Supported Activation Functions:**
- **TANH**: `f(x) = tanh(x)`, derivative: `f'(x) = 1 - f(x)²`
- **RELU**: `f(x) = max(0, x)`, derivative: `f'(x) = 1 if x > 0, else 0`
- **LINEAR**: `f(x) = x`, derivative: `f'(x) = 1`

---

## Layer Implementation

### layer (layer.h)

**PURPOSE:**
This class is used to store the input(Input class), output(Activation_function class), weights(Weights class), input size, output size and activation function name of a layer.

**Architecture:**
```
Input -> Weights -> Activation_function -> Output
```

**Attributes:**
- `in`: pointer to the input (Input class)
- `out`: pointer to the output (Activation_function class)
- `W`: pointer to the weights (Weights class)
- `input_size`: size of the input
- `output_size`: size of the output
- `function_name`: name of the activation function

**Constructors:**
- `layer(double input_size, double output_size, Activation_name activation_function)`: creates a new layer with the passed input size, output size and activation function name

**Methods:**
- `operator()(input *in)`: evaluates the output of the layer
- `zero_grad()`: sets all the gradients to 0
- `update(double learning_rate)`: updates the weights using the computed gradients
- `get_output()`: returns the output of the layer
- `print_weights()`: prints the weights
- `print_grad_weights()`: prints the gradients of the weights

---

## Multi-Layer Perceptron (MLP)

### mlp (mlp.h)

**PURPOSE:**
This class is used to store the layers, input size, output size and activation function name of a multi-layer perceptron.

**Architecture:**
```
layer_0 -> layer_1 -> ... -> layer_n-1 -> layer_n
```

**Attributes:**
- `layers`: pointer to the layers (Layer class)
- `num_layers`: number of layers
- `input_size`: size of the input
- `output_size`: size of the output
- `function_name`: name of the activation function

**Constructors:**
- `mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, Activation_name activation_function)`: creates a new mlp with the passed input size, output size, number of layers and hidden sizes
- `~mlp()`: destructor to delete the layers

**Methods:**
- `operator()(input *in)`: evaluates the output of the mlp
- `compute_loss(input *target)`: computes the loss of the mlp. It also computes the gradients of the whole neural network calling the backward pass on the loss that will be linked with the last layer and so on until the input layer
- `update(double learning_rate)`: updates the weights using the computed gradients
- `zero_grad()`: sets all the gradients to 0
- `print_weights()`: prints the weights
- `print_grad_weights()`: prints the gradients of the weights

---

## Loss Function

### loss (loss.h)

**PURPOSE:**
This class is used to store the loss value and the gradients of the loss function. It also stores a pointer to the target values and the predecessor pointer to perform the backward pass on the whole neural network.

**Attributes:**
- `pred`: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)
- `target`: pointer to the target values
- `grad`: pointer to the gradients
- `loss_value`: pointer to the loss values
- `size`: size of the loss_value and grad arrays

**Constructors:**
- `loss(BackwardClass *pred, int size)`: creates a new array for the gradients and loss value arrays and sets the predecessor to the passed pointer (The target pointer will be set when the operator() is called)
- `loss(BackwardClass *pred, int size, double *target)`: creates a new array for the gradients and loss value arrays and sets the predecessor to the passed pointer and the target to the passed pointer

**Methods:**
- `operator()(double *target)`: sets the target values and calculates the loss value
- `operator()()`: calculates the loss value
- `zero_grad()`: sets all the gradients to 0
- `backward(double *derivatives)`: accumulates the gradients and propagates them to the predecessor
- `print_loss()`: prints the loss value
- `print_grad()`: prints the gradients

**Loss Function Implementation:**
Currently implements **Mean Squared Error (MSE)**:
- Forward: `loss = (prediction - target)²`
- Backward: `gradient = 2 * (prediction - target) * derivatives`

---

## TODO Items

The following TODO items have been identified across the codebase for future development:

### Activation Function (activation_function.h)
1. Create a function to evaluate the output of a whole batch
2. Create a function to evaluate the gradient of a whole batch
3. Optimize the batch operations using personalized cuda kernels

### Layer (layer.h)
1. Create functions to evaluate the output and gradient of a whole batch

### Loss Function (loss.h)
1. Write a backward function that does not need to know the derivatives value since it is the first step of the backward pass and the derivatives are known
2. Create a function to evaluate the loss of a whole batch
3. Create a function to evaluate the gradient of a whole batch
4. Optimize the batch operations using personalized cuda kernels

### Multi-Layer Perceptron (mlp.h)
1. Create functions to evaluate the output and gradient of a whole batch
2. Optimize the batch operations using personalized cuda kernels

### Weights (weights.h)
1. Optimize the matrix multiplication using a personalized CUDA kernel
2. Optimize the gradient computation using a personalized CUDA kernel
3. Create a function to evaluate to process a whole batch of data (not only one single data point)
4. Create a function to evaluate the gradient of a whole batch
5. Optimize the batch operations using personalized cuda kernels

---

## Usage Example

```cpp
// Create an MLP with 2 hidden layers
int hidden_sizes[] = {64, 32};
mlp network(784, 10, 2, hidden_sizes, RELU);

// Create input data
input* data = new input(784);
// ... populate data ...

// Forward pass
input* output = network(data);

// Create target
input* target = new input(10);
// ... populate target ...

// Backward pass and update
network.compute_loss(target);
network.update(0.01);  // learning rate = 0.01
network.zero_grad();   // prepare for next iteration
```

This documentation strictly follows the documentation structure and content found in each class header file, preserving all the specific details, method signatures, and architectural decisions described in the original source code.
