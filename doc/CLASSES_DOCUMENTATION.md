# Neural Network Classes Documentation

This documentation provides a comprehensive overview of all classes in the neural network implementation, following the structure and specifications detailed in each class header file.

## Architecture Overview

The neural network follows this computational flow:
```
Input -> Weights -> Activation_function -> Layer -> MLP -> [Softmax (optional)] -> Loss
```

**Detailed Layer Flow:**
```
Input -> [Layer 1: Weights -> Activation] -> [Layer 2: Weights -> Activation] -> ... -> [Layer N: Weights -> Activation] -> [Softmax (optional)] -> Loss (MSE or Cross-Entropy)
```

The backpropagation uses a graph-based approach where each component has a predecessor pointer for gradient flow. The gradient flows backward through the computational graph from the loss function to the input.

**Key Features:**
- Per-layer activation functions (RELU, TANH, LINEAR, SIGMOID, SOFTMAX)
- Multiple loss functions (MSE for regression, Cross-Entropy for classification)
- Optional softmax layer for classification tasks
- Xavier/Glorot weight initialization for improved convergence
- Modular design with clean separation of concerns

## Table of Contents

1. [Core Virtual Interface](#core-virtual-interface)
2. [Enumeration Types](#enumeration-types)
3. [Input Management](#input-management)
4. [Weight Management](#weight-management)
5. [Activation Functions](#activation-functions)
6. [Softmax Activation](#softmax-activation)
7. [Layer Implementation](#layer-implementation)
8. [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
9. [Loss Functions](#loss-functions)
   - [Base Loss Class](#base-loss-class)
   - [MSE Loss](#mse-loss)
   - [Cross-Entropy Loss](#cross-entropy-loss)
10. [TODO Items](#todo-items)

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

### Loss_name (enums.h)

**PURPOSE:**
Defines available loss function types for the neural network.

**Values:**
- `MSE`: Mean Squared Error (for regression tasks)
- `CROSS_ENTROPY`: Cross-Entropy loss (for classification tasks)

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
- `weights(int input_size, int output_size)`: creates a new array for the weights and gradients arrays and sets the predecessor to nullptr. Weights are initialized using Xavier/Glorot initialization (scaled by `sqrt(1/input_size)`) for better convergence.

**Methods:**
- `values_pointer()`: returns the pointer to the weights array
- `grad_pointer()`: returns the pointer to the gradients array
- `operator()(BackwardClass *in)`: performs the matrix multiplication between the weights and the input values
- `zero_grad()`: sets all the gradients to 0
- `backward(double *derivatives)`: accumulates the gradients and propagates them to the predecessor
- `update(double learning_rate)`: updates the weights using gradient descent (w = w - learning_rate * gradient)
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

## Softmax Activation

### softmax (softmax.h)

**PURPOSE:**
This class implements the softmax activation function, commonly used as the final layer in multi-class classification problems. It converts raw scores (logits) into probabilities that sum to 1. The softmax is typically paired with cross-entropy loss for optimal training of classification networks.

**Attributes:**
- `value`: pointer to the values array (probabilities after softmax)
- `grad`: pointer to the gradients array
- `size`: size of the values and gradients arrays
- `temperature`: temperature parameter for controlling the sharpness of the probability distribution (default: 1.0)
- `pred`: pointer to the predecessor (this pointer can be seen as an edge in the computational graph of the neural network)

**Constructors:**
- `softmax(int size, double *value, BackwardClass *pred)`: creates a new softmax layer with default temperature (1.0)
- `softmax(int size, double *value, double temperature, BackwardClass *pred)`: creates a new softmax layer with custom temperature

**Methods:**
- `values_pointer()`: returns the pointer to the values array
- `grad_pointer()`: returns the pointer to the gradients array
- `operator()()`: applies the softmax function to the values array with numerical stability (subtracts max value to prevent overflow)
- `zero_grad()`: no operation (gradients are handled in backward pass)
- `backward(double *derivatives)`: computes the Jacobian-vector product for softmax gradient
- `get_prediction()`: returns the index of the highest probability (predicted class)
- `get_prediction_probability(int index)`: returns the probability for a specific class
- `print_value()`: prints the probability values
- `print_grad()`: prints the gradients array

**Softmax Function Implementation:**
- Forward: `softmax(x_i) = exp((x_i - max(x)) / T) / sum(exp((x_j - max(x)) / T))`
  - Uses numerical stability trick (subtracting max) to prevent overflow
  - Temperature parameter T controls sharpness (lower T = sharper distribution)
- Backward: Uses Jacobian matrix multiplication: `grad_i = softmax_i * (derivative_i - dot(softmax, derivatives)) / T`

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
This class is used to store the layers, input size, output size and activation functions of a multi-layer perceptron. It supports different activation functions per layer and different loss functions (MSE, Cross-Entropy). The MLP can optionally include a softmax layer before the loss, which is recommended for classification tasks using cross-entropy loss.

**Architecture:**
```
layer_0 -> layer_1 -> ... -> layer_n-1 -> layer_n -> [softmax (optional)] -> loss
```

**Attributes:**
- `layers`: pointer array to the layers (Layer class)
- `num_layers`: number of layers
- `input_size`: size of the input
- `output_size`: size of the output
- `activation_functions`: array of activation functions (one per layer)
- `loss_function`: type of loss function to use (MSE or CROSS_ENTROPY)
- `has_softmax`: flag indicating if softmax layer is present
- `softmax_layer`: pointer to optional softmax layer (for classification)
- `softmax_values`: buffer for softmax output values
- `mse_loss_layer`: pointer to MSE loss layer (if using MSE)
- `ce_loss_layer`: pointer to cross-entropy loss layer (if using CROSS_ENTROPY)
- `current_loss`: accumulated loss value

**Constructors:**
- `mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, Activation_name *activation_functions, Loss_name loss_function, bool use_softmax = false)`: creates a new mlp with the passed parameters. Each layer can have its own activation function. If use_softmax is true, adds a softmax layer before the loss (recommended for cross-entropy).
- `mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, Activation_name activation_function)`: legacy constructor where all layers use the same activation function and MSE loss (for backward compatibility)
- `~mlp()`: destructor to delete the layers, softmax layer, and loss layers

**Methods:**
- `operator()(BackwardClass *in)`: evaluates the output of the mlp (forward pass through all layers)
- `compute_loss(double *target)`: computes the loss with target array and performs backward pass
- `compute_loss(int target_index)`: computes the loss with target class index (for classification) and performs backward pass
- `get_loss()`: returns the current accumulated loss value
- `zero_loss()`: resets the accumulated loss to 0
- `get_prediction()`: returns the predicted class index (requires softmax layer)
- `get_prediction_probability(int index)`: returns the probability for a specific class (requires softmax layer)
- `update(double learning_rate)`: updates the weights using the computed gradients
- `zero_grad()`: sets all the gradients to 0
- `print_weights()`: prints the weights of all layers
- `print_grad_weights()`: prints the gradients of the weights of all layers
- `print_loss()`: prints the current loss value

---

## Loss Functions

The neural network implementation provides multiple loss functions for different tasks. All loss functions inherit from `BackwardClass` and participate in the backpropagation graph.

### MSE Loss

#### mse_loss (mse_loss.h)

**PURPOSE:**
Mean Squared Error loss function specifically designed for regression tasks. Returns a scalar loss value (averaged over all outputs) rather than per-element losses. This is the recommended loss function for regression problems.

**Attributes:**
- `pred`: pointer to the predecessor (output layer)
- `target`: pointer to the target values
- `grad`: pointer to the gradients
- `loss_value`: scalar loss value
- `size`: number of outputs

**Constructors:**
- `mse_loss(BackwardClass *pred, int size)`: creates a new MSE loss layer without target (target set later)
- `mse_loss(BackwardClass *pred, int size, double *target)`: creates a new MSE loss layer with target

**Methods:**
- `operator()(double *target)`: sets target and computes loss
- `operator()()`: computes loss with stored target
- `backward()`: simplified backward pass (assumes derivative of loss w.r.t. itself is 1)
- `backward(double *derivatives)`: backward pass with incoming derivatives
- `zero_grad()`: sets all gradients to 0
- `values_pointer()`: returns pointer to scalar loss value
- `grad_pointer()`: returns pointer to gradients array
- `get_loss()`: returns the scalar loss value
- `print_loss()`: prints the loss value
- `print_grad()`: prints the gradients

**Formula:**
- Forward: `L = (1/n) * sum((prediction - target)²)`
- Backward: `dL/dprediction[i] = (2/n) * (prediction[i] - target[i])`

### Cross-Entropy Loss

#### cross_entropy_loss (cross_entropy_loss.h)

**PURPOSE:**
Cross-Entropy loss function for multi-class classification tasks. This is the standard loss function for classification problems and should be used with softmax activation. It provides numerically stable gradients and measures the KL divergence between predicted and true probability distributions.

**Attributes:**
- `pred`: pointer to the predecessor (typically softmax layer)
- `target`: pointer to the target values (one-hot encoded)
- `grad`: pointer to the gradients
- `loss_value`: scalar loss value
- `size`: number of classes

**Constructors:**
- `cross_entropy_loss(BackwardClass *pred, int size)`: creates a new cross-entropy loss layer without target
- `cross_entropy_loss(BackwardClass *pred, int size, double *target)`: creates a new cross-entropy loss layer with one-hot target

**Methods:**
- `operator()(double *target)`: sets one-hot encoded target and computes loss
- `operator()(int target_index)`: sets target using class index (converts to one-hot) and computes loss
- `operator()()`: computes loss with stored target
- `backward()`: simplified backward pass for softmax + cross-entropy combination
- `backward(double *derivatives)`: backward pass with incoming derivatives
- `zero_grad()`: sets all gradients to 0
- `values_pointer()`: returns pointer to scalar loss value
- `grad_pointer()`: returns pointer to gradients array
- `get_loss()`: returns the scalar loss value
- `print_loss()`: prints the loss value
- `print_grad()`: prints the gradients

**Formula:**
- Forward: `L = -sum(target * log(prediction + ε))` where ε = 1e-15 for numerical stability
- Backward (with softmax): `dL/dprediction[i] = prediction[i] - target[i]` (beautiful simplification!)

**Note:** When combined with softmax, the gradient simplifies to `prediction - target`, which provides stable and efficient training for classification tasks.

---

## TODO Items

The following TODO items have been identified across the codebase for future development:

### Input (input.h)
1. Add batch representation

### Activation Function (activation_function.h)
1. Create a function to evaluate the output of a whole batch
2. Create a function to evaluate the gradient of a whole batch

### Layer (layer.h)
1. Create functions to evaluate the output and gradient of a whole batch

### Loss Functions (mse_loss.h and cross_entropy_loss.h)
1. Write a backward function that does not need to know the derivatives value since it is the first step of the backward pass and the derivatives are known (✓ Partially completed in mse_loss and cross_entropy_loss)
2. Create a function to evaluate the loss of a whole batch
3. Create a function to evaluate the gradient of a whole batch
4. Optimize the batch operations using personalized cuda kernels

### Multi-Layer Perceptron (mlp.h)
1. Create functions to evaluate the output and gradient of a whole batch
2. Optimize the batch operations using personalized cuda kernels

### Weights (weights.h)
1. Create a function to evaluate to process a whole batch of data
2. Create a function to evaluate the gradient of a whole batch

---

## Usage Examples

### Example 1: Classification with Cross-Entropy Loss (Recommended for Classification)

```cpp
// Create an MLP for MNIST digit classification (784 inputs -> 10 outputs)
// Architecture: 784 -> 128 (RELU) -> 64 (RELU) -> 10 (LINEAR) -> Softmax -> Cross-Entropy
int hidden_sizes[] = {128, 64};
Activation_name activations[] = {RELU, RELU, LINEAR};
mlp network(784, 10, 3, hidden_sizes, activations, CROSS_ENTROPY, true);

// Create input data
input* data = new input(784);
// ... populate data with pixel values ...

// Forward pass
BackwardClass* output = network(data);

// Compute loss and backward pass (using class index)
int target_label = 7;  // The digit is 7
network.compute_loss(target_label);

// Update weights and reset gradients
network.update(0.01);  // learning rate = 0.01
network.zero_grad();
network.zero_loss();   // reset accumulated loss

// Get predictions
int predicted_class = network.get_prediction();
double confidence = network.get_prediction_probability(predicted_class);
double loss = network.get_loss();
```

### Example 2: Regression with MSE Loss

```cpp
// Create an MLP for regression (10 inputs -> 1 output)
int hidden_sizes[] = {64, 32};
Activation_name activations[] = {RELU, RELU, LINEAR};
mlp network(10, 1, 3, hidden_sizes, activations, MSE, false);

// Create input data
input* data = new input(10);
// ... populate data ...

// Forward pass
BackwardClass* output = network(data);

// Create target (regression target)
double target[] = {42.5};

// Compute loss and backward pass
network.compute_loss(target);

// Update weights
network.update(0.001);
network.zero_grad();
network.zero_loss();

double loss = network.get_loss();
```

### Example 3: Legacy Constructor (Backward Compatible)

```cpp
// All layers use the same activation function (RELU)
// Uses MSE loss by default
int hidden_sizes[] = {64, 32};
mlp network(784, 10, 2, hidden_sizes, RELU);

// Create input data
input* data = new input(784);
// ... populate data ...

// Forward pass
BackwardClass* output = network(data);

// Create target
double target[10] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};  // one-hot encoded

// Backward pass and update
network.compute_loss(target);
network.update(0.01);
network.zero_grad();
```

### Example 4: Training Loop for Classification

```cpp
// Setup network
int hidden_sizes[] = {128, 64};
Activation_name activations[] = {RELU, RELU, LINEAR};
mlp network(784, 10, 3, hidden_sizes, activations, CROSS_ENTROPY, true);

// Training loop
for (int epoch = 0; epoch < 10; epoch++) {
    network.zero_loss();
    
    for (int i = 0; i < num_samples; i++) {
        // Create input
        input* data = new input(784);
        // ... load sample i into data ...
        
        // Forward pass
        network(data);
        
        // Backward pass with target label
        int label = labels[i];
        network.compute_loss(label);
        
        // Update weights
        network.update(0.01);
        network.zero_grad();
        
        delete data;
    }
    
    double avg_loss = network.get_loss() / num_samples;
    cout << "Epoch " << epoch << ", Loss: " << avg_loss << endl;
}
```

---

This documentation strictly follows the documentation structure and content found in each class header file, preserving all the specific details, method signatures, and architectural decisions described in the original source code.
