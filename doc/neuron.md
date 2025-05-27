# Neuron Class

The `Neuron` class implements a single artificial neuron, which is the basic building block of neural networks.

## Overview

A neuron receives multiple inputs, computes a weighted sum with learned weights, adds a bias term, and applies an activation function. The `Neuron` class in this implementation:

1. Maintains weights and bias as `Value` objects
2. Computes forward passes (prediction)
3. Supports gradient-based learning through backpropagation

## Class Structure

```cpp
class Neuron {
private:
    int in_size;             // Number of inputs
    vector<Value*> weights;  // Weight for each input
    Value* bias;             // Bias term

public:
    // Constructor
    Neuron(int in_size);
    
    // Forward pass
    Value* operator()(vector<Value*> inputs);
    
    // Training methods
    void zerograd();
    void backprop(double lr);
    
    // Utility
    string paramsToString();
};
```

## Key Components

### Initialization

When a `Neuron` is created, it initializes weights and bias with random values:

```cpp
Neuron(int in_size) {
    this->in_size = in_size;
    for(int i = 0; i < in_size; i++) {
        weights.push_back(new Value(static_cast<double>(rand()) / RAND_MAX * 2 - 1));
    }
    bias = new Value(static_cast<double>(rand()) / RAND_MAX * 2 - 1);
}
```

This initialization uses a uniform distribution between -1 and 1.

### Forward Pass

The `operator()` method performs a forward pass through the neuron:

```cpp
Value* operator()(vector<Value*> inputs) {
    // Create input Value objects
    vector<Value*> input_vals;
    for(int i = 0; i < in_size; i++) {
        input_vals.push_back(new Value(inputs[i]->getData()));
    }
    
    // Calculate weighted sum
    Value* sum = new Value(0.0);
    for(int i = 0; i < in_size; i++) {
        Value* product = *weights[i] * *input_vals[i];
        sum = *sum + *product;
    }
    
    // Add bias and apply activation
    sum = *sum + *bias;
    return sum->relu();
}
```

The process:
1. Converts input values to `Value` objects
2. Computes the weighted sum (Σ weight_i * input_i)
3. Adds the bias term
4. Applies the ReLU activation function

### Backpropagation

The `backprop` method updates weights and bias based on gradients:

```cpp
void backprop(double lr) {
    for(int i = 0; i < this->in_size; i++) {
        this->weights[i]->backprop(lr);
    }
    bias->backprop(lr);
}
```

And `zerograd` resets gradients to zero:

```cpp
void zerograd() {
    for(int i = 0; i < this->in_size; i++) {
        this->weights[i]->zeroGrad();
    }
    bias->zeroGrad();
}
```

## Example Usage

```cpp
// Create a neuron with 3 inputs
Neuron n = Neuron(3);

// Create input values
vector<Value*> inputs;
inputs.push_back(new Value(1.0));
inputs.push_back(new Value(0.5));
inputs.push_back(new Value(2.0));

// Perform forward pass
Value* output = n(inputs);
cout << "Output: " << output->getData() << endl;

// Calculate loss (e.g., MSE with target 1.0)
Value* loss = output->mse(1.0);

// Backpropagate
loss->backward();

// Update weights
n.backprop(0.01);  // Learning rate 0.01

// Reset gradients for next iteration
n.zerograd();
```

## Implementation Notes

- The neuron uses ReLU activation (`relu()`) rather than tanh or sigmoid
- Training follows a standard gradient descent approach with a fixed learning rate
- The `paramsToString()` method provides a way to visualize the current weights, biases, and their gradients 