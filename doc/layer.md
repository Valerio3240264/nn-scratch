# Layer Class

The `Layer` class represents a layer of neurons in a neural network, where each neuron processes the same input but produces different outputs.

## Overview

A neural network layer contains multiple neurons that operate in parallel. The `Layer` class in this implementation:

1. Manages a collection of neurons
2. Processes inputs through all neurons simultaneously 
3. Provides methods for backpropagation and parameter updates

## Class Structure

```cpp
class Layer {
private:
    int in_size;               // Number of inputs to each neuron
    int out_size;              // Number of neurons (outputs)
    vector<Neuron*> neurons;   // Collection of neurons

public:
    // Constructor
    Layer(int in_size, int out_size);
    
    // Forward pass
    vector<Value*> operator()(vector<Value*> inputs);
    
    // Training methods
    void zerograd();
    void backprop(double lr);
    
    // Utility
    string paramsToString();
};
```

## Key Components

### Initialization

When a `Layer` is created, it initializes a collection of neurons, each with the same number of inputs:

```cpp
Layer(int in_size, int out_size) {
    this->in_size = in_size;
    this->out_size = out_size;
    for(int i = 0; i < out_size; i++) {
        neurons.push_back(new Neuron(in_size));
    }
}
```

### Forward Pass

The `operator()` method performs a forward pass through the layer:

```cpp
vector<Value*> operator()(vector<Value*> inputs) {
    vector<Value*> outputs;
    for(int i = 0; i < out_size; i++) {
        outputs.push_back(neurons[i]->operator()(inputs));
    }
    return outputs;
}
```

The process:
1. Takes a vector of input values
2. Passes the inputs through each neuron
3. Collects each neuron's output into a vector
4. Returns the vector of outputs

### Backpropagation

The `backprop` method updates weights and biases for all neurons in the layer:

```cpp
void backprop(double lr) {
    for(int i = 0; i < out_size; i++) {
        neurons[i]->backprop(lr);
    }
}
```

And `zerograd` resets gradients for all neurons:

```cpp
void zerograd() {
    for(int i = 0; i < out_size; i++) {
        neurons[i]->zerograd();
    }
}
```

## Example Usage

```cpp
// Create a layer with 3 inputs and 2 outputs (2 neurons)
Layer layer = Layer(3, 2);

// Create input values
vector<Value*> inputs;
inputs.push_back(new Value(1.0));
inputs.push_back(new Value(0.5));
inputs.push_back(new Value(2.0));

// Perform forward pass
vector<Value*> outputs = layer(inputs);
cout << "Output 1: " << outputs[0]->getData() << endl;
cout << "Output 2: " << outputs[1]->getData() << endl;

// Calculate loss (e.g., MSE with targets [1.0, 0.0])
Value* loss1 = outputs[0]->mse(1.0);
Value* loss2 = outputs[1]->mse(0.0);

// Backpropagate
loss1->backward();
loss2->backward();

// Update weights
layer.backprop(0.01);  // Learning rate 0.01

// Reset gradients for next iteration
layer.zerograd();
```

## Implementation Notes

- Each neuron in the layer is independent and processes the same inputs
- The layer outputs a vector with one value per neuron
- Training happens by calling `backprop()` after gradients are computed
- The `paramsToString()` method provides a way to visualize the parameters of all neurons in the layer

## Layer in Neural Networks

In the context of a neural network:
- An input layer is implicitly created by providing input values
- Hidden layers process intermediate computations
- The output layer produces the final predictions

Each layer transforms its input data into a representation that the next layer can use, with the goal of making the final output accurate for the given task. 