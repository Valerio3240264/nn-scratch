# MLP Class (Multi-Layer Perceptron)

The `MLP` class implements a complete neural network composed of multiple layers of neurons.

## Overview

A Multi-Layer Perceptron (MLP) is a type of feedforward neural network consisting of multiple layers:
- An input layer (implicitly defined by the input data)
- One or more hidden layers
- An output layer

The `MLP` class in this implementation:

1. Manages a sequence of `Layer` objects
2. Connects layers to form a complete neural network
3. Provides methods for forward and backward passes

## Class Structure

```cpp
class MLP {
private:
    vector<Layer> layers;   // Collection of layers

public:
    // Constructor
    MLP(vector<int> sizes);
    
    // Forward pass
    vector<Value*> operator()(vector<Value*> input);
    
    // Training methods
    void backprop(double learning_rate);
    void zerograd();
    
    // Utility
    string paramsToString();
};
```

## Key Components

### Initialization

When an `MLP` is created, it initializes a sequence of layers based on the specified sizes:

```cpp
MLP(vector<int> sizes) {
    for(int i = 0; i < sizes.size()-1; i++) {
        layers.push_back(Layer(sizes[i], sizes[i+1]));
    }
}
```

The `sizes` vector specifies the number of neurons in each layer:
- `sizes[0]` is the number of inputs
- `sizes[1]` to `sizes[n-1]` are the number of neurons in each hidden layer
- `sizes[n]` is the number of outputs

For example, `MLP({3, 4, 2})` creates a network with 3 inputs, 1 hidden layer with 4 neurons, and 2 outputs.

### Forward Pass

The `operator()` method performs a forward pass through the entire network:

```cpp
vector<Value*> operator()(vector<Value*> input) {
    vector<Value*> output = input;
    for(int i = 0; i < layers.size(); i++) {
        output = layers[i](output);
    }
    return output;
}
```

The process:
1. Takes a vector of input values
2. Passes the inputs through the first layer
3. Passes the outputs of each layer as inputs to the next layer
4. Returns the outputs of the final layer

### Backpropagation

The `backprop` method updates weights in all layers, starting from the output layer:

```cpp
void backprop(double learning_rate) {
    for(int i = layers.size()-1; i >= 0; i--) {
        layers[i].backprop(learning_rate);
    }
}
```

And `zerograd` resets gradients for all layers:

```cpp
void zerograd() {
    for(int i = 0; i < layers.size(); i++) {
        layers[i].zerograd();
    }
}
```

## Example Usage

```cpp
// Create an MLP with 3 inputs, a hidden layer of 4 neurons, and 2 outputs
MLP mlp = MLP({3, 4, 2});

// Create input values
vector<Value*> inputs;
inputs.push_back(new Value(1.0));
inputs.push_back(new Value(0.5));
inputs.push_back(new Value(2.0));

// Perform forward pass
vector<Value*> outputs = mlp(inputs);
cout << "Output 1: " << outputs[0]->getData() << endl;
cout << "Output 2: " << outputs[1]->getData() << endl;

// Calculate loss (e.g., MSE with targets [1.0, 0.0])
Value* loss1 = outputs[0]->mse(1.0);
Value* loss2 = outputs[1]->mse(0.0);

// Backpropagate
loss1->backward();
loss2->backward();

// Update weights
mlp.backprop(0.01);  // Learning rate 0.01

// Reset gradients for next iteration
mlp.zerograd();
```

## Training Loop Example

Here's a typical training loop for an MLP:

```cpp
// Create an MLP with architecture [3, 4, 1] (3 inputs, 1 hidden layer, 1 output)
MLP mlp = MLP({3, 4, 1});

// Training data
vector<Value*> input = /* your input data */;
double target = /* your target value */;

// Training loop
for(int epoch = 0; epoch < 1000; epoch++) {
    // Forward pass
    vector<Value*> output = mlp(input);
    
    // Compute loss
    Value* loss = output[0]->mse(target);
    
    // Backpropagation
    loss->backward();
    
    // Update weights
    mlp.backprop(0.01);
    
    // Reset gradients
    mlp.zerograd();
    
    // Print progress (optional)
    if(epoch % 100 == 0) {
        cout << "Epoch " << epoch << ", Loss: " << loss->getData() 
             << ", Prediction: " << output[0]->getData() << endl;
    }
}
```

## Implementation Notes

- The MLP follows a simple feed-forward architecture
- Each layer uses ReLU activation (through the `Neuron` implementation)
- The network uses MSE loss and simple gradient descent for optimization
- Backpropagation updates weights in reverse order (output to input) to correctly propagate gradients
- The implementation does not include features like dropout, batch normalization, or momentum

## Applications

MLPs can be used for various tasks including:
- Regression (predicting continuous values)
- Classification (predicting categories)
- Function approximation
- Pattern recognition

The simplicity of MLPs makes them a good starting point for understanding neural networks before moving to more complex architectures. 