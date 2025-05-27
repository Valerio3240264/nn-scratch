# Neural Network from Scratch

This documentation describes a simple neural network implementation built from scratch in C++. The implementation includes the fundamental building blocks of neural networks:

- `Value`: A computational graph node that tracks gradients for automatic differentiation
- `Neuron`: A single neuron with weights, bias, and activation function
- `Layer`: A collection of neurons
- `MLP`: A multi-layer perceptron composed of multiple layers

## Table of Contents

1. [Value Class](value.md) - The automatic differentiation engine
2. [Neuron Class](neuron.md) - Individual neuron implementation
3. [Layer Class](layer.md) - Neural network layer implementation
4. [MLP Class](mlp.md) - Multi-layer perceptron implementation
5. [Usage Examples](examples.md) - Examples of using the neural network 