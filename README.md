# Neural Network Framework from Scratch

## Project Overview

This project is an educational implementation of a neural network framework built from scratch in C++. The primary goal is to understand the fundamental components and mechanisms that power modern deep learning frameworks like PyTorch and TensorFlow.

## Purpose

Rather than using existing frameworks as black boxes, this project is an "original" implementation of a neural network framework.
The code architecture has been completely designed by me, I don't know if something similar exists.
I gained a deeper understanding of:
- How to manage complex computational graphs 
- How to write an efficient and correct forward propagation algorithm
- How to write an efficient and correct backpropagation algorithm
- How GPU acceleration can be implemented using CUDA

## What's Implemented

The framework includes:

- **Core Components**: Layers, weights, activation functions (ReLU, Sigmoid, Tanh, Linear), and loss functions (MSE, Cross-Entropy)
- **Network Architecture**: Multi-Layer Perceptron (MLP) that can be configured with any number of layers and sizes
- **Training Pipeline**: Complete forward pass, loss computation, backpropagation, and weight updates
- **Dual Implementation**: Both CPU and CUDA (GPU) versions for performance comparison
- **Real-World Testing**: Validated on the MNIST digit recognition dataset

This project is a learning exercise to bridge the gap between theoretical understanding of neural networks and their practical implementation in production frameworks and also to gain a deeper understanding and insights about how neural networks work internally.

## Resources used
- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Learning CUDA by optimizing softmax: A worklog](https://maharshi.bearblog.dev/optimizing-softmax-cuda/)
- [Learning CUDA by optimizing matrix-vector multiplication (SGEMV) for cuBLAS-like performance - A worklog](https://maharshi.bearblog.dev/optimizing-sgemv-cuda/)
- [Restrict keyword](https://en.wikipedia.org/wiki/Restrict)


A special thanks to chatgpt and claude for helping debugging the code.