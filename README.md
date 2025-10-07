This is a personal project to implement and optimize a neural network from scratch using C++.
The goal of this project is to learn what does a neural network under the hood and try to optimize it as much as possible.

## Layer Architecture and Object Linking

### How Layers Work
Each layer in the neural network follows this architecture:
```
Input -> Weights -> Activation_function -> Output
```

A **layer** class stores:
- `in`: pointer to the input (Input class)
- `out`: pointer to the output (Activation_function class) 
- `W`: pointer to the weights (Weights class)
- `input_size`, `output_size`: dimensions
- `function_name`: activation function type

When a layer processes data via `operator()(input *in)`:
1. Takes an input object
2. Performs matrix multiplication through weights: `double *weights_output = (*this->W)(in)`
3. Creates activation_function object with the weighted output
4. Applies the activation function to get the final layer output

### How Objects Are Linked Together

The neural network uses a **graph-based backpropagation approach** where each component has a **predecessor pointer** (`pred`) that creates edges in the computational graph.

**Object Linking Flow:**
```
layer_0 -> layer_1 -> ... -> layer_n-1 -> layer_n -> Loss
```

**Linking Mechanism:**
- Each `input` object can be linked to a predecessor via `input(int size, BackwardClass *pred)`
- Each `activation_function` stores a predecessor pointer to enable backward pass
- Each `weights` object maintains a predecessor link for gradient flow
- The `loss` function connects to the final layer output

**Gradient Flow (Backward Pass):**
The predecessor pointers enable automatic gradient propagation:
1. Loss function computes gradients and calls `pred->backward(derivatives)`
2. Each layer propagates gradients to its predecessor
3. Chain continues until reaching the input layer

This creates a computational graph where "Layer_i will call Layer_i-1 activation_function and so on until the input layer."

## Current TODO List

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
