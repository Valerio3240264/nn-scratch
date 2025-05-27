# Value Class

The `Value` class is the foundation of the automatic differentiation system. It implements a computational graph where each node represents a value and tracks gradients for backpropagation.

## Overview

The `Value` class provides:

1. Forward computation of operations (addition, multiplication, etc.)
2. Backward propagation of gradients
3. Automatic differentiation for training neural networks

## Class Structure

```cpp
class Value {
private:
    double data;                // The actual value
    set<Value*> childs;         // Child nodes in computational graph
    char op;                    // Operation that created this value
    function<void()> _backward; // Function to compute gradients during backprop

public:
    double grad;                // Gradient value

    // Constructors
    Value(double data, char op, set<Value*> childs);
    Value(double data);
    
    // Destructor
    ~Value();
    
    // Getters
    double getData();
    double getGrad();
    
    // Operators for building computational graph
    Value* operator+(Value &other);
    Value* operator+(double other);
    Value* operator+=(Value &other);
    Value* operator+=(double other);
    Value* operator-(Value &other);
    Value* operator-(double other);
    Value* operator*(Value &other);
    Value* operator*(double other);
    Value* operator/(Value &other);
    Value* operator/(double other);
    Value* operator^(Value &other);
    Value* operator^(double other);
    
    // Activation functions
    Value* tanh();
    Value* relu();
    
    // Loss function
    Value* mse(double exp);
    
    // Backpropagation
    void topoSort(Value* node, vector<Value*>& topo, set<Value*>& visited, vector<Value*>& tovisit);
    void backward();
    void zeroGrad();
    void backprop(double lr);
    
    // Utilities
    string toString();
    void deleteGraph();
};
```

## Key Features

### Computational Graph

Each `Value` object can be a node in a computational graph, with connections to its child nodes through the `childs` set. Operations like addition and multiplication create new `Value` objects that reference their operands as children.

### Automatic Differentiation

When you call `backward()` on a `Value` object, it performs backpropagation through the entire computational graph:

1. First, it performs a topological sort of the graph using `topoSort()`
2. Then it sets the gradient of the root node to 1.0
3. Finally, it applies the gradient calculations in reverse topological order

### Operations

The `Value` class supports standard mathematical operations:

- Arithmetic: `+`, `-`, `*`, `/`, `^` (power)
- Activation functions: `tanh()`, `relu()`
- Loss function: `mse()` (Mean Squared Error)

Each operation creates a new `Value` object with a custom `_backward` function that implements the chain rule for that specific operation.

## Example Usage

```cpp
// Create some values
Value* a = new Value(2);
Value* b = new Value(3);

// Build a computational graph
Value* c = *a + *b;      // c = a + b = 5
Value* d = *c * *a;      // d = c * a = 10

// Compute gradients through backpropagation
d->backward();

// Get gradient values
cout << "a.grad = " << a->getGrad() << endl;  // Should be 5 (∂d/∂a)
cout << "b.grad = " << b->getGrad() << endl;  // Should be 2 (∂d/∂b)
```

## Implementation Details

### ReLU Activation

The implementation uses Leaky ReLU with a small alpha value (0.01) to prevent dead neurons:

```cpp
Value* relu() {
    Value* result = new Value(std::max(0.0, this->data), 'r', {this});
    result->_backward = [this, result]() {
        this->grad += (this->data > 0 ? 1 : LEAKY_RELU_ALPHA) * result->grad;
    };
    return result;
}
```

### Gradient Update

The `backprop()` method updates the value based on the gradient and learning rate:

```cpp
void backprop(double lr) {
    this->data -= this->grad * lr;
}
```

### Memory Management

The `deleteGraph()` method helps prevent memory leaks by deleting all nodes in the computation graph:

```cpp
void deleteGraph() {
    set<Value*> visited;
    vector<Value*> tovisit;
    tovisit.push_back(this);
    while(!tovisit.empty()) {
        Value* current = tovisit.back();
        tovisit.pop_back();
        if(visited.find(current) != visited.end()) continue;
        visited.insert(current);
        for(Value* child : current->childs) {
            tovisit.push_back(child);
        }
    }
    
    for(Value* node : visited) {
        if(node != this) {
            delete node;
        }
    }
}
``` 