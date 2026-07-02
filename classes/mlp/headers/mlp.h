#ifndef MLP_H
#define MLP_H

#include <cstddef>

#include "../../enums.h"
#include "../../virtual_classes.h"
#include "../headers/layer.h"

/*
MLP CLASS DOCUMENTATION
Purpose:
- Orchestrates a stack of `layer` objects plus optional softmax and one loss node.
- Works in CPU or CUDA mode via polymorphic interfaces.

Topology:
- layers[0] ... layers[num_layers-1]
- optional softmax after final layer
- loss node fed by softmax (if present) or last layer output

Training workflow:
1) operator()(input_node) for forward pass
2) compute_loss(target or target_indices) to accumulate current_loss
3) backward() to propagate gradients through the graph
4) update(learning_rate) to apply parameter updates
5) zero_grad()/zero_loss() as needed between steps

Notes:
- current_loss accumulates across calls until zero_loss().
- get_predictions() requires softmax to be enabled.
*/

class mlp{
  private:
    // Layers
    layer **layers;
    size_t num_layers;
    size_t input_size;
    size_t output_size;
    size_t batch_size;
    Activation_name *activation_functions;
    
    // Softmax
    bool has_softmax;
    ActivationClass *softmax_layer;
    
    // Loss function
    Loss_name loss_function;
    LossClass *loss_layer;
    float current_loss;
    
    // Cuda check
    bool use_cuda = false;

    void cuda_init(size_t *hidden_sizes);
    void cpu_init(size_t *hidden_sizes);

  public:
    // Constructor with activation functions per layer and loss function
    mlp(size_t input_size,
        size_t output_size,
        size_t batch_size,
        size_t num_layers,
        size_t *hidden_sizes,
        Activation_name *activation_functions,
        Loss_name loss_function,
        bool use_softmax = false,
        bool use_cuda = false);
    
    // Simple constructor (all layers use same activation)
    mlp(size_t input_size,
        size_t output_size,
        size_t batch_size,
        size_t num_layers,
        size_t *hidden_sizes,
        Activation_name activation_function,
        bool use_cuda = false);
    
    ~mlp();

    // Getters
    void get_predictions(size_t *predictions);

    // Methods
    void operator()(BackwardClass *in);
    void compute_loss(float *target);
    void compute_loss(size_t *target_indeces);
    float get_loss();
    void zero_loss();
    
    // Backpropagation functions
    void update(float learning_rate);
    void backward();
    void zero_grad();

    // Print functions
    void print_weights();
    void print_grad_weights();
    void print_loss();
};

#endif