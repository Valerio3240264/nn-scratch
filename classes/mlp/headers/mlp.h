#ifndef MLP_H
#define MLP_H

#include "../../enums.h"
#include "../../virtual_classes.h"
#include "../headers/layer.h"

/*TODO
1: Create functions to evaluate the output and gradient of a whole batch.
2: Optimize the batch operations using personalized cuda kernels.
*/

/*
MLP CLASS DOCUMENTATION:
PURPOSE:
This class is used to store the layers, input size, output size and activation functions of a multi-layer perceptron.
Supports different activation functions per layer and different loss functions (MSE, Cross-Entropy).

Architecture:
layer_0 -> layer_1 -> ... -> layer_n-1 -> [softmax (optional)] -> loss

Attributes:
- layers: pointer to the layers (Layer class)
- num_layers: number of layers
- input_size: size of the input
- output_size: size of the output
- activation_functions: array of activation functions (one per layer)
- loss_function: type of loss function to use
- softmax_layer: optional softmax layer (for classification with cross-entropy)
- has_softmax: flag indicating if softmax layer is present

Constructors:
- mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, 
      Activation_name *activation_functions, Loss_name loss_function, bool use_softmax):
  Creates a new mlp with the passed parameters. Each layer can have its own activation function.
  If use_softmax is true, adds a softmax layer before the loss (recommended for cross-entropy).
- ~mlp(): destructor to delete the layers and softmax layer.

Methods:
- operator()(input *in): evaluates the output of the mlp.
- compute_loss(double *target): computes the loss with target array.
- compute_loss(int target_index): computes the loss with target index (for classification).
- get_loss(): returns the loss value.
- update(double learning_rate): updates the weights using the computed gradients.
- zero_grad(): sets all the gradients to 0.
- print_weights(): prints the weights.
- print_grad_weights(): prints the gradients of the weights.

*/

class mlp{
  private:
    // Layers
    layer **layers;
    int num_layers;
    int input_size;
    int output_size;
    Activation_name *activation_functions;
    
    // Softmax
    bool has_softmax;
    SoftmaxClass *softmax_layer;
    
    // Loss function
    Loss_name loss_function;
    LossClass *loss_layer;
    float current_loss;
    
    // Cuda check
    bool use_cuda = false;

    void cuda_init(int *hidden_sizes);
    void cpu_init(int *hidden_sizes);

  public:
    // Constructor with activation functions per layer and loss function
    mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, Activation_name *activation_functions, Loss_name loss_function, bool use_softmax = false, bool use_cuda = false);
    
    // Simple constructor (all layers use same activation)
    mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, Activation_name activation_function, bool use_cuda = false);
    
    ~mlp();

    // Getters
    int get_prediction();
    float get_prediction_probability(int index);

    // Methods
    BackwardClass* operator()(BackwardClass *in);
    void compute_loss(float *target);
    void compute_loss(int target_index);
    float get_loss();
    void zero_loss();
    
    // Backpropagation functions
    void update(float learning_rate);
    void zero_grad();

    // Print functions
    void print_weights();
    void print_grad_weights();
    void print_loss();
    void print_last_layer_weights();
    void print_last_layer_grad_weights();
    void print_loss_gradients();
};

#endif