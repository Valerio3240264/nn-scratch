#ifndef LAYER_H
#define LAYER_H

#include "../../virtual_classes.h"
#include "../../enums.h"

/*
LAYER CLASS DOCUMENTATION
Purpose:
- Wraps one affine module (`W`) and one activation module (`out`).
- Supports CPU and CUDA implementations behind shared virtual interfaces.

Execution model:
- Forward: W->operator() then out->operator()
- Backward: initiated from loss; gradients propagate through out -> W -> previous node
- zero_grad() and update() delegate to W

Connectivity:
- set_input() rewires the predecessor of W at runtime.
- get_output() exposes the activation node as the layer output.
*/

using namespace std;

class layer{
  private:
    ActivationClass *out;
    WeightsClass *W;
    size_t input_size;
    size_t output_size;
    size_t batch_size;
    bool use_cuda;

  public:
    layer(size_t input_size, 
          size_t output_size, 
          size_t batch_size,
          Activation_name function_name,
          BackwardClass *input = nullptr,
          bool use_cuda = false);
    ~layer();

    // Methods
    void operator()();

    // Backpropagation functions
    void zero_grad();
    void update(float learning_rate);

    // Getters
    Activation_name get_function();
    BackwardClass *get_output();

    // Setter
    void set_input(BackwardClass *in);

    // Print functions
    void print_weights();
    void print_grad_weights();
};

#endif