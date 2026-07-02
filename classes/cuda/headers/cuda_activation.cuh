#ifndef CUDA_ACTIVATION_CUH
#define CUDA_ACTIVATION_CUH

#include <cstddef>
#include "../../virtual_classes.h"
#include "../../enums.h"

/*
CUDA ACTIVATION CLASS DOCUMENTATION
Purpose:
- Device-side activation node with owned value/gradient buffers.
- Supports TANH, RELU, and LINEAR kernels.

Current behavior:
- operator() applies activation in-place on d_values.
- backward() transforms d_grad into predecessor gradient and calls pred->backward().
- set_pred() rewires predecessor; caller is responsible for shape consistency.
*/

class cuda_activation: public ActivationClass{
  private:
    size_t size;
    size_t batch_size;
    float *d_values;
    float *d_grad;
    BackwardClass *pred;
    Activation_name function_name;

    // Check pred component matches dimensions
    void check_pred();

  public:

    // Constructor
    cuda_activation(
      size_t size,
      size_t batch_size,
      Activation_name function_name,
      BackwardClass *pred
    );
  
    // Destructor
    ~cuda_activation();
  
    // Getters
    Activation_name get_activation_fun() override;
    float *values_pointer() override;
    float *grad_pointer() override;
    size_t get_output_size() override;
    size_t get_batch_size() override;

    // Setters
    void set_pred(BackwardClass *pred);

    // Methods
    void operator()() override;

    // Backpropagation functions
    void zero_grad() override;
    void backward() override;
};

#endif
