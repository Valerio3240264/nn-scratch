#ifndef CUDA_WEIGHTS_CUH
#define CUDA_WEIGHTS_CUH

#include "../../virtual_classes.h"
#include "../headers/cuda_input.cuh"
#include "../../enums.h"
/*
CUDA WEIGHTS CLASS DOCUMENTATION
Purpose:
- Device affine module storing W, b and their gradients.
- Computes forward and backward with CUDA matmul/update kernels.

Implementation notes:
- All matrices use compact row-major storage with their logical dimensions.
- Matrix multiplication kernels use shared-memory tiles and bounds checks.
- Initialization policy matches CPU (Xavier for TANH/LINEAR, He for RELU).

*/

class cuda_weights: public WeightsClass{
  private:
    float *d_w;
    float *d_grad_w;
    float *d_b;
    float *d_grad_b;
    size_t input_size;
    size_t output_size;
    size_t batch_size;
    BackwardClass *pred;
    BackwardClass *next;

    // Initialization based on the activation function name
    void init_weights(Activation_name function_name) override;
    void check_pred();
    void check_next();

  public:

    // Constructor
    cuda_weights(size_t input_size,
                 size_t output_size,
                 size_t batch_size,
                 Activation_name function_name,
                 BackwardClass *pred,
                 BackwardClass *next);
  
    // Destructor
    ~cuda_weights();
  
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float *bias_pointer();
    float *grad_bias_pointer();
    size_t get_output_size() override;
    size_t get_batch_size() override;

    // Setters
    void set_pred(BackwardClass *pred);
    void set_next(BackwardClass *next);
  
    // Methods
    void backward() override;
    void zero_grad() override;
    void operator()() override;
    void update(float learning_rate) override;
    void print_weights() override;
    void print_grad_weights() override;
};

#endif
