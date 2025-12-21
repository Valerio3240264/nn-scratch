#ifndef CUDA_WEIGHTS_CUH
#define CUDA_WEIGHTS_CUH

#include "../../virtual_classes.h"
#include "../headers/cuda_input.cuh"
/*
CUDA WEIGHTS CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the weights class but it is used to store the weights and gradients in device memory.

Note: When using this class, we assume that each class that interacts with this class (in the raw layer) has memory allocated in device memory.
*/

class cuda_weights: public WeightsClass{
  private:
    float *d_w;
    float *d_grad_w;
    float *d_b;
    float *d_grad_b;
    float *d_input_grad_buffer;
    int input_size;
    int output_size;
    float *d_input_values;
    BackwardClass *pred;

  public:

    // Constructor
    cuda_weights(int input_size, int output_size);
  
    // Destructor
    ~cuda_weights();
  
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float *bias_pointer();
    float *grad_bias_pointer();
  
    // Methods
    void backward(float *derivatives) override;
    void zero_grad() override;
    void operator()(BackwardClass *in, float *output_pointer) override;
    void update(float learning_rate);
    void print_weights() override;
    void print_grad_weights() override;
};

#endif