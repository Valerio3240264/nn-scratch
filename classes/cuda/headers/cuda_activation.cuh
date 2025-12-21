#ifndef CUDA_ACTIVATION_CUH
#define CUDA_ACTIVATION_CUH

#include "../../virtual_classes.h"
#include "../../enums.h"

/*
CUDA ACTIVATION CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the activation function class but it is used to store the values and gradients in device memory.

Note: When using this class, we assume that each class that interacts with this class (in the raw layer) has memory allocated in device memory.
*/

class cuda_activation: public ActivationClass{
  private:
    float *d_value;
    float *d_grad;
    int size;
    BackwardClass *pred;
    Activation_name function_name;
  public:

    // Constructor
    cuda_activation(int size, float *value, Activation_name function_name, BackwardClass *pred);
  
    // Destructor
    ~cuda_activation();
  
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
  
    // Methods
    void operator()() override;
    void zero_grad() override;
    void backward(float *derivatives) override;
};

#endif