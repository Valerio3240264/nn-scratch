#ifndef CUDA_INPUT_CUH
#define CUDA_INPUT_CUH

#include "../../virtual_classes.h"

/*
CUDA INPUT CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the input class but it is used to store the values and gradients in device memory.

Note: When using this class, we assume that each class that interacts with this class (in the raw layer) has memory allocated in device memory.
*/

class cuda_input: public BackwardClass{
  private:
    float *d_value;
    float *d_grad;
    int size;
    BackwardClass *pred;

  public:

    // Constructors
    cuda_input(int size);
    cuda_input(int size, float *value);
    cuda_input(int size, BackwardClass *pred);

    // Destructor
    ~cuda_input();

    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
  
    // Methods
    void zero_grad() override;
    void backward(float *derivatives) override;
};

#endif