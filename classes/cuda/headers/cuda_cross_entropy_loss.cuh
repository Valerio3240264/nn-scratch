#ifndef CUDA_CROSS_ENTROPY_LOSS_CUH
#define CUDA_CROSS_ENTROPY_LOSS_CUH

#include "../../virtual_classes.h"

/*
CUDA CROSS-ENTROPY LOSS CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the cross_entropy_loss class but it is used to store the values and gradients in device memory.

Note: When using this class, we assume that each class that interacts with this class (in the raw layer) has memory allocated in device memory.
*/
/*
!!! IMPORTANT NOTE !!!
Currently, in this class we are expecting that the last activation function is a softmax activation function.
I'll work on this in the future to make it more flexible.
*/

class cuda_cross_entropy_loss : public LossClass {
  private:
    BackwardClass *pred;
    float *target;
    float *grad;
    float loss_value;
    float *d_loss_sum;
    int size;
    bool has_target;  // Track if we have a target
    bool owns_target; // Track if we own the target memory
  public:
    // Constructors
    cuda_cross_entropy_loss(BackwardClass *pred, int size);
    cuda_cross_entropy_loss(BackwardClass *pred, int size, float *target);

    // Destructor
    ~cuda_cross_entropy_loss();

    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float get_loss() override;

    // Methods
    void operator()(float *target) override;
    void operator()(int target_index) override;
    void operator()() override;
    void zero_grad() override;
    void backward(float *derivatives) override;
    void backward() override;
};

#endif