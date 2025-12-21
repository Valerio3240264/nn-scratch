#ifndef CUDA_SOFTMAX_CUH
#define CUDA_SOFTMAX_CUH

#include "../../virtual_classes.h"

/*
CUDA SOFTMAX CLASS DOCUMENTATION:
PURPOSE:
This class has the same purpose of the softmax class but it is used to store the values and gradients in device memory.

Note: When using this class, we assume that each class that interacts with this class (in the raw layer) has memory allocated in device memory.
*/

class cuda_softmax: public SoftmaxClass{
  private:
    float *d_value;
    float *d_grad;
    float *d_max;        // Persistent buffer for max value in forward pass (legacy - not used by new kernel)
    float *d_exp_sum;    // Persistent buffer for exp sum in forward pass (legacy - not used by new kernel)
    float *d_dot;        // Persistent buffer for dot product in backward pass (legacy - not used by new kernel)
    int size;
    float temperature;
    BackwardClass *pred;
  public:

    // Constructor
    cuda_softmax(int size, BackwardClass *pred) ;
    cuda_softmax(int size, float temperature, BackwardClass *pred) ;
    cuda_softmax(int size, float *value, BackwardClass *pred) ;
    cuda_softmax(int size, float *value, float temperature, BackwardClass *pred) ;
    
    // Destructor
    ~cuda_softmax() override;
    
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    int get_prediction() override;
    float get_prediction_probability(int index) override;
   
    // Setters
    void set_value(float *value) override;
    void copy_values(float *value) override;

    // Methods
    void backward(float *derivatives) override;
    void zero_grad() override;
    void operator()() override;
};

#endif