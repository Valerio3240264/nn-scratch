#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "../../virtual_classes.h"

/*
MSE LOSS CLASS DOCUMENTATION:
PURPOSE:
Mean Squared Error loss function for regression tasks.
Formula: L = (1/n) * sum((prediction - target)^2)
Gradient: dL/dprediction = 2 * (prediction - target)

Attributes:
- pred: pointer to the predecessor (output layer)
- target: pointer to the target values
- grad: pointer to the gradients
- loss_value: scalar loss value
- size: size of the output vector

Methods:
- operator()(double *target): forward pass with target array
- operator()(): forward pass with stored target
- backward(): backward pass (simplified, assumes derivative = 1)
- backward(double *derivatives): backward pass with incoming derivatives
*/

class mse_loss : public LossClass {
  private:
    BackwardClass *pred;
    float *target;
    float *grad;
    float loss_value;
    int size;

  public:
    // Constructors
    mse_loss(BackwardClass *pred, int size);
    mse_loss(BackwardClass *pred, int size, float *target);
    
    // Destructor
    ~mse_loss() override;
    
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
  
    // Testing functions
    void print_loss();
    void print_grad();
};

#endif