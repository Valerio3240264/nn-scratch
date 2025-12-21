#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "../../virtual_classes.h"

/*
CROSS-ENTROPY LOSS CLASS DOCUMENTATION:
PURPOSE:
Cross-Entropy loss function for multi-class classification tasks (after Softmax).
This is the standard loss function for classification problems.

Formula: L = -log(prediction[correct_class])
Gradient (when combined with Softmax): prediction - one_hot_target

This loss function is numerically stable and provides strong gradients for training.
It measures the KL divergence between the predicted and true probability distributions.

Attributes:
- pred: pointer to the predecessor (softmax layer)
- target: pointer to the target values (one-hot encoded)
- grad: pointer to the gradients
- loss_value: scalar loss value
- size: number of classes

Methods:
- operator()(double *target): forward pass with one-hot encoded target
- operator()(int target_index): forward pass with class index (converts to one-hot)
- operator()(): forward pass with stored target
- backward(): backward pass (simplified, assumes derivative = 1)
- backward(double *derivatives): backward pass with incoming derivatives
*/

class cross_entropy_loss : public LossClass {
  private:
    BackwardClass *pred;
    float *target;
    float *grad;
    float loss_value;
    int size;

  public:
    // Constructors
    cross_entropy_loss(BackwardClass *pred, int size);
    cross_entropy_loss(BackwardClass *pred, int size, float *target);
    
    // Destructor
    ~cross_entropy_loss() override;
    
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