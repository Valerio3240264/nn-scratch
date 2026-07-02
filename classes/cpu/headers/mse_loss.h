#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include <cstddef>
#include "../../virtual_classes.h"

/*
MSE LOSS CLASS DOCUMENTATION
Purpose:
- CPU mean squared error loss over batched outputs.
- Stores target and scalar loss, and writes gradients to predecessor.

Current formula:
- loss = sum((prediction - target)^2) / size
  (normalized by feature size, not by batch size)
- backward writes dL/dy = (2 / size) * (prediction - target)

Target inputs:
- operator()(float* target): expects dense target values for all batch elements.
- operator()(size_t* target_indices): one-hot encodes class indices into target.
*/

class mse_loss : public LossClass {
  private:
    BackwardClass *pred;
    float *target;
    float loss_value;
    size_t size;
    size_t batch_size;

    // Check pred component matches dimensions
    void check_pred();

  public:
    // Constructors
    mse_loss( size_t size, 
              size_t batch_size, 
              BackwardClass *pred);
    
    // Destructor
    ~mse_loss() override;
    
    // Getters
    float *values_pointer() override;
    float get_loss() override;
    size_t get_size();
    size_t get_batch_size();

    // Setters
    void set_pred(BackwardClass *pred);
    
    // Methods
    void operator()(float *target) override;
    void operator()(size_t* target_indices) override;
    void backward() override;
  
    // Testing functions
    void print_loss();
};

#endif