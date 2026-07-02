#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include <cstddef>
#include "../../virtual_classes.h"

/*
CROSS ENTROPY LOSS CLASS DOCUMENTATION
Purpose:
- CPU multi-class cross-entropy loss for batched predictions.
- Intended to be used with a softmax predecessor.

Current formula:
- loss = -sum(target * log(prediction + 1e-15)) / batch_size
- backward writes gradient w.r.t. softmax output:
  dL/ds = -(target / max(prediction, 1e-15)) / batch_size
  then calls pred->backward() for further propagation.

Target inputs:
- operator()(float* target): dense one-hot/soft labels.
- operator()(size_t* target_indices): class indices converted to one-hot.
*/

class cross_entropy_loss : public LossClass {
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
    cross_entropy_loss( size_t size, 
                        size_t batch_size,
                        BackwardClass *pred);
    
    // Destructor
    ~cross_entropy_loss() override;
    
    // Getters
    float *values_pointer() override;
    float get_loss() override;
    size_t get_output_size();
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