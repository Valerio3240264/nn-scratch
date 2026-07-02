#ifndef INPUT_H
#define INPUT_H

#include <cstddef>
#include "../../virtual_classes.h"

/*
INPUT CLASS DOCUMENTATION
Purpose:
- Entry node for model inputs in CPU mode.
- Exposes externally owned input values and an internally owned gradient buffer.

Ownership and layout:
- values: non-owning pointer set by set_values().
- grad: owned buffer with shape (batch_size, size), used for dL/d(input).
- size and batch_size: metadata used by downstream checks.

Behavior:
- backward() is a no-op because this is a graph leaf.
- zero_grad() clears the full batch gradient buffer.
*/

class input : public BackwardClass{
  private:
    float *values;
    float *grad;
    size_t size;
    size_t batch_size;

  public:
    
    // Constructors
    input(size_t size, size_t batch_size);

    // Destructor
    ~input();

    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    size_t get_output_size() override;
    size_t get_batch_size() override;
    
    // Setters
    void set_values(float *new_values);
    void zero_grad() override;

    // Backward 
    void backward() override;

    // Testing functions
    void print_values();
    void print_grad();
};

#endif