#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cstddef>
#include "../../enums.h"
#include "../../virtual_classes.h"

/*
ACTIVATION CLASS DOCUMENTATION
Purpose:
- CPU activation node applied after an affine transform.
- Stores output values and dL/d(output) for a full batch.

Current implementation details:
- Supported functions: TANH, RELU, LINEAR.
- operator() applies the activation in-place on `value`.
- backward() transforms `grad` into dL/d(input) using current `value`,
  then propagates to pred->backward().

Memory and shape:
- value and grad are owned buffers of size (batch_size * size).
- pred is non-owning and must match size and batch_size.
*/

class activation : public ActivationClass {
  private:
    size_t size;
    size_t batch_size;
    float *value;
    float *grad;
    BackwardClass *pred;
    Activation_name function_name;

    // Check pred component matches dimensions
    void check_pred();

  public:
    // Constructors
    activation(
      size_t size,
      size_t batch_size,
      Activation_name function_name,
      BackwardClass *pred
    );
    
    // Destructor
    ~activation();
    
    // Getters
    Activation_name get_activation_fun() override;
    float *values_pointer() override;
    float *grad_pointer() override;
    float get_value(size_t index);
    float get_grad(size_t index);
    size_t get_output_size() override;
    size_t get_batch_size() override;

    // Setters
    void set_pred(BackwardClass *pred);

    // Methods
    void operator()() override;
    
    // Backpropagation functions
    void zero_grad() override;
    void backward() override;
    
    // Testing functions
    void print_value();
    void print_grad();
}; 

#endif