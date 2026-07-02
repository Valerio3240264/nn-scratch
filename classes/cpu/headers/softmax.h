#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cstddef>
#include "../../enums.h"
#include "../../virtual_classes.h"

/*
SOFTMAX CLASS DOCUMENTATION
Purpose:
- CPU softmax activation over each batch row.
- Commonly used as the final probability layer for classification.

Behavior:
- operator() copies logits from pred and applies temperature-scaled softmax
  with row-wise numerical stabilization.
- backward() computes Jacobian-vector product:
  dL/dz_j = s_j * (dL/ds_j - dot(s, dL/ds)) / temperature
  then propagates to pred->backward().
- get_predictions() returns argmax class index per batch row.

Memory:
- values and grad are owned buffers with shape (batch_size * size).
- pred is non-owning and must match size and batch_size.
*/

class softmax : public ActivationClass {
  private:
    float *values;
    float *grad;
    size_t size;
    size_t batch_size;
    float temperature;
    BackwardClass *pred;

    // Check pred component matches dimensions
    void check_pred();
    
  public:

    // Constructors
    softmax(size_t size, 
            size_t batch_size, 
            float temperature, 
            BackwardClass *pred);
    
    // Destructor
    ~softmax();
    
    // Getters
    Activation_name get_activation_fun() override;
    float *values_pointer() override;
    float *grad_pointer() override;
    float get_temperature();
    void get_predictions(size_t *predictions);
    size_t get_output_size() override;
    size_t get_batch_size() override;
    
    // Setters
    void set_pred(BackwardClass *pred);
    
    // Methods
    void backward() override;
    void zero_grad() override;
    void operator()() override;
    
    // Testing functions
    void print_value();
    void print_grad();
};

#endif 