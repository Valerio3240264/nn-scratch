#ifndef WEIGHTS_H
#define WEIGHTS_H
#include <cstddef>
#include "input.h"
#include "../../virtual_classes.h"
#include "../../enums.h"


/*
WEIGHTS CLASS DOCUMENTATION
Purpose:
- CPU affine layer parameters: W and b.
- Computes forward pass Y = XW + b for batched input.
- Computes backward gradients for W, b, and predecessor input.

Initialization:
- TANH/SOFTMAX/LINEAR -> Xavier uniform scale sqrt(6 / (in + out))
- RELU                -> He uniform scale sqrt(2 / in)

Shapes:
- w, grad_w: (input_size, output_size)
- b, grad_b: (output_size)
- pred values/grad: (batch_size, input_size)
- next values/grad: (batch_size, output_size)

Notes:
- `pred` and `next` are required for forward/backward and validated by size.
- backward() accumulates parameter grads and writes dL/dX to pred->grad_pointer().
*/

class weights: public WeightsClass{
  private:
    float *w;
    float *grad_w;
    float *b;
    float *grad_b;
    size_t input_size;
    size_t output_size;
    size_t batch_size;
    BackwardClass *pred;
    BackwardClass *next;

    // Initialization based on the activation function name
    void init_weights(Activation_name function_name) override;

    // Check pred component matches dimensions
    void check_pred();

    // Check next component matches dimensions
    void check_next();

  public:
    // Constructors
    weights(size_t input_size, 
            size_t output_size, 
            size_t batch_size, 
            Activation_name function_name,
            BackwardClass *pred,
            BackwardClass *next);
    
    // Destructor
    ~weights();
    
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    float *bias_pointer();
    float *grad_bias_pointer();
    size_t get_input_size();
    size_t get_output_size() override;
    size_t get_batch_size() override;

    // Setters
    void set_pred(BackwardClass *pred) override;
    void set_next(BackwardClass *next) override;

    // Methods
    void operator()() override;
    
    // Backpropagation functions
    void zero_grad() override;
    void backward() override;
    void update(float learning_rate);
    
    // Testing functions
    void print_weights() override;
    void print_grad_weights() override;
    void print_bias();
    void print_grad_bias();
};

#endif
