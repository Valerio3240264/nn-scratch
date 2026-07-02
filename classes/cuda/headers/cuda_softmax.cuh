#ifndef CUDA_SOFTMAX_CUH
#define CUDA_SOFTMAX_CUH

#include <cstddef>
#include "../../virtual_classes.h"
#include "../../enums.h"

/*
CUDA SOFTMAX CLASS DOCUMENTATION
Purpose:
- Device softmax activation with owned probability/gradient buffers.

Current behavior:
- operator() applies row-wise softmax from predecessor logits into d_value.
- backward() writes the Jacobian-vector product to the predecessor gradient.
- get_predictions() copies probabilities to host and returns argmax per row.
*/

class cuda_softmax: public ActivationClass{
  private:
    float *d_value;
    float *d_grad;
    size_t size;
    size_t batch_size;
    float temperature;
    BackwardClass *pred;

    void check_pred();
  public:

    // Constructor
    cuda_softmax(size_t size, size_t batch_size, float temperature, BackwardClass *pred);
    
    // Destructor
    ~cuda_softmax() override;
    
    // Getters
    Activation_name get_activation_fun() override;
    float *values_pointer() override;
    float *grad_pointer() override;
    size_t get_output_size() override;
    size_t get_batch_size() override;
    float get_temperature();
    void get_predictions(size_t *predictions);
   
    // Methods
    void backward() override;
    void zero_grad() override;
    void operator()() override;

    // Setters
    void set_pred(BackwardClass *pred);
};

#endif
