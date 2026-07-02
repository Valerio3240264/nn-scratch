#ifndef CUDA_INPUT_CUH
#define CUDA_INPUT_CUH

#include <cstddef>
#include "../../virtual_classes.h"

/*
CUDA INPUT CLASS DOCUMENTATION
Purpose:
- CUDA counterpart of `input` used as a graph leaf.
- Exposes a non-owning device pointer for values and owns device gradient memory.

Current behavior:
- d_values is set externally via set_values() (no allocation/copy performed here).
- d_grad is allocated and zeroed by the class.
- backward() is a no-op leaf operation.
*/

class cuda_input: public BackwardClass{
  private:
    float *d_values;
    float *d_grad;
    size_t size;
    size_t batch_size;

  public:

    // Constructors
    cuda_input(size_t size, size_t batch_size);

    // Destructor
    ~cuda_input();

    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    size_t get_output_size() override;
    size_t get_batch_size() override;

    // Setters
    void set_values(float *new_values);
    void zero_grad() override;
  
    // Methods
    void backward() override;
};

#endif
