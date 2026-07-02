#ifndef CUDA_MSE_LOSS_CUH
#define CUDA_MSE_LOSS_CUH

#include <cstddef>
#include "../../virtual_classes.h"

/*
CUDA MSE LOSS CLASS DOCUMENTATION
Purpose:
- Device MSE loss node that stores scalar loss and target tensor on GPU.

Current behavior:
- operator()(float* target): copies host target to device, computes loss.
- operator()(size_t* target_indices): builds a batched one-hot target.
- backward() computes the same normalized gradient as the CPU implementation.

Ownership:
- `target` may be owned by this class or externally provided (owns_target flag).

*/

class cuda_mse_loss : public LossClass {
  private:
    BackwardClass *pred;
    float *target;
    float loss_value;
    float *d_loss_sum;
    size_t size;
    bool has_target;  // Track if we have a target
    bool owns_target; // Track if we own the target memory
    size_t batch_size;

  public:
    // Constructors
    cuda_mse_loss(size_t size, size_t batch_size, BackwardClass *pred);
    cuda_mse_loss(size_t size, size_t batch_size, BackwardClass *pred, float *target);

    // Destructor
    ~cuda_mse_loss();

    // Getters
    float *values_pointer() override;
    float get_loss() override;

    // Methods
    void operator()(float *target) override;
    void operator()(size_t* target_indices) override;
    void backward() override;
};

#endif
