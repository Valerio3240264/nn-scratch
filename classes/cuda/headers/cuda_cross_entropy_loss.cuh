#ifndef CUDA_CROSS_ENTROPY_LOSS_CUH
#define CUDA_CROSS_ENTROPY_LOSS_CUH

#include <cstddef>
#include "../../virtual_classes.h"

/*
CUDA CROSS ENTROPY LOSS CLASS DOCUMENTATION
Purpose:
- Device cross-entropy loss for classification outputs.
- Intended to receive softmax probabilities from predecessor.

Current behavior:
- operator()(float* target): copies dense target to device and computes loss.
- operator()(size_t* target_indices): builds a batched one-hot target.
- backward() computes -target / prediction, averaged over the batch, matching CPU.

Ownership:
- `target` can be internally allocated or externally provided (owns_target flag).

*/

class cuda_cross_entropy_loss : public LossClass {
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
    cuda_cross_entropy_loss(size_t size, size_t batch_size, BackwardClass *pred);
    cuda_cross_entropy_loss(size_t size, size_t batch_size, BackwardClass *pred, float *target);

    // Destructor
    ~cuda_cross_entropy_loss();

    // Getters
    float *values_pointer() override;
    float get_loss() override;

    // Methods
    void operator()(float *target) override;
    void operator()(size_t* target_indices) override;
    void backward() override;
};

#endif
