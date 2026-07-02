#include "enums.h"
#include <cstddef>

#ifndef VIRTUAL_CLASSES_H
#define VIRTUAL_CLASSES_H

/*
Core interface for graph nodes that expose:
- output values
- gradient buffer for dL/d(output)
- output shape metadata (feature size and batch size)
- backward propagation and gradient reset hooks
*/
class BackwardClass {
	public:
		virtual ~BackwardClass() = default;
		// Getters
		virtual float* values_pointer() = 0;
		virtual float* grad_pointer() = 0;
		virtual size_t get_output_size() = 0;
		virtual size_t get_batch_size() = 0;
		// Grads
		virtual void backward() = 0;
		virtual void zero_grad() = 0;
};

/*
Extension of BackwardClass for affine modules (weights/bias).
Implementations are expected to:
- run forward with operator() using linked pred/next nodes
- compute parameter and input gradients in backward()
- update trainable parameters in update()
*/
class WeightsClass : public BackwardClass {
	private:
		virtual void init_weights(Activation_name function_name) = 0;

	public:
		virtual ~WeightsClass() = default;
		virtual void operator()() = 0;
		virtual void update(float learning_rate) = 0;
		virtual void set_pred(BackwardClass *) = 0;
		virtual void set_next(BackwardClass *) = 0;
		virtual void print_weights() = 0;
		virtual void print_grad_weights() = 0;
};

/*
Extension of BackwardClass for activation modules.
operator() must transform the current value buffer in place.
*/
class ActivationClass : public BackwardClass {
	public:
		virtual ~ActivationClass() = default;
		virtual void operator()() = 0;
		virtual Activation_name get_activation_fun() = 0;
};

/*
Loss interface used as the training graph sink.
Loss implementations own the scalar loss state and write gradients to
their predecessor node before calling pred->backward().
*/
class LossClass{
  public:
    // Destructor
    virtual ~LossClass() = default ;
    // Getters
		virtual float *values_pointer() = 0;
    virtual float get_loss() = 0;
    // Methods
    virtual void operator()(float *target) = 0;
    virtual void operator()(size_t* target_indices) = 0;
		virtual void backward() = 0;
};
#endif