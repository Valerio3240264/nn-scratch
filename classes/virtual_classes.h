#include "enums.h"

#ifndef VIRTUAL_CLASSES_H
#define VIRTUAL_CLASSES_H

/* BASE INTERFACE FOR ALL LAYERS */
class BackwardClass {
	public:
		virtual ~BackwardClass() = default;
		virtual float* values_pointer() = 0;
		virtual float* grad_pointer() = 0;
		virtual void backward(float *derivatives) = 0;
		virtual void zero_grad() = 0;
};

/* BASE INTERFACE FOR ALL WEIGHTS classes */
class WeightsClass : public BackwardClass {
	private:
		virtual void init_weights(Activation_name function_name) = 0;

	public:
		virtual ~WeightsClass() = default;
		virtual void operator()(BackwardClass *in, float *output_pointer) = 0;
		virtual void update(float learning_rate) = 0;
		virtual void print_weights() = 0;
		virtual void print_grad_weights() = 0;
};

/* BASE INTERFACE FOR ALL ACTIVATION classes */
class ActivationClass : public BackwardClass {
	public:
		virtual ~ActivationClass() = default;
		virtual void operator()() = 0;
};

/* BASE INTERFACE FOR ALL SOFTMAX classes */
class SoftmaxClass : public ActivationClass {
	public:
		// Destructor
		virtual ~SoftmaxClass() = default ;
		// Getters
		virtual float* values_pointer() = 0 ;
		virtual float* grad_pointer() = 0 ;
		virtual int get_prediction() = 0;
		virtual float get_prediction_probability(int index) = 0;
		// Setters
		virtual void set_value(float *value) = 0;
		virtual void copy_values(float *value) = 0;
		// Methods
		virtual void backward(float *derivatives) = 0 ;
		virtual void zero_grad() = 0 ;
		virtual void operator()() = 0;
};

/* BASE INTERFACE FOR ALL LOSS FUNCTIONS */
class LossClass : public BackwardClass {
  public:
    // Destructor
    virtual ~LossClass() = default ;
    // Getters
    virtual float* values_pointer() = 0 ;
    virtual float* grad_pointer() = 0 ;
    virtual float get_loss() = 0;
    // Methods
    virtual void operator()(float *target) = 0;
    virtual void operator()(int target_index) = 0;
    virtual void operator()() = 0;
    virtual void zero_grad() = 0 ;
    virtual void backward(float *derivatives) = 0 ;
		virtual void backward() = 0;
};
#endif