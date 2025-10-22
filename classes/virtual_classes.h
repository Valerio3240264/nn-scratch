#ifndef VIRTUAL_CLASSES_H
#define VIRTUAL_CLASSES_H

#include <concepts>

/* BASE INTERFACE FOR ALL LAYERS */
class BackwardClass {
public:
    virtual ~BackwardClass() = default;
    virtual float* values_pointer() = 0;
    virtual float* grad_pointer() = 0;
    virtual void backward(float *derivatives) = 0;
    virtual void zero_grad() = 0;
};

#endif 