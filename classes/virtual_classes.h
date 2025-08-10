#ifndef VIRTUAL_CLASSES_H
#define VIRTUAL_CLASSES_H

#include <concepts>

/* BASE INTERFACE FOR ALL LAYERS */
class BackwardClass {
public:
    virtual ~BackwardClass() = default;
    virtual double* values_pointer() = 0;
    virtual double* grad_pointer() = 0;
    virtual void backward(double *derivatives) = 0;
    virtual void zero_grad() = 0;
};

#endif 