#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../../enums.h"
#include "../../virtual_classes.h"

class softmax : public SoftmaxClass {
  private:
    float *value;
    float *grad;
    int size;
    float temperature;
    BackwardClass *pred;

  public:

    // Constructors
    softmax(int size, BackwardClass *pred);
    softmax(int size, float temperature, BackwardClass *pred);
    softmax(int size, float *value, BackwardClass *pred);
    softmax(int size, float *value, float temperature, BackwardClass *pred);
    
    // Destructor
    ~softmax();
    
    // Getters
    float *values_pointer() override;
    float *grad_pointer() override;
    int get_prediction() override;
    float get_prediction_probability(int index) override;
    
    // Setters
    void set_value(float *value) override;
    void copy_values(float *value) override;
    
    // Methods
    void backward(float *derivatives) override;
    void zero_grad() override;
    void operator()() override;
    
    // Testing functions
    void print_value();
    void print_grad();
};

#endif 