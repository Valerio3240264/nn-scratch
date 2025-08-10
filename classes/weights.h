#ifndef WEIGHTS_H
#define WEIGHTS_H
#include "input.h"
#include "virtual_classes.h"

using namespace std;

class weights: public BackwardClass{
  private:
    double *w;
    double *grad_w;
    int input_size;
    int output_size;
    double *input_values;
    BackwardClass *pred;

  public:
    // Constructors
    weights(int input_size, int output_size);
    // Destructor
    ~weights();
    // Getters
    double *values_pointer() override;
    double *grad_pointer() override;
    // Methods
    double *operator()(BackwardClass * in);
    // Backpropagation functions
    void zero_grad() override;
    void backward(double *derivatives) override;
    void update(double learning_rate);
    // Testing functions
    void print_weights();
    void print_grad_weights();
};

/* CONSTRUCTOR AND DESTRUCTOR */
// Constructor
weights::weights(int input_size, int output_size)
{
  this->input_size = input_size;
  this->output_size = output_size;
  this->w = new double[input_size * output_size];
  this->grad_w = new double[input_size * output_size];
  this->input_values = nullptr;
  this->pred = nullptr;
  for (int i = 0; i < input_size * output_size; i++)
  {
    this->w[i] = (rand() % 100)/100.0;
    this->grad_w[i] = 0;
  }
}

// Destructor
weights::~weights(){
  delete[] this->w;
  delete[] this->grad_w;
}

/* GETTERS */
// Get the values pointer
double *weights::values_pointer(){
  return this->w;
}

// Get the gradient pointer
double *weights::grad_pointer(){
  return this->grad_w;
}

/* METHODS */
// Operator to evaluate the output
double *weights::operator()(BackwardClass *in){
  this->input_values = in->values_pointer();
  this->pred = in;
  double *output = new double[output_size];
  for (int i = 0; i < this->output_size; i++){
    output[i] = 0;
    for (int j = 0; j < this->input_size; j++){
      output[i] += this->w[j * this->output_size + i] * this->input_values[j];
    }
  }
  return output;
}

/* BACKPROPAGATION FUNCTIONS */
// Zero the gradient
void weights::zero_grad(){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    this->grad_w[i] = 0.0;
  }
}

// Backward pass
void weights::backward(double *derivatives){
  double *prevGrad = new double[this->input_size];
  for (int i = 0; i <  this->input_size; i++){
    prevGrad[i] = 0;
    for (int j = 0; j < this->output_size; j++){
      prevGrad[i] += derivatives[j] * this->w[i * this->output_size + j];
      this->grad_w[i * this->output_size + j] += this->input_values[i] * derivatives[j];
    }
  }
  this->pred->backward(prevGrad);
  delete[] prevGrad;
}

// Update the weights
void weights::update(double learning_rate){
  for (int i = 0; i < this->input_size * this->output_size; i++){
    this->w[i] += learning_rate * this->grad_w[i];
  }
}

/* TESTING FUNCTIONS */
// Print the weights
void weights::print_weights(){
  for (int i = 0; i < this->input_size * this->output_size; i++)
  {
    cout << this->w[i] << " ";
  }
  cout << endl;
}

// Print the gradient of the weights
void weights::print_grad_weights(){
  for (int i = 0; i < this->input_size * this->output_size; i++)
  {
    cout << this->grad_w[i] << " ";
  }
  cout << endl;
}

#endif