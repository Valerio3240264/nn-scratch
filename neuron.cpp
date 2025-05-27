#include <vector>
#include "value.cpp"
#include <sstream>
#include <iostream>

using namespace std;

class Neuron{
  private:
    int in_size;
    vector<Value*> weights;
    Value* bias;

  public:
  Neuron(int in_size){
    this->in_size = in_size;
    for(int i = 0; i < in_size; i++){
      weights.push_back(new Value(static_cast<double>(rand()) / RAND_MAX * 2 - 1));
    }
    bias = new Value(static_cast<double>(rand()) / RAND_MAX * 2 - 1);
  }

  Value* operator()(vector<Value*> inputs){
    // Create input Value objects
    vector<Value*> input_vals;
    for(int i = 0; i < in_size; i++){
      input_vals.push_back(new Value(inputs[i]->getData()));
    }
    
    // Calculate weighted sum
    Value* sum = new Value(0.0);
    for(int i = 0; i < in_size; i++){
      Value* product = *weights[i] * *input_vals[i];
      sum = *sum + *product;
    }
    
    // Add bias and apply activation
    sum = *sum + *bias;
    return sum->relu();
  } 

  string paramsToString(){
    stringstream ss;
    ss << "Weights: ";
    for(int i = 0; i < in_size; i++){
      ss<<"Weight: " << weights[i]->getData() << " grad: "<< weights[i]->getGrad() << " ";
    }
    ss << "Bias: " << bias->getData() << " grad: " << bias->getGrad();
    return ss.str();
  }

  void zerograd(){
    for(int i = 0; i< this->in_size; i++){
      this->weights[i]->zeroGrad();
    }
    bias->zeroGrad();
  }

  void backprop(double lr){
    for(int i = 0; i< this->in_size; i++){
      this->weights[i]->backprop(lr);
    }
    bias->backprop(lr);
  }
};