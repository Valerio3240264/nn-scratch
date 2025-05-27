#ifndef VALUE_CPP
#define VALUE_CPP

#include <iostream>
#include <set>
#include <vector>
#include <stdio.h>
#include <sstream>
#include <functional>
#include <cmath>

const double LEAKY_RELU_ALPHA = 0.01;

using namespace std;

class Value{
  private:
    double grad;
    double data;
    set<Value*> childs;
    char op;
    function<void()> _backward;

  public:
    // Constructor
    Value(double data, char op, set<Value*> childs){
      this->data = data;
      this->grad = 0;
      this->childs = childs;
      this->op = op;
      this->_backward = [](){};  // Default empty lambda
    }

    Value(double data){
      this->data = data;
      this->grad = 0;
      this->op = 'n';
      this->_backward = [](){};  // Default empty lambda
    }

    // Getter for data
    double getData() {
      return this->data;
    }

    // Getter for grad
    double getGrad(){
      return this->grad;
    }

    // setop
    void setOp(char op){
      this->op = op;
    }

    // Operator sum
    Value* operator+(Value &other){
      Value* childs[] = {this, &other};
      Value* result = new Value(this->data + other.data, '+', set<Value*>(childs, childs + 2));

      // Set the backward function using lambda
      result->_backward = [this, &other, result]() {
        this->grad += result->grad;
        other.grad += result->grad;
      };
      
      return result;
    }
    
    Value* operator+(double other){
      Value* childs[] = {this};
      Value* result = new Value(this->data + other, '+', set<Value*>(childs, childs + 1));
      
      // Set the backward function using lambda
      result->_backward = [this, result]() {
        this->grad += result->grad;
      };
  
      return result;
    }

    // Operator +=
    Value* operator+=(Value &other){
      Value* result = *this + other;
      this->data = result->getData();
      this->childs = result->childs;
      this->op = result->op;
      this->_backward = result->_backward;
      delete result; // Clean up temporary result
      return this;
    }

    Value* operator+=(double other){
      Value* result = *this + other;
      this->data = result->getData();
      this->childs = result->childs;
      this->op = result->op;
      this->_backward = result->_backward;
      delete result; // Clean up temporary result
      return this;
    }

    // Operator subtraction
    Value* operator-(Value &other){
      Value* childs[] = {this, &other};
      Value* result = new Value(this->data - other.data, '-', set<Value*>(childs, childs + 2));
      
      // Set the backward function using lambda
      result->_backward = [this, &other, result]() {
        this->grad += result->grad;
        other.grad -= result->grad;  // Subtract for right operand
      }; 
      
      return result;
    }
    
    Value* operator-(double other){
      Value* childs[] = {this};
      Value* result = new Value(this->data - other, '-', set<Value*>(childs, childs + 1));
      
      // Set the backward function using lambda
      result->_backward = [this, result]() {
        this->grad += result->grad;
      };
      
      return result;
    }

    // Operator multiplication
    Value* operator*(Value &other){
      Value* childs[] = {this, &other};
      Value* result = new Value(this->data * other.data, '*', set<Value*>(childs, childs + 2));
      
      // Set the backward function using lambda
      result->_backward = [this, &other, result]() {
        this->grad += other.data * result->grad;
        other.grad += this->data * result->grad;
      };
      
      return result;
    }
    
    Value* operator*(double other){
      Value* childs[] = {this};
      Value* result = new Value(this->data * other, '*', set<Value*>(childs, childs + 1));
      
      // Set the backward function using lambda
      result->_backward = [this, other, result]() {
        this->grad += other * result->grad;
      };
      
      return result;
    }

    // Operator division
    Value* operator/(Value &other){
      Value* childs[] = {this, &other};
      Value* result = new Value(this->data / other.data, '/', set<Value*>(childs, childs + 2));
      
      // Set the backward function using lambda
      result->_backward = [this, &other, result]() {
        this->grad += (1.0 / other.data) * result->grad;
        other.grad += (-this->data / (other.data * other.data)) * result->grad;
      };
      
      return result;
    }
    
    Value* operator/(double other){
      Value* childs[] = {this};
      Value* result = new Value(this->data / other, '/', set<Value*>(childs, childs + 1));
      
      // Set the backward function using lambda
      result->_backward = [this, other, result]() {
        this->grad += (1.0 / other) * result->grad;
      };
      
      return result;
    }
    
    // Operator power
    Value* operator^(Value &other){
      Value* childs[] = {this, &other};
      Value* result = new Value(pow(this->data, other.data), '^', set<Value*>(childs, childs + 2));
      
      // Set the backward function using lambda
      result->_backward = [this, &other, result]() {
        this->grad += other.data * pow(this->data, other.data - 1) * result->grad;
        other.grad += log(this->data) * pow(this->data, other.data) * result->grad;
      };
      
      return result;
    }

    Value* operator^(double other){
      Value* childs[] = {this};
      Value* result = new Value(pow(this->data, other), '^', set<Value*>(childs, childs + 1));

      // Set the backward function using lambda
      result->_backward = [this, other, result]() {
        this->grad += other * pow(this->data, other - 1) * result->grad;
      };
      
      return result;
    }
    
    // Operator tanh
    Value* tanh(){
      Value* result = new Value(std::tanh(this->data), 't', {this});
      
      // Set the backward function using lambda
      result->_backward = [this, result]() {
        this->grad += (1 - std::pow(std::tanh(this->data), 2)) * result->grad;
      };
      
      return result;
    }

    // Operator relu
    Value* relu(){
      Value* result = new Value(std::max(0.0, this->data), 'r', {this});
      // Set the backward function using lambda
      result->_backward = [this, result]() {
        this->grad += (this->data > 0 ? 1 : LEAKY_RELU_ALPHA) * result->grad;
      };
      
      return result;
    }

    // Operator MSE
    Value* mse(double exp){
      Value* childs[] = {this};
      Value* diff = new Value(this->data - exp, '-', set<Value*>(childs, childs + 1));
      Value* result = new Value(pow(diff->data, 2), '^', {diff});

      // Set the backward function using lambda
      result->_backward = [this, result, diff, exp]() {
        diff->grad += 2 * diff->data * result->grad;
      };
      
      diff->_backward = [this, diff]() {
        this->grad += diff->grad;
      };
      
      return result;
    }

    void topoSort(Value* node, vector<Value*>& topo, set<Value*>& visited, vector<Value*>& tovisit){
      if(visited.find(node) != visited.end()) return;
      visited.insert(node);
      for(Value* child : node->childs){
        topoSort(child, topo, visited, tovisit);
      }
      topo.push_back(node);
    }

    void backward(){
      // Build topological order
      vector<Value*> topo;
      set<Value*> visited;
      vector<Value*> tovisit;
      topoSort(this, topo, visited, tovisit);

      // Set initial gradient to 1.0
      this->grad = 1.0;
      
      // Backward pass - apply gradients in topological order
      for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
      }
    }

    void zeroGrad(){
      this->grad = 0;
    }

    void backprop(double lr){
      this->data -= this->grad*lr;
    }

    string toString(){
      std::ostringstream oss;
      oss << "Value(data=" << this->data << ", grad=" << this->grad << ", childs=" << this->childs.size() << ", op=" << this->op << ")";
      return oss.str();
    }

};

#endif // VALUE_CPP