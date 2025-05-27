#include "neuron.cpp"
#include <vector>
#include <string>
#include <sstream>

using namespace std;

class Layer{
  private:
    int in_size;
    int out_size;
    vector<Neuron*> neurons;
  public:
    Layer(int in_size, int out_size){
      this->in_size = in_size;
      this->out_size = out_size;
      for(int i = 0; i < out_size; i++){
        neurons.push_back(new Neuron(in_size));
      }
    }

    vector<Value*> operator()(vector<Value*> inputs){
      vector<Value*> outputs;
      for(int i = 0; i < out_size; i++){
        outputs.push_back(neurons[i]->operator()(inputs));
      }
      return outputs;
    }

    string paramsToString(){
      stringstream ss;
      for(int i = 0; i < out_size; i++){
        ss << neurons[i]->paramsToString();
      }
      return ss.str();
    }

    void zerograd(){
      for(int i = 0; i < out_size; i++){
        neurons[i]->zerograd();
      }
    }

    void backprop(double lr){
      for(int i = 0; i < out_size; i++){
        neurons[i]->backprop(lr);
      }
    }
};