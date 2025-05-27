#include "layer.cpp"
#include <vector>

using namespace std;

class MLP{
  private:
    vector<Layer> layers;
  public:
    MLP(vector<int> sizes){
      for(int i = 0; i<sizes.size()-1; i++){
        layers.push_back(Layer(sizes[i], sizes[i+1]));
      }
    }

    vector<Value*> operator()(vector<Value*> input){
      vector<Value*> output = input;
      for(int i = 0; i<layers.size(); i++){
        output = layers[i](output);
      }
      return output;
    }

    void backprop(double learning_rate){
      for(int i = layers.size()-1; i>=0; i--){
        layers[i].backprop(learning_rate);
      }
    }
    
    void zerograd(){
      for(int i = 0; i<layers.size(); i++){
        layers[i].zerograd();
      }
    }
    
    string paramsToString(){
      string params = "";
      for(int i = 0; i<layers.size(); i++){
        params += layers[i].paramsToString();
      }
      return params;
    }
};
