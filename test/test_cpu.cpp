#include "../classes/cpu/input.h"
#include "../classes/cpu/weights.h"
#include "../classes/cpu/activation_function.h"
#include "../classes/cpu/softmax.h"
#include "../classes/cpu/cross_entropy_loss.h"
#include "../classes/cpu/mse_loss.h"
#include "../classes/enums.h"
#include "../classes/virtual_classes.h"
#include <iostream>

using namespace std;
const int epochs = 40;
const double learning_rate = 0.1;
const double temperature = 1.0;

int main(){
  cout<<"Starting test..."<<endl;

  // Set target index
  int target_index = 9;
  
  // Create weights layer outside the loop so learning persists across iterations
  cout<<"Creating weights layer..."<<endl;
  weights W(11, 10);
  
  for(int i = 0; i < epochs; i++){
    cout<<"Iteration "<<i<<":"<<endl;
    
    // Zero gradients at the start of each iteration
    W.zero_grad();

    cout<<"Creating input layer..."<<endl;
    // Create input data for this iteration (each iteration gets its own copy)
    double* I = new double[11];
    for(int j = 0; j < 10; j++){
      I[j] = j;
    }
    I[10] = 1;
    
    // Create input layer (input destructor will delete I)
    input in(11, I);
    
    cout<<"Forward pass through weights..."<<endl;
    // Forward pass through weights
    double *weights_output = W(&in);
    
    cout<<"Creating and applying softmax..."<<endl;
    // Create and apply softmax
    softmax sm(10, weights_output, temperature, &W);
    sm();

    cout<<"Creating loss and performing forward pass..."<<endl;
    // Create loss and perform forward pass
    cross_entropy_loss ce_loss(&sm, 10);
    ce_loss(target_index);

    cout<<"Printing the prediction, target and loss..."<<endl;
    // Print the prediction, target and loss
    cout<<"Prediction: ";
    int prediction_index = 0;
    for(int j = 0; j < 10; j++){
      if(sm.values_pointer()[j] > sm.values_pointer()[prediction_index]){
        prediction_index = j;
      }
      cout<<sm.values_pointer()[j]<<" ";
    }
    cout<<endl;
    cout<<"Prediction index: "<<prediction_index<<endl;
    cout<<"Target index: "<<target_index<<endl;
    cout<<"Cross-Entropy Loss: "<<ce_loss.get_loss()<<endl;
    cout<<endl;

    // Perform backward pass
    cout<<"Performing backward pass..."<<endl;
    ce_loss.backward();

    cout<<"Updating the weights..."<<endl;
    // Update the weights
    W.update(learning_rate);
  }
  
  cout<<"End of test..."<<endl;
  
  return 0;
}