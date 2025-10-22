#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "cpu/input.h"
#include "cpu/activation_function.h"
#include "cpu/softmax.h"
#include "cpu/mse_loss.h"
#include "cpu/cross_entropy_loss.h"
#include "enums.h"

/*TODO
1: Create functions to evaluate the output and gradient of a whole batch.
2: Optimize the batch operations using personalized cuda kernels.
*/

/*
MLP CLASS DOCUMENTATION:
PURPOSE:
This class is used to store the layers, input size, output size and activation functions of a multi-layer perceptron.
Supports different activation functions per layer and different loss functions (MSE, Cross-Entropy).

Architecture:
layer_0 -> layer_1 -> ... -> layer_n-1 -> [softmax (optional)] -> loss

Attributes:
- layers: pointer to the layers (Layer class)
- num_layers: number of layers
- input_size: size of the input
- output_size: size of the output
- activation_functions: array of activation functions (one per layer)
- loss_function: type of loss function to use
- softmax_layer: optional softmax layer (for classification with cross-entropy)
- has_softmax: flag indicating if softmax layer is present

Constructors:
- mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, 
      Activation_name *activation_functions, Loss_name loss_function, bool use_softmax):
  Creates a new mlp with the passed parameters. Each layer can have its own activation function.
  If use_softmax is true, adds a softmax layer before the loss (recommended for cross-entropy).
- ~mlp(): destructor to delete the layers and softmax layer.

Methods:
- operator()(input *in): evaluates the output of the mlp.
- compute_loss(double *target): computes the loss with target array.
- compute_loss(int target_index): computes the loss with target index (for classification).
- get_loss(): returns the loss value.
- update(double learning_rate): updates the weights using the computed gradients.
- zero_grad(): sets all the gradients to 0.
- print_weights(): prints the weights.
- print_grad_weights(): prints the gradients of the weights.

*/

using namespace std;

class mlp{
  private:
    layer **layers;
    int num_layers;
    int input_size;
    int output_size;
    Activation_name *activation_functions;
    Loss_name loss_function;
    bool has_softmax;
    softmax *softmax_layer;
    mse_loss *mse_loss_layer;
    cross_entropy_loss *ce_loss_layer;
    float current_loss;

  public:
    // Constructor with activation functions per layer and loss function
    mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, 
        Activation_name *activation_functions, Loss_name loss_function, bool use_softmax = false);
    
    // Simple constructor (all layers use same activation)
    mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, Activation_name activation_function);
    
    ~mlp();

    // Getters
    int get_prediction();
    float get_prediction_probability(int index);

    // Methods
    BackwardClass* operator()(BackwardClass *in);
    void compute_loss(float *target);
    void compute_loss(int target_index);
    float get_loss();
    void zero_loss();
    
    // Backpropagation functions
    void update(float learning_rate);
    void zero_grad();

    // Print functions
    void print_weights();
    void print_grad_weights();
    void print_loss();
};

/* CONSTRUCTOR AND DESTRUCTOR */
mlp::mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, 
         Activation_name *activation_functions, Loss_name loss_function, bool use_softmax){
  this->input_size = input_size;
  this->output_size = output_size;
  this->num_layers = num_layers;
  this->activation_functions = new Activation_name[num_layers];
  for(int i = 0; i < num_layers; i++){
    this->activation_functions[i] = activation_functions[i];
  }
  this->loss_function = loss_function;
  this->has_softmax = use_softmax;
  this->current_loss = 0.0f;
  this->layers = new layer*[num_layers]; 

  // Create layers
  if(num_layers > 1){
    this->layers[0] = new layer(input_size, hidden_sizes[0], activation_functions[0]);
    for(int i = 1; i < num_layers - 1; i++){
      this->layers[i] = new layer(hidden_sizes[i-1], hidden_sizes[i], activation_functions[i]);
    }
    this->layers[num_layers - 1] = new layer(hidden_sizes[num_layers - 2], output_size, activation_functions[num_layers - 1]);
  }
  else if(num_layers == 1){
    this->layers[0] = new layer(input_size, output_size, activation_functions[0]);
  }
  else{
    cout<<"Error: num_layers must be greater than 0"<<endl;
    exit(1);
  }
  
  if(use_softmax){
    BackwardClass *last_layer_act = layers[num_layers - 1]->get_output();
    this->softmax_layer = new softmax(output_size, last_layer_act);
  } else {
    this->softmax_layer = nullptr;
  }
  
  BackwardClass *loss_predecessor = use_softmax ? 
    (BackwardClass*)this->softmax_layer : 
    (BackwardClass*)layers[num_layers - 1]->get_output();
  
  if(loss_function == MSE){
    this->mse_loss_layer = new mse_loss(loss_predecessor, output_size);
    this->ce_loss_layer = nullptr;
  } else {
    this->ce_loss_layer = new cross_entropy_loss(loss_predecessor, output_size);
    this->mse_loss_layer = nullptr;
  }
}

// Legacy constructor (all layers use same activation)
mlp::mlp(int input_size, int output_size, int num_layers, int *hidden_sizes, Activation_name function_name){
  this->input_size = input_size;
  this->output_size = output_size;
  this->num_layers = num_layers;
  this->activation_functions = new Activation_name[num_layers];
  for(int i = 0; i < num_layers; i++){
    this->activation_functions[i] = function_name;
  }
  this->loss_function = MSE;
  this->has_softmax = false;
  this->current_loss = 0.0f;
  this->layers = new layer*[num_layers]; 

  if(num_layers > 1){
    this->layers[0] = new layer(input_size, hidden_sizes[0], function_name);
    for(int i = 1; i < num_layers - 1; i++){
      this->layers[i] = new layer(hidden_sizes[i-1], hidden_sizes[i], function_name);
    }
    this->layers[num_layers - 1] = new layer(hidden_sizes[num_layers - 2], output_size, function_name);
  }
  else if(num_layers == 1){
    this->layers[0] = new layer(input_size, output_size, function_name);
  }
  else{
    cout<<"Error: num_layers must be greater than 0"<<endl;
    exit(1);
  }
  
  this->softmax_layer = nullptr;
  
  BackwardClass *loss_predecessor = layers[num_layers - 1]->get_output();
  this->mse_loss_layer = new mse_loss(loss_predecessor, output_size);
  this->ce_loss_layer = nullptr;
}

mlp::~mlp(){
  for(int i = 0; i < num_layers; i++){
    delete layers[i];
  }
  delete[] layers;
  delete[] activation_functions;
  
  delete softmax_layer;
  
  delete mse_loss_layer;
  delete ce_loss_layer;
}

/* GETTERS */
int mlp::get_prediction(){
  return this->softmax_layer->get_prediction();
}

float mlp::get_prediction_probability(int index){
  return this->softmax_layer->get_prediction_probability(index);
}

/* METHODS */
BackwardClass* mlp::operator()(BackwardClass *in){
  BackwardClass *out = in;
  
  // Forward pass through all layers
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->operator()(out);
    out = layers[i]->get_output();
  }
  
  // Use softmax if needed
  if(this->has_softmax){
    float *last_values = out->values_pointer();
    this->softmax_layer->copy_values(last_values);
    this->softmax_layer->operator()();
    
    return this->softmax_layer;
  }

  return out;
}

// Compute loss with target array
void mlp::compute_loss(float *target){
  if(this->loss_function == MSE){
    this->mse_loss_layer->operator()(target);
    this->current_loss += this->mse_loss_layer->get_loss();
    this->mse_loss_layer->backward();
  } else if(this->loss_function == CROSS_ENTROPY){
    this->ce_loss_layer->operator()(target);
    this->current_loss += this->ce_loss_layer->get_loss();
    this->ce_loss_layer->backward();
  }
}

// Compute loss with target index (for classification)
void mlp::compute_loss(int target_index){
  if(this->loss_function == MSE){
    // Convert target index to one-hot for MSE
    float *target = new float[this->output_size];
    for(int i = 0; i < this->output_size; i++){
      target[i] = (i == target_index) ? 1.0f : 0.0f;
    }
    this->mse_loss_layer->operator()(target);
    this->current_loss += this->mse_loss_layer->get_loss();
    this->mse_loss_layer->backward();
    delete[] target;
  } else if(this->loss_function == CROSS_ENTROPY){
    this->ce_loss_layer->operator()(target_index);
    this->current_loss += this->ce_loss_layer->get_loss();
    this->ce_loss_layer->backward();
  }
}

// Get the loss value
float mlp::get_loss(){
  return this->current_loss;
}

// Zero the loss value
void mlp::zero_loss(){
  this->current_loss = 0.0f;
}

/* BACKPROPAGATION FUNCTIONS */
void mlp::update(float learning_rate){
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->update(learning_rate);
  }
}

void mlp::zero_grad(){
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->zero_grad();
  }
}

/* PRINT FUNCTIONS */
void mlp::print_weights(){
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->print_weights();
  }
}

void mlp::print_grad_weights(){
  for(int i = 0; i < this->num_layers; i++){
    layers[i]->print_grad_weights();
  }
}

void mlp::print_loss(){
  cout << "Loss: " << this->current_loss << endl;
}

#endif