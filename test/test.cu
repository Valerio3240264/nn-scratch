#include "classes/input.h"
#include "classes/virtual_classes.h"
#include "classes/activation_function.h"
#include "classes/weights.h"
#include "classes/loss.h"
#include "classes/mlp.h"
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <iomanip>
#include <cstdlib>

using namespace std;

// Target function: f(x,y,z) = 2x + 4y + 4.5z
double target_function(double x, double y, double z) {
  return 2.0 * x + 4.0 * y + 4.5 * z;
}

// Data generation function
void generate_dataset(
  vector<input*>& inputs, vector<double>& targets, int num_samples, int min_val = -5, int max_val = 5) {
  
    uniform_real_distribution<double> dis(min_val, max_val);

    default_random_engine gen(time(0));

    // Clean up existing inputs
    for (input* inp : inputs) {
        delete inp;
    }
    inputs.clear();
    targets.clear();
    inputs.reserve(num_samples);
    targets.reserve(num_samples);
    
    for (int i = 0; i < num_samples; i++) {
      double x = dis(gen);
      double y = dis(gen);
      double z = dis(gen);
      
      // Create input object and set values manually
      input* inp = new input(3);  // Use default constructor
      // Set values through the value pointer
      double* values = inp->values_pointer();
      values[0] = x;
      values[1] = y;
      values[2] = z;
      
      inputs.push_back(inp);
      targets.push_back(target_function(x, y, z));
    }
}

// Calculate mean squared error
double calculate_mse(const vector<double>& predictions, const vector<double>& targets) {
  double mse = 0.0;
  int n = predictions.size();
  for (int i = 0; i < n; i++) {
    double diff = predictions[i] - targets[i];
    mse += diff * diff;
  }
  return mse / n;
}

int main() {

  cout << "=== MLP Training for f(x,y,z) = 2x + 4y + 4.5z ===" << endl;
  
  // Network architecture
  int input_size = 3;  // x, y, z
  int output_size = 1; // function value
  int num_layers = 2;  // input -> hidden -> output
  int hidden_sizes[] = {10};
  int batch_size = 10;
  Activation_name activation_function = LINEAR;
  
  // Create MLP
  mlp model(input_size, output_size, num_layers, hidden_sizes, activation_function);
  
  // Generate datasets
  vector<input*> train_inputs, val_inputs;
  vector<double> train_targets, val_targets;

  cout << "Generating training data..." << endl;
  generate_dataset(train_inputs, train_targets, 1000, -5.0, 5.0);
  
  cout << "Generating validation data..." << endl;
  generate_dataset(val_inputs, val_targets, 500, -3.0, 3.0);
  
  // Training parameters
  double learning_rate = 0.001;  // Fixed gradient direction, back to positive LR
  int epochs = 100;  // More epochs for better convergence
  int print_every = 1;
  
  cout << "\nStarting training..." << endl;
  cout << "Learning rate: " << learning_rate << endl;
  cout << "Epochs: " << epochs << endl;

  // Training loop
  for (int epoch = 0; epoch < epochs; epoch++) {
    cout<<"Epoch: "<<epoch<<endl;
    vector<double> epoch_predictions;
    epoch_predictions.reserve(train_inputs.size());
    
    // Train on all samples
    for (size_t i = 0; i < train_inputs.size(); i++) {
      // Forward pass (use input pointer directly)
      input* output = model(train_inputs[i]);
      epoch_predictions.push_back(output->get_value(0));
      
      // Create target properly - scale down to fit tanh range
      input target(output_size);
      double* target_values = target.values_pointer();
      target_values[0] = train_targets[i];
      
      // Compute loss and gradients
      model.compute_loss(&target);
      // Update weights
      if(i % batch_size == 0) {
        model.update(learning_rate);
        model.zero_grad();
      }
    }
  
    // Print progress
    if ((epoch + 1) % print_every == 0) {
      double train_mse = calculate_mse(epoch_predictions, train_targets);
      cout << "Epoch " << (epoch + 1) << "/" << epochs << " - Training MSE: " << train_mse << endl;
      cout<<endl;
    }
  }
  
  cout << "\nTraining completed!" << endl;
  
  // Validation
  cout << "\nTesting on validation set..." << endl;
  vector<double> val_predictions;
  val_predictions.reserve(val_inputs.size());
  
  for (size_t i = 0; i < val_inputs.size(); i++) {
    input* output = model(val_inputs[i]);
    val_predictions.push_back(output->get_value(0));  
  }
  
  double val_mse = calculate_mse(val_predictions, val_targets);
  cout << "Validation MSE: " << val_mse << endl;
  
  model.print_weights();
  
  // Clean up dynamically allocated input objects
  cout << "\nCleaning up memory..." << endl;
  for (input* inp : train_inputs) {
    delete inp;
  }
  for (input* inp : val_inputs) {
    delete inp;
  }
  return 0;
}