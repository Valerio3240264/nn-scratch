#include "classes/input.h"
#include "classes/virtual_classes.h"
#include "classes/activation_function.h"
#include "classes/weights.h"
#include <iostream>
#include <vector>
#include <memory>
#include "classes/loss.h"
#include "classes/mlp.h"

using namespace std;

int main(){
  int input_size = 2;
  int output_size = 1;
  int num_layers = 2;
  int hidden_sizes[] = {2};

  mlp model(input_size, output_size, num_layers, hidden_sizes);

  double input_values[] = {1, 1};
  input in(input_size, input_values);

  input *output = model(&in);
  double target_values[] = {1};
  input *target = new input(output_size, target_values);
  model.compute_loss(target);

  cout << "Weights:" << endl;
  model.print_weights();
  cout << "Grad Weights:" << endl;
  model.print_grad_weights();
  cout << "Output:" << endl;
  output->print_value();

  cout << "Updating weights..." << endl;
  model.update(0.1);

  cout << "Updated Weights:" << endl;
  model.print_weights();
  cout << "Updated Grad Weights:" << endl;
  model.print_grad_weights();

  delete target;
  return 0;
}