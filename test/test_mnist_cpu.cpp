#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <random>
#include <vector>

#include "../classes/cpu/input.h"
#include "../classes/enums.h"
#include "../classes/mlp.h"
#include "../classes/enums.h"
#include "../classes/virtual_classes.h"

/* PURPOSE OF THE FILE:
This file is used to test the neural network on the MNIST dataset.
It is a simple test to check if the neural network is working correctly.
The goal of this test is to check if the neural network is able to classify the MNIST dataset correctly, without any memory inefficiencies and logical errors.
Excecution time is very slow, because the network trains entirely on the CPU.
*/


void plot_sample(input *data, int label, int sample_index) {
  std::cout << "\n=== Sample " << sample_index << " (Label: " << label << ") ===" << std::endl;
  
  double *values = data->values_pointer();
  
  // Each MNIST image is 28x28 pixels
  for (int row = 0; row < 28; row++) {
    for (int col = 0; col < 28; col++) {
      double pixel_value = values[row * 28 + col];
      
      // Convert normalized pixel value (0.0-1.0) to ASCII characters
      if (pixel_value < 0.1) {
        std::cout << "  ";  // Very dark/black
      } else if (pixel_value < 0.3) {
        std::cout << "..";  // Dark gray
      } else if (pixel_value < 0.5) {
        std::cout << "::";  // Medium gray
      } else if (pixel_value < 0.7) {
        std::cout << "++";  // Light gray
      } else {
        std::cout << "##";  // White/very light
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void read_dataset(input **data, int *labels, std::string filename, int max_samples){
  std::ifstream file(filename);
  std::string line;
  
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return;
  }
  
  // Skip the header line
  std::getline(file, line);
  
  int sample_index = 0;

  // Read each line of data (limit to max_samples)
  while (std::getline(file, line) && !line.empty() && sample_index < max_samples) {
    std::stringstream ss(line);
    std::string value;
    
    double *values_ptr = data[sample_index]->values_pointer();

    // Read the label (first value)
    if (std::getline(ss, value, ',')) {
      labels[sample_index] = std::stoi(value);
    }
    
    // Read the 784 pixel values
    for (int pixel = 0; pixel < 784; pixel++) {
      if (std::getline(ss, value, ',')) {
        // Normalize pixel values from 0-255 to 0.0-1.0
        values_ptr[pixel] = std::stod(value);
      }
    }
    values_ptr[784] = 1.0; // Add bias term
    sample_index++;
  }
  
  file.close();
  std::cout << "Successfully loaded " << sample_index << " samples from " << filename << std::endl;
}

// Function to shuffle training indices
void shuffle_training_data(std::vector<int>& train_indices) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::shuffle(train_indices.begin(), train_indices.end(), gen);
}

// Function to calculate accuracy on a dataset
double calculate_accuracy(mlp& network, input** dataset, int* labels, const std::vector<int>& indices) {
  int correct_predictions = 0;
  network.zero_loss();
  
  for(int idx : indices) {
    network(dataset[idx]);
    if(network.get_prediction() == labels[idx]) {
      correct_predictions++;
    }
  }
  
  return (double)correct_predictions / indices.size() * 100.0;
}

/* DATASET INFORMATION */
const int total_samples = 42000;  // Total samples in train.csv (excluding header)
const int training_samples = 32000;  // Total samples to use for training
const int test_samples = 10000;  // Total samples to use for validation
const int num_features = 785;   // 28x28 pixels + 1 bias term

/* HYPERPARAMETERS */
int input_size = 785;
int output_size = 10;
int num_layers = 3;
int hidden_sizes[3] = {257, 129, 10};
Activation_name activation_functions[3] = {RELU, RELU, LINEAR};
Loss_name loss_function = CROSS_ENTROPY;
bool use_softmax = true;
int num_epochs = 5;
int batch_size = 100;
double learning_rate = 0.0001;

int main(){
  std::cout << "Starting MNIST training..." << std::endl;
  std::cout << "Allocating memory for dataset and labels..." << std::endl;
  
  // Allocate memory for the full dataset
  input **dataset = new input*[total_samples];
  for(int i = 0; i < total_samples; i++){
    dataset[i] = new input(num_features);
  }
  int *labels = new int[total_samples];

  std::cout << "Reading dataset..." << std::endl;
  read_dataset(dataset, labels, "./test/dataset/train.csv", total_samples);
  std::cout << "Dataset read successfully!" << std::endl;
  
  // Create index vectors for training and validation sets
  std::vector<int> train_indices;
  std::vector<int> validation_indices;
  
  // Samples for training
  for(int i = 0; i < training_samples; i++) {
    train_indices.push_back(i);
  }
  
  // Samples for validation/testing
  for(int i = training_samples; i < total_samples; i++) {
    validation_indices.push_back(i);
  }
  
  std::cout << "Dataset split: " << train_indices.size() << " training, " 
            << validation_indices.size() << " validation samples" << std::endl;
  
  std::cout << "Creating neural network..." << std::endl;
  mlp network(input_size, output_size, num_layers, hidden_sizes, activation_functions, loss_function, use_softmax);
  std::cout << "Network created successfully!" << std::endl;

  // TESTING PHASE (BEFORE TRAINING)
  std::cout << "\n=== TESTING BEFORE TRAINING ===" << std::endl;
  double accuracy_before = calculate_accuracy(network, dataset, labels, validation_indices);
  std::cout << "Accuracy before training: " << accuracy_before << "%" << std::endl;

  // TRAINING PHASE
  std::cout << "\n=== TRAINING PHASE ===" << std::endl;
  
  for(int epoch = 0; epoch < num_epochs; epoch++){
    std::cout << "Epoch " << (epoch + 1) << " started" << std::endl;
    
    // Shuffle training data at the beginning of each epoch
    shuffle_training_data(train_indices);
    std::cout << "Training data shuffled" << std::endl;
    
    network.zero_loss();
    
    // Train on shuffled data
    for(int i = 0; i < train_indices.size(); i++){
      int idx = train_indices[i];
      network(dataset[idx]);
      network.compute_loss(labels[idx]);
      
      // Update after accumulating batch_size gradients
      if((i + 1) % batch_size == 0){
        network.update(learning_rate);
        network.zero_grad();
      }
    }
    
    // Update with remaining samples if training_samples not divisible by batch_size
    if(train_indices.size() % batch_size != 0){
      network.update(learning_rate);
      network.zero_grad();
    }
    
    // Print training loss
    std::cout << "Training ";
    network.print_loss();
    
    // Evaluate on validation set
    /*double validation_accuracy = calculate_accuracy(network, dataset, labels, validation_indices);
    std::cout << "Validation accuracy: " << validation_accuracy << "%" << std::endl;
    std::cout << "Epoch " << (epoch + 1) << " completed" << std::endl;
  */}

  // FINAL TESTING PHASE
  std::cout << "\n=== FINAL TESTING ===" << std::endl;
  double accuracy_after = calculate_accuracy(network, dataset, labels, validation_indices);
  std::cout << "Final accuracy: " << accuracy_after << "%" << std::endl;
  std::cout << "Improvement: " << (accuracy_after - accuracy_before) << "%" << std::endl;

  // Plot the first 10 samples
  
  /*
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "DISPLAYING FIRST 10 MNIST SAMPLES" << std::endl;
  std::cout << std::string(60, '=') << std::endl;
  
  for (int i = 0; i < 10; i++) {
    plot_sample(dataset[i], labels[i], i);
    
    if (i < 9) {
      std::cout << std::string(40, '-') << std::endl;
    }
  }
  
  std::cout << "\nVisualization complete!" << std::endl;
  */
  
  // Clean up memory
  for(int i = 0; i < total_samples; i++){
    delete dataset[i];
  }
  delete[] dataset;
  delete[] labels;
  
  std::cout << "\nTraining completed successfully!" << std::endl;
  return 0;
}