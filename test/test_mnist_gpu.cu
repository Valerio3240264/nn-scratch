#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

#include "../classes/cuda/cuda_manager.cuh"
#include "../classes/cuda/cuda_manager_impl.cuh"
#include "../classes/cuda/headers/cuda_input.cuh"
#include "../classes/enums.h"
#include "../classes/mlp/headers/mlp.h"
#include "../classes/virtual_classes.h"

/* PURPOSE OF THE FILE:
This file is used to test the neural network on the MNIST dataset using GPU acceleration.
The network trains on CUDA-enabled GPUs for significantly faster training compared to CPU.
All forward and backward passes are executed on the GPU, with minimal host-device transfers.
*/

void read_dataset(cuda_input **data, int *labels, std::string filename, int max_samples){
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
    
    // Allocate host buffer for this sample
    float *h_values = new float[784]; // 784 pixels

    // Read the label (first value)
    if (std::getline(ss, value, ',')) {
      labels[sample_index] = std::stoi(value);
    }
    
    // Read the 784 pixel values
    for (int pixel = 0; pixel < 784; pixel++) {
      if (std::getline(ss, value, ',')) {
        // Normalize pixel values from 0-255 to 0.0-1.0
        h_values[pixel] = std::stof(value) / 255.0f;
      }
    }
    
    // Copy to device memory
    float *d_values = data[sample_index]->values_pointer();
    copy_host_to_device(d_values, h_values, 784);
    
    delete[] h_values;
    sample_index++;
  }
  
  file.close();
  std::cout << "Successfully loaded " << sample_index << " samples from " << filename << std::endl;
}

// Function to calculate accuracy on a dataset
float calculate_accuracy(mlp& network, cuda_input** dataset, int* labels, const std::vector<int>& indices) {
  int correct_predictions = 0;
  network.zero_loss();
  
  for(int idx : indices) {
    network(dataset[idx]);
    if(network.get_prediction() == labels[idx]) {
      correct_predictions++;
    }
  }
  
  return (float)correct_predictions / indices.size() * 100.0f;
}

/* DATASET INFORMATION */
const int total_samples = 42000;  // Total samples in train.csv (excluding header)
const int training_samples = 32000;  // Total samples to use for training
const int test_samples = 10000;  // Total samples to use for validation
const int num_features = 784;   // 28x28 pixels

/* HYPERPARAMETERS */
int input_size = 784;
int output_size = 10;
int num_layers = 3;
int hidden_sizes[3] = {256, 128, 10};
Activation_name activation_functions[3] = {RELU, RELU, LINEAR};
Loss_name loss_function = CROSS_ENTROPY;
bool use_softmax = true;
bool use_cuda = true;
int num_epochs = 5;
int batch_size = 100;
float learning_rate = 0.01;

int main(){

  std::cout << "Starting MNIST training on GPU..." << std::endl;
  
  // Check CUDA availability
  if(!is_cuda_available()){
    std::cerr << "Error: CUDA is not available on this system!" << std::endl;
    return 1;
  }
  
  std::cout << "Allocating memory for dataset and labels..." << std::endl;
  
  // Allocate memory for the full dataset (device memory via cuda_input)
  cuda_input **dataset = new cuda_input*[total_samples];
  for(int i = 0; i < total_samples; i++){
    dataset[i] = new cuda_input(num_features);  // Allocates on device
  }
  int *labels = new int[total_samples];  // Labels stay on host

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
  for(int i = 0; i < test_samples; i++) {
    validation_indices.push_back(training_samples + i);
  }
  
  std::cout << "Dataset split: " << train_indices.size() << " training, " 
            << validation_indices.size() << " validation samples" << std::endl;
  
  std::cout << "Creating neural network on GPU..." << std::endl;
  mlp network(input_size, output_size, num_layers, hidden_sizes, activation_functions, loss_function, use_softmax, use_cuda);
  std::cout << "Network created successfully!" << std::endl;

  // TESTING PHASE (BEFORE TRAINING)
  std::cout << "\n=== TESTING BEFORE TRAINING ===" << std::endl;
  float accuracy_before = calculate_accuracy(network, dataset, labels, validation_indices);
  std::cout << "Accuracy before training: " << accuracy_before << "%" << std::endl;

  // TRAINING PHASE
  std::cout << "\n=== TRAINING PHASE ===" << std::endl;
  
  // Global timing accumulators across all epochs
  double total_forward_time_all = 0.0;
  double total_loss_time_all = 0.0;
  double total_update_time_all = 0.0;
  long long total_samples_processed = 0;
  long long total_batches_processed = 0;
  
  for(int epoch = 0; epoch < num_epochs; epoch++){
    std::cout << "Epoch " << (epoch + 1) << " started" << std::endl;
    
    network.zero_loss();
    
    // Timing accumulators
    double total_forward_time = 0.0;
    double total_loss_time = 0.0;
    double total_update_time = 0.0;
    int num_batches = 0;
    
    for(int i = 0; i < train_indices.size(); i++){
      int idx = train_indices[i];
      
      // Forward pass (timed)
      auto start_time = std::chrono::high_resolution_clock::now();
      network(dataset[idx]);
      auto end_time = std::chrono::high_resolution_clock::now();
      auto forward_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0; // Convert to ms
      total_forward_time += forward_time;

      // Compute loss (includes backward pass) (timed)
      start_time = std::chrono::high_resolution_clock::now();
      network.compute_loss(labels[idx]);
      end_time = std::chrono::high_resolution_clock::now();
      auto loss_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0; // Convert to ms
      total_loss_time += loss_time;
      
      // Update after accumulating batch_size gradients
      if((i + 1) % batch_size == 0){
        start_time = std::chrono::high_resolution_clock::now();
        network.update(learning_rate);
        end_time = std::chrono::high_resolution_clock::now();
        auto update_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0; // Convert to ms
        total_update_time += update_time;
        num_batches++;
        network.zero_grad();
      }
    }
    
    // Update with remaining samples if training_samples not divisible by batch_size
    if(train_indices.size() % batch_size != 0){
      auto start_time = std::chrono::high_resolution_clock::now();
      network.update(learning_rate);
      auto end_time = std::chrono::high_resolution_clock::now();
      auto update_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0; // Convert to ms
      total_update_time += update_time;
      num_batches++;
      network.zero_grad();
    }
    
    // Print timing statistics
    std::cout << "\n--- Timing Statistics (Epoch " << (epoch + 1) << ") ---" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mean Forward Pass Time:    " << total_forward_time / train_indices.size() << " ms" << std::endl;
    std::cout << "Mean Compute Loss Time:    " << total_loss_time / train_indices.size() << " ms" << std::endl;
    if(num_batches > 0){
      std::cout << "Mean Update Step Time:     " << total_update_time / num_batches << " ms" << std::endl;
    }
    std::cout << "Total Forward Time:        " << total_forward_time << " ms" << std::endl;
    std::cout << "Total Compute Loss Time:    " << total_loss_time << " ms" << std::endl;
    std::cout << "Total Update Step Time:    " << total_update_time << " ms" << std::endl;
    double total_time = total_forward_time + total_loss_time + total_update_time;
    std::cout << "Total Epoch Time:          " << total_time << " ms (" << total_time / 1000.0 << " seconds)" << std::endl;
    std::cout << "Training ";
    double loss = network.get_loss();
    std::cout << "Loss: " << loss/training_samples << std::endl;
    std::cout << "------------------------------------\n" << std::endl;

    // Accumulate global timings
    total_forward_time_all += total_forward_time;
    total_loss_time_all += total_loss_time;
    total_update_time_all += total_update_time;
    total_samples_processed += train_indices.size();
    total_batches_processed += num_batches;
    
  }
  
  // Print overall timing statistics
  std::cout << "\n=== TOTAL TRAINING TIMINGS ===" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Mean Forward Pass Time:    " << (total_samples_processed ? total_forward_time_all / total_samples_processed : 0.0) << " ms" << std::endl;
  std::cout << "Mean Compute Loss Time:    " << (total_samples_processed ? total_loss_time_all / total_samples_processed : 0.0) << " ms" << std::endl;
  if(total_batches_processed > 0){
    std::cout << "Mean Update Step Time:     " << total_update_time_all / total_batches_processed << " ms" << std::endl;
  }
  std::cout << "Total Forward Time:        " << total_forward_time_all << " ms" << std::endl;
  std::cout << "Total Compute Loss Time:    " << total_loss_time_all << " ms" << std::endl;
  std::cout << "Total Update Step Time:    " << total_update_time_all << " ms" << std::endl;
  double grand_total_time = total_forward_time_all + total_loss_time_all + total_update_time_all;
  std::cout << "Total Training Time:       " << grand_total_time << " ms (" << grand_total_time / 1000.0 << " seconds)" << std::endl;
  
  // FINAL TESTING PHASE
  std::cout << "\n=== FINAL TESTING ===" << std::endl;
  float accuracy_after = calculate_accuracy(network, dataset, labels, validation_indices);
  std::cout << "Final accuracy: " << accuracy_after << "%" << std::endl;
  std::cout << "Improvement: " << (accuracy_after - accuracy_before) << "%" << std::endl;

  // Clean up memory
  for(int i = 0; i < total_samples; i++){
    delete dataset[i];
  }
  delete[] dataset;
  delete[] labels;
  
  std::cout << "\nGPU Training completed successfully!" << std::endl;
  return 0;
}