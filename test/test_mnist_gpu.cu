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

#include "../classes/cuda/cuda_input.cuh"
#include "../classes/enums.h"
#include "../classes/mlp.h"
#include "../classes/virtual_classes.h"

/* PURPOSE OF THE FILE:
This file is used to test the neural network on the MNIST dataset using GPU acceleration.
The network trains on CUDA-enabled GPUs for significantly faster training compared to CPU.
All forward and backward passes are executed on the GPU, with minimal host-device transfers.
*/

void plot_sample(cuda_input *data, int label, int sample_index) {
  std::cout << "\n=== Sample " << sample_index << " (Label: " << label << ") ===" << std::endl;
  
  // Copy values from device to host for visualization
  float *h_values = new float[784];
  float *d_values = data->values_pointer();
  copy_device_to_host(h_values, d_values, 784);
  
  // Each MNIST image is 28x28 pixels
  for (int row = 0; row < 28; row++) {
    for (int col = 0; col < 28; col++) {
      float pixel_value = h_values[row * 28 + col];
      
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
  
  delete[] h_values;
}

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

// Function to shuffle training indices
void shuffle_training_data(std::vector<int>& train_indices) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::shuffle(train_indices.begin(), train_indices.end(), gen);
}

// Function to calculate accuracy on a dataset
float calculate_accuracy(mlp& network, cuda_input** dataset, int* labels, const std::vector<int>& indices) {
  int correct_predictions = 0;
  network.zero_loss();
  
  for(int idx : indices) {
    //std::cout << "Computing loss for sample " << idx << std::endl;
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
  
  // Create CUDA events for timing
  //cudaEvent_t start, stop;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);
  
  for(int epoch = 0; epoch < num_epochs; epoch++){
    std::cout << "Epoch " << (epoch + 1) << " started" << std::endl;
    
    network.zero_loss();
    
    // Timing accumulators for this epoch
    /*
    float total_forward_time = 0.0f;
    float total_loss_time = 0.0f;
    float total_update_time = 0.0f;
    float total_zero_grad_time = 0.0f;
    int num_updates = 0;
    */
    // Train on shuffled data
    for(int i = 0; i < train_indices.size(); i++){
      int idx = train_indices[i];
      
      // Time forward pass
      //cudaEventRecord(start);
      network(dataset[idx]);
      //cudaEventRecord(stop);
      //cudaEventSynchronize(stop);
      //float forward_time;
      //cudaEventElapsedTime(&forward_time, start, stop);
      //total_forward_time += forward_time;
      
      // Time compute loss
      //cudaEventRecord(start);
      network.compute_loss(labels[idx]);
      //cudaEventRecord(stop);
      //cudaEventSynchronize(stop);
      //float loss_time;
      //cudaEventElapsedTime(&loss_time, start, stop);
      //total_loss_time += loss_time;
      
      // Update after accumulating batch_size gradients
      if((i + 1) % batch_size == 0){
        // Synchronize to ensure all backward pass kernels have completed
        //cudaDeviceSynchronize();
        // Time update
        //cudaEventRecord(start);
        network.update(learning_rate);
        //cudaEventRecord(stop);
        //cudaEventSynchronize(stop);
        //float update_time;
        //cudaEventElapsedTime(&update_time, start, stop);
        //total_update_time += update_time;
        // Time zero_grad
        //cudaEventRecord(start);
        network.zero_grad();
        //cudaEventRecord(stop);
        //cudaEventSynchronize(stop);
        //float zero_grad_time;
        //cudaEventElapsedTime(&zero_grad_time, start, stop);
        //total_zero_grad_time += zero_grad_time;
        
        //num_updates++;
      }
    }
    
    // Update with remaining samples if training_samples not divisible by batch_size
    if(train_indices.size() % batch_size != 0){
      // Synchronize to ensure all backward pass kernels have completed
      
      // Time update
      //cudaEventRecord(start);
      network.update(learning_rate);
      //cudaEventRecord(stop);
      //cudaEventSynchronize(stop);
      //float update_time;
      //cudaEventElapsedTime(&update_time, start, stop);
      //total_update_time += update_time;
      
      // Time zero_grad
      //cudaEventRecord(start);
      network.zero_grad();
      //cudaEventRecord(stop);
      //cudaEventSynchronize(stop);
      //float zero_grad_time;
      //cudaEventElapsedTime(&zero_grad_time, start, stop);
      //total_zero_grad_time += zero_grad_time;
      
      //num_updates++;
    }
    
    // Print training loss
    std::cout << "Training ";
    network.print_loss();
    
    // Print timing statistics
    /*
    int num_samples = train_indices.size();
    std::cout << "\n--- Timing Statistics (Epoch " << (epoch + 1) << ") ---" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mean Forward Pass Time:    " << (total_forward_time / num_samples) << " ms" << std::endl;
    std::cout << "Mean Compute Loss Time:    " << (total_loss_time / num_samples) << " ms" << std::endl;
    std::cout << "Mean Update Time:          " << (total_update_time / num_updates) << " ms" << std::endl;
    std::cout << "Mean Zero Grad Time:       " << (total_zero_grad_time / num_updates) << " ms" << std::endl;
    std::cout << "Total Forward Time:        " << total_forward_time << " ms" << std::endl;
    std::cout << "Total Compute Loss Time:   " << total_loss_time << " ms" << std::endl;
    std::cout << "Total Update Time:         " << total_update_time << " ms" << std::endl;
    std::cout << "Total Zero Grad Time:      " << total_zero_grad_time << " ms" << std::endl;
    std::cout << "Total Epoch Time:          " << (total_forward_time + total_loss_time + total_update_time + total_zero_grad_time) << " ms" << std::endl;
    std::cout << "------------------------------------\n" << std::endl;
    */
  }
  
  // Clean up CUDA events
  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);

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