#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstddef>

#include "../classes/cpu/headers/input.h"
#include "../classes/enums.h"
#include "../classes/mlp/headers/mlp.h"

/*
PURPOSE OF THE FILE:
Batch-structured MNIST CPU training test using:
1) one contiguous dataset matrix in host memory
2) in-place shuffle for matrix rows + labels
3) one reusable input object per batch
*/

/* DATASET INFORMATION */
const int total_samples = 42000;
const int training_samples = 32000;
const int test_samples = 10000;
const int num_features = 784;

void read_dataset(float *data_matrix, int *labels, const std::string &filename, int max_samples){
  std::ifstream file(filename);
  std::string line;
  
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return;
  }
  
  std::getline(file, line); // header
  
  int sample_index = 0;
  while (std::getline(file, line) && !line.empty() && sample_index < max_samples) {
    std::stringstream ss(line);
    std::string value;
    float *row_ptr = data_matrix + static_cast<size_t>(sample_index) * static_cast<size_t>(num_features);

    if (std::getline(ss, value, ',')) {
      labels[sample_index] = std::stoi(value);
    }
    
    for (size_t pixel = 0; pixel < static_cast<size_t>(num_features); pixel++) {
      if (std::getline(ss, value, ',')) {
        row_ptr[pixel] = std::stof(value) / 255.0f;
      }
    }
    sample_index++;
  }
  
  file.close();
  std::cout << "Successfully loaded " << sample_index << " samples from " << filename << std::endl;
}

void shuffle_dataset_rows(float *data_matrix, int *labels, size_t rows, size_t cols, std::mt19937 &rng) {
  float *tmp_row = new float[cols];
  for (size_t i = rows; i > 1; i--) {
    std::uniform_int_distribution<size_t> dist(0, i - 1);
    size_t j = dist(rng);
    size_t k = i - 1;
    if (j == k) continue;

    float *row_j = data_matrix + j * cols;
    float *row_k = data_matrix + k * cols;

    for (size_t c = 0; c < cols; c++){ 
      tmp_row[c] = row_j[c];
      row_j[c] = row_k[c];
      row_k[c] = tmp_row[c];
    }

    int tmp_label = labels[j];
    labels[j] = labels[k];
    labels[k] = tmp_label;
  }
  delete[] tmp_row;
}

float calculate_accuracy(mlp &network,
                         float *data_matrix,
                         int *labels,
                         size_t offset,
                         size_t count,
                         size_t batch_size) {
  size_t correct_predictions = 0;
  network.zero_loss();

  input eval_input(static_cast<size_t>(num_features), batch_size);
  size_t *predictions = new size_t[batch_size];
  float *eval_batch = new float[batch_size * static_cast<size_t>(num_features)];

  for (size_t start = 0; start < count; start += batch_size) {
    size_t valid = std::min(batch_size, count - start);

    // Populate full batch buffer. If tail is shorter than batch_size, pad by
    // repeating the last valid sample to satisfy fixed-shape network input.
    for (size_t row = 0; row < batch_size; row++) {
      size_t src_local = (row < valid) ? row : (valid - 1);
      size_t src_idx = offset + start + src_local;
      float *src = data_matrix + src_idx * static_cast<size_t>(num_features);
      float *dst = eval_batch + row * static_cast<size_t>(num_features);
      for (size_t c = 0; c < static_cast<size_t>(num_features); c++) {
        dst[c] = src[c];
      }
    }

    eval_input.set_values(eval_batch);
    network(&eval_input);
    network.get_predictions(predictions);

    for (size_t row = 0; row < valid; row++) {
      size_t label_idx = offset + start + row;
      if (predictions[row] == static_cast<size_t>(labels[label_idx])) {
        correct_predictions++;
      }
    }
  }

  delete[] predictions;
  delete[] eval_batch;

  return static_cast<float>(correct_predictions) / static_cast<float>(count) * 100.0f;
}

/* HYPERPARAMETERS */
size_t input_size = 784;
size_t output_size = 10;
size_t num_layers = 3;
size_t hidden_sizes[3] = {256, 128, 10};
Activation_name activation_functions[3] = {RELU, RELU, LINEAR};
Loss_name loss_function = CROSS_ENTROPY;
bool use_softmax = true;
int num_epochs = 5;
size_t batch_size = 100;
float learning_rate = 1.0f;

int main(){
  std::cout << "Starting MNIST batch-structured training..." << std::endl;

  float *dataset_matrix = new float[static_cast<size_t>(total_samples) * static_cast<size_t>(num_features)];
  int *labels = new int[total_samples];

  read_dataset(dataset_matrix, labels, "./test/dataset/train.csv", total_samples);

  mlp network(
    input_size,
    output_size,
    batch_size,
    num_layers,
    hidden_sizes,
    activation_functions,
    loss_function,
    use_softmax
  );

  float accuracy_before = calculate_accuracy(
    network,
    dataset_matrix,
    labels,
    static_cast<size_t>(training_samples),
    static_cast<size_t>(test_samples),
    batch_size
  );
  std::cout << "Accuracy before training: " << accuracy_before << "%" << std::endl;

  std::random_device rd;
  std::mt19937 rng(rd());

  input batch_input(num_features, batch_size);

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    auto epoch_start_time = std::chrono::high_resolution_clock::now();
    shuffle_dataset_rows(
      dataset_matrix,
      labels,
      static_cast<size_t>(training_samples),
      static_cast<size_t>(num_features),
      rng
    );
    network.zero_loss();
    network.zero_grad();

    double forward_ms = 0.0;
    double backward_ms = 0.0;
    double update_ms = 0.0;
    size_t num_batches = 0;
    size_t *batch_targets = new size_t[batch_size];

    for (size_t batch_start = 0; batch_start < static_cast<size_t>(training_samples); batch_start += static_cast<size_t>(batch_size)) {
      size_t valid = std::min(batch_size, static_cast<size_t>(training_samples) - batch_start);
      if(valid != batch_size){
        break;
      }
      batch_input.set_values(dataset_matrix + (batch_start * static_cast<size_t>(num_features)));
      for(size_t i = 0; i < batch_size; i++){
        batch_targets[i] = static_cast<size_t>(labels[batch_start + i]);
      }

      auto start_time = std::chrono::high_resolution_clock::now();
      network(&batch_input);
      auto end_time = std::chrono::high_resolution_clock::now();
      forward_ms += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

      network.compute_loss(batch_targets);

      start_time = std::chrono::high_resolution_clock::now();
      network.backward();
      end_time = std::chrono::high_resolution_clock::now();
      backward_ms += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

      start_time = std::chrono::high_resolution_clock::now();
      network.update(learning_rate);
      end_time = std::chrono::high_resolution_clock::now();
      update_ms += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

      network.zero_grad();
      num_batches++;
    }
    delete[] batch_targets;
    auto epoch_end_time = std::chrono::high_resolution_clock::now();
    double epoch_total_ms = std::chrono::duration_cast<std::chrono::microseconds>(epoch_end_time - epoch_start_time).count() / 1000.0;

    std::cout << "\nEpoch " << (epoch + 1) << " completed" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Mean Forward Time/batch:  " << (num_batches ? forward_ms / static_cast<double>(num_batches) : 0.0) << " ms" << std::endl;
    std::cout << "Mean Backward Time/batch: " << (num_batches ? backward_ms / static_cast<double>(num_batches) : 0.0) << " ms" << std::endl;
    std::cout << "Mean Update Time/batch:   " << (num_batches ? update_ms / static_cast<double>(num_batches) : 0.0) << " ms" << std::endl;
    std::cout << "Total Epoch Time:         " << epoch_total_ms << " ms (" << (epoch_total_ms / 1000.0) << " s)" << std::endl;
    std::cout << "Training Loss:            " << network.get_loss() / static_cast<float>(training_samples) << std::endl;
  }

  float accuracy_after = calculate_accuracy(
    network,
    dataset_matrix,
    labels,
    static_cast<size_t>(training_samples),
    static_cast<size_t>(test_samples),
    batch_size
  );
  std::cout << "\nFinal accuracy: " << accuracy_after << "%" << std::endl;
  std::cout << "Improvement: " << (accuracy_after - accuracy_before) << "%" << std::endl;

  delete[] dataset_matrix;
  delete[] labels;

  std::cout << "\nBatch-structured training completed successfully!" << std::endl;
  return 0;
}
