#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../classes/cuda/cuda_manager.cuh"
#include "../classes/cuda/cuda_manager_impl.cuh"
#include "../classes/cuda/headers/cuda_input.cuh"
#include "../classes/enums.h"
#include "../classes/mlp/headers/mlp.h"

/*
PURPOSE:
- Exercise the complete batched CUDA MLP path on MNIST.
- Keep the dataset in host memory and copy one compact batch at a time.
- Reuse one device input buffer and one cuda_input graph node.
*/

namespace {

constexpr size_t TOTAL_SAMPLES = 42000;
constexpr size_t TRAINING_SAMPLES = 32000;
constexpr size_t TEST_SAMPLES = 10000;
constexpr size_t INPUT_SIZE = 784;
constexpr size_t OUTPUT_SIZE = 10;
constexpr size_t BATCH_SIZE = 100;
constexpr int NUM_EPOCHS = 5;
constexpr float LEARNING_RATE = 0.01f;

size_t read_dataset(
    std::vector<float>& data,
    std::vector<size_t>& labels,
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: could not open " << filename << std::endl;
    return 0;
  }

  std::string line;
  std::getline(file, line);

  size_t sample = 0;
  while (sample < TOTAL_SAMPLES && std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream row_stream(line);
    std::string value;

    if (!std::getline(row_stream, value, ',')) {
      break;
    }
    labels[sample] = static_cast<size_t>(std::stoul(value));

    float* row = data.data() + sample * INPUT_SIZE;
    for (size_t feature = 0; feature < INPUT_SIZE; ++feature) {
      if (!std::getline(row_stream, value, ',')) {
        std::cerr << "Error: incomplete row " << sample << std::endl;
        return sample;
      }
      row[feature] = std::stof(value) / 255.0f;
    }

    ++sample;
  }

  return sample;
}

void copy_sample_to_batch(
    const std::vector<float>& dataset,
    size_t sample,
    std::vector<float>& batch,
    size_t batch_row) {
  const float* source = dataset.data() + sample * INPUT_SIZE;
  float* destination = batch.data() + batch_row * INPUT_SIZE;
  std::copy(source, source + INPUT_SIZE, destination);
}

float calculate_accuracy(
    mlp& network,
    cuda_input& input,
    float* device_batch,
    const std::vector<float>& dataset,
    const std::vector<size_t>& labels,
    size_t offset,
    size_t count) {
  std::vector<float> host_batch(BATCH_SIZE * INPUT_SIZE);
  std::vector<size_t> predictions(BATCH_SIZE);
  size_t correct = 0;

  for (size_t start = 0; start < count; start += BATCH_SIZE) {
    const size_t valid = std::min(BATCH_SIZE, count - start);

    for (size_t row = 0; row < BATCH_SIZE; ++row) {
      const size_t source_row = row < valid ? row : valid - 1;
      copy_sample_to_batch(
          dataset,
          offset + start + source_row,
          host_batch,
          row);
    }

    copy_host_to_device(
        device_batch,
        host_batch.data(),
        host_batch.size());
    network(&input);
    network.get_predictions(predictions.data());

    for (size_t row = 0; row < valid; ++row) {
      if (predictions[row] == labels[offset + start + row]) {
        ++correct;
      }
    }
  }

  return 100.0f * static_cast<float>(correct) /
         static_cast<float>(count);
}

}  // namespace

int main() {
  if (!is_cuda_available()) {
    std::cerr << "Error: CUDA is not available." << std::endl;
    return 1;
  }

  std::vector<float> dataset(TOTAL_SAMPLES * INPUT_SIZE);
  std::vector<size_t> labels(TOTAL_SAMPLES);
  const size_t loaded =
      read_dataset(dataset, labels, "./test/dataset/train.csv");
  if (loaded != TOTAL_SAMPLES) {
    std::cerr << "Error: loaded " << loaded << " of "
              << TOTAL_SAMPLES << " samples." << std::endl;
    return 1;
  }

  float* device_batch = nullptr;
  allocate_device_memory<float>(
      &device_batch,
      BATCH_SIZE * INPUT_SIZE);

  {
    cuda_input batch_input(INPUT_SIZE, BATCH_SIZE);
    batch_input.set_values(device_batch);

    size_t hidden_sizes[3] = {256, 128, OUTPUT_SIZE};
    Activation_name activation_functions[3] = {
        RELU,
        RELU,
        LINEAR};

    mlp network(
        INPUT_SIZE,
        OUTPUT_SIZE,
        BATCH_SIZE,
        3,
        hidden_sizes,
        activation_functions,
        CROSS_ENTROPY,
        true,
        true);

    float accuracy = calculate_accuracy(
        network,
        batch_input,
        device_batch,
        dataset,
        labels,
        TRAINING_SAMPLES,
        TEST_SAMPLES);
    std::cout << "Accuracy before training: "
              << accuracy << "%" << std::endl;

    std::vector<size_t> training_order(TRAINING_SAMPLES);
    std::iota(training_order.begin(), training_order.end(), 0);
    std::mt19937 random(std::random_device{}());

    std::vector<float> host_batch(BATCH_SIZE * INPUT_SIZE);
    std::vector<size_t> batch_targets(BATCH_SIZE);

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
      std::shuffle(
          training_order.begin(),
          training_order.end(),
          random);
      network.zero_loss();
      network.zero_grad();

      const auto epoch_start = std::chrono::steady_clock::now();
      size_t batches = 0;

      for (size_t start = 0;
           start + BATCH_SIZE <= TRAINING_SAMPLES;
           start += BATCH_SIZE) {
        for (size_t row = 0; row < BATCH_SIZE; ++row) {
          const size_t sample = training_order[start + row];
          copy_sample_to_batch(
              dataset,
              sample,
              host_batch,
              row);
          batch_targets[row] = labels[sample];
        }

        copy_host_to_device(
            device_batch,
            host_batch.data(),
            host_batch.size());

        network(&batch_input);
        network.compute_loss(batch_targets.data());
        network.backward();
        network.update(LEARNING_RATE);
        network.zero_grad();
        ++batches;
      }

      CUDA_CHECK_MANAGER(cudaDeviceSynchronize());
      const auto epoch_end = std::chrono::steady_clock::now();
      const double epoch_seconds =
          std::chrono::duration<double>(epoch_end - epoch_start).count();
      const float mean_batch_loss =
          batches == 0
              ? 0.0f
              : network.get_loss() / static_cast<float>(batches);

      std::cout << std::fixed << std::setprecision(4)
                << "Epoch " << (epoch + 1)
                << " | loss: " << mean_batch_loss
                << " | time: " << epoch_seconds << " s"
                << std::endl;
    }

    accuracy = calculate_accuracy(
        network,
        batch_input,
        device_batch,
        dataset,
        labels,
        TRAINING_SAMPLES,
        TEST_SAMPLES);
    std::cout << "Final accuracy: "
              << accuracy << "%" << std::endl;
  }

  free_device_memory(device_batch);
  return 0;
}
