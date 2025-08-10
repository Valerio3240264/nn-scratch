#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include <iomanip>
#include <direct.h>
#include "../mlp.cpp"

using namespace std;

const int epochs = 1;
const double lr = 0.01;
const int batch_size = 10;

// Function to read a 32-bit integer from a binary file in big-endian format
uint32_t read_int(std::ifstream& file) {
  uint8_t buffer[4];
  file.read(reinterpret_cast<char*>(buffer), 4);
  
  // Convert from big-endian to host endian
  return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

// Function to load MNIST images into a vector of doubles
vector<vector<double>> load_mnist_images(const string& filename) {
  ifstream file(filename, ios::binary);
  if (!file) {
    cerr << "Failed to open file: " << filename << endl;
    return {};
  }

  uint32_t magic_number = read_int(file);
  if (magic_number != 2051) {
    cerr << "Invalid magic number in " << filename << ": " << magic_number << endl;
    return {};
  }

  uint32_t num_images = read_int(file);
  uint32_t rows = read_int(file);
  uint32_t cols = read_int(file);

  cout << "Loading " << num_images << " images of size " << rows << "x" << cols << endl;

  vector<vector<double>> images(num_images);
  
  // Each image is rows*cols pixels
  size_t image_size = rows * cols;
  
  for (uint32_t i = 0; i < num_images; ++i) {
    images[i].resize(image_size);
    
    for (uint32_t j = 0; j < image_size; ++j) {
      uint8_t pixel;
      file.read(reinterpret_cast<char*>(&pixel), 1);
      
      // Normalize pixel value to [0, 1]
      images[i][j] = static_cast<double>(pixel) / 255.0;
    }
  }

  return images;
}

// Function to load MNIST labels into a vector of doubles
vector<double> load_mnist_labels(const string& filename) {
  ifstream file(filename, ios::binary);
  if (!file) {
    cerr << "Failed to open file: " << filename << endl;
    return {};
  }

  uint32_t magic_number = read_int(file);
  if (magic_number != 2049) {
    cerr << "Invalid magic number in " << filename << ": " << magic_number << endl;
    return {};
  }

  uint32_t num_labels = read_int(file);
  cout << "Loading " << num_labels << " labels" << endl;

  vector<double> labels(num_labels);
  
  for (uint32_t i = 0; i < num_labels; ++i) {
    uint8_t label;
    file.read(reinterpret_cast<char*>(&label), 1);
    labels[i] = static_cast<double>(label);
  }

  return labels;
}

// Function to convert a label to one-hot encoded vector
vector<double> one_hot_encode(double label, int num_classes = 10) {
  vector<double> one_hot(num_classes, 0.0);
  int index = static_cast<int>(label);
  if (index >= 0 && index < num_classes) {
    one_hot[index] = 1.0;
  }
  return one_hot;
}

// Function to convert all labels to one-hot encoded vectors
vector<vector<double>> convert_to_one_hot(const vector<double>& labels, int num_classes = 10) {
  vector<vector<double>> one_hot_labels;
  one_hot_labels.reserve(labels.size());
  
  for (const auto& label : labels) {
    one_hot_labels.push_back(one_hot_encode(label, num_classes));
  }
  
  return one_hot_labels;
}

// Utility function to display an image (28x28 pixels) in the console
void display_image(const std::vector<double>& image, int rows = 28, int cols = 28) {
    if (image.size() != static_cast<size_t>(rows * cols)) {
        cerr << "Invalid image dimensions" << endl;
        return;
    }
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double pixel = image[i * cols + j];
            
            // Use different characters to represent different pixel intensities
            if (pixel < 0.2) {
                cout << " ";
            } else if (pixel < 0.4) {
                cout << ".";
            } else if (pixel < 0.6) {
                cout << "+";
            } else if (pixel < 0.8) {
                cout << "o";
            } else {
                cout << "#";
            }
        }
        cout << endl;
    }
}

// Function to save an image as a PPM file (P3 format - ASCII)
void save_image_as_ppm(const std::vector<double>& image, const std::string& filename, int rows = 28, int cols = 28) {
    if (image.size() != static_cast<size_t>(rows * cols)) {
        cerr << "Invalid image dimensions" << endl;
        return;
    }
    
    ofstream file(filename);
    if (!file) {
        cerr << "Failed to create file: " << filename << endl;
        return;
    }
    
    // PPM header
    file << "P3" << endl;                  // P3 format (ASCII)
    file << cols << " " << rows << endl;   // Width and height
    file << "255" << endl;                 // Maximum color value
    
    // Write pixel data
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Convert normalized value [0,1] to [0,255]
            int pixelValue = static_cast<int>(image[i * cols + j] * 255);
            
            // Write the same value for R, G, B (grayscale)
            file << pixelValue << " " << pixelValue << " " << pixelValue;
            
            // Add space or newline
            if (j < cols - 1) {
                file << " ";
            } else {
                file << endl;
            }
        }
    }
    
    cout << "Image saved as " << filename << endl;
}

// Function to save multiple images as PPM files
void save_images_as_ppm(const vector<vector<double>>& images, const vector<double>& labels, 
                        const string& prefix, int num_images = 10, int rows = 28, int cols = 28) {
    if (images.empty() || labels.empty() || images.size() < num_images || labels.size() < num_images) {
        cerr << "Not enough images or labels to save" << endl;
        return;
    }
    
    // Create output directory if it doesn't exist
    // Windows-compatible directory creation
    _mkdir("output");
    
    for (int i = 0; i < num_images; ++i) {
        ostringstream filename;
        filename << "output/" << prefix << "_" << i << "_label_" << static_cast<int>(labels[i]) << ".ppm";
        save_image_as_ppm(images[i], filename.str(), rows, cols);
    }
    
    cout << "Saved " << num_images << " images with prefix '" << prefix << "' to the output directory" << endl;
}

// Function to save an image as a BMP file
void save_image_as_bmp(const std::vector<double>& image, const std::string& filename, int rows = 28, int cols = 28) {
    if (image.size() != static_cast<size_t>(rows * cols)) {
        cerr << "Invalid image dimensions" << endl;
        return;
    }
    
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Failed to create file: " << filename << endl;
        return;
    }
    
    // BMP header constants
    const int fileHeaderSize = 14;
    const int infoHeaderSize = 40;
    const int bytesPerPixel = 3; // RGB
    
    // No padding for 24bpp images with width=28 (padding = (4 - (width * bytesPerPixel) % 4) % 4)
    // For width=28 and bytesPerPixel=3: 28*3 = 84, 84 % 4 = 0, so padding = 0
    const int padding = (4 - (cols * bytesPerPixel) % 4) % 4;
    const int rowSize = cols * bytesPerPixel + padding;
    const int pixelDataSize = rowSize * rows;
    const int fileSize = fileHeaderSize + infoHeaderSize + pixelDataSize;
    
    // File header (14 bytes)
    uint8_t fileHeader[14] = {
        'B', 'M',                   // Signature
        0, 0, 0, 0,                 // File size (bytes)
        0, 0,                       // Reserved
        0, 0,                       // Reserved
        0, 0, 0, 0                  // Pixel data offset
    };
    
    // Update file size
    fileHeader[2] = (uint8_t)(fileSize);
    fileHeader[3] = (uint8_t)(fileSize >> 8);
    fileHeader[4] = (uint8_t)(fileSize >> 16);
    fileHeader[5] = (uint8_t)(fileSize >> 24);
    
    // Update pixel data offset
    fileHeader[10] = (uint8_t)(fileHeaderSize + infoHeaderSize);
    fileHeader[11] = (uint8_t)((fileHeaderSize + infoHeaderSize) >> 8);
    fileHeader[12] = (uint8_t)((fileHeaderSize + infoHeaderSize) >> 16);
    fileHeader[13] = (uint8_t)((fileHeaderSize + infoHeaderSize) >> 24);
    
    // Info header (40 bytes)
    uint8_t infoHeader[40] = {
        0, 0, 0, 0,                 // Info header size
        0, 0, 0, 0,                 // Image width
        0, 0, 0, 0,                 // Image height (negative for top-down)
        0, 0,                       // Number of color planes (must be 1)
        0, 0,                       // Bits per pixel
        0, 0, 0, 0,                 // Compression method (0 = none)
        0, 0, 0, 0,                 // Image size (can be 0 for uncompressed)
        0, 0, 0, 0,                 // Horizontal resolution (pixels/meter)
        0, 0, 0, 0,                 // Vertical resolution (pixels/meter)
        0, 0, 0, 0,                 // Number of colors in palette (0 = default)
        0, 0, 0, 0                  // Number of important colors (0 = all)
    };
    
    // Update info header values
    infoHeader[0] = (uint8_t)(infoHeaderSize);
    infoHeader[4] = (uint8_t)(cols);
    infoHeader[5] = (uint8_t)(cols >> 8);
    infoHeader[6] = (uint8_t)(cols >> 16);
    infoHeader[7] = (uint8_t)(cols >> 24);
    
    // Negative height for top-down image (origin at top-left)
    int32_t negHeight = -rows;
    infoHeader[8] = (uint8_t)(negHeight);
    infoHeader[9] = (uint8_t)(negHeight >> 8);
    infoHeader[10] = (uint8_t)(negHeight >> 16);
    infoHeader[11] = (uint8_t)(negHeight >> 24);
    
    infoHeader[12] = 1; // Color planes
    infoHeader[14] = (uint8_t)(bytesPerPixel * 8); // Bits per pixel
    
    // Write headers
    file.write(reinterpret_cast<char*>(fileHeader), fileHeaderSize);
    file.write(reinterpret_cast<char*>(infoHeader), infoHeaderSize);
    
    // Write pixel data
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Convert normalized value [0,1] to [0,255]
            uint8_t pixelValue = static_cast<uint8_t>(image[i * cols + j] * 255);
            
            // Write the same value for B, G, R (grayscale)
            file.write(reinterpret_cast<char*>(&pixelValue), 1); // B
            file.write(reinterpret_cast<char*>(&pixelValue), 1); // G
            file.write(reinterpret_cast<char*>(&pixelValue), 1); // R
        }
        
        // Write padding (if any)
        uint8_t paddingValue = 0;
        for (int p = 0; p < padding; ++p) {
            file.write(reinterpret_cast<char*>(&paddingValue), 1);
        }
    }
    
    cout << "Image saved as " << filename << endl;
}

// Function to save multiple images as BMP files
void save_images_as_bmp(const std::vector<std::vector<double>>& images, const std::vector<double>& labels, 
                       const std::string& prefix, int num_images = 10, int rows = 28, int cols = 28) {
    if (images.empty() || labels.empty() || images.size() < num_images || labels.size() < num_images) {
        cerr << "Not enough images or labels to save" << endl;
        return;
    }
    
    // Create output directory if it doesn't exist
    _mkdir("output");
    
    for (int i = 0; i < num_images; ++i) {
        std::ostringstream filename;
        filename << "output/" << prefix << "_" << i << "_label_" << static_cast<int>(labels[i]) << ".bmp";
        save_image_as_bmp(images[i], filename.str(), rows, cols);
    }
    
    std::cout << "Saved " << num_images << " images with prefix '" << prefix << "' to the output directory" << std::endl;
}

// Function to visualize an image as a 2D grid with values
void visualize_image_with_values(const std::vector<double>& image, int rows = 28, int cols = 28) {
    if (image.size() != static_cast<size_t>(rows * cols)) {
        std::cerr << "Invalid image dimensions" << std::endl;
        return;
    }
    
    std::cout << "Image as 2D grid with normalized values (0.0-1.0):" << std::endl;
    
    // Print column headers
    std::cout << "    ";
    for (int j = 0; j < cols; j += 2) { // Print every other column to fit in console
        std::cout << std::setw(4) << j;
    }
    std::cout << std::endl;
    
    // Print row separator
    std::cout << "    ";
    for (int j = 0; j < cols; j += 2) {
        std::cout << "----";
    }
    std::cout << std::endl;
    
    // Print each row
    for (int i = 0; i < rows; i += 2) { // Print every other row to fit in console
        std::cout << std::setw(2) << i << " | ";
        
        for (int j = 0; j < cols; j += 2) {
            double pixel = image[i * cols + j];
            
            // Print the value with fixed precision
            if (pixel < 0.01) {
                std::cout << "    "; // Empty for very small values
            } else {
                std::cout << std::fixed << std::setprecision(1) << std::setw(3) << pixel << " ";
            }
        }
        std::cout << std::endl;
    }
}

int main() {
  // Define paths to MNIST dataset files
  // Note: These files need to be downloaded and extracted manually
  string train_images_file = "data/train-images-idx3-ubyte";
  string train_labels_file = "data/train-labels-idx1-ubyte";
  string test_images_file = "data/t10k-images-idx3-ubyte";
  string test_labels_file = "data/t10k-labels-idx1-ubyte";
  
  cout << "Please ensure the following MNIST dataset files are in the current directory:" << endl;
  cout << "- " << train_images_file << endl;
  cout << "- " << train_labels_file << endl;
  cout << "- " << test_images_file << endl;
  cout << "- " << test_labels_file << endl;
    
  // Load training data
  vector<vector<double>> train_images = load_mnist_images(train_images_file);
  vector<double> train_labels = load_mnist_labels(train_labels_file);

  // Load test data
  vector<vector<double>> test_images = load_mnist_images(test_images_file);
  vector<double> test_labels = load_mnist_labels(test_labels_file);

  // Verify data was loaded
  if (train_images.empty() || train_labels.empty() || test_images.empty() || test_labels.empty()) {
    std::cerr << "Failed to load MNIST dataset" << std::endl;
    return 1;
  }

  cout << "Successfully loaded MNIST dataset:" << endl;
  cout << "Training images: " << train_images.size() << endl;
  cout << "Training labels: " << train_labels.size() << endl;
  cout << "Test images: " << test_images.size() << endl;
  cout << "Test labels: " << test_labels.size() << endl;
  
  // Convert labels to one-hot encoded vectors
  vector<vector<double>> train_labels_one_hot = convert_to_one_hot(train_labels);
  vector<vector<double>> test_labels_one_hot = convert_to_one_hot(test_labels);
  
  // Display a sample image and its label
  if (!train_images.empty()) {
    int sample_index = 0;
    cout << "\nSample image (index " << sample_index << "):" << endl;
    display_image(train_images[sample_index]);
    
    // Also show the image with values
    visualize_image_with_values(train_images[sample_index]);
    
    cout << "Label: " << static_cast<int>(train_labels[sample_index]) << endl;
    
    cout << "One-hot encoded label: [";
    for (size_t i = 0; i < train_labels_one_hot[sample_index].size(); ++i) {
      cout << train_labels_one_hot[sample_index][i];
      if (i < train_labels_one_hot[sample_index].size() - 1) {
        cout << ", ";
      }
    }
    cout << "]" << endl;
    
    // Save sample images
    _mkdir("output");
    save_image_as_ppm(train_images[sample_index], "output/sample_image.ppm");
    save_image_as_bmp(train_images[sample_index], "output/sample_image.bmp");
    
    // Save multiple images
    cout << "\nSaving 10 training images to output directory..." << endl;
    save_images_as_bmp(train_images, train_labels, "train", 10);
    
    cout << "\nSaving 10 test images to output directory..." << endl;
    save_images_as_bmp(test_images, test_labels, "test", 10);
  }
  

  // Create and train the MLP
  MLP mlp({784, 128, 10});

  // MLP training
  for(int i = 0; i < epochs; i++){
    
    for(int j = 0; j < train_images.size(); j++){
      if(j % 100 == 0){
        cout<<"Training image "<<j<<" of "<<train_images.size()<<endl;
      }
      
      cout<<"Processing image "<<j<<endl;
      
      // Forward pass
      vector<Value*> out = mlp(train_images[j]);
      
      // Calculate loss for all outputs at once
      double batch_loss = 0.0;
      for(int k = 0; k < 10; k++){
        Value* error = out[k]->mse(train_labels_one_hot[j][k]);
        error->backward();
        delete error;
      }
      
      // Track accuracy during training
      int predicted_label = 0;
      double max_prob = out[0]->getData();
      for(int k = 1; k < 10; k++){
        if(out[k]->getData() > max_prob){
          max_prob = out[k]->getData();
          predicted_label = k;
        }
      }
      
      // Update weights only at batch boundaries or at the end
      if((j+1) % batch_size == 0){
        cout << "Starting backprop for batch" << endl;
        mlp.backprop(lr);
        cout << "Backprop complete, starting cleanup" << endl;
        mlp.cleanup(); // Clean up memory after backprop
        cout << "Cleanup complete, zeroing gradients" << endl;
        mlp.zerograd();
        cout<<"Batch "<<j/batch_size<<" completed"<<endl;
      }
    }
    
    // Print epoch statistics
    cout << "Epoch " << i+1 << " completed. "<<endl;
    break; // Only run one epoch for debugging
  }

  // Test the MLP
  int correct = 0;
  int total = test_images.size();

  for(int i = 0; i < total; i++){
    vector<Value*> out = mlp(test_images[i]);
    int predicted_label = 0;
    double max_prob = out[0]->getData();
    for(int j = 1; j < 10; j++){
      if(out[j]->getData() > max_prob){
        max_prob = out[j]->getData();
        predicted_label = j;
      }
    }
    if(test_labels[i] == predicted_label){
      cout<<"Predicted label: "<<predicted_label<<" True label: "<<test_labels[i]<<endl;
      correct++;
    }
    
    // Clean up after test prediction
    mlp.cleanup();
  }
  cout << "Accuracy: " << (double)correct / total << endl;

  return 0;
}
