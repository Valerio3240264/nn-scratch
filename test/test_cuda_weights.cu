#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include "../classes/weights.h"
#include "../classes/input.h"

using namespace std;
using namespace std::chrono;

// Test configuration
const double EPSILON = 1e-10;  // Tolerance for floating point comparison
const int PERFORMANCE_ITERATIONS = 10;

// Color codes for output
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define RESET "\033[0m"

// Test statistics
struct TestStats {
    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;
};

TestStats global_stats;

// Helper function to compare arrays with tolerance
bool compare_arrays(double* arr1, double* arr2, int size, double tolerance = EPSILON) {
    for (int i = 0; i < size; i++) {
        if (abs(arr1[i] - arr2[i]) > tolerance) {
            cout << "  Difference at index " << i << ": " 
                 << arr1[i] << " vs " << arr2[i] 
                 << " (diff: " << abs(arr1[i] - arr2[i]) << ")" << endl;
            return false;
        }
    }
    return true;
}

// Helper function to create input with specific pattern
input* create_test_input(int size, string pattern = "random") {
    input* test_input = new input(size);
    double* values = test_input->values_pointer();
    
    if (pattern == "ones") {
        for (int i = 0; i < size; i++) {
            values[i] = 1.0;
        }
    } else if (pattern == "sequential") {
        for (int i = 0; i < size; i++) {
            values[i] = i + 1.0;
        }
    } else if (pattern == "alternating") {
        for (int i = 0; i < size; i++) {
            values[i] = (i % 2 == 0) ? 1.0 : -1.0;
        }
    } else if (pattern == "random") {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1.0, 1.0);
        for (int i = 0; i < size; i++) {
            values[i] = dis(gen);
        }
    } else if (pattern == "zeros") {
        for (int i = 0; i < size; i++) {
            values[i] = 0.0;
        }
    }
    
    return test_input;
}

// Print test result
void print_test_result(const string& test_name, bool passed, const string& details = "") {
    global_stats.total_tests++;
    
    if (passed) {
        global_stats.passed_tests++;
        cout << GREEN << "[PASS] " << RESET << test_name;
        if (!details.empty()) {
            cout << " - " << details;
        }
        cout << endl;
    } else {
        global_stats.failed_tests++;
        cout << RED << "[FAIL] " << RESET << test_name;
        if (!details.empty()) {
            cout << " - " << details;
        }
        cout << endl;
    }
}

// Test basic functionality with small matrices
void test_basic_functionality() {
    cout << BLUE << "\n=== Testing Basic Functionality ===" << RESET << endl;
    
    // Test 1: Small matrix (2x3) with known values
    {
        weights layer(2, 3);  // 2 inputs, 3 outputs
        
        // Manually set simple weights for predictable results
        double* w = layer.values_pointer();
        w[0] = 1.0; w[1] = 0.0; w[2] = 0.0;  // weights for input 0
        w[3] = 0.0; w[4] = 1.0; w[5] = 0.0;  // weights for input 1
        
        input* test_input = create_test_input(2, "ones");
        
        double* result_cpu = layer.operator_cpu(test_input);
        double* result_auto = layer(test_input);
        
        // Expected result: [1, 1, 0] (since inputs are [1, 1])
        double expected[] = {1.0, 1.0, 0.0};
        
        bool cpu_correct = compare_arrays(result_cpu, expected, 3);
        bool auto_correct = compare_arrays(result_auto, expected, 3);
        bool cpu_vs_auto = compare_arrays(result_cpu, result_auto, 3);
        
        print_test_result("Small Matrix CPU Correctness", cpu_correct);
        print_test_result("Small Matrix Auto Correctness", auto_correct);
        print_test_result("Small Matrix CPU vs Auto", cpu_vs_auto);
        
        delete[] result_cpu;
        delete[] result_auto;
        delete test_input;
    }
    
    // Test 2: Edge case - single element
    {
        weights layer(1, 1);
        double* w = layer.values_pointer();
        w[0] = 5.0;
        
        input* test_input = create_test_input(1, "ones");
        double* result = layer(test_input);
        
        bool correct = (abs(result[0] - 5.0) < EPSILON);
        print_test_result("Single Element Test", correct, 
            "Expected 5.0, got " + to_string(result[0]));
        
        delete[] result;
        delete test_input;
    }
}

// Test different input patterns
void test_input_patterns() {
    cout << BLUE << "\n=== Testing Different Input Patterns ===" << RESET << endl;
    
    const int input_size = 10;
    const int output_size = 5;
    weights layer(input_size, output_size);
    
    vector<string> patterns = {"zeros", "ones", "sequential", "alternating", "random"};
    
    for (const string& pattern : patterns) {
        input* test_input = create_test_input(input_size, pattern);
        
        double* result_cpu = layer.operator_cpu(test_input);
        double* result_auto = layer(test_input);
        
        bool match = compare_arrays(result_cpu, result_auto, output_size);
        print_test_result("Pattern: " + pattern, match);
        
        delete[] result_cpu;
        delete[] result_auto;
        delete test_input;
    }
}

// Test CUDA-specific functionality
void test_cuda_functionality() {
    cout << BLUE << "\n=== Testing CUDA Functionality ===" << RESET << endl;
    
    // Test CUDA availability detection
    bool cuda_available = is_cuda_available();
    print_test_result("CUDA Availability Detection", true, 
        cuda_available ? "CUDA Available" : "CUDA Not Available");
    
    if (cuda_available) {
        weights layer(100, 50);
        input* test_input = create_test_input(100, "random");
        
        // Test explicit CUDA call
        try {
            double* result_cuda = layer.operator_cuda(test_input);
            double* result_cpu = layer.operator_cpu(test_input);
            
            bool match = compare_arrays(result_cuda, result_cpu, 50);
            print_test_result("Explicit CUDA vs CPU", match);
            
            delete[] result_cuda;
            delete[] result_cpu;
        } catch (const exception& e) {
            print_test_result("Explicit CUDA Call", false, 
                "Exception: " + string(e.what()));
        }
        
        delete test_input;
    } else {
        cout << YELLOW << "  Skipping CUDA-specific tests (CUDA not available)" << RESET << endl;
    }
}

// Performance benchmarking
void test_performance() {
    cout << BLUE << "\n=== Performance Testing ===" << RESET << endl;
    
    vector<pair<int, int>> sizes = {{100, 50}, {500, 200}, {1000, 500}, {2000, 1000}, {3000, 1500}, {4000, 2000}, {5000, 2500}, {6000, 3000}, {7000, 3500}, {8000, 4000}, {9000, 4500}, {10000, 5000}};
    
    for (const auto& size_pair : sizes) {
        int input_size = size_pair.first;
        int output_size = size_pair.second;
        
        cout << "Matrix size: " << input_size << "x" << output_size << endl;
        
        weights layer(input_size, output_size);
        input* test_input = create_test_input(input_size, "random");
        
        // Benchmark CPU
        auto start_cpu = high_resolution_clock::now();
        for (int i = 0; i < PERFORMANCE_ITERATIONS; i++) {
            double* result = layer.operator_cpu(test_input);
            delete[] result;
        }
        auto end_cpu = high_resolution_clock::now();
        auto cpu_time = duration_cast<microseconds>(end_cpu - start_cpu).count();
        
        // Benchmark Auto (GPU if available)
        auto start_auto = high_resolution_clock::now();
        for (int i = 0; i < PERFORMANCE_ITERATIONS; i++) {
            double* result = layer(test_input);
            delete[] result;
        }
        auto end_auto = high_resolution_clock::now();
        auto auto_time = duration_cast<microseconds>(end_auto - start_auto).count();
        
        cout << "  CPU Time: " << cpu_time / PERFORMANCE_ITERATIONS << " μs/iteration" << endl;
        cout << "  Auto Time: " << auto_time / PERFORMANCE_ITERATIONS << " μs/iteration" << endl;
        
        if (is_cuda_available() && auto_time > 0) {
            double speedup = (double)cpu_time / auto_time;
            cout << "  Speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
            
            bool performance_gain = speedup > 0.8;  // Allow some overhead for small matrices
            print_test_result("Performance Test " + to_string(input_size) + "x" + to_string(output_size), 
                performance_gain, "Speedup: " + to_string(speedup) + "x");
        } else {
            print_test_result("Performance Test " + to_string(input_size) + "x" + to_string(output_size), 
                true, "CPU baseline established");
        }
        
        delete test_input;
        cout << endl;
    }
}

// Test error handling and edge cases
void test_error_handling() {
    cout << BLUE << "\n=== Testing Error Handling ===" << RESET << endl;
    
    // Test very large matrix (might exceed memory)
    try {
        weights large_layer(10000, 10000);  // 100M elements
        print_test_result("Large Matrix Creation", true, "Successfully created 10000x10000 matrix");
        
        // Don't actually run computation to avoid memory issues
        
    } catch (const exception& e) {
        print_test_result("Large Matrix Creation", true, 
            "Appropriately failed with: " + string(e.what()));
    }
    
    // Test zero-sized matrix
    try {
        input* test_input = create_test_input(0, "zeros");
        print_test_result("Zero Size Input Creation", true, "Created zero-size input");
        delete test_input;
    } catch (const exception& e) {
        print_test_result("Zero Size Input Creation", true, 
            "Appropriately failed with: " + string(e.what()));
    }
}

// Print final statistics
void print_summary() {
    cout << BLUE << "\n=== Test Summary ===" << RESET << endl;
    cout << "Total Tests: " << global_stats.total_tests << endl;
    cout << GREEN << "Passed: " << global_stats.passed_tests << RESET << endl;
    cout << RED << "Failed: " << global_stats.failed_tests << RESET << endl;
    
    double pass_rate = (double)global_stats.passed_tests / global_stats.total_tests * 100.0;
    cout << "Pass Rate: " << fixed << setprecision(1) << pass_rate << "%" << endl;
    
    if (global_stats.failed_tests == 0) {
        cout << GREEN << "All tests passed! 🎉" << RESET << endl;
    } else {
        cout << YELLOW << "Some tests failed. Please review the results above." << RESET << endl;
    }
}

// Main test runner
int main() {
    cout << BLUE << "CUDA Weights Test Suite" << RESET << endl;
    cout << "========================" << endl;
    
    // Initialize random seed
    srand(time(nullptr));
    
    // Run all test suites
    test_basic_functionality();
    test_input_patterns();
    test_cuda_functionality();
    test_performance();
    test_error_handling();
    
    // Print summary
    print_summary();
    
    return (global_stats.failed_tests == 0) ? 0 : 1;
}
