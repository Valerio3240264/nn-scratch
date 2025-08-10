# Neural Network Testing Framework

This directory contains comprehensive tests for the neural network classes using the Catch2 testing framework.

## Test Structure

### Individual Class Tests

1. **test_input.cpp** - Tests for the `input` class
   - Constructor tests (size-only, with values, with predecessor)
   - Backward propagation tests
   - Edge cases (single element, large inputs)
   - Memory management

2. **test_activation_function.cpp** - Tests for the `activation_function` class
   - Constructor and basic setup
   - TANH activation forward pass
   - TANH backward propagation
   - Edge cases (single element, large values)
   - Error handling for unsupported activation functions

3. **test_weights.cpp** - Tests for the `weights` class
   - Constructor and initialization
   - Forward pass variations (1→1, N→1, 1→N, N→M)
   - Backward propagation and gradient computation
   - Edge cases (zero inputs, large values, negative inputs)
   - Memory management

4. **test_integration.cpp** - Integration tests
   - Complete neural network layers (input → weights → activation)
   - Multi-layer networks
   - Learning simulation (gradient descent step)
   - Memory management across multiple operations
   - Edge cases in full network context

## Running Tests

### Prerequisites
- C++ compiler with C++20 support (g++, clang++)
- Make (optional, for convenience)

### Using Make (Recommended)

```bash
# Build and run all tests
make test

# Run tests with verbose output
make test-verbose

# Run specific test categories
make test-input        # Only input class tests
make test-activation   # Only activation function tests
make test-weights      # Only weights class tests
make test-integration  # Only integration tests

# Build test executable without running
make run_tests

# Clean build artifacts
make clean
```

### Manual Compilation

```bash
# Compile Catch2 implementation
g++ -std=c++20 -c catch_amalgamated.cpp -o catch_amalgamated.o

# Compile and run tests
g++ -std=c++20 -Wall -Wextra -g catch_amalgamated.o tests/test_runner.cpp -o run_tests
./run_tests
```

## Test Categories and Tags

Tests are organized with tags for easy filtering:

- `[input]` - Input class tests
- `[activation_function]` - Activation function tests
- `[weights]` - Weights class tests
- `[integration]` - Integration tests

### Running Specific Tests

```bash
# Run only input tests
./run_tests "[input]"

# Run multiple categories
./run_tests "[input],[weights]"

# Run with verbose output
./run_tests -v

# List all available tests
./run_tests --list-tests
```

## Test Coverage

### Input Class Coverage
- ✅ Constructor variations
- ✅ Value and gradient access
- ✅ Backward propagation
- ✅ Predecessor chaining
- ✅ Edge cases

### Activation Function Coverage
- ✅ TANH activation function
- ✅ Forward pass computation
- ✅ Backward pass (gradient computation)
- ✅ Interface compliance (BackwardClass)
- ✅ Error handling for unsupported functions

### Weights Class Coverage
- ✅ Matrix multiplication (forward pass)
- ✅ Gradient computation (backward pass)
- ✅ Various input/output size combinations
- ✅ Weight initialization
- ✅ Interface compliance (BackwardClass)

### Integration Coverage
- ✅ Full neural network layers
- ✅ Multi-layer networks
- ✅ Gradient flow through entire network
- ✅ Learning simulation
- ✅ Memory management
- ✅ Edge cases in realistic scenarios

## Test Quality Features

### Numerical Precision
- Uses `1e-10` tolerance for floating-point comparisons
- Tests mathematical correctness of computations
- Verifies gradient calculations against analytical derivatives

### Memory Safety
- Tests for memory leaks in repeated operations
- Verifies proper cleanup in destructors
- Tests edge cases that might cause memory issues

### Error Handling
- Tests for proper exception throwing
- Verifies error messages for invalid operations
- Tests boundary conditions

### Realistic Scenarios
- Integration tests simulate actual neural network usage
- Tests gradient descent learning steps
- Verifies end-to-end functionality

## Adding New Tests

### Test Structure
```cpp
TEST_CASE("Test description", "[tag]") {
    SECTION("Specific scenario") {
        // Setup
        // Execute
        // Verify with REQUIRE()
    }
}
```

### Best Practices
1. Use descriptive test names
2. Test one concept per SECTION
3. Use appropriate tolerance for floating-point comparisons
4. Clean up dynamically allocated memory
5. Test both success and failure cases
6. Include edge cases

### Example Test
```cpp
TEST_CASE("New feature test", "[your_class]") {
    SECTION("Basic functionality") {
        YourClass obj(params);
        
        double result = obj.someMethod();
        
        REQUIRE(abs(result - expected) < 1e-10);
    }
    
    SECTION("Edge case") {
        // Test edge case
    }
}
```

## Known Issues

1. **Memory Leak in weights::operator()**: The current implementation allocates memory for output but may not properly manage it in some scenarios.

2. **Weight Matrix Indexing**: The weight matrix indexing in the weights class might need verification for correctness.

3. **Gradient Accumulation**: Some tests document current behavior that might need adjustment based on intended neural network semantics.

## Future Improvements

1. Add performance benchmarks
2. Add more activation functions (ReLU, Sigmoid, etc.)
3. Add bias term support to weights class
4. Add more sophisticated learning algorithms
5. Add serialization/deserialization tests
6. Add numerical gradient checking 