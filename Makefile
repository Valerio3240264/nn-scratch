# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -g
LDFLAGS = 

# Directories
CLASSES_DIR = classes
TESTS_DIR = tests

# Source files
CATCH_SOURCES = catch_amalgamated.cpp
TEST_SOURCES = $(TESTS_DIR)/test_runner.cpp
MAIN_SOURCE = test.cpp

# Object files
CATCH_OBJECTS = catch_amalgamated.o
TEST_OBJECTS = $(TESTS_DIR)/test_runner.o
MAIN_OBJECTS = test.o

# Executables
TEST_EXECUTABLE = run_tests
MAIN_EXECUTABLE = test

# Default target
all: $(MAIN_EXECUTABLE) $(TEST_EXECUTABLE)

# Build main application
$(MAIN_EXECUTABLE): $(MAIN_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build test executable
$(TEST_EXECUTABLE): $(CATCH_OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile Catch2 implementation
$(CATCH_OBJECTS): $(CATCH_SOURCES)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compile test runner
$(TEST_OBJECTS): $(TEST_SOURCES)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compile main application
$(MAIN_OBJECTS): $(MAIN_SOURCE)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Run tests
test: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE)

# Run tests with verbose output
test-verbose: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE) -v

# Run specific test tags
test-input: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE) "[input]"

test-activation: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE) "[activation_function]"

test-weights: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE) "[weights]"

test-integration: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE) "[integration]"

# Run main application
run: $(MAIN_EXECUTABLE)
	./$(MAIN_EXECUTABLE)

# Clean build artifacts
clean:
	rm -f *.o $(TESTS_DIR)/*.o $(TEST_EXECUTABLE) $(MAIN_EXECUTABLE)

# Clean everything including downloads
clean-all: clean
	rm -f catch_amalgamated.hpp catch_amalgamated.cpp

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build both main application and tests"
	@echo "  $(MAIN_EXECUTABLE)           - Build main application"
	@echo "  $(TEST_EXECUTABLE)       - Build test suite"
	@echo "  test             - Run all tests"
	@echo "  test-verbose     - Run all tests with verbose output"
	@echo "  test-input       - Run only input class tests"
	@echo "  test-activation  - Run only activation function tests"
	@echo "  test-weights     - Run only weights class tests"
	@echo "  test-integration - Run only integration tests"
	@echo "  run              - Run the main application"
	@echo "  clean            - Remove build artifacts"
	@echo "  clean-all        - Remove everything including downloads"
	@echo "  help             - Show this help message"

.PHONY: all test test-verbose test-input test-activation test-weights test-integration run clean clean-all help 