# Compilers
CXX = g++
NVCC = nvcc

# Compilation flags
CXXFLAGS = -std=c++17 -O2 -I.
NVCCFLAGS = -std=c++17 -O2 -use_fast_math -rdc=true -I.

# Folders
BUILD_DIR = build
BIN_DIR = bin

#-------------------------
# Sources and Objects
#-------------------------

# CPU sources and objects
CPU_SRC = classes/cpu/src/input.cpp \
          classes/cpu/src/weights.cpp \
          classes/cpu/src/activation.cpp \
          classes/cpu/src/softmax.cpp \
          classes/cpu/src/mse_loss.cpp \
          classes/cpu/src/cross_entropy_loss.cpp

CPU_OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CPU_SRC))

# CUDA sources and objects
CUDA_SRC = classes/cuda/src/cuda_input.cu \
           classes/cuda/src/cuda_weights.cu \
           classes/cuda/src/cuda_activation.cu \
           classes/cuda/src/cuda_softmax.cu \
           classes/cuda/src/cuda_mse_loss.cu \
           classes/cuda/src/cuda_cross_entropy_loss.cu

CUDA_OBJ = $(patsubst %.cu,$(BUILD_DIR)/%.o,$(CUDA_SRC))

# MLP sources
MLP_SRC = classes/mlp/src/mlp.cu \
          classes/mlp/src/layer.cu

# CPU MLP objects (compiled as C++ with g++)
MLP_CPU_OBJ = $(BUILD_DIR)/classes/mlp/src/mlp_cpu.o \
              $(BUILD_DIR)/classes/mlp/src/layer_cpu.o

# GPU MLP objects (compiled with nvcc)
MLP_GPU_OBJ = $(patsubst %.cu,$(BUILD_DIR)/%.o,$(MLP_SRC))

# Kernel operations
KERNEL_SRC = Kernels/activation.cu \
             Kernels/loss.cu \
             Kernels/matrix.cu \
             Kernels/softmax.cu

KERNEL_OBJ = $(patsubst %.cu,$(BUILD_DIR)/%.o,$(KERNEL_SRC))

#-------------------------
# Test files
#-------------------------
CPU_TEST = $(BUILD_DIR)/test/test_mnist_cpu.o
GPU_TEST = $(BUILD_DIR)/test/test_mnist_gpu.o

#-------------------------
# Executables
#-------------------------
CPU_EXE = $(BIN_DIR)/test_mnist_cpu.exe
GPU_EXE = $(BIN_DIR)/test_mnist_gpu.exe

#-------------------------
# Compile rules
#-------------------------

# Compile CPU objects
$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA objects (for GPU build)
$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile CPU MLP objects (treat .cu files as C++)
$(BUILD_DIR)/classes/mlp/src/mlp_cpu.o: classes/mlp/src/mlp.cu
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

$(BUILD_DIR)/classes/mlp/src/layer_cpu.o: classes/mlp/src/layer.cu
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

#-------------------------
# Build executables
#-------------------------
$(CPU_EXE): $(CPU_OBJ) $(MLP_CPU_OBJ) $(CPU_TEST)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(GPU_EXE): $(CPU_OBJ) $(CUDA_OBJ) $(MLP_GPU_OBJ) $(KERNEL_OBJ) $(GPU_TEST)
	$(NVCC) $(NVCCFLAGS) $^ -lcurand -o $@

#-------------------------
# Default target
#-------------------------
all: $(CPU_EXE) $(GPU_EXE)

#-------------------------
# Clean build files
#-------------------------
clean:
	rm -rf $(BUILD_DIR)/*.o $(BIN_DIR)/*
	rm -rf $(BUILD_DIR)/classes/mlp/src/*.o
	rm -rf $(BUILD_DIR)/classes/cuda/src/*.o
	rm -rf $(BUILD_DIR)/Kernels/*.o
	rm -rf $(BUILD_DIR)/test/*.o
	rm -rf $(BUILD_DIR)/classes/cpu/src/*.o
	rm -rf $(BUILD_DIR)/classes/cuda/src/*.o
	rm -rf $(BUILD_DIR)/Kernels/*.o
	rm -rf $(BUILD_DIR)/test/*.o
	rm -rf $(BUILD_DIR)/classes/cpu/src/*.o
	rm -rf $(BUILD_DIR)/classes/cuda/src/*.o
	rm -rf $(BUILD_DIR)/*.o $(BIN_DIR)/*