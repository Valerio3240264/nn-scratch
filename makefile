# Compilers
CXX = g++
NVCC = nvcc

# Compilation flags
CXXFLAGS = -std=c++17 -O2 -I.
NVCCFLAGS = -std=c++17 -O2 -use_fast_math -rdc=true -I.

# Folders
BUILD_DIR = build
BIN_DIR = bin

.PHONY: all run_cpu_batch_test test_cuda clean

#-------------------------
# Sources and Objects
#-------------------------

# CPU sources and objects
CPU_SRC = classes/cpu/src/input.cpp \
          classes/cpu/src/weights.cpp \
          classes/cpu/src/activation.cpp \
          classes/cpu/src/mse_loss.cpp \
          classes/cpu/src/cross_entropy_loss.cpp \
					utils/MatricesOp.cpp

CPU_OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CPU_SRC))

# CUDA sources and objects
CUDA_SRC = classes/cuda/src/cuda_input.cu \
           classes/cuda/src/cuda_weights.cu \
           classes/cuda/src/cuda_activation.cu \
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
             Kernels/matrix.cu

KERNEL_OBJ = $(patsubst %.cu,$(BUILD_DIR)/%.o,$(KERNEL_SRC))

#-------------------------
# Test files
#-------------------------
CPU_BATCH_TEST = $(BUILD_DIR)/test/test_cpu.o
CUDA_BATCH_TEST = $(BUILD_DIR)/test/test_cuda.o

#-------------------------
# Executables
#-------------------------
CPU_BATCH_EXE = $(BIN_DIR)/test_cpu.exe
CUDA_BATCH_EXE = $(BIN_DIR)/test_cuda.exe

#-------------------------
# Compile rules
#-------------------------

# Compile CPU objects
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test objects explicitly
$(CPU_BATCH_TEST): test/test_cpu.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CUDA_BATCH_TEST): test/test_cuda.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile CUDA objects (for GPU build)
$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile CPU MLP objects (treat .cu files as C++)
$(BUILD_DIR)/classes/mlp/src/mlp_cpu.o: classes/mlp/src/mlp.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

$(BUILD_DIR)/classes/mlp/src/layer_cpu.o: classes/mlp/src/layer.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

#-------------------------
# Build executables
#-------------------------
$(CPU_BATCH_EXE): $(CPU_OBJ) $(MLP_CPU_OBJ) $(CPU_BATCH_TEST)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(CUDA_BATCH_EXE): $(CPU_OBJ) $(CUDA_OBJ) $(MLP_GPU_OBJ) $(KERNEL_OBJ) $(CUDA_BATCH_TEST)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $^ -lcurand -o $@

#-------------------------
# Default target
#-------------------------
all: $(CPU_BATCH_EXE) $(CUDA_BATCH_EXE)

run_cpu_batch_test: $(CPU_BATCH_EXE)
	./$(CPU_BATCH_EXE)

test_cuda: $(CUDA_BATCH_EXE)

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
	rm -rf $(BUILD_DIR)/utils/*.o
	rm -rf $(BUILD_DIR)/*.o $(BIN_DIR)/*
