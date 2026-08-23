NVCC ?= nvcc
CUDA_ARCH ?= sm_90
TARGET = backprop_wmma

.PHONY: all clean

all: $(TARGET)

$(TARGET): backprop_wmma.cu ../../cuda_common/h100_mixed_precision.cuh ../../cuda_common/h100_profiling.cuh
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) \
		backprop_wmma.cu -o $(TARGET)

clean:
	$(RM) $(TARGET)
