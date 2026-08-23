NVCC ?= nvcc
CUDA_ARCH ?= sm_90
TARGET = dense_lu_panel_update_tc

.PHONY: all clean

all: $(TARGET)

$(TARGET): dense_lu_panel_update_tc.cu
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) \
		-lcublas \
		dense_lu_panel_update_tc.cu -o $(TARGET)

clean:
	$(RM) $(TARGET)
