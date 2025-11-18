# Copyright (c) 2024 SparseEnd2End. All rights reserved.

# 默认设置（可被外部环境覆盖）
TARGET          ?= lib/customLayerNorm.so
BUILD_DIR       ?= build
SRC_FILE        ?= custom_layernorm.cu

CUDA_BIN        ?= $(ENV_CUDA_BIN)
NVCC            ?= $(CUDA_BIN)/nvcc
CUDASM          ?= $(cudasm)

# 兼容未加载 set_env.sh 的情况
ifeq ($(strip $(CUDASM)),)
  CUDASM := 86
endif

INCLUDES        := -I$(ENV_CUDA_INC) -I$(ENV_TensorRT_INC)
LIBRARIES       := -L$(ENV_CUDA_LIB) -L$(ENV_TensorRT_LIB) -lcudart -lnvinfer

CUDAFLAGS       := --shared -Xcompiler -fPIC -O3 \
                   -gencode=arch=compute_$(CUDASM),code=compute_$(CUDASM) \
                   -gencode=arch=compute_$(CUDASM),code=sm_$(CUDASM)

all: $(TARGET)

$(TARGET): $(BUILD_DIR)/custom_layernorm.o
	@mkdir -p $(dir $@)
	@echo "[Link] $@"
	@$(NVCC) $(CUDAFLAGS) $^ -o $@ $(INCLUDES) $(LIBRARIES)

$(BUILD_DIR)/custom_layernorm.o: $(SRC_FILE)
	@mkdir -p $(BUILD_DIR)
	@echo "[NVCC] $<"
	@$(NVCC) $(CUDAFLAGS) -c $< -o $@ $(INCLUDES)

clean:
	@rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean

