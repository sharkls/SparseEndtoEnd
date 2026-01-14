# Copyright (c) 2024 SparseEnd2End. All rights reserved.

TARGET          ?= lib/SparseBox3DKeyPointsPlugin.so
BUILD_DIR       ?= build
SRC_CPP         := SparseBox3DKeyPointsPlugin.cpp
SRC_CU          := SparseBox3DKeyPointsKernel.cu

CUDA_BIN        ?= $(ENV_CUDA_BIN)
NVCC            ?= $(CUDA_BIN)/nvcc
CUDASM          ?= $(cudasm)

# 检查环境变量是否设置
ifeq ($(strip $(ENV_CUDA_BIN)),)
  $(error ENV_CUDA_BIN is not set. Please run: source ../tools/set_env.sh or use ./build.sh)
endif

ifeq ($(strip $(CUDASM)),)
  # 检测平台架构，设置默认 CUDASM
  ARCH := $(shell uname -m)
  ifeq ($(ARCH),aarch64)
    CUDASM := 87  # Jetson Orin 使用 sm_87
  else
    CUDASM := 86  # x86_64 平台使用 sm_86
  endif
endif

# 检查 nvcc 是否存在
ifeq ($(wildcard $(NVCC)),)
  $(error nvcc not found at $(NVCC). Please check ENV_CUDA_BIN in set_env.sh)
endif

INCLUDES        := -I$(ENV_CUDA_INC) -I$(ENV_TensorRT_INC)
LIBRARIES       := -L$(ENV_CUDA_LIB) -L$(ENV_TensorRT_LIB) -lcudart -lnvinfer

# 支持 DEBUG_NAN 编译选项
DEBUG_NAN       ?= 0
CUDAFLAGS_BASE  := --shared -Xcompiler -fPIC -O3 \
                   -gencode=arch=compute_$(CUDASM),code=compute_$(CUDASM) \
                   -gencode=arch=compute_$(CUDASM),code=sm_$(CUDASM)

# 如果启用 DEBUG_NAN，添加编译选项
ifeq ($(DEBUG_NAN),1)
    CUDAFLAGS := $(CUDAFLAGS_BASE) -DDEBUG_NAN
else
    CUDAFLAGS := $(CUDAFLAGS_BASE)
endif

OBJS            := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRC_CPP)) \
                   $(patsubst %.cu,$(BUILD_DIR)/%.o,$(SRC_CU))

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	@echo "[Link] $@"
	@$(NVCC) $(CUDAFLAGS) $^ -o $@ $(INCLUDES) $(LIBRARIES)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	@echo "[NVCC] $<"
	@$(NVCC) $(CUDAFLAGS) -c $< -o $@ $(INCLUDES)

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(BUILD_DIR)
	@echo "[NVCC] $<"
	@$(NVCC) $(CUDAFLAGS) -c $< -o $@ $(INCLUDES)

clean:
	@rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean


