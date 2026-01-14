#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved.
# SparseBox3DKeyPointsPlugin 编译脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 加载环境变量
if [ -f "${DEPLOY_DIR}/tools/set_env.sh" ]; then
    # 切换到 deploy 目录，因为 set_env.sh 中的路径是相对路径
    cd "${DEPLOY_DIR}"
    source "tools/set_env.sh"
    cd "${SCRIPT_DIR}"
else
    echo "[ERROR] Cannot find set_env.sh at ${DEPLOY_DIR}/tools/set_env.sh"
    exit 1
fi

# 如果 CUDASM 未设置，设置默认值（Jetson Orin 使用 sm_87）
if [ -z "${CUDASM}" ]; then
    # 检测平台架构
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        CUDASM=87  # Jetson Orin 使用 sm_87
    else
        CUDASM=86  # x86_64 平台使用 sm_86
    fi
    echo "[WARNING] CUDASM not set, using default for ${ARCH}: ${CUDASM}"
    export CUDASM
fi

# 检查必要的环境变量
if [ -z "${ENV_CUDA_BIN}" ] || [ -z "${ENV_TensorRT_INC}" ]; then
    echo "[ERROR] Environment variables not set. Please check set_env.sh"
    exit 1
fi

# 检查 nvcc 是否存在
if [ ! -f "${ENV_CUDA_BIN}/nvcc" ]; then
    echo "[ERROR] nvcc not found at ${ENV_CUDA_BIN}/nvcc"
    exit 1
fi

echo "[INFO] Building SparseBox3DKeyPointsPlugin..."
echo "[INFO] CUDA_BIN: ${ENV_CUDA_BIN}"
echo "[INFO] TensorRT_INC: ${ENV_TensorRT_INC}"
echo "[INFO] CUDASM: ${CUDASM:-87}"

# 切换到插件目录
cd "${SCRIPT_DIR}"

# 清理旧的构建文件（确保重新编译）
echo "[INFO] Cleaning previous build artifacts..."
make clean 2>/dev/null || true
rm -rf build/*.o 2>/dev/null || true

# 运行 make
make

if [ $? -eq 0 ]; then
    echo "[INFO] Build successful!"
    if [ -f "lib/SparseBox3DKeyPointsPlugin.so" ]; then
        echo "[INFO] Plugin library: lib/SparseBox3DKeyPointsPlugin.so"
        ls -lh lib/SparseBox3DKeyPointsPlugin.so
    fi
else
    echo "[ERROR] Build failed!"
    exit 1
fi

