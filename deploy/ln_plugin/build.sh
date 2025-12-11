#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved.
# LayerNorm Plugin编译脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取deploy目录 (父目录)
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到deploy目录并source环境设置
cd "$DEPLOY_DIR" || exit 1
if [ -f "tools/set_env.sh" ]; then
    source tools/set_env.sh || exit 1
else
    echo "[ERROR] tools/set_env.sh not found in $DEPLOY_DIR"
    exit 1
fi

# 切换回ln_plugin目录
cd "$SCRIPT_DIR" || exit 1

# 清理旧的构建文件
echo "[INFO] Cleaning previous build..."
make clean

# 编译插件 (Release Mode by default via Makefile O3)
echo "[INFO] Building LayerNorm plugin..."
make

# 检查编译结果
if [ $? -eq 0 ]; then
    echo "[INFO] Plugin compiled successfully!"
    echo "[INFO] Output: $(pwd)/lib/customLayerNorm.so"
else
    echo "[ERROR] Plugin compilation failed!"
    exit 1
fi

