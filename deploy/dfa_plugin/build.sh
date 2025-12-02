#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
# DFA Plugin编译脚本

# 获取脚本所在目录的父目录（deploy目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到deploy目录并source环境设置
cd "$DEPLOY_DIR" || exit 1
source tools/set_env.sh || exit 1

# 切换到dfa_plugin目录
cd "$SCRIPT_DIR" || exit 1

# 清理旧的构建文件
echo "[INFO] Cleaning previous build..."
make clean

# 编译插件（使用DEBUG=1以包含调试符号）
echo "[INFO] Building DFA plugin with DEBUG=1..."
make DEBUG=1

# 检查编译结果
if [ $? -eq 0 ]; then
    echo "[INFO] Plugin compiled successfully!"
    echo "[INFO] Output: $(pwd)/lib/deformableAttentionAggr.so"
else
    echo "[ERROR] Plugin compilation failed!"
    exit 1
fi

