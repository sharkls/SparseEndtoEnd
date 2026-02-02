#!/bin/bash

# 加载环境设置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# 设置错误即停止
set -e

# 1. 确保必要的目录存在
mkdir -p deploy/engine
mkdir -p deploy/onnx

# 2. 检查并生成修复后的 ONNX
ORIG_ONNX="deploy/onnx/sparse4dbackbone.onnx"
FIXED_ONNX="deploy/onnx/sparse4dbackbone_fixed.onnx"

echo "[INFO] Checking ONNX model..."
if [ ! -f "$ORIG_ONNX" ]; then
    echo "❌ ERROR: Original ONNX $ORIG_ONNX not found!"
    exit 1
fi

# 始终重新运行 fix_onnx.py 以确保使用的是最新的修复逻辑
echo "[INFO] Running fix_onnx.py to prepare model..."
python3 deploy/fix_onnx.py --input_onnx "$ORIG_ONNX" --output_onnx "$FIXED_ONNX"

if [ ! -f "$FIXED_ONNX" ]; then
    echo "❌ ERROR: fix_onnx.py failed to generate $FIXED_ONNX"
    exit 1
fi

# 3. 定义构建参数
ENGINE_PATH="deploy/engine/sparse4dbackbone_int8_real.engine"
CACHE_PATH="deploy/engine/backbone_int8.cache"
DATA_LOADER="deploy/backbone_calib_data.py:get_data_loader"

# 4. 环境变量准备
export PATH=$PATH:$HOME/.local/bin
# 确保能够找到系统安装的 tensorrt
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.10/dist-packages

# 检查 polygraphy 是否可用
if ! command -v polygraphy &> /dev/null; then
    echo "❌ ERROR: polygraphy not found. Please run: pip3 install polygraphy"
    exit 1
fi

echo "========================================================"
echo "🚀 Building Backbone INT8 Engine using Polygraphy"
echo "[Model]  : $FIXED_ONNX"
echo "[Output] : $ENGINE_PATH"
echo "[Config] : INT8 + FP16 fallback, Workspace 2G"
echo "========================================================"

# 5. 执行构建
# 注意：Polygraphy 0.49+ 的参数：
# --calibration-cache (旧版可能是 --calib-cache)
# --data-loader-script (旧版可能是 --data-loader-fetch-factory)

polygraphy run "$FIXED_ONNX" \
    --trt \
    --int8 \
    --fp16 \
    --save-engine "$ENGINE_PATH" \
    --data-loader-script "$DATA_LOADER" \
    --calibration-cache "$CACHE_PATH" \
    --trt-min-shapes img:[1,6,3,256,704] \
    --trt-opt-shapes img:[1,6,3,256,704] \
    --trt-max-shapes img:[1,6,3,256,704] \
    --pool-limit workspace:2G \
    --verbose

echo "✅ SUCCESS: Backbone Engine has been saved to $ENGINE_PATH"
