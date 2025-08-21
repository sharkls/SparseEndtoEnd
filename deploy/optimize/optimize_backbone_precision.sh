#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

# Sparse4D Backbone 精度优化一键脚本

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=== Sparse4D Backbone 精度优化脚本 ==="
echo "当前目录: $(pwd)"
echo ""

# 检查环境
echo "1. 检查环境..."
if ! command -v python3 &> /dev/null; then
    echo "✗ Python3 未安装"
    exit 1
fi

if ! command -v trtexec &> /dev/null; then
    echo "✗ TensorRT trtexec 未安装或不在PATH中"
    exit 1
fi

echo "✓ 环境检查通过"
echo ""

# 检查ONNX文件
echo "2. 检查ONNX文件..."
ONNX_PATH="onnx/sparse4dbackbone.onnx"
if [ ! -f "${ONNX_PATH}" ]; then
    echo "✗ ONNX文件不存在: ${ONNX_PATH}"
    echo "请先运行 export_backbone_onnx.py 导出ONNX模型"
    exit 1
fi
echo "✓ ONNX文件存在: ${ONNX_PATH}"
echo ""

# 创建优化后的引擎目录
echo "3. 创建输出目录..."
mkdir -p "engine"
mkdir -p "logs"
echo "✓ 目录创建完成"
echo ""

# 构建高精度引擎
echo "4. 构建高精度TensorRT引擎..."
echo "使用Python脚本构建，提供更精细的精度控制..."

if python3 build_backbone_engine_python.py \
    --onnx_path "${ONNX_PATH}" \
    --engine_path "engine/sparse4dbackbone_highprec.engine" \
    --workspace_size 8192 \
    --precision "fp32" \
    --log_level "info"; then
    echo "✓ 高精度引擎构建成功"
else
    echo "✗ 高精度引擎构建失败"
    echo "尝试使用trtexec构建..."
    
    # 备用方案：使用trtexec
    echo "使用trtexec构建高精度引擎..."
    trtexec --onnx="${ONNX_PATH}" \
        --memPoolSize=workspace:8192 \
        --saveEngine="engine/sparse4dbackbone_highprec.engine" \
        --verbose \
        --noTF32 \
        --strictTypeConstraints \
        --maxBatch=1 \
        --optShapes=img:1x6x3x256x704 \
        --minShapes=img:1x6x3x256x704 \
        --maxShapes=img:1x6x3x256x704 \
        >"logs/build_backbone_highprec.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ trtexec高精度引擎构建成功"
    else
        echo "✗ trtexec高精度引擎构建也失败"
        exit 1
    fi
fi
echo ""

# 构建平衡精度引擎
echo "5. 构建平衡精度TensorRT引擎..."
echo "使用FP16精度，平衡性能和精度..."

trtexec --onnx="${ONNX_PATH}" \
    --memPoolSize=workspace:4096 \
    --saveEngine="engine/sparse4dbackbone_balanced.engine" \
    --verbose \
    --fp16 \
    --strictTypeConstraints \
    --maxBatch=1 \
    --optShapes=img:1x6x3x256x704 \
    --minShapes=img:1x6x3x256x704 \
    --maxShapes=img:1x6x3x256x704 \
    >"logs/build_backbone_balanced.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ 平衡精度引擎构建成功"
else
    echo "✗ 平衡精度引擎构建失败"
fi
echo ""

# 显示结果
echo "6. 构建结果汇总..."
echo "引擎文件:"
if [ -f "engine/sparse4dbackbone_highprec.engine" ]; then
    echo "  ✓ 高精度引擎: engine/sparse4dbackbone_highprec.engine"
    ls -lh "engine/sparse4dbackbone_highprec.engine"
fi

if [ -f "engine/sparse4dbackbone_balanced.engine" ]; then
    echo "  ✓ 平衡精度引擎: engine/sparse4dbackbone_balanced.engine"
    ls -lh "engine/sparse4dbackbone_balanced.engine"
fi

echo ""
echo "日志文件:"
ls -la "logs/" 2>/dev/null || echo "  无日志文件"

echo ""
echo "=== 优化完成 ==="
echo ""
echo "建议:"
echo "1. 使用高精度引擎进行一致性验证:"
echo "   python3 unit_test/img_backbone_infer-consistency-val_vs_onnxrt_unit_test_preprocessed.py --trtengine engine/sparse4dbackbone_highprec.engine"
echo ""
echo "2. 使用平衡精度引擎进行性能测试:"
echo "   python3 unit_test/img_backbone_infer-consistency-val_vs_onnxrt_unit_test_preprocessed.py --trtengine engine/sparse4dbackbone_balanced.engine"
echo ""
echo "3. 如果仍有精度问题，可以尝试:"
echo "   - 增加工作空间大小到16GB"
echo "   - 使用更低级别的优化"
echo "   - 检查输入数据的数值范围" 