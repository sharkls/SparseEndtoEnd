#!/bin/bash

# 数据对比脚本
# 对比030生成的第一帧数据和010生成的第一帧原始数据

echo "🔍 开始对比030和010脚本生成的数据..."

# 设置路径
SCRIPT_DIR="/share/Code/SparseEnd2End/C++/script/compare"
DATA_010_DIR="/share/Code/SparseEnd2End/C++/Output"
DATA_030_DIR="/share/Code/SparseEnd2End/C++/Data/sparse"

# 检查目录是否存在
if [ ! -d "$DATA_010_DIR" ]; then
    echo "❌ 010数据目录不存在: $DATA_010_DIR"
    echo "请先运行010脚本生成数据"
    exit 1
fi

if [ ! -d "$DATA_030_DIR" ]; then
    echo "❌ 030数据目录不存在: $DATA_030_DIR"
    echo "请先运行030脚本生成数据"
    exit 1
fi

# 检查Python脚本是否存在
PYTHON_SCRIPT="$SCRIPT_DIR/compare_030_vs_010_data.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

# 运行对比脚本
echo "📊 运行数据对比..."
python3 "$PYTHON_SCRIPT" \
    --data-010 "$DATA_010_DIR" \
    --data-030 "$DATA_030_DIR" \
    --frame-idx 0 \
    2>&1 | tee "$OUTPUT_DIR/compare_result.log"

echo "✅ 对比完成！结果保存在: $OUTPUT_DIR/compare_result.log" 