#!/bin/bash

# C++与Python输出结果对比测试脚本

set -e  # 遇到错误时退出

echo "=========================================="
echo "C++与Python输出结果对比测试"
echo "=========================================="

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CPP_OUTPUT_DIR="$PROJECT_ROOT/Output/val_bin"
PYTHON_ASSET_DIR="/share/Code/SparseEnd2End/script/tutorial/asset"
COMPARE_RESULTS_DIR="$SCRIPT_DIR/compare/results"

echo "脚本目录: $SCRIPT_DIR"
echo "项目根目录: $PROJECT_ROOT"
echo "C++输出目录: $CPP_OUTPUT_DIR"
echo "Python资产目录: $PYTHON_ASSET_DIR"
echo "对比结果目录: $COMPARE_RESULTS_DIR"

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p "$CPP_OUTPUT_DIR"
mkdir -p "$COMPARE_RESULTS_DIR"

# 检查Python资产目录是否存在
if [ ! -d "$PYTHON_ASSET_DIR" ]; then
    echo "错误: Python资产目录不存在: $PYTHON_ASSET_DIR"
    echo "请先运行Python脚本生成参考数据"
    exit 1
fi

# 检查是否有Python参考数据
python_files=(
    "sample_0_imgs_1*6*3*256*704_float32.bin"
    "sample_0_feature_1*1536*256_float32.bin"
    "sample_0_time_interval_1_float32.bin"
    "sample_0_image_wh_2_float32.bin"
    "sample_0_lidar2img_6*4*4_float32.bin"
)

echo "检查Python参考数据..."
for file in "${python_files[@]}"; do
    if [ ! -f "$PYTHON_ASSET_DIR/$file" ]; then
        echo "警告: Python参考文件不存在: $file"
    else
        echo "✓ 找到Python参考文件: $file"
    fi
done

# 编译C++代码
echo "编译C++代码..."
cd "$PROJECT_ROOT"
if [ -f "build.sh" ]; then
    ./build.sh
else
    echo "错误: 找不到build.sh脚本"
    exit 1
fi

# 运行C++测试
echo "运行C++测试..."
if [ -f "TestSparseBEVAlgv2" ]; then
    ./TestSparseBEVAlgv2
    echo "✓ C++测试完成"
else
    echo "错误: 找不到TestSparseBEVAlgv2可执行文件"
    exit 1
fi

# 检查C++输出文件
echo "检查C++输出文件..."
cpp_files=(
    "sample_0_imgs_1*6*3*256*704_float32.bin"
    "sample_0_feature_1*1536*256_float32.bin"
    "sample_0_time_interval_1_float32.bin"
    "sample_0_image_wh_2_float32.bin"
    "sample_0_lidar2img_6*4*4_float32.bin"
)

for file in "${cpp_files[@]}"; do
    if [ ! -f "$CPP_OUTPUT_DIR/$file" ]; then
        echo "警告: C++输出文件不存在: $file"
    else
        echo "✓ 找到C++输出文件: $file"
    fi
done

# 运行对比脚本
echo "运行对比脚本..."
cd "$SCRIPT_DIR"
python3 compare_cpp_vs_python.py \
    --cpp_dir "$CPP_OUTPUT_DIR" \
    --python_dir "$PYTHON_ASSET_DIR" \
    --save_dir "$COMPARE_RESULTS_DIR" \
    --tolerance 0.01

# 检查对比结果
if [ $? -eq 0 ]; then
    echo "✓ 所有对比测试通过!"
else
    echo "⚠ 部分对比测试失败，请查看详细报告"
fi

# 显示结果摘要
echo ""
echo "=========================================="
echo "测试结果摘要"
echo "=========================================="
echo "C++输出目录: $CPP_OUTPUT_DIR"
echo "对比结果目录: $COMPARE_RESULTS_DIR"

if [ -f "$COMPARE_RESULTS_DIR/comparison_report.txt" ]; then
    echo ""
    echo "对比报告内容:"
    cat "$COMPARE_RESULTS_DIR/comparison_report.txt"
fi

echo ""
echo "测试完成!"
echo "==========================================" 