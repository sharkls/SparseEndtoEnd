#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

set -e

echo "=== 特征提取精度优化 - 重新编译和测试 ==="

# 设置路径
ONBOARD_DIR="/share/Code/SparseEnd2End/onboard"
BUILD_DIR="${ONBOARD_DIR}/build"
ASSETS_DIR="${ONBOARD_DIR}/assets"

# 1. 重新编译onboard版本
echo "1. 重新编译onboard版本..."
cd "${ONBOARD_DIR}"

# 清理之前的构建
if [ -d "${BUILD_DIR}" ]; then
    echo "清理之前的构建..."
    rm -rf "${BUILD_DIR}"
fi

# 创建构建目录
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 配置和编译
echo "配置CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "编译..."
make -j$(nproc)

echo "编译完成！"

# 2. 运行特征提取测试
echo "2. 运行特征提取精度测试..."
cd "${BUILD_DIR}/unit_test/sparse4d_extract_feat_trt_infer_unit_test"

if [ -f "./sparse4d_extract_feat_trt_infer_unit_test.bin" ]; then
    echo "运行测试..."
    ./sparse4d_extract_feat_trt_infer_unit_test.bin
else
    echo "错误：测试可执行文件不存在"
    exit 1
fi

# 3. 运行精度分析脚本
echo "3. 运行精度分析..."
cd "/share/Code/SparseEnd2End"

# 检查测试数据是否存在
PRED_FILE="/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_feature_1*89760*256_float32.bin"
EXPECTED_FILE="/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_feature_1*89760*256_float32.bin"

if [ -f "${PRED_FILE}" ] && [ -f "${EXPECTED_FILE}" ]; then
    echo "运行精度分析脚本..."
    python3 script/optimize_feature_extraction.py \
        --pred_file "${PRED_FILE}" \
        --expected_file "${EXPECTED_FILE}" \
        --shape "1,89760,256" \
        --save_dir "feature_extraction_analysis"
else
    echo "警告：测试数据文件不存在，跳过精度分析"
fi

echo "=== 特征提取精度优化完成 ==="
echo "请检查以下文件："
echo "1. 测试结果：查看上面的测试输出"
echo "2. 精度分析：feature_extraction_analysis/analysis_report.txt"
echo "3. 误差分布图：feature_extraction_analysis/error_distribution.png" 