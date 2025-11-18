#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
# 在构建引擎前优化 ONNX 模型，减少 ForeignNode 中的 Slice/Transpose 操作

# 加载环境设置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tools/set_env.sh"

echo "开始优化 ONNX 模型以减少 ForeignNode 耗时..."

# 优化 head2nd.onnx（主要问题所在）
if [ -f "${ENV_HEAD2_ONNX}" ]; then
    echo "优化 ${ENV_HEAD2_ONNX}..."
    
    # 备份原始文件
    BACKUP_FILE="${ENV_HEAD2_ONNX}.backup"
    if [ ! -f "${BACKUP_FILE}" ]; then
        cp "${ENV_HEAD2_ONNX}" "${BACKUP_FILE}"
        echo "已备份原始文件到: ${BACKUP_FILE}"
    fi
    
    # 使用 simplify_onnx.py 优化
    OPTIMIZED_ONNX="${ENV_HEAD2_ONNX%.onnx}_optimized.onnx"
    python "${SCRIPT_DIR}/tools/simplify_onnx.py" \
        --inp "${ENV_HEAD2_ONNX}" \
        --out "${OPTIMIZED_ONNX}" \
        --shapes "feature:1x89760x256,instance_feature:1x900x256,anchor:1x900x11,temp_instance_feature:1x600x256,temp_anchor:1x600x11,track_id:1x900,mask:1x900,image_wh:1x2,lidar2img:1x6x4x4,time_interval:1"
    
    if [ $? -eq 0 ]; then
        echo "✓ ONNX 优化成功: ${OPTIMIZED_ONNX}"
        echo "提示: 如需使用优化后的 ONNX，请修改构建脚本中的 ONNX 路径"
        echo "或者运行: cp ${OPTIMIZED_ONNX} ${ENV_HEAD2_ONNX}"
        read -p "是否替换原始 ONNX 文件? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp "${OPTIMIZED_ONNX}" "${ENV_HEAD2_ONNX}"
            echo "✓ 已替换原始 ONNX 文件"
        fi
    else
        echo "✗ ONNX 优化失败，使用原始文件"
    fi
else
    echo "警告: 未找到 ${ENV_HEAD2_ONNX}"
fi

# 可选：优化 head1st.onnx
if [ -f "${ENV_HEAD1_ONNX}" ]; then
    echo ""
    echo "优化 ${ENV_HEAD1_ONNX}..."
    
    BACKUP_FILE="${ENV_HEAD1_ONNX}.backup"
    if [ ! -f "${BACKUP_FILE}" ]; then
        cp "${ENV_HEAD1_ONNX}" "${BACKUP_FILE}"
        echo "已备份原始文件到: ${BACKUP_FILE}"
    fi
    
    OPTIMIZED_ONNX="${ENV_HEAD1_ONNX%.onnx}_optimized.onnx"
    python "${SCRIPT_DIR}/tools/simplify_onnx.py" \
        --inp "${ENV_HEAD1_ONNX}" \
        --out "${OPTIMIZED_ONNX}" \
        --shapes "feature:1x89760x256,instance_feature:1x900x256,anchor:1x900x11,image_wh:1x2,lidar2img:1x6x4x4,time_interval:1"
    
    if [ $? -eq 0 ]; then
        echo "✓ ONNX 优化成功: ${OPTIMIZED_ONNX}"
    else
        echo "✗ ONNX 优化失败，使用原始文件"
    fi
fi

echo ""
echo "ONNX 优化完成！"
echo "现在可以运行: bash build_sparse4d_engine_optimized.sh fp16"

