#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
# 检查 TensorRT Engine 是否支持 FP16 并测试 warmup

# 使用方法: ./check_engine_fp16.sh <engine_path> [warmup_iterations]
# 示例: ./check_engine_fp16.sh engine/sparse4dhead1st.engine 100

ENGINE_PATH=${1:-"engine/sparse4dhead2nd.engine"}
WARMUP_ITER=${2:-100}

if [ ! -f "$ENGINE_PATH" ]; then
    echo "[ERROR] Engine 文件不存在: $ENGINE_PATH"
    exit 1
fi

echo "===================================================================================================================="
echo "检查 Engine: $ENGINE_PATH"
echo "===================================================================================================================="

# 加载环境设置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tools/set_env.sh" 2>/dev/null || {
    echo "[WARN] 无法加载环境设置，使用默认路径"
    ENV_TensorRT_BIN=${ENV_TensorRT_BIN:-"/mnt/env/tensorrt/TensorRT-8.5.1.7/bin"}
    ENVTARGETPLUGIN=${ENVTARGETPLUGIN:-"dfa_plugin/lib/deformableAttentionAggr.so"}
}

# 方法1: 使用 trtexec 检查 engine 信息
echo ""
echo "=== 方法1: 使用 trtexec 检查 Engine 信息 ==="
echo "执行命令: trtexec --loadEngine=$ENGINE_PATH --dumpLayerInfo --verbose"
echo ""

${ENV_TensorRT_BIN}/trtexec --loadEngine="$ENGINE_PATH" \
    --plugins=$ENVTARGETPLUGIN \
    --dumpLayerInfo \
    --verbose 2>&1 | tee /tmp/engine_info_check.log | head -100

# 提取精度信息（格式: [时间戳] [I] Precision: FP32 或 [时间戳] [I] Precision: FP32+FP16）
ENGINE_PRECISION=$(grep "Precision:" /tmp/engine_info_check.log | head -1 | awk '{print $4}')
echo ""
echo "--- Engine 精度检查结果 ---"
if [ "$ENGINE_PRECISION" == "FP16" ]; then
    echo "✓ Engine 精度: FP16 - 支持 FP16 推理"
elif [[ "$ENGINE_PRECISION" == *"FP16"* ]]; then
    echo "⚠ Engine 精度: $ENGINE_PRECISION - 混合精度（包含 FP16 和 FP32）"
    echo "  提示: 虽然包含 FP16，但可能部分层使用了 FP32"
elif [ "$ENGINE_PRECISION" == "FP32" ]; then
    echo "⚠ Engine 精度: FP32 - 这是 FP32 engine，不是 FP16"
    echo "  可能原因:"
    echo "  1. ONNX 模型可能不是 FP16 精度"
    echo "  2. TensorRT 构建时某些层不支持 FP16，自动回退到 FP32"
    echo "  3. 构建时没有正确使用 --fp16 参数"
    echo "  建议:"
    echo "  1. 验证 ONNX: python deploy/verify_onnx_fp16.py onnx/sparse4dhead1st.onnx"
    echo "  2. 检查构建日志: tail -50 deploy/engine/build_head1.log"
    echo "  3. 重新构建: ./deploy/build_sparse4d_engine.sh fp16"
else
    echo "? Engine 精度: $ENGINE_PRECISION (未知)"
fi

echo ""
echo "===================================================================================================================="

# 方法2: 使用 Python 脚本检查精度类型（如果 Python 脚本失败，跳过）
echo ""
echo "=== 方法2: 使用 Python 检查 Engine 精度类型 ==="
# 使用独立的 Python 脚本，避免 heredoc 导致的 segmentation fault
if python3 "${SCRIPT_DIR}/check_engine_fp16.py" "$ENGINE_PATH" --no-warmup 2>/dev/null; then
    echo "[INFO] Python 检查完成"
else
    echo "[WARN] Python 检查失败，跳过（可能是 TensorRT Python API 问题或 engine 文件问题）"
fi

echo ""
echo "===================================================================================================================="

# 方法3: 使用 trtexec 进行 FP16 warmup 测试
echo ""
echo "=== 方法3: 测试 FP16 Warmup ==="
echo "执行 warmup 测试 ($WARMUP_ITER 次迭代)..."
echo ""

# 根据 engine 类型设置不同的 shapes
ENGINE_NAME=$(basename "$ENGINE_PATH" .engine)

if [[ "$ENGINE_NAME" == *"backbone"* ]]; then
    # Backbone 输入: img (1, 6, 3, 256, 704)
    SHAPES="img:1x6x3x256x704"
elif [[ "$ENGINE_NAME" == *"head1"* ]]; then
    # Head1 输入
    SHAPES="feature:1x89760x256,spatial_shapes:6x4x2,level_start_index:6x4,instance_feature:1x900x256,anchor:1x900x11,time_interval:1,image_wh:1x6x2,lidar2img:1x6x4x4"
elif [[ "$ENGINE_NAME" == *"head2"* ]]; then
    # Head2 输入
    SHAPES="feature:1x89760x256,spatial_shapes:6x4x2,level_start_index:6x4,instance_feature:1x900x256,anchor:1x900x11,time_interval:1,temp_instance_feature:1x600x256,temp_anchor:1x600x11,mask:1,track_id:1x900,image_wh:1x6x2,lidar2img:1x6x4x4"
else
    echo "[WARN] 未知的 engine 类型，跳过 warmup 测试"
    SHAPES=""
fi

if [ -n "$SHAPES" ]; then
    echo "使用 shapes: $SHAPES"
    echo ""
    
    ${ENV_TensorRT_BIN}/trtexec --loadEngine="$ENGINE_PATH" \
        --plugins=$ENVTARGETPLUGIN \
        --shapes="$SHAPES" \
        --warmUp=$WARMUP_ITER \
        --iterations=10 \
        --verbose \
        2>&1 | tee /tmp/engine_warmup_test.log
    
    WARMUP_RESULT=$?
    
    echo ""
    # 检查 engine 的实际精度（格式: [时间戳] [I] Precision: FP32）
    ENGINE_PRECISION=$(grep "Precision:" /tmp/engine_warmup_test.log | head -1 | awk '{print $4}')
    
    if [ $WARMUP_RESULT -eq 0 ]; then
        echo "✓ Warmup 测试成功！"
        
        # 根据实际精度显示信息
        if [ "$ENGINE_PRECISION" == "FP16" ]; then
            echo "  Engine 精度: FP16 - 支持 FP16 推理 ✓"
        elif [ "$ENGINE_PRECISION" == "FP32" ]; then
            echo "  ⚠ Engine 精度: FP32 - 这是 FP32 engine，不是 FP16"
            echo "  提示: 如果需要 FP16 engine，请使用 './build_sparse4d_engine.sh fp16' 重新构建"
        else
            echo "  Engine 精度: $ENGINE_PRECISION"
        fi
        
        # 提取性能信息
        if grep -q "GPU Compute" /tmp/engine_warmup_test.log; then
            echo ""
            echo "性能摘要:"
            grep -A 5 "GPU Compute" /tmp/engine_warmup_test.log | head -10
        fi
    else
        echo "✗ Warmup 测试失败！请检查错误信息"
        echo "详细日志: /tmp/engine_warmup_test.log"
    fi
fi

echo ""
echo "===================================================================================================================="
echo "检查完成！"
echo "===================================================================================================================="

