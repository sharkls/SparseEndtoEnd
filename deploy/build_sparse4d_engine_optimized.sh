#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
# 优化版本：针对 ForeignNode 耗时优化

# 使用方法: ./build_sparse4d_engine_optimized.sh [fp32|fp16|int8]
# 默认精度: fp16

# 解析精度参数
PRECISION=${1:-fp16}

# 验证精度参数
if [[ "$PRECISION" != "fp32" && "$PRECISION" != "fp16" && "$PRECISION" != "int8" ]]; then
    echo "错误: 不支持的精度类型 '$PRECISION'"
    echo "支持的精度: fp32, fp16, int8"
    echo "使用方法: $0 [fp32|fp16|int8]"
    exit 1
fi

echo "选择的精度: $PRECISION (优化版本)"

# 加载环境设置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tools/set_env.sh"

if [ ! -d "${ENVTRTDIR}" ]; then
    mkdir -p "${ENVTRTDIR}"
fi

# 根据精度参数生成TensorRT参数
get_precision_args() {
    case "$PRECISION" in
        "fp32")
            echo ""
            ;;
        "fp16")
            echo "--fp16"
            ;;
        "int8")
            echo "--int8 --strictTypeConstraints"
            ;;
    esac
}

# 获取精度参数
PRECISION_ARGS=$(get_precision_args)

# 优化选项：
# --builderOptimizationLevel=5: 最高优化级别，更积极地融合操作
# --tacticSources=-CUBLAS,-CUBLAS_LT: 禁用某些较慢的策略源（可选，根据实际情况调整）
# --maxAuxStreams=4: 增加辅助流数量，可能有助于并行化
# --timingCache=: 使用时序缓存加速构建（如果存在）
TIMING_CACHE="${ENVTRTDIR}/timing_cache.cache"
OPTIMIZATION_ARGS="--builderOptimizationLevel=5 --maxAuxStreams=4"
if [ -f "${TIMING_CACHE}" ]; then
    OPTIMIZATION_ARGS="${OPTIMIZATION_ARGS} --timingCache=${TIMING_CACHE}"
fi

# STEP1: build sparse4dbackbone engine
echo "STEP1: build sparse4dbackbone ${PRECISION} engine (优化版本) -> saving in ${ENV_BACKBONE_ENGINE}..."
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_BACKBONE_ONNX} \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_BACKBONE_ENGINE} \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTDIR}/buildOutput_backbone.json \
    --exportProfile=${ENVTRTDIR}/buildProfile_backbone.json \
    --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_backbone.json \
    --profilingVerbosity=detailed \
    ${PRECISION_ARGS} \
    ${OPTIMIZATION_ARGS} \
    >${ENVTRTDIR}/build_backbone.log 2>&1

# STEP2: build 1st frame sparse4dhead engine
echo "STEP2: build 1st frame sparse4dhead ${PRECISION} engine (优化版本) -> saving in ${ENV_HEAD1_ENGINE}..."
sleep 2s
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD1_ONNX} \
    --plugins=$ENVTARGETPLUGIN \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_HEAD1_ENGINE} \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTDIR}/buildOutput_head1.json \
    --exportProfile=${ENVTRTDIR}/buildProfile_head1.json \
    --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_head1.json \
    --profilingVerbosity=detailed \
    ${PRECISION_ARGS} \
    ${OPTIMIZATION_ARGS} \
    >${ENVTRTDIR}/build_head1.log 2>&1

# STEP3: build frame > 2 sparse4dhead engine (重点优化这个，因为 ForeignNode 问题主要在这里)
echo "STEP3: build frame > 2 sparse4dhead ${PRECISION} engine (优化版本，重点优化 ForeignNode) -> saving in ${ENV_HEAD2_ENGINE}..."
sleep 2s
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD2_ONNX} \
    --plugins=$ENVTARGETPLUGIN \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_HEAD2_ENGINE} \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTDIR}/buildOutput_head2.json \
    --exportProfile=${ENVTRTDIR}/buildProfile_head2.json \
    --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_head2.json \
    --profilingVerbosity=detailed \
    ${PRECISION_ARGS} \
    ${OPTIMIZATION_ARGS} \
    >${ENVTRTDIR}/build_head2.log 2>&1

echo "success build ${PRECISION} engines (优化版本)."
echo "提示: 如果 ForeignNode 耗时仍然较高，可以考虑："
echo "  1. 检查 ONNX 模型中的 Slice 和 Transpose 操作是否可以优化"
echo "  2. 尝试修改 export_head_onnx.py 中的导出选项"
echo "  3. 检查 Plugin 是否支持 FP16，减少精度转换开销"

