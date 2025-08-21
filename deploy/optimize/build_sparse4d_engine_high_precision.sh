#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

# 加载环境设置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tools/set_env.sh"

if [ ! -d "${ENVTRTDIR}" ]; then
    mkdir -p "${ENVTRTDIR}"
fi

# STEP1: build sparse4dbackbone engine with high precision
echo "STEP1: build sparse4dbackbone engine with HIGH PRECISION -> saving in ${ENV_BACKBONE_ENGINE}..."
echo "使用FP32精度模式，禁用FP16和INT8量化以提高推理一致性..."

${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_BACKBONE_ONNX} \
    --memPoolSize=workspace:8192 \
    --saveEngine=${ENV_BACKBONE_ENGINE} \
    --verbose \
    --warmUp=500 \
    --iterations=100 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTDIR}/buildOutput_backbone_highprec.json \
    --exportProfile=${ENVTRTDIR}/buildProfile_backbone_highprec.json \
    --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_backbone_highprec.json \
    --profilingVerbosity=detailed \
    --noTF32 \
    --strictTypeConstraints \
    --maxBatch=1 \
    --optShapes=img:1x6x3x256x704 \
    --minShapes=img:1x6x3x256x704 \
    --maxShapes=img:1x6x3x256x704 \
    --buildOnly \
    --timingCache=${ENVTRTDIR}/backbone_timing_cache.cache \
    >${ENVTRTDIR}/build_backbone_highprec.log 2>&1

# 检查构建结果
if [ $? -eq 0 ]; then
    echo "✓ Backbone高精度引擎构建成功!"
    echo "日志文件: ${ENVTRTDIR}/build_backbone_highprec.log"
else
    echo "✗ Backbone高精度引擎构建失败!"
    echo "请检查日志文件: ${ENVTRTDIR}/build_backbone_highprec.log"
    exit 1
fi

echo "高精度backbone引擎构建完成!" 