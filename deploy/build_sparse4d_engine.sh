#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

# 使用方法: ./build_sparse4d_engine.sh [fp32|fp16|int8]
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

echo "选择的精度: $PRECISION"

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

# 组合插件参数
PLUGIN_ARGS=""
# 检查DFA插件（使用绝对路径）
if [[ -n "${ENVTARGETPLUGIN}" ]]; then
    # 转换为绝对路径
    if [[ ! "${ENVTARGETPLUGIN}" = /* ]]; then
        DFA_PLUGIN_PATH="${SCRIPT_DIR}/${ENVTARGETPLUGIN}"
    else
        DFA_PLUGIN_PATH="${ENVTARGETPLUGIN}"
    fi
    if [[ -f "${DFA_PLUGIN_PATH}" ]]; then
        PLUGIN_ARGS="${PLUGIN_ARGS} --plugins=${DFA_PLUGIN_PATH}"
        echo "[INFO] DeformableAttentionAggrPlugin enabled: ${DFA_PLUGIN_PATH}"
    else
        echo "[WARNING] DeformableAttentionAggrPlugin not found: ${DFA_PLUGIN_PATH}"
        echo "[WARNING] Engine will be built without DeformableAttentionAggrPlugin"
    fi
fi
# 检查LayerNorm插件（使用绝对路径）
if [[ -n "${ENV_LAYER_NORM_PLUGIN}" ]]; then
    if [[ ! "${ENV_LAYER_NORM_PLUGIN}" = /* ]]; then
        LN_PLUGIN_PATH="${SCRIPT_DIR}/${ENV_LAYER_NORM_PLUGIN}"
    else
        LN_PLUGIN_PATH="${ENV_LAYER_NORM_PLUGIN}"
    fi
    if [[ -f "${LN_PLUGIN_PATH}" ]]; then
        PLUGIN_ARGS="${PLUGIN_ARGS} --plugins=${LN_PLUGIN_PATH}"
        echo "[INFO] LayerNormPlugin enabled: ${LN_PLUGIN_PATH}"
    else
        echo "[WARNING] LayerNormPlugin not found: ${LN_PLUGIN_PATH}"
        echo "[WARNING] Engine will be built without LayerNormPlugin optimization"
    fi
fi
# 检查SparseBox插件（使用绝对路径）
if [[ -n "${ENV_SPARSEBOX_PLUGIN}" ]]; then
    if [[ ! "${ENV_SPARSEBOX_PLUGIN}" = /* ]]; then
        SPARSEBOX_PLUGIN_PATH="${SCRIPT_DIR}/${ENV_SPARSEBOX_PLUGIN}"
    else
        SPARSEBOX_PLUGIN_PATH="${ENV_SPARSEBOX_PLUGIN}"
    fi
    if [[ -f "${SPARSEBOX_PLUGIN_PATH}" ]]; then
        PLUGIN_ARGS="${PLUGIN_ARGS} --plugins=${SPARSEBOX_PLUGIN_PATH}"
        echo "[INFO] SparseBox3DKeyPointsPlugin enabled: ${SPARSEBOX_PLUGIN_PATH}"
    else
        echo "[WARNING] SparseBox3DKeyPointsPlugin not found: ${SPARSEBOX_PLUGIN_PATH}"
        echo "[WARNING] Engine will be built without SparseBox3DKeyPointsPlugin optimization"
    fi
fi

# STEP1: build sparse4dbackbone engine
echo "STEP1: build sparse4dbackbone ${PRECISION} engine -> saving in ${ENV_BACKBONE_ENGINE}..."
# TensorRT工作内存大小
# 优化后的TensorRT引擎文件的保存路径
# 启动详细日志输出
# 在性能测试前进行200次预热，预热可以让GPU达到稳定的工作状态，获得更准确的性能数据
# 性能测试时进行50次迭代， 用于计算平均推理时间和性能指标
# 导出模型输出结果
# 导出性能分析数据profile
# 导出每一层的详细信息（如层类型、输入输出形状等）
# 设置性能分析的详细程度为详细模式
# 将所有标准输出和错误输出重定向到日志文件，2>&1表示将标准错误也重定向到同一个文件
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
    >${ENVTRTDIR}/build_backbone.log 2>&1

# STEP2: build 1st frame sparse4dhead engine
echo "STEP2: build 1st frame sparse4dhead ${PRECISION} engine -> saving in ${ENV_HEAD1_ENGINE}..."
sleep 2s
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD1_ONNX} \
    ${PLUGIN_ARGS} \
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
    >${ENVTRTDIR}/build_head1.log 2>&1

# STEP3: build frame > 2 sparse4dhead engine
echo "STEP3: build frame > 2 sparse4dhead ${PRECISION} engine -> saving in ${ENV_HEAD2_ENGINE}..."
sleep 2s
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD2_ONNX} \
    ${PLUGIN_ARGS} \
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
    >${ENVTRTDIR}/build_head2.log 2>&1

echo "success build ${PRECISION} engines."