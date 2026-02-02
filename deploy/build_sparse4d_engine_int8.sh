#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
# Modified for Explicit PTQ INT8 Engine Build

# 使用方法: ./build_sparse4d_engine_int8.sh
# 默认精度: int8

PRECISION="int8"
echo "选择的精度: $PRECISION (Explicit Quantization)"

# 加载环境设置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tools/set_env.sh"

# 强制设置 ENVTRTDIR 为 deploy/engine 的绝对路径
export ENVTRTDIR="${SCRIPT_DIR}/engine"

if [ ! -d "${ENVTRTDIR}" ]; then
    mkdir -p "${ENVTRTDIR}"
fi

# ==========================================
# PATHS FOR INT8 ENGINES & PROFILES
# ==========================================
export ENV_BACKBONE_ONNX=${SCRIPT_DIR}/onnx/sparse4dbackbone_int8.onnx
export ENV_BACKBONE_ENGINE=${ENVTRTDIR}/sparse4dbackbone_int8.engine
export ENV_HEAD1_ONNX=${SCRIPT_DIR}/onnx/sparse4dhead1st_int8.onnx
export ENV_HEAD1_ENGINE=${ENVTRTDIR}/sparse4dhead1st_int8.engine
export ENV_HEAD2_ONNX=${SCRIPT_DIR}/onnx/sparse4dhead2nd_int8.onnx
export ENV_HEAD2_ENGINE=${ENVTRTDIR}/sparse4dhead2nd_int8.engine

# 创建 profile 目录
export PROFILE_DIR="${ENVTRTDIR}/profile"
if [ ! -d "${PROFILE_DIR}" ]; then
    mkdir -p "${PROFILE_DIR}"
fi

echo "=========================================="
echo " Building INT8 Engines (PTQ) "
echo "=========================================="
echo "Backbone ONNX: ${ENV_BACKBONE_ONNX}"
echo "Backbone ENGINE: ${ENV_BACKBONE_ENGINE}"
echo "Head1 ONNX: ${ENV_HEAD1_ONNX}"
echo "Head1 ENGINE: ${ENV_HEAD1_ENGINE}"
echo "Head2 ONNX: ${ENV_HEAD2_ONNX}"
echo "Head2 ENGINE: ${ENV_HEAD2_ENGINE}"
echo "=========================================="

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
# 注意：在ONNX导出阶段我们跳过了插件替换，但在TensorRT构建阶段是否加载插件取决于ONNX中是否包含自定义算子。
# 由于我们导出的ONNX使用了原生算子，这里即使加载插件也不会被用到，但加载它是无害的。
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
    fi
fi

# INT8 构建参数
# 关键点：我们使用的是 Explicit Quantization (Q/DQ 节点已在 ONNX 中)，所以必须开启 --int8
# --fp16 是推荐开启的，以便非 INT8 层回退到 FP16 而不是 FP32，进一步加速
# --profilingVerbosity=layer_names_only: 开启层级耗时统计
INT8_ARGS="--int8 --fp16 --profilingVerbosity=layer_names_only"

# STEP1: build sparse4dbackbone engine
echo "STEP1: build sparse4dbackbone INT8 engine..."
if [ ! -f "${ENV_BACKBONE_ONNX}" ]; then
    echo "Error: Backbone ONNX file not found: ${ENV_BACKBONE_ONNX}"
    exit 1
fi

# Backbone 不需要 Head 的插件，为了稳健性不加载 PLUGIN_ARGS
# 使用 --shapes 指定真实输入维度 (1x6x3x256x704)
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_BACKBONE_ONNX} \
    --shapes=img:1x6x3x256x704 \
    --memPoolSize=workspace:4096 \
    --saveEngine=${ENV_BACKBONE_ENGINE} \
    --exportProfile=${ENVTRTDIR}/log/backbone_int8_profile.json \
    --exportLayerInfo=${ENVTRTDIR}/log/backbone_int8_layer_info.json \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    ${INT8_ARGS} \
    >${ENVTRTDIR}/log/build_backbone_int8.log 2>&1
# --dumpProfile 

if [ $? -ne 0 ]; then
    echo "Error: Backbone Engine build failed. Check ${ENVTRTDIR}/log/build_backbone_int8.log"
    exit 1
fi
echo "Success: Backbone INT8 Engine saved to ${ENV_BACKBONE_ENGINE}"

# STEP2: build 1st frame sparse4dhead engine
echo "STEP2: build 1st frame sparse4dhead INT8 engine..."
if [ ! -f "${ENV_HEAD1_ONNX}" ]; then
    echo "Error: Head1 ONNX file not found: ${ENV_HEAD1_ONNX}"
    exit 1
fi

# 转换 NPZ 数据为 BIN (供 trtexec --loadInputs 使用)
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_npz_to_bin.py"
NPZ_HEAD1="${SCRIPT_DIR}/calibration_data/head1/000000.npz"
BIN_HEAD1_DIR="${SCRIPT_DIR}/calibration_data_bin/head1"

if [ -f "${NPZ_HEAD1}" ]; then
    echo "[INFO] Found Head1 NPZ calibration data. Converting to BIN for trtexec..."
    python3 "${CONVERT_SCRIPT}" --head1-npz "${NPZ_HEAD1}" --output-dir "${SCRIPT_DIR}/calibration_data_bin"
else
    echo "[WARNING] Head1 NPZ calibration data not found at ${NPZ_HEAD1}. Skipping data loading."
fi

# 加载安全输入以避免随机数据导致的非法内存访问 (illegal memory access)
LOAD_INPUTS_HEAD1=""
if [ -d "${BIN_HEAD1_DIR}" ]; then
    echo "[INFO] Loading safe inputs for Head1 from ${BIN_HEAD1_DIR}"
    LOAD_INPUTS_HEAD1="--loadInputs=feature:${BIN_HEAD1_DIR}/input_feature.bin,spatial_shapes:${BIN_HEAD1_DIR}/input_spatial_shapes.bin,level_start_index:${BIN_HEAD1_DIR}/input_level_start_index.bin,instance_feature:${BIN_HEAD1_DIR}/input_instance_feature.bin,anchor:${BIN_HEAD1_DIR}/input_anchor.bin,time_interval:${BIN_HEAD1_DIR}/input_time_interval.bin,image_wh:${BIN_HEAD1_DIR}/input_image_wh.bin,lidar2img:${BIN_HEAD1_DIR}/input_lidar2img.bin"
fi

# Head1 是静态模型，移除 --shapes 参数
# 注意：如果推理耗时异常短，请检查导出 ONNX 时使用的输入数据维度是否为全尺寸
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD1_ONNX} \
    ${PLUGIN_ARGS} \
    ${LOAD_INPUTS_HEAD1} \
    --memPoolSize=workspace:4096 \
    --saveEngine=${ENV_HEAD1_ENGINE} \
    --exportProfile=${ENVTRTDIR}/log/head1_int8_profile.json \
    --exportLayerInfo=${ENVTRTDIR}/log/head1_int8_layer_info.json \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    ${INT8_ARGS} \
    >${ENVTRTDIR}/log/build_head1_int8.log 2>&1
# --dumpProfile 

if [ $? -ne 0 ]; then
    echo "Error: Head1 Engine build failed. Check ${ENVTRTDIR}/log/build_head1_int8.log"
    exit 1
fi
echo "Success: Head1 INT8 Engine saved to ${ENV_HEAD1_ENGINE}"

# STEP3: build frame > 2 sparse4dhead engine
echo "STEP3: build frame > 2 sparse4dhead INT8 engine..."
if [ ! -f "${ENV_HEAD2_ONNX}" ]; then
    echo "Error: Head2 ONNX file not found: ${ENV_HEAD2_ONNX}"
    exit 1
fi

# 转换 Head2 NPZ 数据为 BIN
NPZ_HEAD2="${SCRIPT_DIR}/calibration_data/head2/000001.npz"
BIN_HEAD2_DIR="${SCRIPT_DIR}/calibration_data_bin/head2"

if [ -f "${NPZ_HEAD2}" ]; then
    echo "[INFO] Found Head2 NPZ calibration data. Converting to BIN for trtexec..."
    python3 "${CONVERT_SCRIPT}" --head2-npz "${NPZ_HEAD2}" --output-dir "${SCRIPT_DIR}/calibration_data_bin"
fi

# 准备安全输入路径
LOAD_INPUTS_HEAD2=""
if [ -d "${BIN_HEAD2_DIR}" ]; then
    echo "[INFO] Loading safe inputs for Head2 from ${BIN_HEAD2_DIR}"
    LOAD_INPUTS_HEAD2="--loadInputs=feature:${BIN_HEAD2_DIR}/input_feature.bin,spatial_shapes:${BIN_HEAD2_DIR}/input_spatial_shapes.bin,level_start_index:${BIN_HEAD2_DIR}/input_level_start_index.bin,instance_feature:${BIN_HEAD2_DIR}/input_instance_feature.bin,anchor:${BIN_HEAD2_DIR}/input_anchor.bin,time_interval:${BIN_HEAD2_DIR}/input_time_interval.bin,image_wh:${BIN_HEAD2_DIR}/input_image_wh.bin,lidar2img:${BIN_HEAD2_DIR}/input_lidar2img.bin,temp_instance_feature:${BIN_HEAD2_DIR}/input_temp_instance_feature.bin,temp_anchor:${BIN_HEAD2_DIR}/input_temp_anchor.bin,mask:${BIN_HEAD2_DIR}/input_mask.bin,track_id:${BIN_HEAD2_DIR}/input_track_id.bin"
else
    # 回退到旧的逻辑（如果 calibration_data 不存在）
    SAFE_INPUT_DIR="${SCRIPT_DIR}/val_data_trtexec"
    if [[ -f "${SAFE_INPUT_DIR}/spatial_shapes.bin" && -f "${SAFE_INPUT_DIR}/level_start_index.bin" ]]; then
        echo "[INFO] Using fallback safe inputs for trtexec."
        LOAD_INPUTS_HEAD2="--loadInputs=spatial_shapes:${SAFE_INPUT_DIR}/spatial_shapes.bin,level_start_index:${SAFE_INPUT_DIR}/level_start_index.bin"
    fi
fi

# 使用 --shapes 指定真实输入维度
# temp_instance_feature: 1x600x256, temp_anchor: 1x600x11 (TopK=600)
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD2_ONNX} \
    ${PLUGIN_ARGS} \
    ${LOAD_INPUTS_HEAD2} \
    --memPoolSize=workspace:4096 \
    --saveEngine=${ENV_HEAD2_ENGINE} \
    --exportProfile=${ENVTRTDIR}/log/head2_int8_profile.json \
    --exportLayerInfo=${ENVTRTDIR}/log/head2_int8_layer_info.json \
    --verbose \
    --warmUp=200 \
    --iterations=50 \
    ${INT8_ARGS} \
    >${ENVTRTDIR}/log/build_head2_int8.log 2>&1
# --dumpProfile 

if [ $? -ne 0 ]; then
    echo "Error: Head2 Engine build failed. Check ${ENVTRTDIR}/log/build_head2_int8.log"
    exit 1
fi
echo "Success: Head2 INT8 Engine saved to ${ENV_HEAD2_ENGINE}"

echo "=========================================="
echo " All INT8 Engines Built Successfully! "
echo "=========================================="
