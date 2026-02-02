#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

# 使用方法: ./build_sparse4d_engine_v3.sh [fp32|fp16|int8]
# 默认精度: fp16

# 解析精度参数
PRECISION=${1:-fp16}

# 验证精度参数
if [[ "$PRECISION" != "fp32" && "$PRECISION" != "fp16" && "$PRECISION" != "int8" && "$PRECISION" != "mixed" ]]; then
    echo "错误: 不支持的精度类型 '$PRECISION'"
    echo "支持的精度: fp32, fp16, int8, mixed"
    echo "使用方法: $0 [fp32|fp16|int8|mixed]"
    exit 1
fi

echo "选择的精度: $PRECISION"

# 加载环境设置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tools/set_env.sh"

# 强制设置 ENVTRTDIR 为 deploy/engine 的绝对路径
export ENVTRTDIR="${SCRIPT_DIR}/engine"

if [ ! -d "${ENVTRTDIR}" ]; then
    mkdir -p "${ENVTRTDIR}"
fi

# ==========================================
# OVERRIDE PATHS FOR V3 OPTIMIZATION
# ==========================================
export ENV_HEAD1_ONNX=${SCRIPT_DIR}/onnx/sparse4dhead1st_v3.onnx
export ENV_HEAD1_ENGINE=${ENVTRTDIR}/sparse4dhead1st_v3.engine
export ENV_HEAD2_ONNX=${SCRIPT_DIR}/onnx/sparse4dhead2nd_v3.onnx
export ENV_HEAD2_ENGINE=${ENVTRTDIR}/sparse4dhead2nd_v3.engine

echo "=========================================="
echo " Building V3 Engines (Optimized Layout) "
echo "=========================================="
echo "Head1 ONNX: ${ENV_HEAD1_ONNX}"
echo "Head1 ENGINE: ${ENV_HEAD1_ENGINE}"
echo "Head2 ONNX: ${ENV_HEAD2_ONNX}"
echo "Head2 ENGINE: ${ENV_HEAD2_ENGINE}"
echo "=========================================="

# 根据精度参数生成TensorRT参数
get_precision_args() {
    case "$PRECISION" in
        "fp32")
            echo ""
            echo ""
            ;;
        "fp16")
            echo "--fp16"
            echo "--fp16"
            ;;
        "int8")
            echo "--int8 --strictTypeConstraints"
            echo "--int8 --strictTypeConstraints"
            ;;
        "mixed")
            echo "--fp16"
            echo ""
            ;;
    esac
}

# 获取精度参数
args_output="$(get_precision_args)"
# BACKBONE_ARGS=$(echo "$args_output" | head -n 1) # Skip backbone for this script
HEAD_ARGS=$(echo "$args_output" | tail -n 1)

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

# STEP2: build 1st frame sparse4dhead engine
if [ "$PRECISION" == "int8" ]; then
    echo "STEP2: build 1st frame sparse4dhead (V3) INT8 engine using Python API..."
    
    # Use FP32 ONNX for INT8 calibration to avoid type conflicts
    ENV_HEAD1_ONNX_FP32="${SCRIPT_DIR}/onnx/sparse4dhead1st_v3.onnx"
    if [ -f "${ENV_HEAD1_ONNX_FP32}" ]; then
        echo "[INFO] Using FP32 ONNX for INT8 calibration: ${ENV_HEAD1_ONNX_FP32}"
        TARGET_ONNX=${ENV_HEAD1_ONNX_FP32}
    else
        echo "[WARNING] FP32 ONNX not found, falling back to default ONNX: ${ENV_HEAD1_ONNX}"
        TARGET_ONNX=${ENV_HEAD1_ONNX}
    fi
    
    # 构造插件列表参数
    PLUGIN_LIST_ARGS=""
    if [[ -n "${DFA_PLUGIN_PATH}" ]]; then PLUGIN_LIST_ARGS="${PLUGIN_LIST_ARGS} ${DFA_PLUGIN_PATH}"; fi
    if [[ -n "${LN_PLUGIN_PATH}" ]]; then PLUGIN_LIST_ARGS="${PLUGIN_LIST_ARGS} ${LN_PLUGIN_PATH}"; fi
    if [[ -n "${SPARSEBOX_PLUGIN_PATH}" ]]; then PLUGIN_LIST_ARGS="${PLUGIN_LIST_ARGS} ${SPARSEBOX_PLUGIN_PATH}"; fi

    python3 deploy/tools/build_engine.py \
        --onnx ${TARGET_ONNX} \
        --engine ${ENV_HEAD1_ENGINE} \
        --head head1 \
        --mode int8 \
        --calib_dir ${SCRIPT_DIR}/calibration_data/head1 \
        --explicit_precision \
        --plugins ${PLUGIN_LIST_ARGS} > ${ENVTRTDIR}/build_head1_v3_python.log 2>&1

    if [ $? -ne 0 ]; then
        echo "Error: Python build script failed. Check ${ENVTRTDIR}/build_head1_v3_python.log"
        exit 1
    fi

    echo "Profiling Head1 INT8 engine with Custom Python Script (trtexec causes segfault)..."
    python3 ${SCRIPT_DIR}/tools/test_int8_engine.py --engine ${ENV_HEAD1_ENGINE} > ${ENVTRTDIR}/build_head1_v3.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Warning: Python Profiling Failed. Check ${ENVTRTDIR}/build_head1_v3.log"
    else
        cat ${ENVTRTDIR}/build_head1_v3.log | grep "Average Latency"
    fi

else
    echo "STEP2: build 1st frame sparse4dhead (V3) ${PRECISION} engine -> saving in ${ENV_HEAD1_ENGINE}..."
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
        --exportOutput=${ENVTRTDIR}/buildOutput_head1_v3.json \
        --exportProfile=${ENVTRTDIR}/buildProfile_head1_v3.json \
        --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_head1_v3.json \
        --profilingVerbosity=detailed \
        ${HEAD_ARGS} \
        >${ENVTRTDIR}/build_head1_v3.log 2>&1
fi

# STEP3: build frame > 2 sparse4dhead engine
if [ "$PRECISION" == "int8" ]; then
    echo "STEP3: build frame > 2 sparse4dhead (V3) INT8 engine using Python API..."
    
    # Use FP32 ONNX for INT8 calibration
    ENV_HEAD2_ONNX_FP32="${SCRIPT_DIR}/onnx/sparse4dhead2nd_v3.onnx"
    if [ -f "${ENV_HEAD2_ONNX_FP32}" ]; then
        echo "[INFO] Using FP32 ONNX for INT8 calibration: ${ENV_HEAD2_ONNX_FP32}"
        TARGET_ONNX=${ENV_HEAD2_ONNX_FP32}
    else
        echo "[WARNING] FP32 ONNX not found, falling back to default ONNX: ${ENV_HEAD2_ONNX}"
        TARGET_ONNX=${ENV_HEAD2_ONNX}
    fi
    
    python3 deploy/tools/build_engine.py \
        --onnx ${TARGET_ONNX} \
        --engine ${ENV_HEAD2_ENGINE} \
        --head head2 \
        --mode int8 \
        --calib_dir ${SCRIPT_DIR}/calibration_data/head2 \
        --explicit_precision \
        --plugins ${PLUGIN_LIST_ARGS} > ${ENVTRTDIR}/build_head2_v3_python.log 2>&1
        
    if [ $? -ne 0 ]; then
        echo "Error: Python build script failed. Check ${ENVTRTDIR}/build_head2_v3_python.log"
        exit 1
    fi

    echo "Profiling Head2 INT8 engine with Custom Python Script (trtexec causes segfault)..."
    python3 ${SCRIPT_DIR}/tools/test_int8_engine.py --engine ${ENV_HEAD2_ENGINE} > ${ENVTRTDIR}/build_head2_v3.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Warning: Python Profiling Failed. Check ${ENVTRTDIR}/build_head2_v3.log"
    else
        cat ${ENVTRTDIR}/build_head2_v3.log | grep "Average Latency"
    fi

else
    echo "STEP3: build frame > 2 sparse4dhead (V3) ${PRECISION} engine -> saving in ${ENV_HEAD2_ENGINE}..."
    sleep 2s
    
    # 准备安全输入路径
    SAFE_INPUT_DIR="${SCRIPT_DIR}/val_data_trtexec"
    LOAD_INPUTS_ARGS=""
    if [[ -f "${SAFE_INPUT_DIR}/spatial_shapes.bin" && -f "${SAFE_INPUT_DIR}/level_start_index.bin" ]]; then
        echo "[INFO] Using safe inputs for trtexec to avoid segmentation fault."
        LOAD_INPUTS_ARGS="--loadInputs=spatial_shapes:${SAFE_INPUT_DIR}/spatial_shapes.bin,level_start_index:${SAFE_INPUT_DIR}/level_start_index.bin"
    fi

    ${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD2_ONNX} \
        ${PLUGIN_ARGS} \
        ${LOAD_INPUTS_ARGS} \
        --memPoolSize=workspace:2048 \
        --saveEngine=${ENV_HEAD2_ENGINE} \
        --verbose \
        --warmUp=200 \
        --iterations=50 \
        --dumpOutput \
        --dumpProfile \
        --dumpLayerInfo \
        --exportOutput=${ENVTRTDIR}/buildOutput_head2_v3.json \
        --exportProfile=${ENVTRTDIR}/buildProfile_head2_v3.json \
        --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_head2_v3.json \
        --profilingVerbosity=detailed \
        ${HEAD_ARGS} \
        >${ENVTRTDIR}/build_head2_v3.log 2>&1
fi

echo "success build ${PRECISION} engines (V3)."
