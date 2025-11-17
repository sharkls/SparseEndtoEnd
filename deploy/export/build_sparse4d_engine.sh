#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

# 使用方法: ./build_sparse4d_engine.sh [fp32|fp16|int8]
# 默认精度: fp32

# 解析精度参数
PRECISION=${1:-fp32}

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
            # TensorRT 8.5.1.7 不支持 --strictTypeConstraints
            # 使用 --fp16 --noTF32 让 TensorRT 尽可能使用 FP16
            # --noTF32 禁用 TF32，有助于确保使用 FP16
            echo "--fp16 "
            ;;
        "int8")
            # TensorRT 8.5.1.7 不支持 --strictTypeConstraints
            echo "--int8"
            ;;
    esac
}

# 获取精度参数
PRECISION_ARGS=$(get_precision_args)

# 根据精度设置IO格式参数（仅用于backbone，因为它的输入是图像格式CHW）
# 注意：head1和head2的输入不是标准图像格式，不能使用IO格式参数
if [[ "$PRECISION" == "fp16" ]]; then
    BACKBONE_IO_FORMAT_ARGS="--inputIOFormats=fp16:chw --outputIOFormats=fp16:chw"
elif [[ "$PRECISION" == "fp32" ]]; then
    BACKBONE_IO_FORMAT_ARGS="--inputIOFormats=fp32:chw --outputIOFormats=fp32:chw"
else
    BACKBONE_IO_FORMAT_ARGS=""
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
    ${BACKBONE_IO_FORMAT_ARGS} \
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

# 检查 backbone engine 构建是否成功
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to build backbone engine. Check ${ENVTRTDIR}/build_backbone.log for details."
    exit 1
fi
echo "[SUCCESS] Backbone engine built successfully."

# STEP2: build 1st frame sparse4dhead engine
echo "STEP2: build 1st frame sparse4dhead ${PRECISION} engine -> saving in ${ENV_HEAD1_ENGINE}..."
sleep 2s

# 创建timing cache文件路径（用于head1）
TIMING_CACHE_HEAD1="${ENVTRTDIR}/timing_cache_head1_${PRECISION}.cache"

${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD1_ONNX} \
    --plugins=$ENVTARGETPLUGIN \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_HEAD1_ENGINE} \
    --timingCacheFile=${TIMING_CACHE_HEAD1} \
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

# 检查 head1 engine 构建是否成功
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to build head1 engine. Check ${ENVTRTDIR}/build_head1.log for details."
    exit 1
fi
echo "[SUCCESS] Head1 engine built successfully."

# STEP3: simplify second head ONNX to reduce Transpose/Gather chains
echo "STEP3: simplify second head ONNX to reduce Transpose/Gather chains"
# 导出侧布局/类型对齐开关（用于减少 TRT Reformat）。
# 注意：实际导出 ONNX 在 export_head_onnx.py 中进行，
# 若你在同一终端先导出，再执行本脚本，请确保当时也设置了该环境变量：
#   export SPARSE4D_HEAD2_LAYOUT_OPT=1
export SPARSE4D_HEAD2_LAYOUT_OPT=${SPARSE4D_HEAD2_LAYOUT_OPT:-1}
SIM_HEAD2_ONNX=${ENVTRTDIR}/sparse4d_head2.simplified.onnx
# 只指定需要固定的输入形状（维度匹配的），其他使用 ONNX 原始形状
${ENV_PYTHON} ${SCRIPT_DIR}/tools/simplify_onnx.py \
    --inp "${ENV_HEAD2_ONNX}" \
    --out "${SIM_HEAD2_ONNX}" \
    --shapes "feature:1x89760x256,spatial_shapes:1x6x4x2,level_start_index:1x6x4,instance_feature:1x900x256,anchor:1x900x11,time_interval:1,temp_instance_feature:1x600x256,temp_anchor:1x600x11,mask:1,track_id:1x900,image_wh:1x6x2,lidar2img:1x6x4x4" \
    || { echo "[WARN] simplify failed, fallback to original"; SIM_HEAD2_ONNX=${ENV_HEAD2_ONNX}; }

# # STEP4: build frame > 2 sparse4dhead engine
echo "STEP4: build frame > 2 sparse4dhead ${PRECISION} engine -> saving in ${ENV_HEAD2_ENGINE}..."
sleep 2s

# 创建timing cache文件路径（用于head2）
TIMING_CACHE_HEAD2="${ENVTRTDIR}/timing_cache_head2_${PRECISION}.cache"

${ENV_TensorRT_BIN}/trtexec --onnx=${SIM_HEAD2_ONNX} \
    --plugins=$ENVTARGETPLUGIN \
    --memPoolSize=workspace:4096 \
    --saveEngine=${ENV_HEAD2_ENGINE} \
    --timingCacheFile=${TIMING_CACHE_HEAD2} \
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

# 检查 head2 engine 构建是否成功
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to build head2 engine. Check ${ENVTRTDIR}/build_head2.log for details."
    exit 1
fi
echo "[SUCCESS] Head2 engine built successfully."

echo "===================================================================================================================="
echo "[SUCCESS] All ${PRECISION} engines built successfully!"
echo "  - Backbone: ${ENV_BACKBONE_ENGINE}"
echo "  - Head1:    ${ENV_HEAD1_ENGINE}"
echo "  - Head2:    ${ENV_HEAD2_ENGINE}"
echo "===================================================================================================================="