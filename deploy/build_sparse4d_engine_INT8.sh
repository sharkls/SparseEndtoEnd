#!/bin/bash
# Copyright (c) 2024 SparseEnd2End.

# 加载环境设置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tools/set_env.sh"

if [ ! -d "${ENVTRTDIR}" ]; then
    mkdir -p "${ENVTRTDIR}"
fi

# ============ INT8 校准缓存文件路径 ============
ENV_BACKBONE_CALIB="${ENVTRTDIR}/backbone_int8.calib"
ENV_HEAD1_CALIB="${ENVTRTDIR}/head1_int8.calib"
ENV_HEAD2_CALIB="${ENVTRTDIR}/head2_int8.calib"

# ============ 统一的固定 Shape（与导出一致） ============
# Backbone (export_backbone_onnx.py):
#   input name: img, shape: 1x6x3x256x704
BACKBONE_INPUT_NAME="img"
BACKBONE_SHAPE="1x6x3x256x704"

# Head-1st frame (export_head_onnx.py):
#   feature            : 1x89760x256
#   spatial_shapes     : 6x4x2
#   level_start_index  : 6x4
#   instance_feature   : 1x900x256
#   anchor             : 1x900x11
#   time_interval      : 1
#   image_wh           : 1x6x2
#   lidar2img          : 1x6x4x4
HEAD1_FEATURE_SHAPE="1x89760x256"
HEAD1_SPATIAL_SHAPES_SHAPE="6x4x2"
HEAD1_LEVEL_START_INDEX_SHAPE="6x4"
HEAD1_INSTANCE_FEATURE_SHAPE="1x900x256"
HEAD1_ANCHOR_SHAPE="1x900x11"
HEAD1_TIME_INTERVAL_SHAPE="1"
HEAD1_IMAGE_WH_SHAPE="1x6x2"
HEAD1_LIDAR2IMG_SHAPE="1x6x4x4"

# Head-2nd frame (export_head_onnx.py):
#   与1st相同 + 额外：
#   temp_instance_feature: 1x600x256
#   temp_anchor          : 1x600x11
#   mask                 : 1
#   track_id             : 1x900
HEAD2_TEMP_INSTANCE_FEATURE_SHAPE="1x600x256"
HEAD2_TEMP_ANCHOR_SHAPE="1x600x11"
HEAD2_MASK_SHAPE="1"
HEAD2_TRACK_ID_SHAPE="1x900"

# ===================== STEP1: Backbone =====================
echo "STEP1: build sparse4dbackbone INT8 engine -> ${ENV_BACKBONE_ENGINE}"
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_BACKBONE_ONNX} \
    --int8 \
    --calib=${ENV_BACKBONE_CALIB} \
    --shapes=${BACKBONE_INPUT_NAME}:${BACKBONE_SHAPE} \
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
    >${ENVTRTDIR}/build_backbone.log 2>&1

# ===================== STEP2: Head (1st) =====================
echo "STEP2: build sparse4dhead 1st INT8 engine -> ${ENV_HEAD1_ENGINE}"
sleep 2s
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD1_ONNX} \
    --plugins=$ENVTARGETPLUGIN \
    --int8 \
    --calib=${ENV_HEAD1_CALIB} \
    --shapes=feature:${HEAD1_FEATURE_SHAPE} \
    --shapes=spatial_shapes:${HEAD1_SPATIAL_SHAPES_SHAPE} \
    --shapes=level_start_index:${HEAD1_LEVEL_START_INDEX_SHAPE} \
    --shapes=instance_feature:${HEAD1_INSTANCE_FEATURE_SHAPE} \
    --shapes=anchor:${HEAD1_ANCHOR_SHAPE} \
    --shapes=time_interval:${HEAD1_TIME_INTERVAL_SHAPE} \
    --shapes=image_wh:${HEAD1_IMAGE_WH_SHAPE} \
    --shapes=lidar2img:${HEAD1_LIDAR2IMG_SHAPE} \
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
    >${ENVTRTDIR}/build_head1.log 2>&1

# ===================== STEP3: Head (>=2nd) =====================
echo "STEP3: build sparse4dhead 2nd INT8 engine -> ${ENV_HEAD2_ENGINE}"
sleep 2s
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_HEAD2_ONNX} \
    --plugins=$ENVTARGETPLUGIN \
    --int8 \
    --calib=${ENV_HEAD2_CALIB} \
    --shapes=feature:${HEAD1_FEATURE_SHAPE} \
    --shapes=spatial_shapes:${HEAD1_SPATIAL_SHAPES_SHAPE} \
    --shapes=level_start_index:${HEAD1_LEVEL_START_INDEX_SHAPE} \
    --shapes=instance_feature:${HEAD1_INSTANCE_FEATURE_SHAPE} \
    --shapes=anchor:${HEAD1_ANCHOR_SHAPE} \
    --shapes=time_interval:${HEAD1_TIME_INTERVAL_SHAPE} \
    --shapes=temp_instance_feature:${HEAD2_TEMP_INSTANCE_FEATURE_SHAPE} \
    --shapes=temp_anchor:${HEAD2_TEMP_ANCHOR_SHAPE} \
    --shapes=mask:${HEAD2_MASK_SHAPE} \
    --shapes=track_id:${HEAD2_TRACK_ID_SHAPE} \
    --shapes=image_wh:${HEAD1_IMAGE_WH_SHAPE} \
    --shapes=lidar2img:${HEAD1_LIDAR2IMG_SHAPE} \
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
    >${ENVTRTDIR}/build_head2.log 2>&1

echo "success build."