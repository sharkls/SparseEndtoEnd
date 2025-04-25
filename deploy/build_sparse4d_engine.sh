#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

if [ ! -d "${ENVTRTDIR}" ]; then
    mkdir -p "${ENVTRTDIR}"
fi

# STEP1: build sparse4dbackbone engine
echo "STEP1: build sparse4dbackbone engine -> saving in ${ENV_BACKBONE_ENGINE}..."
${ENV_TensorRT_BIN}/trtexec --onnx=${ENV_BACKBONE_ONNX} \
    --memPoolSize=workspace:2048 \                  # TensorRT工作内存大小
    --saveEngine=${ENV_BACKBONE_ENGINE} \           # 优化后的TensorRT引擎文件的保存路径
    --verbose \                                     # 启动详细日志输出
    --warmUp=200 \                                  # 在性能测试前进行200次预热，预热可以让GPU达到稳定的工作状态，获得更准确的性能数据
    --iterations=50 \                               # 性能测试时进行50次迭代， 用于计算平均推理时间和性能指标
    --dumpOutput \                                  # 导出模型输出结果
    --dumpProfile \                                 # 导出性能分析数据profile
    --dumpLayerInfo \                               # 导出每一层的详细信息（如层类型、输入输出形状等）
    --exportOutput=${ENVTRTDIR}/buildOutput_backbone.json \
    --exportProfile=${ENVTRTDIR}/buildProfile_backbone.json \
    --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_backbone.json \
    --profilingVerbosity=detailed \                 # 设置性能分析的详细程度为详细模式
    >${ENVTRTDIR}/build_backbone.log 2>&1           # 将所有标准输出和错误输出重定向到日志文件，2>&1表示将标准错误也重定向到同一个文件

# STEP2: build 1st frame sparse4dhead engine
echo "STEP2: build 1st frame sparse4dhead engine -> saving in ${ENV_HEAD1_ENGINE}..."
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
    >${ENVTRTDIR}/build_head1.log 2>&1

# STEP3: build frame > 2 sparse4dhead engine
echo "STEP3: build frame > 2 sparse4dhead engine -> saving in ${ENV_HEAD2_ENGINE}..."
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
    --exportOutput=${ENVTRTDIR}/buildOutput_head2.json --exportProfile=${ENVTRTDIR}/buildProfile_head2.json \
    --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_head2.json \
    --profilingVerbosity=detailed \
    >${ENVTRTDIR}/build_head2.log 2>&1
