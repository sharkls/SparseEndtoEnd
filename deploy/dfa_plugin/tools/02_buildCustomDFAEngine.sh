#!/bin/bash
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

source ./deploy/dfa_plugin/tools/01_setEnv.sh
echo "ENVTRTLOGSDIR = ${ENVTRTLOGSDIR}"
if [ ! -d "${ENVTRTLOGSDIR}" ]; then
    mkdir -p "${ENVTRTLOGSDIR}"
fi

${ENV_TensorRT_BIN}/trtexec --onnx=${ENVONNX} \
    --plugins=deploy/dfa_plugin/$ENVTARGETPLUGIN \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENVEINGINENAME} \
    --verbose \
    --warmUp=50 \
    --iterations=20 \
    >${ENVTRTLOGSDIR}/build.log 2>&1

# --dumpOutput \
#   --dumpProfile \
#   --dumpLayerInfo \
#    --exportOutput=${ENVTRTLOGSDIR}/buildOutput.json \
#    --exportProfile=${ENVTRTLOGSDIR}/buildProfile.json \
#    --exportLayerInfo=${ENVTRTLOGSDIR}/buildLayerInfo.json \
#    --profilingVerbosity=detailed \

