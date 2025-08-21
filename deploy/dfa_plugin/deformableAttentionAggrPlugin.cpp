// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "deformableAttentionAggrPlugin.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "NvInferVersion.h"

// 声明CUDA函数
int thomas_deform_attn_cuda_forward(cudaStream_t stream,
                                    const float* value,
                                    const int* spatialShapes,
                                    const int* levelStartIndex,
                                    const float* samplingLoc,
                                    const float* attnWeight,
                                    float* output,
                                    int batch,
                                    int mSpatialSize,
                                    int mChannels,
                                    int mNumCams,
                                    int mNumLevels,
                                    int mNumQuery,
                                    int mNumPoint,
                                    int mNumGroups);

namespace custom
{

REGISTER_TENSORRT_PLUGIN(DeformableAttentionAggrPluginCreator); // 注册插件到TensorRT

nvinfer1::IPluginV2DynamicExt* DeformableAttentionAggrPlugin::clone() const noexcept
{
    DeformableAttentionAggrPlugin* plugin = new DeformableAttentionAggrPlugin();
    plugin->setPluginNamespace(mNamespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs DeformableAttentionAggrPlugin::getOutputDimensions(int32_t outputIndex,
                                                                       const nvinfer1::DimsExprs* inputs,
                                                                       int32_t nbInputs,
                                                                       nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[3].d[1];
    ret.d[2] = inputs[0].d[2];
    return ret;
}

bool DeformableAttentionAggrPlugin::supportsFormatCombination(int32_t pos,
                                                              const nvinfer1::PluginTensorDesc* inOut,
                                                              int32_t nbInputs,
                                                              int32_t nbOutputs) noexcept
{
    if (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR)
    {
        if ((pos == 1) || (pos == 2))
        {
            return (inOut[pos].type == nvinfer1::DataType::kINT32);
        }
        return ((inOut[pos].type == inOut[0].type) &&
                ((inOut[pos].type == nvinfer1::DataType::kFLOAT) || (inOut[pos].type == nvinfer1::DataType::kHALF)));
    }
    return false;
}

void DeformableAttentionAggrPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
                                                    int32_t nbInputs,
                                                    nvinfer1::DynamicPluginTensorDesc const* out,
                                                    int32_t nbOutputs) noexcept
{
    return;
}

size_t DeformableAttentionAggrPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                                       int32_t nbInputs,
                                                       const nvinfer1::PluginTensorDesc* outputs,
                                                       int32_t nbOutputs) const noexcept
{
    return 0;
}

// 推理函数
int32_t DeformableAttentionAggrPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                               const nvinfer1::PluginTensorDesc* outputDesc,
                                               const void* const* inputs,
                                               void* const* outputs,
                                               void* workspace,
                                               cudaStream_t stream) noexcept
{
    int32_t const batch = inputDesc[0].dims.d[0];
    int32_t spatial_size = inputDesc[0].dims.d[1];
    int32_t channels = inputDesc[0].dims.d[2];
    int32_t num_cams = inputDesc[1].dims.d[0];
    int32_t num_levels = inputDesc[1].dims.d[1];
    int32_t num_query = inputDesc[3].dims.d[1];
    int32_t num_point = inputDesc[3].dims.d[2];
    int32_t num_groups = inputDesc[4].dims.d[5];
    int32_t rc = 0;

    const float* value = static_cast<const float*>(inputs[0]);                  // [1, 89760, 128]
    const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);      // [6, 4, 2]
    const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);    // [6, 4]
    const float* samplingLoc = static_cast<const float*>(inputs[3]);            // [1, 900, 13, 6, 2]
    const float* attnWeight = static_cast<const float*>(inputs[4]);             // [1, 900, 13, 6, 4, 8]

    float* output = static_cast<float*>(outputs[0]);

    rc = thomas_deform_attn_cuda_forward(stream,
                                        value,
                                        spatialShapes,
                                        levelStartIndex,
                                        samplingLoc,
                                        attnWeight,
                                        output,
                                        batch,           // batch_size
                                        num_cams,        // num_cams
                                        spatial_size,    // num_feat (spatial_size)
                                        channels,        // num_embeds (channels)
                                        num_levels,      // num_scale (num_levels)
                                        num_query,       // num_anchors (num_query)
                                        num_point,       // num_pts (num_point)
                                        num_groups);     // num_groups

    return rc;
}

void DeformableAttentionAggrPlugin::attachToContext(cudnnContext* contextCudnn,
                                                    cublasContext* contextCublas,
                                                    nvinfer1::IGpuAllocator* gpuAllocator) noexcept
{
    return;
}

void DeformableAttentionAggrPlugin::detachFromContext() noexcept
{
    return;
}

nvinfer1::DataType DeformableAttentionAggrPlugin::getOutputDataType(int32_t index,
                                                                    nvinfer1::DataType const* inputTypes,
                                                                    int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

const char* DeformableAttentionAggrPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* DeformableAttentionAggrPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t DeformableAttentionAggrPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t DeformableAttentionAggrPlugin::initialize() noexcept
{
    return 0;
}

size_t DeformableAttentionAggrPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void DeformableAttentionAggrPlugin::serialize(void* buffer) const noexcept
{
    return;
}

void DeformableAttentionAggrPlugin::destroy() noexcept
{
    delete this;
    return;
}

void DeformableAttentionAggrPlugin::terminate() noexcept
{
    return;
}

void DeformableAttentionAggrPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace_ = pluginNamespace;
    return;
}

const char* DeformableAttentionAggrPlugin::getPluginNamespace() const noexcept
{
    return mNamespace_.c_str();
}




// 插件创建类
DeformableAttentionAggrPluginCreator::DeformableAttentionAggrPluginCreator()
{
    mAttrs_.clear();
    mFC_.nbFields = mAttrs_.size();
    mFC_.fields = mAttrs_.data();
}

const char* DeformableAttentionAggrPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* DeformableAttentionAggrPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* DeformableAttentionAggrPluginCreator::getFieldNames() noexcept
{
    return &mFC_;
}

nvinfer1::IPluginV2* DeformableAttentionAggrPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept
{
    return new DeformableAttentionAggrPlugin();
}

nvinfer1::IPluginV2* DeformableAttentionAggrPluginCreator::deserializePlugin(const char* name,
                                                                             const void* serialData,
                                                                             size_t serialLength) noexcept
{
    return new DeformableAttentionAggrPlugin();
}

void DeformableAttentionAggrPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace_ = pluginNamespace;
    return;
}

const char* DeformableAttentionAggrPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace_.c_str();
}

}  // namespace custom
