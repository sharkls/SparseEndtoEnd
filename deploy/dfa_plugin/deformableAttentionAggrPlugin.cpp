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
#include <cuda_fp16.h>

// 声明CUDA函数
int thomas_deform_attn_cuda_forward(cudaStream_t stream,
                                    const float* value,
                                    const int* spatialShapes,
                                    const int* levelStartIndex,
                                    const float* key_points,
                                    const float* lidar2img,
                                    const float* image_wh,
                                    const float* attnWeight,
                                    float* output,
                                    int batch_size,
                                    int num_cams,
                                    int num_feat,
                                    int num_embeds,
                                    int num_scale,
                                    int num_anchors,
                                    int num_pts,
                                    int num_groups);

// 声明FP16版本的CUDA函数
int thomas_deform_attn_cuda_forward_half(cudaStream_t stream,
                                         const __half* value,
                                         const int* spatialShapes,
                                         const int* levelStartIndex,
                                         const __half* key_points,
                                         const __half* lidar2img,
                                         const __half* image_wh,
                                         const __half* attnWeight,
                                         __half* output,
                                         float* workspace,
                                         int batch_size,
                                         int num_cams,
                                         int num_feat,
                                         int num_embeds,
                                         int num_scale,
                                         int num_anchors,
                                         int num_pts,
                                         int num_groups);

// 声明混合精度版本的CUDA函数
int thomas_deform_attn_cuda_forward_mixed(cudaStream_t stream,
                                          const __half* value,          // FP16特征值
                                          const int* spatialShapes,
                                          const int* levelStartIndex,
                                          const float* key_points,      // FP32关键点位置
                                          const float* lidar2img,       // FP32投影矩阵
                                          const float* image_wh,        // FP32图像宽高
                                          const float* attnWeight,      // FP32注意力权重
                                          __half* output,               // FP16输出
                                          float* workspace,
                                          int batch_size,
                                          int num_cams,
                                          int num_feat,
                                          int num_embeds,
                                          int num_scale,
                                          int num_anchors,
                                          int num_pts,
                                          int num_groups);

// 声明 INT8 版本的 CUDA 函数
int thomas_deform_attn_cuda_forward_int8(cudaStream_t stream,
                                         const int8_t* value,          // INT8特征值
                                         float value_scale,            // Dequantization scale
                                         const int* spatialShapes,
                                         const int* levelStartIndex,
                                         const float* key_points,
                                         const float* lidar2img,
                                         const float* image_wh,
                                         const float* attnWeight,      // FP32注意力权重
                                         float* output,                // FP32输出
                                         int batch_size,
                                         int num_cams,
                                         int num_feat,
                                         int num_embeds,
                                         int num_scale,
                                         int num_anchors,
                                         int num_pts,
                                         int num_groups);

namespace custom
{

// 序列化辅助函数
template<typename T>
void writeToBuffer(char*& buffer, T const& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template<typename T>
void readFromBuffer(char const*& buffer, T& val)
{
    val = *reinterpret_cast<T const*>(buffer);
    buffer += sizeof(T);
}

REGISTER_TENSORRT_PLUGIN(DeformableAttentionAggrPluginCreator); // 注册插件到TensorRT

nvinfer1::IPluginV2DynamicExt* DeformableAttentionAggrPlugin::clone() const noexcept
{
    DeformableAttentionAggrPlugin* plugin = new DeformableAttentionAggrPlugin(mBatch_, mNumAnchors_, mNumEmbeds_, mNumCams_, mNumPts_, mValueScale_);
    plugin->setPluginNamespace(mNamespace_.c_str());
    // 复制运行时缓存
    plugin->mCachedBatch_ = mCachedBatch_;
    plugin->mCachedNumAnchors_ = mCachedNumAnchors_;
    plugin->mCachedNumEmbeds_ = mCachedNumEmbeds_;
    plugin->mCachedNumCams_ = mCachedNumCams_;
    plugin->mCachedNumPts_ = mCachedNumPts_;
    return plugin;
}

nvinfer1::DimsExprs DeformableAttentionAggrPlugin::getOutputDimensions(int32_t outputIndex,
                                                                       const nvinfer1::DimsExprs* inputs,
                                                                       int32_t nbInputs,
                                                                       nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // 安全检查：nbInputs 现在应该是 7 (value, spatial, start, key_points, lidar2img, image_wh, weights)
    if (nbInputs < 7 || !inputs)
    {
        nvinfer1::DimsExprs ret;
        ret.nbDims = 3;
        ret.d[0] = exprBuilder.constant(1);
        ret.d[1] = exprBuilder.constant(900);
        ret.d[2] = exprBuilder.constant(256);
        return ret;
    }
    
    // 所有检查通过，返回正常维度
    nvinfer1::DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0] = inputs[0].d[0];  // Batch
    ret.d[1] = inputs[3].d[1];  // Num Anchors (from key_points [BS*Q, P, 3], wait, key_points shape is [BS, Q, P, 3] or [BS*Q, P, 3]?)
    // 按照 PyTorch 导出，key_points 应该是 [BS, Q, P, 3]
    ret.d[1] = inputs[3].d[1]; 
    ret.d[2] = inputs[0].d[2];  // Embed dims
    return ret;
}

bool DeformableAttentionAggrPlugin::supportsFormatCombination(int32_t pos,
                                                              const nvinfer1::PluginTensorDesc* inOut,
                                                              int32_t nbInputs,
                                                              int32_t nbOutputs) noexcept
{
    if (!inOut || pos < 0 || pos >= (nbInputs + nbOutputs)) return false;
    if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) return false;
    
    // Position 1 and 2 are INT32 (spatialShapes and levelStartIndex)
    if ((pos == 1) || (pos == 2))
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT32);
    }
    
    // Position 0 is value
    if (pos == 0)
    {
        return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) || 
                (inOut[pos].type == nvinfer1::DataType::kHALF) ||
                (inOut[pos].type == nvinfer1::DataType::kINT8));
    }
    
    // Position 3, 4, 5, 6 are key_points, lidar2img, image_wh, attnWeight
    if (pos >= 3 && pos <= 6)
    {
        nvinfer1::DataType valueType = inOut[0].type;
        nvinfer1::DataType currentType = inOut[pos].type;
        
        // 1. FP32 mode
        if (valueType == nvinfer1::DataType::kFLOAT && currentType == nvinfer1::DataType::kFLOAT)
            return true;
        // 2. Mixed precision or INT8 mode (Inputs 3-6 remain FP32)
        if ((valueType == nvinfer1::DataType::kHALF || valueType == nvinfer1::DataType::kINT8) && 
            currentType == nvinfer1::DataType::kFLOAT)
            return true;
        // 3. Pure FP16 mode
        if (valueType == nvinfer1::DataType::kHALF && currentType == nvinfer1::DataType::kHALF)
            return true;
            
        return false;
    }
    
    // Output type
    if (pos >= nbInputs)
    {
        nvinfer1::DataType valueType = inOut[0].type;
        if (valueType == nvinfer1::DataType::kINT8) return (inOut[pos].type == nvinfer1::DataType::kFLOAT);
        return (inOut[pos].type == valueType);
    }
    
    return false;
}

void DeformableAttentionAggrPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                                    int32_t nbInputs,
                                                    const nvinfer1::DynamicPluginTensorDesc* out,
                                                    int32_t nbOutputs) noexcept
{
    if (!in || nbInputs < 7)
    {
        return;
    }
    
    // 提取 INT8 scale
    if (in[0].desc.type == nvinfer1::DataType::kINT8) {
        mValueScale_ = in[0].desc.scale;
    } else {
        mValueScale_ = 1.0f;
    }

    // in[0]: value [BS, L, C]
    mBatch_ = in[0].desc.dims.d[0];
    mNumEmbeds_ = in[0].desc.dims.d[2];
    
    // in[3]: key_points [BS, Q, P, 3]
    mNumAnchors_ = in[3].desc.dims.d[1];
    mNumPts_ = in[3].desc.dims.d[2];

    // in[4]: lidar2img [BS, C, 4, 4]
    mNumCams_ = in[4].desc.dims.d[1];
    
    return;
}

size_t DeformableAttentionAggrPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                                       int32_t nbInputs,
                                                       const nvinfer1::PluginTensorDesc* outputs,
                                                       int32_t nbOutputs) const noexcept
{
    // Gather 模式不需要额外的 workspace 进行原子操作累加
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
    if (!inputDesc || !outputDesc || !inputs || !outputs) return 1;
    
    // 维度提取 (从 inputDesc 中提取，保证动态形状下的正确性)
    int32_t const batch = inputDesc[0].dims.d[0];
    int32_t num_feat = inputDesc[0].dims.d[1];
    int32_t channels = inputDesc[0].dims.d[2];
    
    int32_t num_cams = inputDesc[1].dims.d[0];
    int32_t num_levels = inputDesc[1].dims.d[1];
    
    int32_t num_query = inputDesc[3].dims.d[1];
    int32_t num_point = inputDesc[3].dims.d[2];
    
    int32_t num_groups = inputDesc[6].dims.d[5]; // attnWeight: [BS, Q, P, Cam, L, G]
    int32_t rc = 0;

    nvinfer1::DataType dataType = inputDesc[0].type;
    
    // 统一使用 FP32 处理坐标相关的投影输入 (lidar2img, image_wh, key_points 在非全 FP16 模式下默认为 FP32)
    // 除非是全 FP16 模式
    bool isPureHalf = (dataType == nvinfer1::DataType::kHALF) && (inputDesc[3].type == nvinfer1::DataType::kHALF);
    bool isMixedPrecision = (dataType == nvinfer1::DataType::kHALF) && (inputDesc[3].type == nvinfer1::DataType::kFLOAT);

    if (dataType == nvinfer1::DataType::kINT8)
    {
        const int8_t* value = static_cast<const int8_t*>(inputs[0]);
        const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);
        const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);
        const float* key_points = static_cast<const float*>(inputs[3]);
        const float* lidar2img = static_cast<const float*>(inputs[4]);
        const float* image_wh = static_cast<const float*>(inputs[5]);
        const float* attnWeight = static_cast<const float*>(inputs[6]);
        float* output = static_cast<float*>(outputs[0]);

        rc = thomas_deform_attn_cuda_forward_int8(stream, value, mValueScale_, spatialShapes, levelStartIndex, 
                                                 key_points, lidar2img, image_wh, attnWeight, output,
                                                 batch, num_cams, num_feat, channels, num_levels, num_query, num_point, num_groups);
    }
    else if (isMixedPrecision)
    {
        const __half* value = static_cast<const __half*>(inputs[0]);
        const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);
        const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);
        const float* key_points = static_cast<const float*>(inputs[3]);
        const float* lidar2img = static_cast<const float*>(inputs[4]);
        const float* image_wh = static_cast<const float*>(inputs[5]);
        const float* attnWeight = static_cast<const float*>(inputs[6]);
        __half* output = static_cast<__half*>(outputs[0]);

        rc = thomas_deform_attn_cuda_forward_mixed(stream, value, spatialShapes, levelStartIndex, 
                                                  key_points, lidar2img, image_wh, attnWeight, output, nullptr,
                                                  batch, num_cams, num_feat, channels, num_levels, num_query, num_point, num_groups);
    }
    else if (dataType == nvinfer1::DataType::kFLOAT)
    {
        const float* value = static_cast<const float*>(inputs[0]);
        const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);
        const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);
        const float* key_points = static_cast<const float*>(inputs[3]);
        const float* lidar2img = static_cast<const float*>(inputs[4]);
        const float* image_wh = static_cast<const float*>(inputs[5]);
        const float* attnWeight = static_cast<const float*>(inputs[6]);
        float* output = static_cast<float*>(outputs[0]);

        rc = thomas_deform_attn_cuda_forward(stream, value, spatialShapes, levelStartIndex, 
                                            key_points, lidar2img, image_wh, attnWeight, output,
                                            batch, num_cams, num_feat, channels, num_levels, num_query, num_point, num_groups);
    }
    else if (isPureHalf)
    {
        const __half* value = static_cast<const __half*>(inputs[0]);
        const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);
        const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);
        const __half* key_points = static_cast<const __half*>(inputs[3]);
        const __half* lidar2img = static_cast<const __half*>(inputs[4]);
        const __half* image_wh = static_cast<const __half*>(inputs[5]);
        const __half* attnWeight = static_cast<const __half*>(inputs[6]);
        __half* output = static_cast<__half*>(outputs[0]);

        rc = thomas_deform_attn_cuda_forward_half(stream, value, spatialShapes, levelStartIndex, 
                                                 key_points, lidar2img, image_wh, attnWeight, output, nullptr,
                                                 batch, num_cams, num_feat, channels, num_levels, num_query, num_point, num_groups);
    }
    else
    {
        return 1;
    }

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
    if (!inputTypes || nbInputs < 1) return nvinfer1::DataType::kFLOAT;
    
    // 如果输入是 INT8，输出为 FP32 (反量化后处理)
    if (inputTypes[0] == nvinfer1::DataType::kINT8)
    {
        return nvinfer1::DataType::kFLOAT;
    }
    
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
    // batch, num_anchors, num_embeds, num_cams, num_pts (5 * int32) + scale (float)
    return sizeof(int32_t) * 5 + sizeof(float);
}

void DeformableAttentionAggrPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    writeToBuffer(d, mBatch_);
    writeToBuffer(d, mNumAnchors_);
    writeToBuffer(d, mNumEmbeds_);
    writeToBuffer(d, mNumCams_);
    writeToBuffer(d, mNumPts_);
    writeToBuffer(d, mValueScale_);
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
    if (!serialData || serialLength < sizeof(int32_t) * 3)
    {
        return new DeformableAttentionAggrPlugin();
    }
    
    const char* d = static_cast<const char*>(serialData);
    int32_t batch, numAnchors, numEmbeds, numCams = 6, numPts = 13;
    float valueScale = 1.0f;
    
    readFromBuffer(d, batch);
    readFromBuffer(d, numAnchors);
    readFromBuffer(d, numEmbeds);
    
    // 兼容逻辑：检查序列化数据长度
    if (serialLength >= sizeof(int32_t) * 5 + sizeof(float)) {
        readFromBuffer(d, numCams);
        readFromBuffer(d, numPts);
        readFromBuffer(d, valueScale);
    } else if (serialLength >= sizeof(int32_t) * 3 + sizeof(float)) {
        readFromBuffer(d, valueScale);
    }
    
    return new DeformableAttentionAggrPlugin(batch, numAnchors, numEmbeds, numCams, numPts, valueScale);
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
