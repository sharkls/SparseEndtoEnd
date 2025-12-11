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
                                    const float* samplingLoc,
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

// 声明FP16版本的CUDA函数（优化版本：使用FP32临时缓冲区，通过workspace提供）
int thomas_deform_attn_cuda_forward_half(cudaStream_t stream,
                                         const __half* value,
                                         const int* spatialShapes,
                                         const int* levelStartIndex,
                                         const __half* samplingLoc,
                                         const __half* attnWeight,
                                         __half* output,
                                         float* workspace,  // TensorRT提供的workspace
                                         int batch_size,
                                         int num_cams,
                                         int num_feat,
                                         int num_embeds,
                                         int num_scale,
                                         int num_anchors,
                                         int num_pts,
                                         int num_groups);

// 声明混合精度版本的CUDA函数：FP16 value + FP32 keypoints（关键点保持FP32精度）
int thomas_deform_attn_cuda_forward_mixed(cudaStream_t stream,
                                          const __half* value,          // FP16特征值
                                          const int* spatialShapes,
                                          const int* levelStartIndex,
                                          const float* samplingLoc,    // FP32关键点位置
                                          const float* attnWeight,      // FP32注意力权重
                                          __half* output,               // FP16输出
                                          float* workspace,              // TensorRT提供的workspace
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
                                         const float* samplingLoc,     // FP32关键点位置
                                         const float* attnWeight,      // FP32注意力权重
                                         float* output,                // FP32输出 (或根据需求改为Half/Int8)
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
    DeformableAttentionAggrPlugin* plugin = new DeformableAttentionAggrPlugin(mBatch_, mNumAnchors_, mNumEmbeds_, mValueScale_);
    plugin->setPluginNamespace(mNamespace_.c_str());
    // 复制运行时缓存
    plugin->mCachedBatch_ = mCachedBatch_;
    plugin->mCachedNumAnchors_ = mCachedNumAnchors_;
    plugin->mCachedNumEmbeds_ = mCachedNumEmbeds_;
    return plugin;
}

nvinfer1::DimsExprs DeformableAttentionAggrPlugin::getOutputDimensions(int32_t outputIndex,
                                                                       const nvinfer1::DimsExprs* inputs,
                                                                       int32_t nbInputs,
                                                                       nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // 安全检查：确保有足够的输入和有效的指针
    if (nbInputs < 5 || !inputs)
    {
        // 返回默认维度（如果输入不足）
        nvinfer1::DimsExprs ret;
        ret.nbDims = 3;
        ret.d[0] = exprBuilder.constant(1);
        ret.d[1] = exprBuilder.constant(900);
        ret.d[2] = exprBuilder.constant(256);
        return ret;
    }
    
    // 安全检查：确保inputs[0]有效
    if (inputs[0].nbDims < 3 || !inputs[0].d || !inputs[0].d[0] || !inputs[0].d[2])
    {
        nvinfer1::DimsExprs ret;
        ret.nbDims = 3;
        ret.d[0] = exprBuilder.constant(1);
        ret.d[1] = exprBuilder.constant(900);
        ret.d[2] = exprBuilder.constant(256);
        return ret;
    }
    
    // 安全检查：确保inputs[3]有效
    if (inputs[3].nbDims < 2 || !inputs[3].d || !inputs[3].d[1])
    {
        // 如果inputs[3]无效，使用inputs[0]的维度，但确保inputs[0]有效
        nvinfer1::DimsExprs ret;
        ret.nbDims = 3;
        ret.d[0] = inputs[0].d[0];  // 此时inputs[0]已经验证过
        ret.d[1] = exprBuilder.constant(900);
        ret.d[2] = inputs[0].d[2];  // 此时inputs[0]已经验证过
        return ret;
    }
    
    // 所有检查通过，返回正常维度
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
    // 安全检查：确保inOut指针有效，pos在有效范围内
    if (!inOut || pos < 0 || pos >= (nbInputs + nbOutputs))
    {
        return false;
    }
    
    // 检查格式：只支持LINEAR格式
    if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
    {
        return false;
    }
    
    // 位置1和2是INT32类型（spatialShapes和levelStartIndex）
    if ((pos == 1) || (pos == 2))
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT32);
    }
    
    // 位置0是value（特征值）
    if (pos == 0)
    {
        return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) || 
                (inOut[pos].type == nvinfer1::DataType::kHALF));
                // (inOut[pos].type == nvinfer1::DataType::kINT8)); // 支持 INT8 输入
    }
    
    // 位置3是samplingLoc（关键点位置），位置4是attnWeight（注意力权重）
    if (pos == 3 || pos == 4)
    {
        // 安全检查：确保有至少1个输入（value）
        if (nbInputs < 1)
        {
            return false;
        }
        
        nvinfer1::DataType valueType = inOut[0].type;
        nvinfer1::DataType keypointType = inOut[pos].type;
        
        // 1. 全FP32模式
        if (valueType == nvinfer1::DataType::kFLOAT && keypointType == nvinfer1::DataType::kFLOAT)
        {
            return true;
        }
        // 2. 全FP16模式 (禁用，强制混合精度以保证坐标/权重精度)
        // else if (valueType == nvinfer1::DataType::kHALF && keypointType == nvinfer1::DataType::kHALF)
        // {
        //     return true;
        // }
        // 3. 混合精度模式：value=FP16 + keypoints=FP32
        else if (valueType == nvinfer1::DataType::kHALF && keypointType == nvinfer1::DataType::kFLOAT)
        {
            return true;
        }
        // 4. INT8 模式：value=INT8 + keypoints=FP32 (Coords & Weights 保持高精度)
        // else if (valueType == nvinfer1::DataType::kINT8 && keypointType == nvinfer1::DataType::kFLOAT)
        // {
        //     return true;
        // }
        
        return false;
    }
    
    // 输出类型
    if (pos >= nbInputs)
    {
        if (nbInputs < 1) return false;
        
        nvinfer1::DataType valueType = inOut[0].type;
        
        // 如果输入是 INT8，输出可以是 FP32 (反量化后处理)
        // if (valueType == nvinfer1::DataType::kINT8)
        // {
        //     return (inOut[pos].type == nvinfer1::DataType::kFLOAT);
        // }
        
        // 否则输出与输入同类型
        if (valueType == nvinfer1::DataType::kFLOAT || valueType == nvinfer1::DataType::kHALF)
        {
            return (inOut[pos].type == valueType);
        }
        
        return false;
    }
    
    return false;
}

void DeformableAttentionAggrPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                                                    int32_t nbInputs,
                                                    const nvinfer1::DynamicPluginTensorDesc* out,
                                                    int32_t nbOutputs) noexcept
{
    // 关键修复：在configurePlugin中安全提取维度信息并保存
    
    if (!in || nbInputs < 5)
    {
        printf("[DFA-PLUGIN] configurePlugin: Invalid inputs, using default values\n");
        return;
    }
    
    // 提取 INT8 scale
    // if (in[0].desc.type == nvinfer1::DataType::kINT8) {
    //     mValueScale_ = in[0].desc.scale;
    //     // printf("[DFA-PLUGIN] configurePlugin: INT8 mode detected, scale = %f\n", mValueScale_);
    // } else {
        mValueScale_ = 1.0f;
    // }

    // 安全提取inputs[0]的维度信息（batch和embeds）
    if (in[0].desc.dims.nbDims >= 3 && in[0].desc.dims.d != nullptr)
    {
        mBatch_ = in[0].desc.dims.d[0];
        mNumEmbeds_ = in[0].desc.dims.d[2];
        
        if (mBatch_ <= 0 || mBatch_ > 100) mBatch_ = 1;
        if (mNumEmbeds_ <= 0 || mNumEmbeds_ > 10000) mNumEmbeds_ = 256;
    }
    
    // 安全提取inputs[3]的维度信息（num_anchors）
    if (in[3].desc.dims.nbDims >= 2 && in[3].desc.dims.d != nullptr)
    {
        mNumAnchors_ = in[3].desc.dims.d[1];
        
        if (mNumAnchors_ <= 0 || mNumAnchors_ > 10000) mNumAnchors_ = 900;
    }
    
    // printf("[DFA-PLUGIN] configurePlugin: Saved dimensions (batch=%d, anchors=%d, embeds=%d, scale=%f)\n",
    //        mBatch_, mNumAnchors_, mNumEmbeds_, mValueScale_);
    
    return;
}

size_t DeformableAttentionAggrPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                                       int32_t nbInputs,
                                                       const nvinfer1::PluginTensorDesc* outputs,
                                                       int32_t nbOutputs) const noexcept
{
    // 动态计算所需的 workspace 大小
    size_t workspaceSize = 4 * 1024 * 1024;
    
    if (inputs && nbInputs >= 5)
    {
        int32_t batch = 1;
        int32_t num_query = 900;
        int32_t channels = 256;
        
        if (inputs[0].dims.nbDims >= 3)
        {
            batch = inputs[0].dims.d[0] > 0 ? inputs[0].dims.d[0] : 1;
            channels = inputs[0].dims.d[2] > 0 ? inputs[0].dims.d[2] : 256;
        }
        
        if (inputs[3].dims.nbDims >= 2)
        {
            num_query = inputs[3].dims.d[1] > 0 ? inputs[3].dims.d[1] : 900;
        }
        
        // 计算实际需求
        size_t required = static_cast<size_t>(batch) * num_query * channels * sizeof(float);
        
        if (required > workspaceSize)
        {
            workspaceSize = static_cast<size_t>(required * 1.5);
        }
        else
        {
            workspaceSize = std::max(workspaceSize, required + 1024 * 1024);
        }
    }
    
    return workspaceSize;
}

// 推理函数
int32_t DeformableAttentionAggrPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                               const nvinfer1::PluginTensorDesc* outputDesc,
                                               const void* const* inputs,
                                               void* const* outputs,
                                               void* workspace,
                                               cudaStream_t stream) noexcept
{
    // 安全检查
    if (!inputDesc || !outputDesc || !inputs || !outputs)
    {
        return 1;
    }
    
    // 安全检查：确保dims结构有效
    if (inputDesc[0].dims.nbDims < 3 || inputDesc[1].dims.nbDims < 2 || 
        inputDesc[3].dims.nbDims < 2 || inputDesc[4].dims.nbDims < 6)
    {
        return 1;
    }
    
    int32_t const batch = inputDesc[0].dims.d[0];
    int32_t spatial_size = inputDesc[0].dims.d[1];
    int32_t channels = inputDesc[0].dims.d[2];
    int32_t num_cams = inputDesc[1].dims.d[0];
    int32_t num_levels = inputDesc[1].dims.d[1];
    int32_t num_query = inputDesc[3].dims.d[1];
    int32_t num_point = inputDesc[3].dims.d[2];
    int32_t num_groups = inputDesc[4].dims.d[5];
    int32_t rc = 0;

    nvinfer1::DataType dataType = inputDesc[0].type;
    nvinfer1::DataType samplingLocType = inputDesc[3].type;
    nvinfer1::DataType attnWeightType = inputDesc[4].type;
    
    bool isMixedPrecision = (dataType == nvinfer1::DataType::kHALF) && 
                            (samplingLocType == nvinfer1::DataType::kFLOAT) && 
                            (attnWeightType == nvinfer1::DataType::kFLOAT);
    
    if (dataType == nvinfer1::DataType::kINT8)
    {
        // INT8 模式：value=INT8, keypoints/weights=FP32, output=FP32
        // // printf("[DFA-PLUGIN] Enqueue: INT8 Mode\n");
        // const int8_t* value = static_cast<const int8_t*>(inputs[0]);
        // const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);
        // const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);
        // const float* samplingLoc = static_cast<const float*>(inputs[3]);
        // const float* attnWeight = static_cast<const float*>(inputs[4]);
        // float* output = static_cast<float*>(outputs[0]);

        // rc = thomas_deform_attn_cuda_forward_int8(stream,
        //                                           value,
        //                                           mValueScale_,  // Use stored scale
        //                                           spatialShapes,
        //                                           levelStartIndex,
        //                                           samplingLoc,
        //                                           attnWeight,
        //                                           output,
        //                                           batch,
        //                                           num_cams,
        //                                           spatial_size,
        //                                           channels,
        //                                           num_levels,
        //                                           num_query,
        //                                           num_point,
        //                                           num_groups);
        return 1; // Not supported for now
    }
    else if (isMixedPrecision)
    {
        // 混合精度模式
        // printf("[DFA-PLUGIN] Enqueue: Mixed Precision Mode (FP16 Feat + FP32 Loc/Weight)\n");
        const __half* value = static_cast<const __half*>(inputs[0]);
        const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);
        const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);
        const float* samplingLoc = static_cast<const float*>(inputs[3]);
        const float* attnWeight = static_cast<const float*>(inputs[4]);
        __half* output = static_cast<__half*>(outputs[0]);
        
        // Remove workspace check as it is not used in gathered mixed kernel
        // if (workspace == nullptr) return 1;
        // float* workspace_ptr = static_cast<float*>(workspace);
        float* workspace_ptr = nullptr; 

        rc = thomas_deform_attn_cuda_forward_mixed(stream,
                                                  value,
                                                  spatialShapes,
                                                  levelStartIndex,
                                                  samplingLoc,
                                                  attnWeight,
                                                  output,
                                                  workspace_ptr,
                                                  batch,
                                                  num_cams,
                                                  spatial_size,
                                                  channels,
                                                  num_levels,
                                                  num_query,
                                                  num_point,
                                                  num_groups);
    }
    else if (dataType == nvinfer1::DataType::kFLOAT)
    {
        // printf("[DFA-PLUGIN] Enqueue: FP32 Mode\n");
        const float* value = static_cast<const float*>(inputs[0]);
        const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);
        const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);
        const float* samplingLoc = static_cast<const float*>(inputs[3]);
        const float* attnWeight = static_cast<const float*>(inputs[4]);
        float* output = static_cast<float*>(outputs[0]);

        rc = thomas_deform_attn_cuda_forward(stream,
                                            value,
                                            spatialShapes,
                                            levelStartIndex,
                                            samplingLoc,
                                            attnWeight,
                                            output,
                                            batch,
                                            num_cams,
                                            spatial_size,
                                            channels,
                                            num_levels,
                                            num_query,
                                            num_point,
                                            num_groups);
    }
    else if (dataType == nvinfer1::DataType::kHALF)
    {
        // printf("[DFA-PLUGIN] Enqueue: Pure FP16 Mode\n");
        const __half* value = static_cast<const __half*>(inputs[0]);
        const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);
        const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);
        const __half* samplingLoc = static_cast<const __half*>(inputs[3]);
        const __half* attnWeight = static_cast<const __half*>(inputs[4]);
        __half* output = static_cast<__half*>(outputs[0]);
        
        // float* workspace_ptr = static_cast<float*>(workspace);
        float* workspace_ptr = nullptr;

        rc = thomas_deform_attn_cuda_forward_half(stream,
                                                  value,
                                                  spatialShapes,
                                                  levelStartIndex,
                                                  samplingLoc,
                                                  attnWeight,
                                                  output,
                                                  workspace_ptr,
                                                  batch,
                                                  num_cams,
                                                  spatial_size,
                                                  channels,
                                                  num_levels,
                                                  num_query,
                                                  num_point,
                                                  num_groups);
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
    
    // 如果输入是 INT8，输出为 FP32
    // if (inputTypes[0] == nvinfer1::DataType::kINT8)
    // {
    //     return nvinfer1::DataType::kFLOAT;
    // }
    
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
    // 序列化维度参数：batch, num_anchors, num_embeds (每个int32_t = 4字节) + scale (float)
    return sizeof(int32_t) * 3 + sizeof(float);
}

void DeformableAttentionAggrPlugin::serialize(void* buffer) const noexcept
{
    // 序列化维度参数到buffer
    char* d = static_cast<char*>(buffer);
    writeToBuffer(d, mBatch_);
    writeToBuffer(d, mNumAnchors_);
    writeToBuffer(d, mNumEmbeds_);
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
    // 从序列化数据中恢复维度参数
    if (!serialData || serialLength < sizeof(int32_t) * 3)
    {
        return new DeformableAttentionAggrPlugin();  // 使用默认值
    }
    
    const char* d = static_cast<const char*>(serialData);
    int32_t batch, numAnchors, numEmbeds;
    float valueScale = 1.0f;
    
    readFromBuffer(d, batch);
    readFromBuffer(d, numAnchors);
    readFromBuffer(d, numEmbeds);
    
    // 检查是否有额外的数据（scale）
    // 兼容旧版本序列化数据
    size_t expectedSizeWithScale = sizeof(int32_t) * 3 + sizeof(float);
    if (serialLength >= expectedSizeWithScale) {
        readFromBuffer(d, valueScale);
    }
    
    // 验证合理性
    if (batch <= 0 || batch > 100) batch = 1;
    if (numAnchors <= 0 || numAnchors > 10000) numAnchors = 900;
    if (numEmbeds <= 0 || numEmbeds > 10000) numEmbeds = 256;
    
    return new DeformableAttentionAggrPlugin(batch, numAnchors, numEmbeds, valueScale);
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
