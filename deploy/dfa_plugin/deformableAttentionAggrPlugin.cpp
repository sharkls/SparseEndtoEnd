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
    DeformableAttentionAggrPlugin* plugin = new DeformableAttentionAggrPlugin(mBatch_, mNumAnchors_, mNumEmbeds_);
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
        return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) || (inOut[pos].type == nvinfer1::DataType::kHALF));
    }
    
    // 位置3是samplingLoc（关键点位置），位置4是attnWeight（注意力权重）
    // 支持以下format组合：
    // 1. 全FP32模式：value=FP32, keypoints=FP32
    // 2. 全FP16模式：value=FP16, keypoints=FP16
    // 3. 混合精度模式：value=FP16, keypoints=FP32（优化精度）
    // 严格禁止：value=FP32, keypoints=FP16（不支持的反向混合精度）
    if (pos == 3 || pos == 4)
    {
        // 安全检查：确保有至少1个输入（value）
        if (nbInputs < 1)
        {
            return false;
        }
        
        // 获取value的类型（位置0）- 已经验证过pos在有效范围内
        nvinfer1::DataType valueType = inOut[0].type;
        nvinfer1::DataType keypointType = inOut[pos].type;
        
        // 严格检查：只允许以下三种组合
        // 1. 全FP32模式：value=FP32 + keypoints=FP32
        if (valueType == nvinfer1::DataType::kFLOAT && keypointType == nvinfer1::DataType::kFLOAT)
        {
            return true;
        }
        // 2. 全FP16模式：value=FP16 + keypoints=FP16
        else if (valueType == nvinfer1::DataType::kHALF && keypointType == nvinfer1::DataType::kHALF)
        {
            return true;
        }
        // 3. 混合精度模式：value=FP16 + keypoints=FP32（保持关键点高精度）
        else if (valueType == nvinfer1::DataType::kHALF && keypointType == nvinfer1::DataType::kFLOAT)
        {
            return true;
        }
        // 严格禁止：value=FP32 + keypoints=FP16（不支持的反向混合精度）
        else if (valueType == nvinfer1::DataType::kFLOAT && keypointType == nvinfer1::DataType::kHALF)
        {
            printf("[DFA-PLUGIN-WARNING] Rejected unsupported format: FP32 value + FP16 keypoints\n");
            return false;
        }
        
        // 拒绝其他无效的format组合
        return false;
    }
    
    // 输出类型与value类型相同
    if (pos >= nbInputs)
    {
        // 安全检查：确保有至少1个输入（value）
        if (nbInputs < 1)
        {
            return false;
        }
        
        // 获取value的类型（位置0）
        nvinfer1::DataType valueType = inOut[0].type;
        
        // 如果value类型有效，输出类型必须与value类型相同
        if (valueType == nvinfer1::DataType::kFLOAT || valueType == nvinfer1::DataType::kHALF)
        {
            return (inOut[pos].type == valueType);
        }
        
        // 如果value类型无效，拒绝该格式组合（更严格）
        return false;
    }
    
    return false;
}

void DeformableAttentionAggrPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
                                                    int32_t nbInputs,
                                                    nvinfer1::DynamicPluginTensorDesc const* out,
                                                    int32_t nbOutputs) noexcept
{
    // 关键修复：在configurePlugin中安全提取维度信息并保存
    // configurePlugin在插件创建后、autotuning之前调用，此时输入信息是完整且安全的
    
    if (!in || nbInputs < 5)
    {
        printf("[DFA-PLUGIN] configurePlugin: Invalid inputs, using default values\n");
        return;  // 使用默认值
    }
    
    // 安全提取inputs[0]的维度信息（batch和embeds）
    if (in[0].desc.dims.nbDims >= 3 && in[0].desc.dims.d != nullptr)
    {
        mBatch_ = in[0].desc.dims.d[0];
        mNumEmbeds_ = in[0].desc.dims.d[2];
        
        // 验证维度值合理性
        if (mBatch_ <= 0 || mBatch_ > 100) mBatch_ = 1;
        if (mNumEmbeds_ <= 0 || mNumEmbeds_ > 10000) mNumEmbeds_ = 256;
    }
    
    // 安全提取inputs[3]的维度信息（num_anchors）
    if (in[3].desc.dims.nbDims >= 2 && in[3].desc.dims.d != nullptr)
    {
        mNumAnchors_ = in[3].desc.dims.d[1];
        
        // 验证维度值合理性
        if (mNumAnchors_ <= 0 || mNumAnchors_ > 10000) mNumAnchors_ = 900;
    }
    
    printf("[DFA-PLUGIN] configurePlugin: Saved dimensions (batch=%d, anchors=%d, embeds=%d)\n",
           mBatch_, mNumAnchors_, mNumEmbeds_);
    
    return;
}

size_t DeformableAttentionAggrPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                                       int32_t nbInputs,
                                                       const nvinfer1::PluginTensorDesc* outputs,
                                                       int32_t nbOutputs) const noexcept
{
    // 动态计算所需的 workspace 大小，避免硬编码导致的越界
    // 基本需求：batch * num_query * channels * sizeof(float)
    
    // 默认值（保底 4MB）
    size_t workspaceSize = 4 * 1024 * 1024;
    
    if (inputs && nbInputs >= 5)
    {
        // 尝试从输入推断维度
        // inputs[0]: value [batch, spatial, channels]
        // inputs[3]: samplingLoc [batch, num_query, ...]
        
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
        
        // 如果计算值大于默认值，使用计算值并增加 50% 余量
        if (required > workspaceSize)
        {
            workspaceSize = static_cast<size_t>(required * 1.5);
        }
        else
        {
            // 即使计算值较小，也至少保留 required + 1MB 的空间
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
    // 安全检查：确保所有指针有效
    if (!inputDesc || !outputDesc || !inputs || !outputs)
    {
        printf("[DFA-PLUGIN-ERROR] Invalid input/output pointers (inputDesc=%p, outputDesc=%p, inputs=%p, outputs=%p)\n",
               inputDesc, outputDesc, inputs, outputs);
        return 1;
    }
    
    // 安全检查：确保有足够的输入
    // 需要至少5个输入：value, spatialShapes, levelStartIndex, samplingLoc, attnWeight
    // 但TensorRT在构建时可能会用不同的输入数量调用，所以需要更宽松的检查
    // 这里只检查必要的输入是否存在
    
    // 安全检查：确保dims结构有效
    if (inputDesc[0].dims.nbDims < 3 || inputDesc[1].dims.nbDims < 2 || 
        inputDesc[3].dims.nbDims < 2 || inputDesc[4].dims.nbDims < 6)
    {
        printf("[DFA-PLUGIN-ERROR] Invalid input dimensions (input[0].nbDims=%d, input[1].nbDims=%d, input[3].nbDims=%d, input[4].nbDims=%d)\n",
               inputDesc[0].dims.nbDims, inputDesc[1].dims.nbDims, 
               inputDesc[3].dims.nbDims, inputDesc[4].dims.nbDims);
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

    // 根据输入数据类型选择FP32、FP16或混合精度路径
    nvinfer1::DataType dataType = inputDesc[0].type;
    nvinfer1::DataType samplingLocType = inputDesc[3].type;  // 关键点位置类型
    nvinfer1::DataType attnWeightType = inputDesc[4].type;   // 注意力权重类型
    
    // 检测混合精度模式：FP16 value + FP32 keypoints
    bool isMixedPrecision = (dataType == nvinfer1::DataType::kHALF) && 
                            (samplingLocType == nvinfer1::DataType::kFLOAT) && 
                            (attnWeightType == nvinfer1::DataType::kFLOAT);
    
    if (isMixedPrecision)
    {
        // 混合精度模式：FP16 value + FP32 keypoints
        // 安全检查：确保所有输入指针有效
        if (!inputs || !inputs[0] || !inputs[1] || !inputs[2] || !inputs[3] || !inputs[4] || !outputs || !outputs[0])
        {
            printf("[DFA-PLUGIN-ERROR] Mixed precision: Invalid input/output pointers\n");
            return 1;
        }
        
        const __half* value = static_cast<const __half*>(inputs[0]);                  // [1, 89760, 128] FP16
        const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);      // [6, 4, 2]
        const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);    // [6, 4]
        const float* samplingLoc = static_cast<const float*>(inputs[3]);            // [1, 900, 13, 6, 2] FP32
        const float* attnWeight = static_cast<const float*>(inputs[4]);             // [1, 900, 13, 6, 4, 8] FP32

        __half* output = static_cast<__half*>(outputs[0]);
        
        // 修复方案：完全依赖TensorRT提供的workspace，如果为nullptr则返回错误
        // 原因：在TensorRT builder阶段使用cudaMallocAsync会导致异常和段错误
        if (workspace == nullptr)
        {
            size_t required_workspace_size = batch * num_query * channels * sizeof(float);
            printf("[DFA-PLUGIN-ERROR] Mixed precision: Workspace is null. Required workspace size: %zu bytes (batch=%d, anchors=%d, embeds=%d)\n",
                   required_workspace_size, batch, num_query, channels);
            printf("[DFA-PLUGIN-ERROR] This format combination is not supported. TensorRT should have allocated workspace in getWorkspaceSize.\n");
            return 1;  // 返回错误码，让TensorRT知道该format组合不可用
        }
        
        float* workspace_ptr = static_cast<float*>(workspace);

        // 调用混合精度版本（FP16 value + FP32 keypoints）
        // 修复参数顺序：必须与thomas_deform_attn_cuda_forward保持一致
        rc = thomas_deform_attn_cuda_forward_mixed(stream,
                                                  value,
                                                  spatialShapes,
                                                  levelStartIndex,
                                                  samplingLoc,
                                                  attnWeight,
                                                  output,
                                                  workspace_ptr,
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
    else if (dataType == nvinfer1::DataType::kFLOAT)
    {
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
    }
    else if (dataType == nvinfer1::DataType::kHALF)
    {
        const __half* value = static_cast<const __half*>(inputs[0]);                  // [1, 89760, 128]
        const int32_t* spatialShapes = static_cast<const int32_t*>(inputs[1]);      // [6, 4, 2]
        const int32_t* levelStartIndex = static_cast<const int32_t*>(inputs[2]);    // [6, 4]
        const __half* samplingLoc = static_cast<const __half*>(inputs[3]);            // [1, 900, 13, 6, 2]
        const __half* attnWeight = static_cast<const __half*>(inputs[4]);             // [1, 900, 13, 6, 4, 8]

        __half* output = static_cast<__half*>(outputs[0]);
        
        // 修复方案：完全依赖TensorRT提供的workspace，如果为nullptr则返回错误
        // 原因：在TensorRT builder阶段使用cudaMallocAsync会导致异常和段错误
        if (workspace == nullptr)
        {
            size_t required_workspace_size = batch * num_query * channels * sizeof(float);
            printf("[DFA-PLUGIN-ERROR] Workspace is null for FP16 mode. Required workspace size: %zu bytes (batch=%d, anchors=%d, embeds=%d)\n",
                   required_workspace_size, batch, num_query, channels);
            printf("[DFA-PLUGIN-ERROR] This format combination is not supported. TensorRT should have allocated workspace in getWorkspaceSize.\n");
            return 1;  // 返回错误码，让TensorRT知道该format组合不可用
        }
        
        float* workspace_ptr = static_cast<float*>(workspace);

        // FP16优化版本：内部使用FP32临时缓冲区，atomicAdd更快
        // workspace由TensorRT在getWorkspaceSize中分配
        rc = thomas_deform_attn_cuda_forward_half(stream,
                                                  value,
                                                  spatialShapes,
                                                  levelStartIndex,
                                                  samplingLoc,
                                                  attnWeight,
                                                  output,
                                                  workspace_ptr,  // 使用TensorRT提供的workspace
                                                  batch,           // batch_size
                                                  num_cams,        // num_cams
                                                  spatial_size,    // num_feat (spatial_size)
                                                  channels,        // num_embeds (channels)
                                                  num_levels,      // num_scale (num_levels)
                                                  num_query,       // num_anchors (num_query)
                                                  num_point,       // num_pts (num_point)
                                                  num_groups);     // num_groups
    }
    else
    {
        printf("[DFA-PLUGIN-ERROR] Unsupported data type: %d\n", static_cast<int>(dataType));
        // 返回错误码，让TensorRT知道该format组合不可用
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
    // 安全检查：确保inputTypes指针有效，且有足够的输入
    if (!inputTypes || nbInputs < 1)
    {
        // 如果输入无效，返回默认的FP32类型
        printf("[DFA-PLUGIN-WARNING] getOutputDataType: Invalid inputTypes or nbInputs < 1, returning kFLOAT\n");
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
    // 序列化维度参数：batch, num_anchors, num_embeds (每个int32_t = 4字节)
    return sizeof(int32_t) * 3;  // 3个int32_t
}

void DeformableAttentionAggrPlugin::serialize(void* buffer) const noexcept
{
    // 序列化维度参数到buffer
    char* d = static_cast<char*>(buffer);
    writeToBuffer(d, mBatch_);
    writeToBuffer(d, mNumAnchors_);
    writeToBuffer(d, mNumEmbeds_);
    
    printf("[DFA-PLUGIN] serialize: Saved dimensions (batch=%d, anchors=%d, embeds=%d)\n",
           mBatch_, mNumAnchors_, mNumEmbeds_);
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
        printf("[DFA-PLUGIN] deserializePlugin: Invalid serial data, using default values\n");
        return new DeformableAttentionAggrPlugin();  // 使用默认值
    }
    
    const char* d = static_cast<const char*>(serialData);
    int32_t batch, numAnchors, numEmbeds;
    readFromBuffer(d, batch);
    readFromBuffer(d, numAnchors);
    readFromBuffer(d, numEmbeds);
    
    // 验证合理性
    if (batch <= 0 || batch > 100) batch = 1;
    if (numAnchors <= 0 || numAnchors > 10000) numAnchors = 900;
    if (numEmbeds <= 0 || numEmbeds > 10000) numEmbeds = 256;
    
    printf("[DFA-PLUGIN] deserializePlugin: Restored dimensions (batch=%d, anchors=%d, embeds=%d)\n",
           batch, numAnchors, numEmbeds);
    
    return new DeformableAttentionAggrPlugin(batch, numAnchors, numEmbeds);
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
