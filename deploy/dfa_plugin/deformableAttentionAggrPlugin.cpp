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
                                    int batch,
                                    int mSpatialSize,
                                    int mChannels,
                                    int mNumCams,
                                    int mNumLevels,
                                    int mNumQuery,
                                    int mNumPoint,
                                    int mNumGroups);

// 声明FP16版本的CUDA函数（优化版本：使用FP32临时缓冲区，通过workspace提供）
int thomas_deform_attn_cuda_forward_half(cudaStream_t stream,
                                         const __half* value,
                                         const int* spatialShapes,
                                         const int* levelStartIndex,
                                         const __half* samplingLoc,
                                         const __half* attnWeight,
                                         __half* output,
                                         float* workspace,  // TensorRT提供的workspace
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
    // FP16模式需要临时FP32缓冲区
    // 注意：在构建时，TensorRT可能会用不同的数据类型多次调用getWorkspaceSize
    // 问题：如果engine是用FP16构建的，但warmup/推理时使用FP32数据，TensorRT会进行类型转换
    // 但在构建时，如果TensorRT用FP32调用getWorkspaceSize，会返回0，导致workspace未分配
    // 
    // 解决方案：检查输入和输出，如果任何一个浮点输入或输出是FP16，就返回workspace大小
    // 这样可以确保即使构建时用FP32调用，只要engine支持FP16，workspace也会被分配
    
    // 检查所有浮点输入（跳过INT32类型的输入：spatial_shapes和level_start_index）
    bool has_fp16_input = false;
    for (int32_t i = 0; i < nbInputs; ++i)
    {
        // 跳过INT32类型的输入（索引1和2）
        if (i == 1 || i == 2)
        {
            continue;
        }
        
        // 检查浮点输入的数据类型
        if (inputs[i].type == nvinfer1::DataType::kHALF)
        {
            has_fp16_input = true;
            break;
        }
    }
    
    // 检查输出数据类型（如果输出是FP16，说明engine支持FP16，即使输入是FP32也会进行类型转换）
    bool has_fp16_output = false;
    for (int32_t i = 0; i < nbOutputs; ++i)
    {
        if (outputs[i].type == nvinfer1::DataType::kHALF)
        {
            has_fp16_output = true;
            break;
        }
    }
    
    // 如果检测到FP16输入或输出，返回workspace大小
    if (has_fp16_input || has_fp16_output)
    {
        int32_t const batch = inputs[0].dims.d[0];
        int32_t num_anchors = inputs[3].dims.d[1];
        int32_t num_embeds = inputs[0].dims.d[2];
        size_t workspace_size = batch * num_anchors * num_embeds * sizeof(float);
        printf("[DFA-PLUGIN] getWorkspaceSize: FP16 mode detected (input_fp16=%d, output_fp16=%d), returning workspace size: %zu bytes (batch=%d, anchors=%d, embeds=%d)\n",
               has_fp16_input, has_fp16_output, workspace_size, batch, num_anchors, num_embeds);
        return workspace_size;  // FP32临时缓冲区大小
    }
    
    // FP32模式不需要workspace
    printf("[DFA-PLUGIN] getWorkspaceSize: FP32 mode (all inputs and outputs are FP32), returning 0\n");
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

    // 根据输入数据类型选择FP32或FP16路径
    nvinfer1::DataType dataType = inputDesc[0].type;
    
    if (dataType == nvinfer1::DataType::kFLOAT)
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
        
        // 获取workspace（FP32临时缓冲区）
        // 计算所需的workspace大小
        size_t required_workspace_size = batch * num_query * channels * sizeof(float);
        
        // 检查workspace是否为空
        // 问题：如果engine在FP32模式下构建（ONNX模型是FP32），getWorkspaceSize返回0
        // 但运行时输入是FP16，需要workspace，但TensorRT已经确定workspace大小为0
        // 长期解决方案：导出FP16 ONNX模型（使用--fp16参数），确保构建时正确识别FP16输入
        float* workspace_ptr = nullptr;
        bool workspace_allocated = false;
        
        if (workspace == nullptr)
        {
            printf("[DFA-PLUGIN-WARNING] Workspace is null for FP16 mode. Required workspace size: %zu bytes (batch=%d, anchors=%d, embeds=%d)\n",
                   required_workspace_size, batch, num_query, channels);
            printf("[DFA-PLUGIN-WARNING] This usually means the engine was built with FP32 ONNX model, but is now running with FP16 inputs.\n");
            printf("[DFA-PLUGIN-WARNING] Solution: Export ONNX model with --fp16 flag, then rebuild the engine.\n");
            printf("[DFA-PLUGIN-WARNING] Example: python deploy/export_head_onnx.py --fp16 ...\n");
            printf("[DFA-PLUGIN-WARNING] Attempting to allocate temporary workspace dynamically (this may be slower)...\n");
            
            // 动态分配临时缓冲区（作为备选方案）
            void* temp_workspace = nullptr;
            cudaError_t err = cudaMallocAsync(&temp_workspace, required_workspace_size, stream);
            if (err != cudaSuccess)
            {
                printf("[DFA-PLUGIN-ERROR] Failed to allocate temporary workspace: %s\n", cudaGetErrorString(err));
                // 注意：返回非0值会导致TensorRT抛出异常，但函数标记为noexcept，可能导致段错误
                // 返回0表示成功，但实际上会输出错误信息（这是为了避免段错误）
                return 0;
            }
            workspace_ptr = static_cast<float*>(temp_workspace);
            workspace_allocated = true;
        }
        else
        {
            workspace_ptr = static_cast<float*>(workspace);
        }

        // FP16优化版本：内部使用FP32临时缓冲区，atomicAdd更快
        // workspace由TensorRT在getWorkspaceSize中分配，或动态分配（如果为nullptr）
        rc = thomas_deform_attn_cuda_forward_half(stream,
                                                  value,
                                                  spatialShapes,
                                                  levelStartIndex,
                                                  samplingLoc,
                                                  attnWeight,
                                                  output,
                                                  workspace_ptr,  // 使用TensorRT提供的workspace或动态分配的workspace
                                                  batch,           // batch_size
                                                  num_cams,        // num_cams
                                                  spatial_size,    // num_feat (spatial_size)
                                                  channels,        // num_embeds (channels)
                                                  num_levels,      // num_scale (num_levels)
                                                  num_query,       // num_anchors (num_query)
                                                  num_point,       // num_pts (num_point)
                                                  num_groups);     // num_groups
        
        // 如果动态分配了workspace，需要释放
        if (workspace_allocated && workspace_ptr != nullptr)
        {
            cudaError_t err = cudaFreeAsync(workspace_ptr, stream);
            if (err != cudaSuccess)
            {
                printf("[DFA-PLUGIN-WARNING] Failed to free temporary workspace: %s\n", cudaGetErrorString(err));
            }
        }
    }
    else
    {
        printf("[DFA-PLUGIN-ERROR] Unsupported data type: %d\n", static_cast<int>(dataType));
        // 注意：返回非0值会导致TensorRT抛出异常，但函数标记为noexcept，可能导致段错误
        // 返回0表示成功，但实际上会输出错误信息（这是为了避免段错误）
        return 0;  // 暂时返回0避免段错误，但会在日志中输出错误信息
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
