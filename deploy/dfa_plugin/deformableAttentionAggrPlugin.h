// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef DEPLOY_DFA_PLUGIN_DEFORMABLEATTENTIONAGGRPLUGIN_H
#define DEPLOY_DFA_PLUGIN_DEFORMABLEATTENTIONAGGRPLUGIN_H

#include <string>
#include <vector>

#include <NvInfer.h>

#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"

namespace custom
{
static const char* PLUGIN_NAME{"DeformableAttentionAggrPlugin"};
static const char* PLUGIN_VERSION{"1"};

///@brief First define a Plugin Class: Including Implementation of DeformableAttentionAggrPlugin.
class DeformableAttentionAggrPlugin : public nvinfer1::IPluginV2DynamicExt
{
  public:
    // 默认构造函数：使用保守的默认值
    DeformableAttentionAggrPlugin() = default;
    
    // 带参数的构造函数：在插件创建时传入维度信息
    DeformableAttentionAggrPlugin(int32_t batch, int32_t numAnchors, int32_t numEmbeds)
        : mBatch_(batch), mNumAnchors_(numAnchors), mNumEmbeds_(numEmbeds) {}
    
    ~DeformableAttentionAggrPlugin() = default;

    /// @brief PART1: Custom Plugin Class: DeformableAttentionAggrPlugin -> nvinfer1::IPluginV2DynamicExt Methods
    /*
     * clone()
     * getOutputDimensions()
     * supportsFormatCombination()
     * configurePlugin()
     * getWorkspaceSize()
     * enqueue()
     * attachToContext()
     * detachFromContext()
     */
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex,
                                            const nvinfer1::DimsExprs* inputs,
                                            int32_t nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos,
                                   const nvinfer1::PluginTensorDesc* inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                         int32_t nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out,
                         int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                            int32_t nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs,
                            int32_t nbOutputs) const noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override;
    void attachToContext(cudnnContext* contextCudnn,
                         cublasContext* contextCublas,
                         nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;

    /// @brief PART2: Custom Plugin Class: DeformableAttentionAggrPlugin -> nvinfer1::IPluginV2Ext Methods
    /*
     * getOutputDataType()
     */
    nvinfer1::DataType getOutputDataType(int32_t index,
                                         nvinfer1::DataType const* inputTypes,
                                         int32_t nbInputs) const noexcept override;

    /// @brief PART3: Custom Plugin Class: DeformableAttentionAggrPlugin -> nvinfer1::IPluginV2 Methods
    /*
     * getPluginType()
     * getPluginVersion()
     * getNbOutputs()
     * initialize()
     * getSerializationSize()
     * serialize()
     * destroy()
     * terminate()
     * setPluginNamespace()
     * getPluginNamespace()
     */
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void terminate() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

  private:
    std::string mNamespace_;
    
    // 序列化的维度参数：在插件创建时确定，避免运行时访问未初始化的内存
    int32_t mBatch_ = 1;           // batch size
    int32_t mNumAnchors_ = 900;    // number of anchors (从inputs[3]获取)
    int32_t mNumEmbeds_ = 256;     // embedding dimension (从inputs[0]获取)
    
    // 运行时缓存：在enqueue中安全获取后缓存，用于后续调用
    mutable int32_t mCachedBatch_ = -1;
    mutable int32_t mCachedNumAnchors_ = -1;
    mutable int32_t mCachedNumEmbeds_ = -1;
};

/// @brief Second define a PluginCreator Class.
/// @brief PART4 Custom Pluginv1 Creator: DeformableAttentionAggrPluginCreator
/*
 * DeformableAttentionAggrPluginCreator()
 * ~DeformableAttentionAggrPluginCreator()
 * getPluginName()
 * getPluginVersion()
 * getFieldNames()
 * createPlugin()
 * deserializePlugin()
 * setPluginNamespace()
 * getPluginNamespace()
 */
class DeformableAttentionAggrPluginCreator : public nvinfer1::IPluginCreator
{
  public:
    DeformableAttentionAggrPluginCreator();
    ~DeformableAttentionAggrPluginCreator() = default;
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                           const void* serialData,
                                           size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

  private:
    nvinfer1::PluginFieldCollection mFC_;         // 插件字段集合，用于存储插件的配置信息
    std::vector<nvinfer1::PluginField> mAttrs_;   // 存储插件属性的向量
    std::string mNamespace_;
};

}  // namespace custom

#endif  // DEPLOY_DFA_PLUGIN_DEFORMABLEATTENTIONAGGRPLUGIN_H
