// Copyright (c) 2024 SparseEnd2End. All rights reserved.
#pragma once

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

namespace sparse4d
{
struct SparseBox3DKeyPointsParams
{
    int32_t embedDims;
    int32_t numPts;
    int32_t numLearnablePts;
    std::vector<float> fixScale;
    std::vector<float> fcWeight;
    std::vector<float> fcBias;
    float inputScale = 1.0f; // Scale for INT8 input instance feature
};

class SparseBox3DKeyPointsPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    explicit SparseBox3DKeyPointsPlugin(const SparseBox3DKeyPointsParams& params);
    SparseBox3DKeyPointsPlugin(const void* data, size_t length);
    ~SparseBox3DKeyPointsPlugin() override;

    // IPluginV2DynamicExt
    int getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& builder) noexcept override;

    bool supportsFormatCombination(
        int pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs,
        int nbOutputs) noexcept override;

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* inputs,
        int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* outputs,
        int nbOutputs) noexcept override;

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override
    {
        return 0;
    }

    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    // IPluginV2Ext / IPluginV2
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    nvinfer1::DataType getOutputDataType(
        int index,
        const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept override;

private:
    SparseBox3DKeyPointsParams mParams;
    std::string mNamespace;
    float* mDeviceFixScale{nullptr};
    float* mDeviceFcWeight{nullptr};
    float* mDeviceFcBias{nullptr};

    void allocateDeviceBuffers();
    void freeDeviceBuffers() noexcept;
};

class SparseBox3DKeyPointsPluginCreator : public nvinfer1::IPluginCreator
{
public:
    SparseBox3DKeyPointsPluginCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength) noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFieldCollection{};
    std::vector<nvinfer1::PluginField> mFields;
};
} // namespace sparse4d
