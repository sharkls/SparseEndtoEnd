#include "SparseBox3DKeyPointsPlugin.h"
#include "SparseBox3DKeyPointsKernel.h"

#include "NvInferPlugin.h"
#include <cstring>
#include <stdexcept>

using namespace nvinfer1;

namespace sparse4d
{
namespace
{
constexpr char PLUGIN_NAME[] = "SparseBox3DKeyPointsPlugin";
constexpr char PLUGIN_VERSION[] = "1";

inline size_t vectorBytes(const std::vector<float>& vec)
{
    return sizeof(int32_t) + vec.size() * sizeof(float);
}

inline void serializeVector(char*& dst, const std::vector<float>& vec)
{
    int32_t size = static_cast<int32_t>(vec.size());
    std::memcpy(dst, &size, sizeof(int32_t));
    dst += sizeof(int32_t);
    if (size > 0)
    {
        std::memcpy(dst, vec.data(), size * sizeof(float));
        dst += size * sizeof(float);
    }
}

inline void deserializeVector(const char*& src, std::vector<float>& vec)
{
    int32_t size = 0;
    std::memcpy(&size, src, sizeof(int32_t));
    src += sizeof(int32_t);
    vec.resize(size);
    if (size > 0)
    {
        std::memcpy(vec.data(), src, size * sizeof(float));
        src += size * sizeof(float);
    }
}
} // namespace

SparseBox3DKeyPointsPlugin::SparseBox3DKeyPointsPlugin(
    const SparseBox3DKeyPointsParams& params)
    : mParams(params)
{
    allocateDeviceBuffers();
}

SparseBox3DKeyPointsPlugin::SparseBox3DKeyPointsPlugin(
    const void* data,
    size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;

    std::memcpy(&mParams.embedDims, a, sizeof(int32_t));
    a += sizeof(int32_t);
    std::memcpy(&mParams.numPts, a, sizeof(int32_t));
    a += sizeof(int32_t);
    std::memcpy(&mParams.numLearnablePts, a, sizeof(int32_t));
    a += sizeof(int32_t);

    deserializeVector(a, mParams.fixScale);
    deserializeVector(a, mParams.fcWeight);
    deserializeVector(a, mParams.fcBias);
    
    // Deserialize input scale if available
    if (static_cast<size_t>(a - d) < length) {
        std::memcpy(&mParams.inputScale, a, sizeof(float));
        a += sizeof(float);
    } else {
        mParams.inputScale = 1.0f;
    }

    if (static_cast<size_t>(a - d) > length) // Relaxed check to allow backward compatibility if length > read
    {
       // If strict check is needed: != length
    }
    allocateDeviceBuffers();
}

SparseBox3DKeyPointsPlugin::~SparseBox3DKeyPointsPlugin()
{
    freeDeviceBuffers();
}

nvinfer1::DimsExprs SparseBox3DKeyPointsPlugin::getOutputDimensions(
    int,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& builder) noexcept
{
    DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[0].d[1];
    output.d[2] = builder.constant(mParams.numPts);
    output.d[3] = builder.constant(3);
    return output;
}

bool SparseBox3DKeyPointsPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) noexcept
{
    const auto& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    // Output (pos == nbInputs) must be FLOAT (forcing FP32 output)
    if (pos == nbInputs)
    {
        return desc.type == DataType::kFLOAT;
    }

    // Input 0 (Anchor) can be Float or Half (usually regressed values, not Int8)
    if (pos == 0)
    {
        return desc.type == DataType::kFLOAT || desc.type == DataType::kHALF;
    }
    
    // Input 1 (Feature)
    if (pos == 1) {
        // Can be Float, Half, or Int8
        if (inOut[0].type == DataType::kFLOAT) {
             return desc.type == DataType::kFLOAT || desc.type == DataType::kINT8;
        }
        if (inOut[0].type == DataType::kHALF) {
             return desc.type == DataType::kHALF || desc.type == DataType::kINT8;
        }
        return false;
    }

    return false;
}

void SparseBox3DKeyPointsPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int nbOutputs) noexcept
{
    // Capture scale if input 1 (feature) is INT8
    if (nbInputs > 1 && inputs[1].desc.type == DataType::kINT8) {
        mParams.inputScale = inputs[1].desc.scale;
    } else {
        mParams.inputScale = 1.0f;
    }
}

int SparseBox3DKeyPointsPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc*,
    const void* const* inputs,
    void* const* outputs,
    void*,
    cudaStream_t stream) noexcept
{
    // 参数验证
    if (inputDesc[0].dims.nbDims < 2)
    {
        return 1;
    }
    
    const int32_t batch = inputDesc[0].dims.d[0];
    const int32_t numAnchor = inputDesc[0].dims.d[1];
    
    if (batch <= 0 || numAnchor <= 0)
    {
        return 1;
    }
    
    // 验证参数完整性
    if (mParams.fixScale.size() < static_cast<size_t>((mParams.numPts - mParams.numLearnablePts) * 3))
    {
        return 1;
    }
    
    if (mParams.numLearnablePts > 0)
    {
        if (mParams.fcWeight.size() < static_cast<size_t>(mParams.numLearnablePts * 3 * mParams.embedDims))
        {
            return 1;
        }
        if (mParams.fcBias.size() < static_cast<size_t>(mParams.numLearnablePts * 3))
        {
            return 1;
        }
    }
    
    const bool useFP16 = inputDesc[0].type == DataType::kHALF;
    const bool useInt8 = (mParams.numLearnablePts > 0) && (inputDesc[1].type == DataType::kINT8);
    const bool outputFP32 = true; // Always output FP32

    SparseBox3DKeyPointsKernelParams params{};
    params.batch = batch;
    params.numAnchor = numAnchor;
    params.embedDims = mParams.embedDims;
    params.numPts = mParams.numPts;
    params.numLearnablePts = mParams.numLearnablePts;
    params.anchor = inputs[0];
    params.instanceFeature = (mParams.numLearnablePts > 0) ? inputs[1] : nullptr;
    params.output = outputs[0];
    params.fixScale = mDeviceFixScale;
    params.fcWeight = mDeviceFcWeight;
    params.fcBias = mDeviceFcBias;
    params.useFP16 = useFP16;
    params.useInt8 = useInt8;
    params.featureScale = mParams.inputScale;
    params.outputFP32 = outputFP32;

    return launchSparseBox3DKeyPointsKernel(params, stream);
}

size_t SparseBox3DKeyPointsPlugin::getSerializationSize() const noexcept
{
    return sizeof(int32_t) * 3
        + vectorBytes(mParams.fixScale)
        + vectorBytes(mParams.fcWeight)
        + vectorBytes(mParams.fcBias)
        + sizeof(float); // inputScale
}

void SparseBox3DKeyPointsPlugin::serialize(void* buffer) const noexcept
{
    char* dst = reinterpret_cast<char*>(buffer);
    std::memcpy(dst, &mParams.embedDims, sizeof(int32_t));
    dst += sizeof(int32_t);
    std::memcpy(dst, &mParams.numPts, sizeof(int32_t));
    dst += sizeof(int32_t);
    std::memcpy(dst, &mParams.numLearnablePts, sizeof(int32_t));
    dst += sizeof(int32_t);

    serializeVector(dst, mParams.fixScale);
    serializeVector(dst, mParams.fcWeight);
    serializeVector(dst, mParams.fcBias);
    
    std::memcpy(dst, &mParams.inputScale, sizeof(float));
    dst += sizeof(float);
}

nvinfer1::IPluginV2DynamicExt* SparseBox3DKeyPointsPlugin::clone() const noexcept
{
    try
    {
        return new SparseBox3DKeyPointsPlugin(mParams);
    }
    catch (...)
    {
        return nullptr;
    }
}

nvinfer1::DataType SparseBox3DKeyPointsPlugin::getOutputDataType(
    int,
    const nvinfer1::DataType* inputTypes,
    int) const noexcept
{
    return DataType::kFLOAT;
}

const char* SparseBox3DKeyPointsPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* SparseBox3DKeyPointsPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

void SparseBox3DKeyPointsPlugin::allocateDeviceBuffers()
{
    auto allocate = [](const std::vector<float>& src, float*& dst) {
        if (src.empty())
        {
            dst = nullptr;
            return cudaSuccess;
        }
        const size_t bytes = src.size() * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dst), bytes);
        if (err != cudaSuccess)
        {
            dst = nullptr;
            return err;
        }
        err = cudaMemcpy(dst, src.data(), bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cudaFree(dst);
            dst = nullptr;
        }
        return err;
    };

    if (allocate(mParams.fixScale, mDeviceFixScale) != cudaSuccess
        || allocate(mParams.fcWeight, mDeviceFcWeight) != cudaSuccess
        || allocate(mParams.fcBias, mDeviceFcBias) != cudaSuccess)
    {
        freeDeviceBuffers();
        throw std::runtime_error("SparseBox3DKeyPointsPlugin: failed to allocate device memory.");
    }
}

void SparseBox3DKeyPointsPlugin::freeDeviceBuffers() noexcept
{
    if (mDeviceFixScale)
    {
        cudaFree(mDeviceFixScale);
        mDeviceFixScale = nullptr;
    }
    if (mDeviceFcWeight)
    {
        cudaFree(mDeviceFcWeight);
        mDeviceFcWeight = nullptr;
    }
    if (mDeviceFcBias)
    {
        cudaFree(mDeviceFcBias);
        mDeviceFcBias = nullptr;
    }
}

SparseBox3DKeyPointsPluginCreator::SparseBox3DKeyPointsPluginCreator()
{
    mFields.reserve(6);
    mFields.emplace_back("embed_dims", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("num_pts", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("num_learnable_pts", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("fix_scale", nullptr, PluginFieldType::kFLOAT32, 0);
    mFields.emplace_back("fc_weight", nullptr, PluginFieldType::kFLOAT32, 0);
    mFields.emplace_back("fc_bias", nullptr, PluginFieldType::kFLOAT32, 0);
    mFieldCollection.nbFields = static_cast<int>(mFields.size());
    mFieldCollection.fields = mFields.data();
}

const char* SparseBox3DKeyPointsPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* SparseBox3DKeyPointsPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* SparseBox3DKeyPointsPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

nvinfer1::IPluginV2* SparseBox3DKeyPointsPluginCreator::createPlugin(
    const char*,
    const nvinfer1::PluginFieldCollection* fc) noexcept
{
    SparseBox3DKeyPointsParams params{};
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const PluginField& field = fc->fields[i];
        if (!std::strcmp(field.name, "embed_dims"))
        {
            params.embedDims = *static_cast<const int32_t*>(field.data);
        }
        else if (!std::strcmp(field.name, "num_pts"))
        {
            params.numPts = *static_cast<const int32_t*>(field.data);
        }
        else if (!std::strcmp(field.name, "num_learnable_pts"))
        {
            params.numLearnablePts = *static_cast<const int32_t*>(field.data);
        }
        else if (!std::strcmp(field.name, "fix_scale"))
        {
            params.fixScale.assign(
                static_cast<const float*>(field.data),
                static_cast<const float*>(field.data) + field.length);
        }
        else if (!std::strcmp(field.name, "fc_weight"))
        {
            params.fcWeight.assign(
                static_cast<const float*>(field.data),
                static_cast<const float*>(field.data) + field.length);
        }
        else if (!std::strcmp(field.name, "fc_bias"))
        {
            params.fcBias.assign(
                static_cast<const float*>(field.data),
                static_cast<const float*>(field.data) + field.length);
        }
    }
    return new SparseBox3DKeyPointsPlugin(params);
}

nvinfer1::IPluginV2* SparseBox3DKeyPointsPluginCreator::deserializePlugin(
    const char*,
    const void* serialData,
    size_t serialLength) noexcept
{
    try
    {
        return new SparseBox3DKeyPointsPlugin(serialData, serialLength);
    }
    catch (...)
    {
        return nullptr;
    }
}

REGISTER_TENSORRT_PLUGIN(SparseBox3DKeyPointsPluginCreator);
} // namespace sparse4d
