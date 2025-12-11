#include "SparseBox3DKeyPointsKernel.h"

#include <cuda_fp16.h>
#include <math.h>
#include <cstdint>

// CUDA device 函数：检查浮点数是否为有限值（非 NaN 非 Inf）
__device__ __forceinline__ bool isFinite(float x)
{
    return isfinite(x);
}

// 关键优化：使用位操作检查NaN，确保100%可靠
__device__ __forceinline__ bool isNaN_strict(float x)
{
    const unsigned int bits = __float_as_uint(x);
    const unsigned int exp_mask = 0x7F800000;
    const unsigned int mantissa_mask = 0x007FFFFF;
    if ((bits & exp_mask) == exp_mask && (bits & mantissa_mask) != 0) {
        return true;
    }
    return false;
}

// 关键优化：使用位操作检查Inf，确保100%可靠
__device__ __forceinline__ bool isInf_strict(float x)
{
    const unsigned int bits = __float_as_uint(x);
    const unsigned int exp_mask = 0x7F800000;
    const unsigned int mantissa_mask = 0x007FFFFF;
    if ((bits & exp_mask) == exp_mask && (bits & mantissa_mask) == 0) {
        return true;
    }
    return false;
}

namespace sparse4d
{
template <typename T>
__device__ __forceinline__ float toFloat(T x)
{
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float toFloat<__half>(__half x)
{
    const unsigned short bits = __half_as_ushort(x);
    const unsigned short exp_mask = 0x7C00;
    const unsigned short mantissa_mask = 0x03FF;
    if ((bits & exp_mask) == exp_mask && (bits & mantissa_mask) != 0) {
        return 0.0f;
    }
    if ((bits & exp_mask) == exp_mask && (bits & mantissa_mask) == 0) {
        return 0.0f;
    }
    float result = __half2float(x);
    if (!isFinite(result)) return 0.0f;
    
    const float max_input_safe = 1e4f;
    const float min_input_safe = -1e4f;
    if (result > max_input_safe || result < min_input_safe) {
        result = fmaxf(fminf(result, max_input_safe), min_input_safe);
        if (!isFinite(result)) return 0.0f;
    }
    return result;
}

template <>
__device__ __forceinline__ float toFloat<int8_t>(int8_t x)
{
    return static_cast<float>(x);
}


template <typename AnchorT, typename FeatureT>
__global__ void sparseBox3DKeyPointsKernel(
    const SparseBox3DKeyPointsKernelParams params)
{
    // Flattened index: processes one specific point per thread
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t total_points = params.batch * params.numAnchor * params.numPts;
    
    if (idx >= total_points)
    {
        return;
    }
    
    if (params.anchor == nullptr || params.output == nullptr) {
        return;
    }

    // Decode indices
    // idx = ((b * numAnchor + n) * numPts + p)
    const int32_t p = idx % params.numPts;
    const int32_t temp = idx / params.numPts;
    const int32_t n = temp % params.numAnchor;
    const int32_t b = temp / params.numAnchor;

    const AnchorT* anchor = reinterpret_cast<const AnchorT*>(params.anchor);
    const FeatureT* inst = reinterpret_cast<const FeatureT*>(params.instanceFeature);
    // Output pointer (float*)
    const int32_t outStrideBytes = params.numAnchor * params.numPts * 3 * sizeof(float);
    float* outPtrFloat = reinterpret_cast<float*>(static_cast<char*>(params.output) + b * outStrideBytes) + n * params.numPts * 3;

    const int32_t anchorStride = params.numAnchor * 11;
    const int32_t instStride = params.numAnchor * params.embedDims;

    const AnchorT* anchorPtr = anchor + b * anchorStride + n * 11;
    const FeatureT* instPtr = (inst && params.numLearnablePts > 0)
        ? inst + b * instStride + n * params.embedDims
        : nullptr;
    
    // Load Anchor (Repeated load, but L1 cache handles it)
    float centerX_raw = toFloat(anchorPtr[0]);
    float centerY_raw = toFloat(anchorPtr[1]);
    float centerZ_raw = toFloat(anchorPtr[2]);
    float log_size_x_raw = toFloat(anchorPtr[3]);
    float log_size_y_raw = toFloat(anchorPtr[4]);
    float log_size_z_raw = toFloat(anchorPtr[5]);
    float sinYaw_raw = toFloat(anchorPtr[6]);
    float cosYaw_raw = toFloat(anchorPtr[7]);
    
    // Safety checks and clamps
    const float max_center_input = 1e4f;
    const float min_center_input = -1e4f;
    
    auto safe_val = [&](float val, float min_v, float max_v, float default_v) {
        if (!isNaN_strict(val) && !isInf_strict(val) && isFinite(val)) {
            val = fmaxf(fminf(val, max_v), min_v);
            if (isNaN_strict(val) || isInf_strict(val) || !isfinite(val)) return default_v;
            return val;
        }
        return default_v;
    };

    float centerX = safe_val(centerX_raw, min_center_input, max_center_input, 0.0f);
    float centerY = safe_val(centerY_raw, min_center_input, max_center_input, 0.0f);
    float centerZ = safe_val(centerZ_raw, min_center_input, max_center_input, 0.0f);

    const float max_log_size = 11.0f;
    const float min_log_size = -11.0f;
    float log_size_x = safe_val(log_size_x_raw, min_log_size, max_log_size, 0.0f);
    float log_size_y = safe_val(log_size_y_raw, min_log_size, max_log_size, 0.0f);
    float log_size_z = safe_val(log_size_z_raw, min_log_size, max_log_size, 0.0f);
    
    float sizeX = expf(log_size_x);
    float sizeY = expf(log_size_y);
    float sizeZ = expf(log_size_z);
    
    const float max_size_safe = 50000.0f;
    const float min_size_safe = 1e-6f;
    sizeX = safe_val(sizeX, min_size_safe, max_size_safe, 1.0f);
    sizeY = safe_val(sizeY, min_size_safe, max_size_safe, 1.0f);
    sizeZ = safe_val(sizeZ, min_size_safe, max_size_safe, 1.0f);

    float sinYaw = safe_val(sinYaw_raw, -10.0f, 10.0f, 0.0f); 
    float cosYaw = safe_val(cosYaw_raw, -10.0f, 10.0f, 1.0f);

    const int32_t fixedPts = params.numPts - params.numLearnablePts;

    // Process ONLY the current point 'p'
    float localX = 0.f;
    float localY = 0.f;
    float localZ = 0.f;
    
    if (!isFinite(sizeX)) sizeX = 1.0f;
    if (!isFinite(sizeY)) sizeY = 1.0f;
    if (!isFinite(sizeZ)) sizeZ = 1.0f;

    if (p < fixedPts)
    {
        // Fixed points
        const int offset = p * 3;
        float fx = params.fixScale[offset + 0];
        float fy = params.fixScale[offset + 1];
        float fz = params.fixScale[offset + 2];
        
        fx = safe_val(fx, -10.0f, 10.0f, 0.0f);
        fy = safe_val(fy, -10.0f, 10.0f, 0.0f);
        fz = safe_val(fz, -10.0f, 10.0f, 0.0f);
        
        localX = fx * sizeX;
        localY = fy * sizeY;
        localZ = fz * sizeZ;
    }
    else if (instPtr != nullptr)
    {
        // Learnable points
        const int learnIdx = p - fixedPts;
        const int rowBase = learnIdx * 3;
        const float* rowWeight = params.fcWeight + rowBase * params.embedDims;
        const float* rowBias = params.fcBias + rowBase;

        float accum[3];
        accum[0] = safe_val(rowBias[0], -10.0f, 10.0f, 0.0f);
        accum[1] = safe_val(rowBias[1], -10.0f, 10.0f, 0.0f);
        accum[2] = safe_val(rowBias[2], -10.0f, 10.0f, 0.0f);
        
        float featureScale = params.featureScale;

        for (int k = 0; k < params.embedDims; ++k)
        {
            // Dequantize if INT8, else just load
            float val = toFloat(instPtr[k]);
            if (params.useInt8) {
                val *= featureScale;
            }
            
            val = safe_val(val, -50.0f, 50.0f, 0.0f);
            if (val == 0.0f) continue;
            
            float w0 = rowWeight[0 * params.embedDims + k];
            float w1 = rowWeight[1 * params.embedDims + k];
            float w2 = rowWeight[2 * params.embedDims + k];
            
            // Accumulate
            accum[0] = fmaf(val, w0, accum[0]);
            accum[1] = fmaf(val, w1, accum[1]);
            accum[2] = fmaf(val, w2, accum[2]);
        }
        
        // Sigmoid centered
        auto sigmoid_centered = [&](float x) {
            x = safe_val(x, -70.0f, 70.0f, 0.0f);
            if (x == 0.0f) return 0.0f;
            float s = 1.0f / (1.0f + expf(-x));
            return s - 0.5f;
        };
        
        localX = sigmoid_centered(accum[0]) * sizeX;
        localY = sigmoid_centered(accum[1]) * sizeY;
        localZ = sigmoid_centered(accum[2]) * sizeZ;
    }
    
    localX = safe_val(localX, -1e5f, 1e5f, 0.0f);
    localY = safe_val(localY, -1e5f, 1e5f, 0.0f);
    localZ = safe_val(localZ, -1e5f, 1e5f, 0.0f);
    
    // Rotation
    float rotX = cosYaw * localX - sinYaw * localY;
    float rotY = sinYaw * localX + cosYaw * localY;
    
    // Translation
    float finalX = rotX + centerX;
    float finalY = rotY + centerY;
    float finalZ = localZ + centerZ;
    
    // Write output
    finalX = safe_val(finalX, -1e5f, 1e5f, centerX);
    finalY = safe_val(finalY, -1e5f, 1e5f, centerY);
    finalZ = safe_val(finalZ, -1e5f, 1e5f, centerZ);
    
    const int out_offset = p * 3;
    outPtrFloat[out_offset + 0] = finalX;
    outPtrFloat[out_offset + 1] = finalY;
    outPtrFloat[out_offset + 2] = finalZ;
}

int launchSparseBox3DKeyPointsKernel(
    const SparseBox3DKeyPointsKernelParams& params,
    cudaStream_t stream)
{
    // Total threads = batch * numAnchor * numPts
    // This significantly increases parallelism (e.g. 900 -> 11700)
    const int32_t total = params.batch * params.numAnchor * params.numPts;
    if (total <= 0) return 1;
    
    const int32_t threads = 256;
    const int32_t blocks = (total + threads - 1) / threads;

    if (params.useFP16) // Anchor is FP16
    {
        if (params.useInt8) // Feature is Int8
        {
             sparseBox3DKeyPointsKernel<__half, int8_t><<<blocks, threads, 0, stream>>>(params);
        }
        else // Feature is FP16 (matches anchor)
        {
             sparseBox3DKeyPointsKernel<__half, __half><<<blocks, threads, 0, stream>>>(params);
        }
    }
    else // Anchor is FP32
    {
        if (params.useInt8) // Feature is Int8
        {
             sparseBox3DKeyPointsKernel<float, int8_t><<<blocks, threads, 0, stream>>>(params);
        }
        else // Feature is FP32
        {
             sparseBox3DKeyPointsKernel<float, float><<<blocks, threads, 0, stream>>>(params);
        }
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1;
    
    return 0;
}
} // namespace sparse4d
