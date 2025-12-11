#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace sparse4d
{
struct SparseBox3DKeyPointsKernelParams
{
    int32_t batch;
    int32_t numAnchor;
    int32_t embedDims;
    int32_t numPts;
    int32_t numLearnablePts;
    const void* anchor;           // FP32 or FP16
    const void* instanceFeature;  // FP32, FP16, or INT8
    void* output;                 // FP32
    const float* fixScale;
    const float* fcWeight;
    const float* fcBias;
    bool useFP16;                 // true if anchor/feature is FP16
    bool useInt8;                 // true if feature is INT8
    float featureScale;           // Scale factor for INT8 feature
    bool outputFP32;              // true if output should be FP32 (always true now)
};

int launchSparseBox3DKeyPointsKernel(
    const SparseBox3DKeyPointsKernelParams& params,
    cudaStream_t stream);
} // namespace sparse4d
