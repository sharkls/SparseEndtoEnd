#pragma once

#include <cuda_runtime_api.h>

namespace sparse4d
{
struct SparseBox3DKeyPointsKernelParams
{
    int32_t batch;
    int32_t numAnchor;
    int32_t embedDims;
    int32_t numPts;
    int32_t numLearnablePts;
    const void* anchor;          // [B, N, 11]
    const void* instanceFeature; // [B, N, embedDims] or nullptr
    void* output;                // [B, N, numPts, 3]
    const float* fixScale;       // numPts * 3
    const float* fcWeight;       // numLearnablePts*3 x embedDims
    const float* fcBias;         // numLearnablePts*3
    bool useFP16;
    bool outputFP32;            // 新增：输出是否为 FP32（当输入是 FP16 但输出是 FP32 时）
};

int launchSparseBox3DKeyPointsKernel(
    const SparseBox3DKeyPointsKernelParams& params,
    cudaStream_t stream);
} // namespace sparse4d


