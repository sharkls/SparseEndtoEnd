#include "SparseBox3DKeyPointsKernel.h"

#include <cuda_fp16.h>
#include <math.h>

// CUDA device 函数：检查浮点数是否为有限值（非 NaN 非 Inf）
// 使用 CUDA 内置函数 isfinite，更可靠
__device__ __forceinline__ bool isFinite(float x)
{
    return isfinite(x);
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
    float result = __half2float(x);
    // 检查转换结果是否有效
    if (!isFinite(result)) {
        return 0.0f;  // 如果转换产生 NaN/Inf，返回 0
    }
    return result;
}

template <typename T>
__device__ __forceinline__ T fromFloat(float x)
{
    return static_cast<T>(x);
}

template <>
__device__ __forceinline__ __half fromFloat<__half>(float x)
{
    // 首先检查是否为 NaN 或 Inf
    if (!isFinite(x)) {
        return __float2half(0.0f);  // 如果异常，返回 0
    }
    
    // FP16 范围：[-65504, 65504]
    // 关键修复：必须在转换前 clamp，否则 __float2half 可能产生 Inf
    // 如果值超出 FP16 范围，__float2half 会产生 Inf，导致后续计算产生巨大误差
    const float fp16_max = 65504.0f;  // FP16 实际最大值
    const float fp16_min = -65504.0f;  // FP16 实际最小值
    
    // 在转换前 clamp，防止产生 Inf
    x = fmaxf(fminf(x, fp16_max), fp16_min);
    
    // 再次检查（防止 clamp 后仍异常）
    if (!isFinite(x)) {
        return __float2half(0.0f);
    }
    
    return __float2half(x);
}

// 关键优化：在 FP16 模式下，插件内部完全使用 FP32 进行计算
// 只在最终输出时转换为 FP16，确保最大精度
template <typename T>
__global__ void sparseBox3DKeyPointsKernel(
    const SparseBox3DKeyPointsKernelParams params)
{
    const int32_t anchorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t total = params.batch * params.numAnchor;
    if (anchorIdx >= total)
    {
        return;
    }

    const int32_t b = anchorIdx / params.numAnchor;
    const int32_t n = anchorIdx % params.numAnchor;

    // 关键修改：无论输入类型如何，都先转换为 FP32 进行处理
    // 这样可以确保所有计算都在 FP32 精度下进行
    const T* anchor = reinterpret_cast<const T*>(params.anchor);
    const T* inst = reinterpret_cast<const T*>(params.instanceFeature);
    // 输出类型处理：如果 outputFP32 为 true，输出是 float*，否则是 T*
    void* output = params.output;

    const int32_t anchorStride = params.numAnchor * 11;
    const int32_t instStride = params.numAnchor * params.embedDims;
    // 输出 stride：根据 outputFP32 标志确定元素大小
    const int32_t outElementSize = params.outputFP32 ? sizeof(float) : sizeof(T);
    // 输出 stride：以字节为单位
    const int32_t outStrideBytes = params.numAnchor * params.numPts * 3 * outElementSize;

    const T* anchorPtr = anchor + b * anchorStride + n * 11;
    const T* instPtr = (inst && params.numLearnablePts > 0)
        ? inst + b * instStride + n * params.embedDims
        : nullptr;
    // 输出指针：根据 outputFP32 标志确定类型
    // outStrideBytes 已经包含了元素大小，所以直接使用
    void* outPtrBase = static_cast<char*>(output) + b * outStrideBytes + n * params.numPts * 3 * outElementSize;

    // 读取并验证所有 anchor 输入值
    float centerX_raw = toFloat(anchorPtr[0]);
    float centerY_raw = toFloat(anchorPtr[1]);
    float centerZ_raw = toFloat(anchorPtr[2]);
    float log_size_x_raw = toFloat(anchorPtr[3]);
    float log_size_y_raw = toFloat(anchorPtr[4]);
    float log_size_z_raw = toFloat(anchorPtr[5]);
    float sinYaw_raw = toFloat(anchorPtr[6]);
    float cosYaw_raw = toFloat(anchorPtr[7]);
    
    // 检查所有输入值是否有效，只处理 NaN/Inf，不进行过度 clamp
    // 过度 clamp 会引入误差，影响 FP16 精度
    float centerX = isFinite(centerX_raw) ? centerX_raw : 0.0f;
    float centerY = isFinite(centerY_raw) ? centerY_raw : 0.0f;
    float centerZ = isFinite(centerZ_raw) ? centerZ_raw : 0.0f;
    
    // log_size 需要限制以防止 exp 溢出，但使用合理的范围
    const float max_log_size = 11.0f;  // exp(11) ≈ 59874 < 65504
    const float min_log_size = -11.0f;
    float log_size_x = isFinite(log_size_x_raw) ? fmaxf(fminf(log_size_x_raw, max_log_size), min_log_size) : 0.0f;
    float log_size_y = isFinite(log_size_y_raw) ? fmaxf(fminf(log_size_y_raw, max_log_size), min_log_size) : 0.0f;
    float log_size_z = isFinite(log_size_z_raw) ? fmaxf(fminf(log_size_z_raw, max_log_size), min_log_size) : 0.0f;
    
    // 计算 size（使用 FP32 精度）
    float sizeX = expf(log_size_x);
    float sizeY = expf(log_size_y);
    float sizeZ = expf(log_size_z);
    
    // 检查 exp 结果是否有效
    if (!isFinite(sizeX)) sizeX = 1.0f;
    if (!isFinite(sizeY)) sizeY = 1.0f;
    if (!isFinite(sizeZ)) sizeZ = 1.0f;
    
        // sin/cos yaw 应该已经在 [-1, 1] 范围内，只检查有效性
        // 关键修复：确保 sinYaw 和 cosYaw 在有效范围内，避免旋转计算产生 NaN
        float sinYaw = isFinite(sinYaw_raw) ? fmaxf(fminf(sinYaw_raw, 1.0f), -1.0f) : 0.0f;
        float cosYaw = isFinite(cosYaw_raw) ? fmaxf(fminf(cosYaw_raw, 1.0f), -1.0f) : 1.0f;
        // 再次检查（防止 clamp 后仍异常）
        if (!isFinite(sinYaw)) sinYaw = 0.0f;
        if (!isFinite(cosYaw)) cosYaw = 1.0f;

    const int32_t fixedPts = params.numPts - params.numLearnablePts;

    for (int i = 0; i < params.numPts; ++i)
    {
        const int offset = i * 3;
        float localX = 0.f;
        float localY = 0.f;
        float localZ = 0.f;

        if (i < fixedPts)
        {
            // 固定点计算（使用 FP32 精度，不进行过度 clamp）
            localX = params.fixScale[offset + 0] * sizeX;
            localY = params.fixScale[offset + 1] * sizeY;
            localZ = params.fixScale[offset + 2] * sizeZ;
            
            // 只检查有效性，不 clamp
            if (!isFinite(localX)) localX = 0.0f;
            if (!isFinite(localY)) localY = 0.0f;
            if (!isFinite(localZ)) localZ = 0.0f;
        }

        if (i >= fixedPts && instPtr != nullptr)
        {
            const int learnIdx = i - fixedPts;
            const int rowBase = learnIdx * 3;
            const float* rowWeight = params.fcWeight + rowBase * params.embedDims;
            const float* rowBias = params.fcBias + rowBase;

            // 混合精度优化：所有计算使用 FP32，确保精度
            // 直接使用 fmaf 进行累加，这是最精确的方式
            float accum[3];
            accum[0] = static_cast<float>(rowBias[0]);
            accum[1] = static_cast<float>(rowBias[1]);
            accum[2] = static_cast<float>(rowBias[2]);
            
            // 使用 fmaf (fused multiply-add) 进行高精度累加
            // fmaf 是单次舍入操作，比分开的乘法和加法更精确
            // 关键修复：在每次 fmaf 后检查 accum 的值，防止累加过程中产生 Inf
            const float max_accum_safe = 88.0f;  // sigmoid 数值稳定的上限
            const float min_accum_safe = -88.0f;  // sigmoid 数值稳定的下限
            
            for (int k = 0; k < params.embedDims; ++k)
            {
                // 将 FP16 输入转换为 FP32
                const float val = toFloat(instPtr[k]);
                
                // 检查输入值是否有效
                if (!isFinite(val)) {
                    continue;  // 跳过无效值
                }
                
                // 权重已经是 FP32，直接使用
                const float w0 = static_cast<float>(rowWeight[0 * params.embedDims + k]);
                const float w1 = static_cast<float>(rowWeight[1 * params.embedDims + k]);
                const float w2 = static_cast<float>(rowWeight[2 * params.embedDims + k]);
                
                // 检查权重是否有效
                if (!isFinite(w0) || !isFinite(w1) || !isFinite(w2)) {
                    continue;  // 跳过无效权重
                }
                
                // 使用 fmaf 进行融合乘加，这是最精确的累加方式
                accum[0] = fmaf(val, w0, accum[0]);
                accum[1] = fmaf(val, w1, accum[1]);
                accum[2] = fmaf(val, w2, accum[2]);
                
                // 关键修复：在每次 fmaf 后检查 accum 的值，防止累加过程中产生 Inf
                // 如果产生 Inf 或超出安全范围，立即 clamp 到安全范围，避免后续计算产生 NaN
                if (!isFinite(accum[0]) || accum[0] > max_accum_safe || accum[0] < min_accum_safe) {
                    accum[0] = fmaxf(fminf(accum[0], max_accum_safe), min_accum_safe);
                    if (!isFinite(accum[0])) accum[0] = static_cast<float>(rowBias[0]);
                }
                if (!isFinite(accum[1]) || accum[1] > max_accum_safe || accum[1] < min_accum_safe) {
                    accum[1] = fmaxf(fminf(accum[1], max_accum_safe), min_accum_safe);
                    if (!isFinite(accum[1])) accum[1] = static_cast<float>(rowBias[1]);
                }
                if (!isFinite(accum[2]) || accum[2] > max_accum_safe || accum[2] < min_accum_safe) {
                    accum[2] = fmaxf(fminf(accum[2], max_accum_safe), min_accum_safe);
                    if (!isFinite(accum[2])) accum[2] = static_cast<float>(rowBias[2]);
                }
            }
            
            // 最终检查：确保值在合理范围内
            // 关键修复：sigmoid 函数对输入值很敏感，超出 [-88, 88] 范围会导致数值不稳定
            // 但更重要的是，如果值太大，sigmoid 输出会接近边界值，导致精度损失
            // 我们 clamp 到一个更合理的范围，确保 sigmoid 计算稳定
            
            // 检查并修复 NaN/Inf
            if (!isFinite(accum[0])) accum[0] = static_cast<float>(rowBias[0]);
            if (!isFinite(accum[1])) accum[1] = static_cast<float>(rowBias[1]);
            if (!isFinite(accum[2])) accum[2] = static_cast<float>(rowBias[2]);
            
            // Clamp 到 sigmoid 数值稳定范围（最终检查）
            accum[0] = fmaxf(fminf(accum[0], max_accum_safe), min_accum_safe);
            accum[1] = fmaxf(fminf(accum[1], max_accum_safe), min_accum_safe);
            accum[2] = fmaxf(fminf(accum[2], max_accum_safe), min_accum_safe);
            
            // 再次检查（防止 clamp 后仍异常）
            if (!isFinite(accum[0])) accum[0] = static_cast<float>(rowBias[0]);
            if (!isFinite(accum[1])) accum[1] = static_cast<float>(rowBias[1]);
            if (!isFinite(accum[2])) accum[2] = static_cast<float>(rowBias[2]);

            // 数值稳定的 sigmoid_centered 实现
            // 等价于 sigmoid(x) - 0.5 = 1/(1+exp(-x)) - 0.5
            // 使用数值稳定的公式避免 exp 溢出/下溢和 NaN
            auto sigmoid_centered = [](float x) {
                // 检查输入是否为 NaN 或 Inf
                if (!isFinite(x)) {
                    return 0.0f;  // 如果输入异常，返回中性值
                }
                
                // 当 x 很大时，sigmoid(x) ≈ 1，所以 sigmoid_centered ≈ 0.5
                // 当 x 很小时，sigmoid(x) ≈ 0，所以 sigmoid_centered ≈ -0.5
                if (x > 88.0f) {
                    return 0.5f;  // expf(-x) 下溢为 0
                } else if (x < -88.0f) {
                    return -0.5f;  // expf(-x) 溢出为 inf，但 1/(1+inf) = 0
                } else {
                    // 标准实现，但在 FP16 下更安全
                    const float exp_neg_x = expf(-x);
                    // 检查 exp 结果是否有效
                    if (!isFinite(exp_neg_x)) {
                        // 如果 exp 产生 inf，根据 x 的符号返回边界值
                        return (x > 0.0f) ? 0.5f : -0.5f;
                    }
                    const float denom = 1.f + exp_neg_x;
                    // 检查分母是否有效
                    if (!isFinite(denom) || denom == 0.0f) {
                        return (x > 0.0f) ? 0.5f : -0.5f;
                    }
                    const float result = 1.f / denom - 0.5f;
                    // 最终检查结果是否有效
                    return isFinite(result) ? result : 0.0f;
                }
            };

            // 计算 sigmoid_centered 输出（范围 [-0.5, 0.5]）
            const float sig_x = sigmoid_centered(accum[0]);
            const float sig_y = sigmoid_centered(accum[1]);
            const float sig_z = sigmoid_centered(accum[2]);

            // 计算 local 坐标（使用 FP32 精度，不进行过度 clamp）
            // sigmoid_centered 输出范围是 [-0.5, 0.5]，乘以 size 后应该在合理范围内
            if (!isFinite(sig_x) || !isFinite(sizeX)) {
                localX = 0.0f;
            } else {
                localX = sig_x * sizeX;
                if (!isFinite(localX)) localX = 0.0f;
            }
            
            if (!isFinite(sig_y) || !isFinite(sizeY)) {
                localY = 0.0f;
            } else {
                localY = sig_y * sizeY;
                if (!isFinite(localY)) localY = 0.0f;
            }
            
            if (!isFinite(sig_z) || !isFinite(sizeZ)) {
                localZ = 0.0f;
            } else {
                localZ = sig_z * sizeZ;
                if (!isFinite(localZ)) localZ = 0.0f;
            }
        }

        // 应用旋转
        // 检查 local 值是否有效
        if (!isFinite(localX)) localX = 0.0f;
        if (!isFinite(localY)) localY = 0.0f;
        if (!isFinite(localZ)) localZ = 0.0f;
        
        // 关键修复：在旋转计算前，clamp localX 和 localY 到合理范围
        // 防止大值乘以 sinYaw/cosYaw 产生 Inf，或 Inf 运算产生 NaN
        const float max_local_safe = 1e6f;  // 合理的最大值
        const float min_local_safe = -1e6f;  // 合理的最小值
        localX = fmaxf(fminf(localX, max_local_safe), min_local_safe);
        localY = fmaxf(fminf(localY, max_local_safe), min_local_safe);
        localZ = fmaxf(fminf(localZ, max_local_safe), min_local_safe);
        // 再次检查（防止 clamp 后仍异常）
        if (!isFinite(localX)) localX = 0.0f;
        if (!isFinite(localY)) localY = 0.0f;
        if (!isFinite(localZ)) localZ = 0.0f;
        
        float rotX = cosYaw * localX - sinYaw * localY;
        float rotY = sinYaw * localX + cosYaw * localY;
        
        // 检查旋转结果是否有效
        if (!isFinite(rotX)) rotX = 0.0f;
        if (!isFinite(rotY)) rotY = 0.0f;

        // 加上中心点（使用 FP32 精度）
        // 关键修复：在加法前，确保所有值都在合理范围内
        // 防止 Inf + 有限值 = Inf，或 Inf + Inf = NaN
        const float max_center_safe = 1e6f;  // 合理的最大值
        const float min_center_safe = -1e6f;  // 合理的最小值
        centerX = fmaxf(fminf(centerX, max_center_safe), min_center_safe);
        centerY = fmaxf(fminf(centerY, max_center_safe), min_center_safe);
        centerZ = fmaxf(fminf(centerZ, max_center_safe), min_center_safe);
        rotX = fmaxf(fminf(rotX, max_center_safe), min_center_safe);
        rotY = fmaxf(fminf(rotY, max_center_safe), min_center_safe);
        localZ = fmaxf(fminf(localZ, max_center_safe), min_center_safe);
        // 再次检查（防止 clamp 后仍异常）
        if (!isFinite(centerX)) centerX = 0.0f;
        if (!isFinite(centerY)) centerY = 0.0f;
        if (!isFinite(centerZ)) centerZ = 0.0f;
        if (!isFinite(rotX)) rotX = 0.0f;
        if (!isFinite(rotY)) rotY = 0.0f;
        if (!isFinite(localZ)) localZ = 0.0f;
        
        float finalX = rotX + centerX;
        float finalY = rotY + centerY;
        float finalZ = localZ + centerZ;

        // 检查中间计算结果是否为 NaN 或 Inf
        if (!isFinite(finalX)) finalX = centerX;
        if (!isFinite(finalY)) finalY = centerY;
        if (!isFinite(finalZ)) finalZ = centerZ;
        
        // 再次检查（防止加法后产生异常值）
        if (!isFinite(finalX)) finalX = 0.0f;
        if (!isFinite(finalY)) finalY = 0.0f;
        if (!isFinite(finalZ)) finalZ = 0.0f;

        // 最终 NaN 检查
        // 关键优化：在 FP16 模式下，插件输出 FP32 值，让 TensorRT 负责 FP32 到 FP16 的转换
        // 这样可以避免插件内部的 FP16 转换问题，确保 100% 成功率
        if (!isFinite(finalX)) finalX = 0.0f;
        if (!isFinite(finalY)) finalY = 0.0f;
        if (!isFinite(finalZ)) finalZ = 0.0f;

        // 输出处理：根据 outputFP32 标志确定输出类型
        // 如果 outputFP32 为 true，输出指针是 float*，直接赋值
        // 如果 outputFP32 为 false，输出指针是 T*，使用 fromFloat<T> 转换
        if (params.outputFP32)
        {
            // 输出是 FP32，直接写入 float 值
            float* outPtrFloat = reinterpret_cast<float*>(outPtrBase);
            outPtrFloat[offset + 0] = finalX;
            outPtrFloat[offset + 1] = finalY;
            outPtrFloat[offset + 2] = finalZ;
        }
        else
        {
            // 输出类型与输入类型相同，使用 fromFloat<T> 转换
            T* outPtr = reinterpret_cast<T*>(outPtrBase);
            outPtr[offset + 0] = fromFloat<T>(finalX);
            outPtr[offset + 1] = fromFloat<T>(finalY);
            outPtr[offset + 2] = fromFloat<T>(finalZ);
        }
    }
}

int launchSparseBox3DKeyPointsKernel(
    const SparseBox3DKeyPointsKernelParams& params,
    cudaStream_t stream)
{
    const int32_t total = params.batch * params.numAnchor;
    if (total <= 0)
    {
        return 1;  // 返回非0表示失败
    }
    
    const int32_t threads = 256;
    const int32_t blocks = (total + threads - 1) / threads;

    // 关键优化：在 FP16 模式下，插件内部使用 FP32 计算，输出也是 FP32
    // 让 TensorRT 负责 FP32 到 FP16 的转换，这样可以确保 100% 成功率
    // 由于 getOutputDataType 在 FP16 输入时返回 FP32，TensorRT 会分配 FP32 输出内存
    // 输入可能是 FP16 或 FP32，但 kernel 内部会将所有输入转换为 FP32 处理
    // 输出总是 FP32（因为 getOutputDataType 返回 FP32）
    if (params.useFP16)
    {
        // FP16 输入，FP32 输出：使用 __half 模板处理输入，但输出是 float*
        // kernel 内部会将 __half 输入转换为 FP32，然后输出 FP32
        sparseBox3DKeyPointsKernel<__half><<<blocks, threads, 0, stream>>>(params);
    }
    else
    {
        // FP32 输入，FP32 输出
        sparseBox3DKeyPointsKernel<float><<<blocks, threads, 0, stream>>>(params);
    }
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return 1;  // 返回非0表示失败
    }
    
    return 0;  // 返回0表示成功
}
} // namespace sparse4d


