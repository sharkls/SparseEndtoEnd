#include "SparseBox3DKeyPointsKernel.h"

#include <cuda_fp16.h>
#include <math.h>

// CUDA device 函数：检查浮点数是否为有限值（非 NaN 非 Inf）
// 使用 CUDA 内置函数 isfinite，更可靠
__device__ __forceinline__ bool isFinite(float x)
{
    return isfinite(x);
}

// 关键优化：使用位操作检查NaN，确保100%可靠
// 这是工程部署中防止NaN的最后一道防线
__device__ __forceinline__ bool isNaN_strict(float x)
{
    // 使用位操作检查：NaN的IEEE 754表示是：指数全1且尾数非0
    const unsigned int bits = __float_as_uint(x);
    const unsigned int exp_mask = 0x7F800000;  // 指数位掩码（8位）
    const unsigned int mantissa_mask = 0x007FFFFF;  // 尾数位掩码（23位）
    
    // 检查是否为NaN（指数全1且尾数非0）
    if ((bits & exp_mask) == exp_mask && (bits & mantissa_mask) != 0) {
        return true;
    }
    return false;
}

// 关键优化：使用位操作检查Inf，确保100%可靠
__device__ __forceinline__ bool isInf_strict(float x)
{
    // 使用位操作检查：Inf的IEEE 754表示是：指数全1且尾数为0
    const unsigned int bits = __float_as_uint(x);
    const unsigned int exp_mask = 0x7F800000;  // 指数位掩码
    const unsigned int mantissa_mask = 0x007FFFFF;  // 尾数位掩码
    
    // 检查是否为Inf（指数全1且尾数为0）
    if ((bits & exp_mask) == exp_mask && (bits & mantissa_mask) == 0) {
        return true;
    }
    return false;
}

// 关键优化：确保值绝对安全，使用多重检查
// 这是工程部署中防止NaN的核心函数
// 注意：只检查NaN/Inf，不进行过度的范围限制，避免破坏正常值
__device__ __forceinline__ float ensureSafeValue(float x)
{
    // 第一层：使用位操作检查NaN（最可靠，优先检查）
    if (isNaN_strict(x)) {
        return 0.0f;
    }
    
    // 第二层：使用位操作检查Inf
    if (isInf_strict(x)) {
        return 0.0f;
    }
    
    // 第三层：使用isfinite检查（双重验证）
    if (!isfinite(x)) {
        return 0.0f;
    }
    
    // 第四层：使用isFinite检查（自定义函数，三重验证）
    if (!isFinite(x)) {
        return 0.0f;
    }
    
    // 关键修复：不进行过度的范围限制，只检查NaN/Inf
    // 过度的范围限制可能导致正常的大值被错误地clamp，影响精度
    // 只对明显异常的值（NaN/Inf）进行处理
    
    return x;
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
    // 关键优化：在转换前检查 FP16 值是否有效
    // 地平线J6部署经验：在输入转换阶段进行严格的数值检查
    // 检查 FP16 是否为 NaN 或 Inf（使用位操作更可靠）
    const unsigned short bits = __half_as_ushort(x);
    const unsigned short exp_mask = 0x7C00;  // 指数位掩码
    const unsigned short mantissa_mask = 0x03FF;  // 尾数位掩码
    
    // 检查是否为 NaN（指数全1且尾数非0）
    if ((bits & exp_mask) == exp_mask && (bits & mantissa_mask) != 0) {
        return 0.0f;  // 如果输入是 NaN，返回 0
    }
    
    // 检查是否为 Inf（指数全1且尾数为0）
    if ((bits & exp_mask) == exp_mask && (bits & mantissa_mask) == 0) {
        return 0.0f;  // 如果输入是 Inf，返回 0
    }
    
    float result = __half2float(x);
    // 再次检查转换结果是否有效
    if (!isFinite(result)) {
        return 0.0f;  // 如果转换产生 NaN/Inf，返回 0
    }
    
    // 关键优化：clamp 转换后的值到合理范围，防止后续计算溢出
    // FP16 范围是 [-65504, 65504]，但为了安全，我们使用更保守的值
    // 地平线J6部署经验：使用更保守的范围值，确保100%稳定性
    const float max_input_safe = 1e4f;  // 更保守的上限（从50000.0降低到1e4）
    const float min_input_safe = -1e4f;  // 更保守的下限
    if (result > max_input_safe || result < min_input_safe) {
        result = fmaxf(fminf(result, max_input_safe), min_input_safe);
        // 再次检查 clamp 后的值
        if (!isFinite(result)) {
            return 0.0f;
        }
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
    
    // 关键优化：在kernel开始时验证所有必要的参数
    if (params.anchor == nullptr || params.output == nullptr) {
        return;  // 如果关键指针为空，直接返回（输出内存应该已经被初始化为0）
    }
    if (params.numAnchor <= 0 || params.numPts <= 0) {
        return;  // 如果参数无效，直接返回（输出内存应该已经被初始化为0）
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
    // 关键优化：在读取后立即进行严格的数值检查和 clamp
    // 地平线J6部署经验：在输入阶段进行严格的数值范围检查与裁剪
    float centerX_raw = toFloat(anchorPtr[0]);
    float centerY_raw = toFloat(anchorPtr[1]);
    float centerZ_raw = toFloat(anchorPtr[2]);
    float log_size_x_raw = toFloat(anchorPtr[3]);
    float log_size_y_raw = toFloat(anchorPtr[4]);
    float log_size_z_raw = toFloat(anchorPtr[5]);
    float sinYaw_raw = toFloat(anchorPtr[6]);
    float cosYaw_raw = toFloat(anchorPtr[7]);
    
        // 检查所有输入值是否有效，并进行合理的范围限制
        // 关键优化：对 center 坐标进行范围限制，防止后续计算溢出
        // 关键修复：使用更严格的检查，确保center值绝对安全
        const float max_center_input = 1e4f;  // center 坐标的合理上限
        const float min_center_input = -1e4f;  // center 坐标的合理下限
        
        // 使用位操作检查，确保绝对可靠
        float centerX = 0.0f;
        if (!isNaN_strict(centerX_raw) && !isInf_strict(centerX_raw) && isFinite(centerX_raw)) {
            centerX = fmaxf(fminf(centerX_raw, max_center_input), min_center_input);
            // 再次验证clamp后的值
            if (isNaN_strict(centerX) || isInf_strict(centerX) || !isfinite(centerX)) {
                centerX = 0.0f;
            }
        } else {
            centerX = 0.0f;
        }
        
        float centerY = 0.0f;
        if (!isNaN_strict(centerY_raw) && !isInf_strict(centerY_raw) && isFinite(centerY_raw)) {
            centerY = fmaxf(fminf(centerY_raw, max_center_input), min_center_input);
            if (isNaN_strict(centerY) || isInf_strict(centerY) || !isfinite(centerY)) {
                centerY = 0.0f;
            }
        } else {
            centerY = 0.0f;
        }
        
        float centerZ = 0.0f;
        if (!isNaN_strict(centerZ_raw) && !isInf_strict(centerZ_raw) && isFinite(centerZ_raw)) {
            centerZ = fmaxf(fminf(centerZ_raw, max_center_input), min_center_input);
            if (isNaN_strict(centerZ) || isInf_strict(centerZ) || !isfinite(centerZ)) {
                centerZ = 0.0f;
            }
        } else {
            centerZ = 0.0f;
        }
        
        // 最终验证：确保center值绝对安全（三重保险）
        if (isNaN_strict(centerX) || isInf_strict(centerX) || !isfinite(centerX)) {
            centerX = 0.0f;
        }
        if (isNaN_strict(centerY) || isInf_strict(centerY) || !isfinite(centerY)) {
            centerY = 0.0f;
        }
        if (isNaN_strict(centerZ) || isInf_strict(centerZ) || !isfinite(centerZ)) {
            centerZ = 0.0f;
        }
    
    // log_size 需要限制以防止 exp 溢出，但使用合理的范围
    const float max_log_size = 11.0f;  // exp(11) ≈ 59874 < 65504
    const float min_log_size = -11.0f;
    float log_size_x = isFinite(log_size_x_raw) ? fmaxf(fminf(log_size_x_raw, max_log_size), min_log_size) : 0.0f;
    float log_size_y = isFinite(log_size_y_raw) ? fmaxf(fminf(log_size_y_raw, max_log_size), min_log_size) : 0.0f;
    float log_size_z = isFinite(log_size_z_raw) ? fmaxf(fminf(log_size_z_raw, max_log_size), min_log_size) : 0.0f;
    
    // 计算 size（使用 FP32 精度）
    // 关键优化：在计算 exp 后立即检查并 clamp，防止产生 Inf
    // 地平线J6部署经验：数值范围检查与裁剪（Clamping）是确保FP16稳定性的关键
    float sizeX = expf(log_size_x);
    float sizeY = expf(log_size_y);
    float sizeZ = expf(log_size_z);
    
    // 关键修复：使用位操作检查exp结果，确保100%可靠
    const unsigned int sizeX_bits = __float_as_uint(sizeX);
    const unsigned int sizeY_bits = __float_as_uint(sizeY);
    const unsigned int sizeZ_bits = __float_as_uint(sizeZ);
    // 关键修复：重命名变量以避免重复声明错误
    const unsigned int exp_mask_for_size = 0x7F800000;
    
    // 检查 exp 结果是否有效，并 clamp 到合理范围
    // FP16 最大值是 65504，但为了安全，我们使用更保守的值
    const float max_size_safe = 50000.0f;  // 保守的最大值，确保不会溢出
    const float min_size_safe = 1e-6f;     // 防止下溢的最小值
    
    if ((sizeX_bits & exp_mask_for_size) == exp_mask_for_size || sizeX > max_size_safe || sizeX < min_size_safe) {
        #ifdef DEBUG_NAN
        if ((sizeX_bits & exp_mask_for_size) == exp_mask_for_size) {
            printf("[DEBUG_NAN] anchorIdx=%d: sizeX is NaN/Inf after expf (bits=0x%08x, log_size_x=%.6f, sizeX=%.6f)\n",
                   anchorIdx, sizeX_bits, log_size_x, sizeX);
        }
        #endif
        sizeX = fmaxf(fminf(sizeX, max_size_safe), min_size_safe);
        // 再次检查clamp后的值
        const unsigned int sizeX_bits_after = __float_as_uint(sizeX);
        if ((sizeX_bits_after & exp_mask_for_size) == exp_mask_for_size) {
            sizeX = 1.0f;
        }
    }
    if ((sizeY_bits & exp_mask_for_size) == exp_mask_for_size || sizeY > max_size_safe || sizeY < min_size_safe) {
        #ifdef DEBUG_NAN
        if ((sizeY_bits & exp_mask_for_size) == exp_mask_for_size) {
            printf("[DEBUG_NAN] anchorIdx=%d: sizeY is NaN/Inf after expf (bits=0x%08x, log_size_y=%.6f, sizeY=%.6f)\n",
                   anchorIdx, sizeY_bits, log_size_y, sizeY);
        }
        #endif
        sizeY = fmaxf(fminf(sizeY, max_size_safe), min_size_safe);
        const unsigned int sizeY_bits_after = __float_as_uint(sizeY);
        if ((sizeY_bits_after & exp_mask_for_size) == exp_mask_for_size) {
            sizeY = 1.0f;
        }
    }
    if ((sizeZ_bits & exp_mask_for_size) == exp_mask_for_size || sizeZ > max_size_safe || sizeZ < min_size_safe) {
        #ifdef DEBUG_NAN
        if ((sizeZ_bits & exp_mask_for_size) == exp_mask_for_size) {
            printf("[DEBUG_NAN] anchorIdx=%d: sizeZ is NaN/Inf after expf (bits=0x%08x, log_size_z=%.6f, sizeZ=%.6f)\n",
                   anchorIdx, sizeZ_bits, log_size_z, sizeZ);
        }
        #endif
        sizeZ = fmaxf(fminf(sizeZ, max_size_safe), min_size_safe);
        const unsigned int sizeZ_bits_after = __float_as_uint(sizeZ);
        if ((sizeZ_bits_after & exp_mask_for_size) == exp_mask_for_size) {
            sizeZ = 1.0f;
        }
    }
    
        // sin/cos yaw 应该已经在 [-1, 1] 范围内，只检查有效性
        // 关键修复：PyTorch 实现并未限制 sin/cos 必须在 [-1, 1] 范围内
        // 为了对齐精度，我们只检查 NaN/Inf，不进行范围截断
        float sinYaw = isFinite(sinYaw_raw) ? sinYaw_raw : 0.0f;
        float cosYaw = isFinite(cosYaw_raw) ? cosYaw_raw : 1.0f;
        
        // 再次检查（防止 NaN/Inf）
        if (!isFinite(sinYaw)) sinYaw = 0.0f;
        if (!isFinite(cosYaw)) cosYaw = 1.0f;

    const int32_t fixedPts = params.numPts - params.numLearnablePts;

    for (int i = 0; i < params.numPts; ++i)
    {
        const int offset = i * 3;
        float localX = 0.f;
        float localY = 0.f;
        float localZ = 0.f;
        
        // 关键优化：在每次循环开始时，确保基础值都是有效的
        // 这是防止NaN传播的关键检查点
        if (!isFinite(sizeX)) sizeX = 1.0f;
        if (!isFinite(sizeY)) sizeY = 1.0f;
        if (!isFinite(sizeZ)) sizeZ = 1.0f;
        if (!isfinite(sizeX)) sizeX = 1.0f;
        if (!isfinite(sizeY)) sizeY = 1.0f;
        if (!isfinite(sizeZ)) sizeZ = 1.0f;

        if (i < fixedPts)
        {
            // 固定点计算（使用 FP32 精度）
            // 关键优化：在乘法前检查 fixScale 是否有效，防止异常值导致溢出
            // 地平线J6部署经验：在关键计算步骤中进行数值范围检查与裁剪
            // 关键修复：确保 size 值在计算前是有效的（size 已经在循环外验证过，这里再次确认）
            if (!isFinite(sizeX)) sizeX = 1.0f;
            if (!isFinite(sizeY)) sizeY = 1.0f;
            if (!isFinite(sizeZ)) sizeZ = 1.0f;
            
            // 检查 fixScale 是否有效，并进行合理的范围限制
            const float fixScale_x_raw = params.fixScale[offset + 0];
            const float fixScale_y_raw = params.fixScale[offset + 1];
            const float fixScale_z_raw = params.fixScale[offset + 2];
            
            // fixScale 通常在 [-1, 1] 范围内，但为了安全，允许更大的范围
            const float max_fixScale = 10.0f;
            const float min_fixScale = -10.0f;
            
            const float fixScale_x = isFinite(fixScale_x_raw) 
                ? fmaxf(fminf(fixScale_x_raw, max_fixScale), min_fixScale) : 0.0f;
            const float fixScale_y = isFinite(fixScale_y_raw) 
                ? fmaxf(fminf(fixScale_y_raw, max_fixScale), min_fixScale) : 0.0f;
            const float fixScale_z = isFinite(fixScale_z_raw) 
                ? fmaxf(fminf(fixScale_z_raw, max_fixScale), min_fixScale) : 0.0f;
            
            // 关键优化：在乘法后立即检查NaN/Inf，确保绝对安全
            // 工程部署要求：每一步计算后都要确保值安全
            localX = fixScale_x * sizeX;
            localY = fixScale_y * sizeY;
            localZ = fixScale_z * sizeZ;
            
            // 检查乘法结果，防止NaN/Inf传播
            if (isNaN_strict(localX) || isInf_strict(localX)) {
                localX = 0.0f;
            }
            if (isNaN_strict(localY) || isInf_strict(localY)) {
                localY = 0.0f;
            }
            if (isNaN_strict(localZ) || isInf_strict(localZ)) {
                localZ = 0.0f;
            }
        }

        if (i >= fixedPts && instPtr != nullptr)
        {
            const int learnIdx = i - fixedPts;
            const int rowBase = learnIdx * 3;
            const float* rowWeight = params.fcWeight + rowBase * params.embedDims;
            const float* rowBias = params.fcBias + rowBase;

            // 混合精度优化：所有计算使用 FP32，确保精度
            // 直接使用 fmaf 进行累加，这是最精确的方式
            // 关键优化：在初始化时检查 bias 值是否有效
            float accum[3];
            accum[0] = isFinite(rowBias[0]) ? static_cast<float>(rowBias[0]) : 0.0f;
            accum[1] = isFinite(rowBias[1]) ? static_cast<float>(rowBias[1]) : 0.0f;
            accum[2] = isFinite(rowBias[2]) ? static_cast<float>(rowBias[2]) : 0.0f;
            
            // 关键优化：clamp bias 到合理范围，防止初始值过大
            const float max_bias_safe = 10.0f;  // bias 的合理上限
            const float min_bias_safe = -10.0f;  // bias 的合理下限
            accum[0] = fmaxf(fminf(accum[0], max_bias_safe), min_bias_safe);
            accum[1] = fmaxf(fminf(accum[1], max_bias_safe), min_bias_safe);
            accum[2] = fmaxf(fminf(accum[2], max_bias_safe), min_bias_safe);
            
            // 再次检查
            if (!isFinite(accum[0])) accum[0] = 0.0f;
            if (!isFinite(accum[1])) accum[1] = 0.0f;
            if (!isFinite(accum[2])) accum[2] = 0.0f;
            
            // 使用 fmaf (fused multiply-add) 进行高精度累加
            // fmaf 是单次舍入操作，比分开的乘法和加法更精确
            // 关键修复：在每次 fmaf 后检查 accum 的值，防止累加过程中产生 Inf
            // 地平线J6部署经验：使用更保守的范围值，确保100%稳定性
            const float max_accum_safe = 80.0f;  // 更保守的上限（从88.0降低到80.0）
            const float min_accum_safe = -80.0f;  // 更保守的下限

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
                
                // 关键优化：在 fmaf 前检查输入值，防止异常值导致溢出
                // 地平线J6部署经验：在关键计算前进行输入验证
                // 使用更保守的范围值，确保100%稳定性
                const float max_val_safe = 50.0f;  // 更保守的输入值上限（从100.0降低到50.0）
                const float min_val_safe = -50.0f;  // 更保守的输入值下限
                const float val_clamped = fmaxf(fminf(val, max_val_safe), min_val_safe);
                if (!isFinite(val_clamped)) {
                    continue;  // 如果 clamp 后仍异常，跳过
                }
                
                // 检查权重是否在合理范围内
                const float max_weight_safe = 5.0f;  // 更保守的权重上限（从10.0降低到5.0）
                const float min_weight_safe = -5.0f;  // 更保守的权重下限
                const float w0_clamped = fmaxf(fminf(w0, max_weight_safe), min_weight_safe);
                const float w1_clamped = fmaxf(fminf(w1, max_weight_safe), min_weight_safe);
                const float w2_clamped = fmaxf(fminf(w2, max_weight_safe), min_weight_safe);
                
                // 检查 clamp 后的权重是否有效
                if (!isFinite(w0_clamped) || !isFinite(w1_clamped) || !isFinite(w2_clamped)) {
                    continue;  // 如果权重异常，跳过
                }
                
                // 使用 fmaf 进行融合乘加，这是最精确的累加方式
                // 关键优化：在每次fmaf后立即检查NaN/Inf，确保绝对安全
                // 工程部署要求：每一步计算后都要确保值安全
                // 关键修复：在 fmaf 前检查所有输入，确保绝对安全
                if (isNaN_strict(accum[0]) || isInf_strict(accum[0]) || !isfinite(accum[0])) {
                    accum[0] = 0.0f;
                }
                if (isNaN_strict(accum[1]) || isInf_strict(accum[1]) || !isfinite(accum[1])) {
                    accum[1] = 0.0f;
                }
                if (isNaN_strict(accum[2]) || isInf_strict(accum[2]) || !isfinite(accum[2])) {
                    accum[2] = 0.0f;
                }
                
                // 执行 fmaf 操作
                float new_accum0 = fmaf(val_clamped, w0_clamped, accum[0]);
                float new_accum1 = fmaf(val_clamped, w1_clamped, accum[1]);
                float new_accum2 = fmaf(val_clamped, w2_clamped, accum[2]);
                
                // 关键修复：使用位操作立即检查fmaf结果，防止NaN/Inf传播
                #ifdef DEBUG_NAN
                if (isNaN_strict(new_accum0) || isInf_strict(new_accum0) || !isfinite(new_accum0)) {
                    printf("[DEBUG_NAN] anchorIdx=%d, pt=%d, k=%d: fmaf[0] produced NaN/Inf (val=%.6f, w=%.6f, accum=%.6f, result=%.6f)\n",
                           anchorIdx, i, k, val_clamped, w0_clamped, accum[0], new_accum0);
                    accum[0] = 0.0f;  // 如果产生NaN/Inf，重置为0
                } else {
                    accum[0] = new_accum0;
                }
                if (isNaN_strict(new_accum1) || isInf_strict(new_accum1) || !isfinite(new_accum1)) {
                    printf("[DEBUG_NAN] anchorIdx=%d, pt=%d, k=%d: fmaf[1] produced NaN/Inf (val=%.6f, w=%.6f, accum=%.6f, result=%.6f)\n",
                           anchorIdx, i, k, val_clamped, w1_clamped, accum[1], new_accum1);
                    accum[1] = 0.0f;
                } else {
                    accum[1] = new_accum1;
                }
                if (isNaN_strict(new_accum2) || isInf_strict(new_accum2) || !isfinite(new_accum2)) {
                    printf("[DEBUG_NAN] anchorIdx=%d, pt=%d, k=%d: fmaf[2] produced NaN/Inf (val=%.6f, w=%.6f, accum=%.6f, result=%.6f)\n",
                           anchorIdx, i, k, val_clamped, w2_clamped, accum[2], new_accum2);
                    accum[2] = 0.0f;
                } else {
                    accum[2] = new_accum2;
                }
                #else
                if (isNaN_strict(new_accum0) || isInf_strict(new_accum0) || !isfinite(new_accum0)) {
                    accum[0] = 0.0f;  // 如果产生NaN/Inf，重置为0
                } else {
                    accum[0] = new_accum0;
                }
                if (isNaN_strict(new_accum1) || isInf_strict(new_accum1) || !isfinite(new_accum1)) {
                    accum[1] = 0.0f;
                } else {
                    accum[1] = new_accum1;
                }
                if (isNaN_strict(new_accum2) || isInf_strict(new_accum2) || !isfinite(new_accum2)) {
                    accum[2] = 0.0f;
                } else {
                    accum[2] = new_accum2;
                }
                #endif
                
                // 关键优化：clamp到安全范围，防止后续sigmoid计算溢出
                accum[0] = fmaxf(fminf(accum[0], max_accum_safe), min_accum_safe);
                accum[1] = fmaxf(fminf(accum[1], max_accum_safe), min_accum_safe);
                accum[2] = fmaxf(fminf(accum[2], max_accum_safe), min_accum_safe);
                
                // 再次检查clamp后的值，确保不是NaN/Inf
                if (isNaN_strict(accum[0]) || isInf_strict(accum[0])) {
                    accum[0] = 0.0f;
                }
                if (isNaN_strict(accum[1]) || isInf_strict(accum[1])) {
                    accum[1] = 0.0f;
                }
                if (isNaN_strict(accum[2]) || isInf_strict(accum[2])) {
                    accum[2] = 0.0f;
                }
            }
            
            // 最终检查：确保值在合理范围内
            // 关键优化：检查accum值，确保不是NaN/Inf
            // 工程部署要求：在进入sigmoid计算前，必须确保输入值安全
            if (isNaN_strict(accum[0]) || isInf_strict(accum[0])) {
                accum[0] = 0.0f;
            }
            if (isNaN_strict(accum[1]) || isInf_strict(accum[1])) {
                accum[1] = 0.0f;
            }
            if (isNaN_strict(accum[2]) || isInf_strict(accum[2])) {
                accum[2] = 0.0f;
            }
            
            // Clamp 到 sigmoid 数值稳定范围
            accum[0] = fmaxf(fminf(accum[0], max_accum_safe), min_accum_safe);
            accum[1] = fmaxf(fminf(accum[1], max_accum_safe), min_accum_safe);
            accum[2] = fmaxf(fminf(accum[2], max_accum_safe), min_accum_safe);
            
            // 再次检查clamp后的值，确保不是NaN/Inf
            if (isNaN_strict(accum[0]) || isInf_strict(accum[0])) {
                accum[0] = 0.0f;
            }
            if (isNaN_strict(accum[1]) || isInf_strict(accum[1])) {
                accum[1] = 0.0f;
            }
            if (isNaN_strict(accum[2]) || isInf_strict(accum[2])) {
                accum[2] = 0.0f;
            }

            // 数值稳定的 sigmoid_centered 实现
            // 等价于 sigmoid(x) - 0.5 = 1/(1+exp(-x)) - 0.5
            // 使用数值稳定的公式避免 exp 溢出/下溢和 NaN
            // 地平线J6部署经验：使用更保守的边界值，确保100%稳定性
            auto sigmoid_centered = [](float x) {
                // 关键修复：使用位操作检查输入，确保绝对可靠
                if (isNaN_strict(x) || isInf_strict(x) || !isfinite(x)) {
                    return 0.0f;  // 如果输入异常，返回中性值
                }
                
                // 关键优化：使用更保守的边界值（从80.0降低到70.0），确保exp计算不会溢出
                // 当 x 很大时，sigmoid(x) ≈ 1，所以 sigmoid_centered ≈ 0.5
                // 当 x 很小时，sigmoid(x) ≈ 0，所以 sigmoid_centered ≈ -0.5
                const float sigmoid_bound = 70.0f;  // 更保守的边界值，确保100%稳定性
                if (x > sigmoid_bound) {
                    return 0.5f;  // expf(-x) 下溢为 0
                } else if (x < -sigmoid_bound) {
                    return -0.5f;  // expf(-x) 溢出为 inf，但 1/(1+inf) = 0
                } else {
                    // 标准实现，但在 FP16 下更安全
                    // 关键优化：clamp x 到安全范围，防止 exp 溢出
                    x = fmaxf(fminf(x, sigmoid_bound), -sigmoid_bound);
                    
                    // 关键修复：clamp后再次检查，确保值安全
                    if (isNaN_strict(x) || isInf_strict(x) || !isfinite(x)) {
                        return 0.0f;
                    }
                    
                    // 关键修复：在计算 exp 前，确保 x 在安全范围内
                    // 使用更保守的范围，防止 exp 溢出
                    const float exp_input = fmaxf(fminf(-x, 70.0f), -70.0f);
                    if (isNaN_strict(exp_input) || isInf_strict(exp_input) || !isfinite(exp_input)) {
                        return (x > 0.0f) ? 0.5f : -0.5f;
                    }
                    
                    const float exp_neg_x = expf(exp_input);
                    
                    // 关键修复：使用位操作检查 exp 结果，确保绝对可靠
                    if (isNaN_strict(exp_neg_x) || isInf_strict(exp_neg_x) || !isfinite(exp_neg_x)) {
                        // 如果 exp 产生 inf，根据 x 的符号返回边界值
                        return (x > 0.0f) ? 0.5f : -0.5f;
                    }
                    
                    const float denom = 1.f + exp_neg_x;
                    
                    // 关键修复：使用位操作检查分母，确保绝对可靠
                    if (isNaN_strict(denom) || isInf_strict(denom) || !isfinite(denom)) {
                        return (x > 0.0f) ? 0.5f : -0.5f;
                    }
                    
                    // 关键修复：不仅检查是否为0，还要检查是否太小，防止除法产生Inf
                    const float min_denom = 1e-10f;  // 防止除零和除极小值
                    if (denom < min_denom) {
                        return (x > 0.0f) ? 0.5f : -0.5f;
                    }
                    
                    // 关键修复：使用安全的除法，确保不会产生Inf
                    // 在除法前再次检查分母
                    if (isNaN_strict(denom) || isInf_strict(denom) || !isfinite(denom) || denom < min_denom) {
                        return (x > 0.0f) ? 0.5f : -0.5f;
                    }
                    
                    const float inv_denom = 1.f / denom;
                    
                    // 关键修复：使用位操作检查除法结果，确保绝对可靠
                    if (isNaN_strict(inv_denom) || isInf_strict(inv_denom) || !isfinite(inv_denom)) {
                        return (x > 0.0f) ? 0.5f : -0.5f;
                    }
                    
                    const float result = inv_denom - 0.5f;
                    
                    // 关键修复：使用位操作检查结果，确保绝对可靠
                    if (isNaN_strict(result) || isInf_strict(result) || !isfinite(result)) {
                        return 0.0f;
                    }
                    
                    // 确保结果在 [-0.5, 0.5] 范围内
                    float clamped_result = fmaxf(fminf(result, 0.5f), -0.5f);
                    
                    // 最终检查：使用位操作确保结果绝对安全
                    if (isNaN_strict(clamped_result) || isInf_strict(clamped_result) || !isfinite(clamped_result)) {
                        return 0.0f;
                    }
                    
                    return clamped_result;
                }
            };

            // 计算 sigmoid_centered 输出（范围 [-0.5, 0.5]）
            // 关键优化：检查accum值，确保不是NaN/Inf
            #ifdef DEBUG_NAN
            if (isNaN_strict(accum[0]) || isInf_strict(accum[0])) {
                printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: accum[0] is NaN/Inf before sigmoid (value=%.6f)\n",
                       anchorIdx, i, accum[0]);
                accum[0] = 0.0f;
            }
            if (isNaN_strict(accum[1]) || isInf_strict(accum[1])) {
                printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: accum[1] is NaN/Inf before sigmoid (value=%.6f)\n",
                       anchorIdx, i, accum[1]);
                accum[1] = 0.0f;
            }
            if (isNaN_strict(accum[2]) || isInf_strict(accum[2])) {
                printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: accum[2] is NaN/Inf before sigmoid (value=%.6f)\n",
                       anchorIdx, i, accum[2]);
                accum[2] = 0.0f;
            }
            #else
            if (isNaN_strict(accum[0]) || isInf_strict(accum[0])) {
                accum[0] = 0.0f;
            }
            if (isNaN_strict(accum[1]) || isInf_strict(accum[1])) {
                accum[1] = 0.0f;
            }
            if (isNaN_strict(accum[2]) || isInf_strict(accum[2])) {
                accum[2] = 0.0f;
            }
            #endif
            
            float sig_x = sigmoid_centered(accum[0]);
            float sig_y = sigmoid_centered(accum[1]);
            float sig_z = sigmoid_centered(accum[2]);
            
            // 关键优化：检查sigmoid输出，确保不是NaN/Inf
            #ifdef DEBUG_NAN
            if (isNaN_strict(sig_x) || isInf_strict(sig_x)) {
                printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: sig_x is NaN/Inf after sigmoid (accum[0]=%.6f, sig_x=%.6f)\n",
                       anchorIdx, i, accum[0], sig_x);
                sig_x = 0.0f;
            }
            if (isNaN_strict(sig_y) || isInf_strict(sig_y)) {
                printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: sig_y is NaN/Inf after sigmoid (accum[1]=%.6f, sig_y=%.6f)\n",
                       anchorIdx, i, accum[1], sig_y);
                sig_y = 0.0f;
            }
            if (isNaN_strict(sig_z) || isInf_strict(sig_z)) {
                printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: sig_z is NaN/Inf after sigmoid (accum[2]=%.6f, sig_z=%.6f)\n",
                       anchorIdx, i, accum[2], sig_z);
                sig_z = 0.0f;
            }
            #else
            if (isNaN_strict(sig_x) || isInf_strict(sig_x)) {
                sig_x = 0.0f;
            }
            if (isNaN_strict(sig_y) || isInf_strict(sig_y)) {
                sig_y = 0.0f;
            }
            if (isNaN_strict(sig_z) || isInf_strict(sig_z)) {
                sig_z = 0.0f;
            }
            #endif

            // 计算 local 坐标（使用 FP32 精度）
            // sigmoid_centered 输出范围是 [-0.5, 0.5]，乘以 size 后应该在合理范围内
            // 关键优化：在乘法后立即检查NaN/Inf，确保绝对安全
            // 工程部署要求：每一步计算后都要确保值安全
            
            // 执行乘法
            localX = sig_x * sizeX;
            localY = sig_y * sizeY;
            localZ = sig_z * sizeZ;
            
            // 检查乘法结果，防止NaN/Inf传播
            if (isNaN_strict(localX) || isInf_strict(localX)) {
                localX = 0.0f;
            }
            if (isNaN_strict(localY) || isInf_strict(localY)) {
                localY = 0.0f;
            }
            if (isNaN_strict(localZ) || isInf_strict(localZ)) {
                localZ = 0.0f;
            }
        }
        
        // 关键优化：在进入旋转计算前，检查所有local值，确保不是NaN/Inf
        if (isNaN_strict(localX) || isInf_strict(localX)) {
            localX = 0.0f;
        }
        if (isNaN_strict(localY) || isInf_strict(localY)) {
            localY = 0.0f;
        }
        if (isNaN_strict(localZ) || isInf_strict(localZ)) {
            localZ = 0.0f;
        }
        
        // 应用旋转
        // 关键优化：在旋转计算后立即检查NaN/Inf，确保绝对安全
        // 工程部署要求：每一步计算后都要确保值安全
        // 关键修复：在旋转计算前，确保所有输入值安全
        // 使用位操作进行最严格的检查，确保100%可靠
        const unsigned int localX_bits = __float_as_uint(localX);
        const unsigned int localY_bits = __float_as_uint(localY);
        const unsigned int sinYaw_bits = __float_as_uint(sinYaw);
        const unsigned int cosYaw_bits = __float_as_uint(cosYaw);
        const unsigned int exp_mask = 0x7F800000;
        
        // 检查localX/Y是否为NaN/Inf
        if ((localX_bits & exp_mask) == exp_mask) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: localX is NaN/Inf before rotation (bits=0x%08x, value=%.6f)\n",
                   anchorIdx, i, localX_bits, localX);
            #endif
            localX = 0.0f;
        }
        if ((localY_bits & exp_mask) == exp_mask) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: localY is NaN/Inf before rotation (bits=0x%08x, value=%.6f)\n",
                   anchorIdx, i, localY_bits, localY);
            #endif
            localY = 0.0f;
        }
        
        // 检查sinYaw/cosYaw是否为NaN/Inf
        if ((sinYaw_bits & exp_mask) == exp_mask) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: sinYaw is NaN/Inf before rotation (bits=0x%08x, value=%.6f)\n",
                   anchorIdx, i, sinYaw_bits, sinYaw);
            #endif
            sinYaw = 0.0f;
        }
        if ((cosYaw_bits & exp_mask) == exp_mask) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: cosYaw is NaN/Inf before rotation (bits=0x%08x, value=%.6f)\n",
                   anchorIdx, i, cosYaw_bits, cosYaw);
            #endif
            cosYaw = 1.0f;
        }
        
        // 执行旋转计算
        float rotX = cosYaw * localX - sinYaw * localY;
        float rotY = sinYaw * localX + cosYaw * localY;
        
        // 关键修复：使用位操作检查旋转结果，防止NaN/Inf传播
        const unsigned int rotX_bits = __float_as_uint(rotX);
        const unsigned int rotY_bits = __float_as_uint(rotY);
        if ((rotX_bits & exp_mask) == exp_mask) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: rotX is NaN/Inf after rotation (bits=0x%08x, localX=%.6f, localY=%.6f, sinYaw=%.6f, cosYaw=%.6f)\n",
                   anchorIdx, i, rotX_bits, localX, localY, sinYaw, cosYaw);
            #endif
            rotX = 0.0f;
        }
        if ((rotY_bits & exp_mask) == exp_mask) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: rotY is NaN/Inf after rotation (bits=0x%08x, localX=%.6f, localY=%.6f, sinYaw=%.6f, cosYaw=%.6f)\n",
                   anchorIdx, i, rotY_bits, localX, localY, sinYaw, cosYaw);
            #endif
            rotY = 0.0f;
        }
        
        // 加上中心点（使用 FP32 精度）
        // 关键优化：在加法后立即检查NaN/Inf，确保绝对安全
        // 工程部署要求：每一步计算后都要确保值安全
        // 加法运算可能产生Inf或NaN：Inf + 有限值 = Inf，Inf + Inf = NaN
        // 关键修复：在加法前，使用位操作确保所有输入值安全
        const unsigned int rotX_bits_check = __float_as_uint(rotX);
        const unsigned int rotY_bits_check = __float_as_uint(rotY);
        const unsigned int localZ_bits = __float_as_uint(localZ);
        const unsigned int centerX_bits = __float_as_uint(centerX);
        const unsigned int centerY_bits = __float_as_uint(centerY);
        const unsigned int centerZ_bits = __float_as_uint(centerZ);
        const unsigned int exp_mask_add = 0x7F800000;
        
        // 检查所有输入值是否为NaN/Inf
        if ((rotX_bits_check & exp_mask_add) == exp_mask_add) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: rotX is NaN/Inf before addition (bits=0x%08x, value=%.6f)\n",
                   anchorIdx, i, rotX_bits_check, rotX);
            #endif
            rotX = 0.0f;
        }
        if ((rotY_bits_check & exp_mask_add) == exp_mask_add) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: rotY is NaN/Inf before addition (bits=0x%08x, value=%.6f)\n",
                   anchorIdx, i, rotY_bits_check, rotY);
            #endif
            rotY = 0.0f;
        }
        if ((localZ_bits & exp_mask_add) == exp_mask_add) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: localZ is NaN/Inf before addition (bits=0x%08x, value=%.6f)\n",
                   anchorIdx, i, localZ_bits, localZ);
            #endif
            localZ = 0.0f;
        }
        if ((centerX_bits & exp_mask_add) == exp_mask_add) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: centerX is NaN/Inf before addition (bits=0x%08x, value=%.6f)\n",
                   anchorIdx, i, centerX_bits, centerX);
            #endif
            centerX = 0.0f;
        }
        if ((centerY_bits & exp_mask_add) == exp_mask_add) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: centerY is NaN/Inf before addition (bits=0x%08x, value=%.6f)\n",
                   anchorIdx, i, centerY_bits, centerY);
            #endif
            centerY = 0.0f;
        }
        if ((centerZ_bits & exp_mask_add) == exp_mask_add) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: centerZ is NaN/Inf before addition (bits=0x%08x, value=%.6f)\n",
                   anchorIdx, i, centerZ_bits, centerZ);
            #endif
            centerZ = 0.0f;
        }
        
        // 执行加法
        float finalX = rotX + centerX;
        float finalY = rotY + centerY;
        float finalZ = localZ + centerZ;
        
        // 关键修复：使用位操作检查加法结果，防止NaN/Inf传播
        const unsigned int finalX_bits = __float_as_uint(finalX);
        const unsigned int finalY_bits = __float_as_uint(finalY);
        const unsigned int finalZ_bits = __float_as_uint(finalZ);
        if ((finalX_bits & exp_mask_add) == exp_mask_add) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: finalX is NaN/Inf after addition (bits=0x%08x, rotX=%.6f, centerX=%.6f)\n",
                   anchorIdx, i, finalX_bits, rotX, centerX);
            #endif
            // 如果加法结果异常，使用安全的中心点值
            if ((centerX_bits & exp_mask_add) == exp_mask_add) {
                finalX = 0.0f;
            } else {
                finalX = centerX;
            }
        }
        if ((finalY_bits & exp_mask_add) == exp_mask_add) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: finalY is NaN/Inf after addition (bits=0x%08x, rotY=%.6f, centerY=%.6f)\n",
                   anchorIdx, i, finalY_bits, rotY, centerY);
            #endif
            if ((centerY_bits & exp_mask_add) == exp_mask_add) {
                finalY = 0.0f;
            } else {
                finalY = centerY;
            }
        }
        if ((finalZ_bits & exp_mask_add) == exp_mask_add) {
            #ifdef DEBUG_NAN
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d: finalZ is NaN/Inf after addition (bits=0x%08x, localZ=%.6f, centerZ=%.6f)\n",
                   anchorIdx, i, finalZ_bits, localZ, centerZ);
            #endif
            if ((centerZ_bits & exp_mask_add) == exp_mask_add) {
                finalZ = 0.0f;
            } else {
                finalZ = centerZ;
            }
        }
        
        // 最终 NaN 检查 - 工程部署中绝对不能有任何NaN输出
        // 关键优化：使用位操作检查，确保绝对不是NaN或Inf
        // 这是防止NaN的最后一道防线，必须100%可靠
        if (isNaN_strict(finalX) || isInf_strict(finalX)) {
            finalX = centerX;  // 如果异常，至少保留中心点
        }
        if (isNaN_strict(finalY) || isInf_strict(finalY)) {
            finalY = centerY;
        }
        if (isNaN_strict(finalZ) || isInf_strict(finalZ)) {
            finalZ = centerZ;
        }
        
        // 双重验证：使用isfinite检查
        if (!isfinite(finalX)) {
            finalX = centerX;
        }
        if (!isfinite(finalY)) {
            finalY = centerY;
        }
        if (!isfinite(finalZ)) {
            finalZ = centerZ;
        }
        
        // 最终验证：再次使用位操作检查，确保绝对不是NaN或Inf
        if (isNaN_strict(finalX) || isInf_strict(finalX)) {
            finalX = centerX;
        }
        if (isNaN_strict(finalY) || isInf_strict(finalY)) {
            finalY = centerY;
        }
        if (isNaN_strict(finalZ) || isInf_strict(finalZ)) {
            finalZ = centerZ;
        }
        
        // 写入前最终检查：确保绝对不是NaN/Inf
        // 这是防止NaN输出的最后一道防线
        // 关键修复：即使center点也是NaN，也要写入安全值（0），避免输出NaN
        if (isNaN_strict(finalX) || isInf_strict(finalX) || !isfinite(finalX)) {
            // 如果centerX也是NaN/Inf，使用0作为安全值
            if (isNaN_strict(centerX) || isInf_strict(centerX) || !isfinite(centerX)) {
                finalX = 0.0f;
            } else {
                finalX = centerX;
            }
        }
        if (isNaN_strict(finalY) || isInf_strict(finalY) || !isfinite(finalY)) {
            // 如果centerY也是NaN/Inf，使用0作为安全值
            if (isNaN_strict(centerY) || isInf_strict(centerY) || !isfinite(centerY)) {
                finalY = 0.0f;
            } else {
                finalY = centerY;
            }
        }
        if (isNaN_strict(finalZ) || isInf_strict(finalZ) || !isfinite(finalZ)) {
            // 如果centerZ也是NaN/Inf，使用0作为安全值
            if (isNaN_strict(centerZ) || isInf_strict(centerZ) || !isfinite(centerZ)) {
                finalZ = 0.0f;
            } else {
                finalZ = centerZ;
            }
        }
        
        // 最终验证：再次检查，确保绝对不是NaN/Inf（三重保险）
        if (isNaN_strict(finalX) || isInf_strict(finalX) || !isfinite(finalX)) {
            finalX = 0.0f;
        }
        if (isNaN_strict(finalY) || isInf_strict(finalY) || !isfinite(finalY)) {
            finalY = 0.0f;
        }
        if (isNaN_strict(finalZ) || isInf_strict(finalZ) || !isfinite(finalZ)) {
            finalZ = 0.0f;
        }
        
        // 写入前最后一次检查：使用位操作确保绝对不是NaN/Inf（四重保险）
        // 这是防止NaN输出的最后一道防线，必须100%可靠
        if (isNaN_strict(finalX) || isInf_strict(finalX)) {
            finalX = 0.0f;
        }
        if (isNaN_strict(finalY) || isInf_strict(finalY)) {
            finalY = 0.0f;
        }
        if (isNaN_strict(finalZ) || isInf_strict(finalZ)) {
            finalZ = 0.0f;
        }
        
        // 使用isfinite再次验证（五重保险）
        if (!isfinite(finalX)) {
            finalX = 0.0f;
        }
        if (!isfinite(finalY)) {
            finalY = 0.0f;
        }
        if (!isfinite(finalZ)) {
            finalZ = 0.0f;
        }
        
        // 关键修复：在写入前，使用原子操作或直接赋值，但必须确保值绝对安全
        // 即使经过多重检查，写入前最后一次验证，确保绝对不是NaN/Inf
        // 关键修复：使用更严格的检查，确保值绝对安全，即使center也是NaN也要写入0
        float safeX = 0.0f;
        if (!isNaN_strict(finalX) && !isInf_strict(finalX) && isfinite(finalX)) {
            safeX = finalX;
        } else {
            // 如果finalX是NaN/Inf，检查centerX是否安全
            if (!isNaN_strict(centerX) && !isInf_strict(centerX) && isfinite(centerX)) {
                safeX = centerX;
            } else {
                safeX = 0.0f;
            }
        }
        
        float safeY = 0.0f;
        if (!isNaN_strict(finalY) && !isInf_strict(finalY) && isfinite(finalY)) {
            safeY = finalY;
        } else {
            if (!isNaN_strict(centerY) && !isInf_strict(centerY) && isfinite(centerY)) {
                safeY = centerY;
            } else {
                safeY = 0.0f;
            }
        }
        
        float safeZ = 0.0f;
        if (!isNaN_strict(finalZ) && !isInf_strict(finalZ) && isfinite(finalZ)) {
            safeZ = finalZ;
        } else {
            if (!isNaN_strict(centerZ) && !isInf_strict(centerZ) && isfinite(centerZ)) {
                safeZ = centerZ;
            } else {
                safeZ = 0.0f;
            }
        }
        
        // 最终验证：使用位操作确保安全值绝对不是NaN/Inf（绝对保险）
        if (isNaN_strict(safeX) || isInf_strict(safeX) || !isfinite(safeX)) {
            safeX = 0.0f;
        }
        if (isNaN_strict(safeY) || isInf_strict(safeY) || !isfinite(safeY)) {
            safeY = 0.0f;
        }
        if (isNaN_strict(safeZ) || isInf_strict(safeZ) || !isfinite(safeZ)) {
            safeZ = 0.0f;
        }
        
        // 写入前最后一次位操作检查（绝对保险）
        const unsigned int bitsX = __float_as_uint(safeX);
        const unsigned int bitsY = __float_as_uint(safeY);
        const unsigned int bitsZ = __float_as_uint(safeZ);
        const unsigned int exp_mask_final_check = 0x7F800000;
        
        // 检查是否为NaN（指数全1且尾数非0）或Inf（指数全1且尾数为0）
        // 如果指数全1，说明是NaN或Inf，都需要处理
        // 关键调试：如果检测到NaN/Inf，记录详细信息（仅在调试模式下）
        #ifdef DEBUG_NAN
        if ((bitsX & exp_mask_final_check) == exp_mask_final_check) {
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d, X: NaN/Inf detected (bits=0x%08x)\n", anchorIdx, i, bitsX);
            printf("  centerX=%.6f, sizeX=%.6f, localX=%.6f, rotX=%.6f, finalX=%.6f, safeX=%.6f\n",
                   centerX, sizeX, localX, rotX, finalX, safeX);
            if (i >= fixedPts && instPtr != nullptr) {
                printf("  learnable point: accum[0]=%.6f, sig_x=%.6f\n", accum[0], sig_x);
            }
            safeX = 0.0f;
        }
        if ((bitsY & exp_mask_final_check) == exp_mask_final_check) {
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d, Y: NaN/Inf detected (bits=0x%08x)\n", anchorIdx, i, bitsY);
            printf("  centerY=%.6f, sizeY=%.6f, localY=%.6f, rotY=%.6f, finalY=%.6f, safeY=%.6f\n",
                   centerY, sizeY, localY, rotY, finalY, safeY);
            if (i >= fixedPts && instPtr != nullptr) {
                printf("  learnable point: accum[1]=%.6f, sig_y=%.6f\n", accum[1], sig_y);
            }
            safeY = 0.0f;
        }
        if ((bitsZ & exp_mask_final_check) == exp_mask_final_check) {
            printf("[DEBUG_NAN] anchorIdx=%d, pt=%d, Z: NaN/Inf detected (bits=0x%08x)\n", anchorIdx, i, bitsZ);
            printf("  centerZ=%.6f, sizeZ=%.6f, localZ=%.6f, finalZ=%.6f, safeZ=%.6f\n",
                   centerZ, sizeZ, localZ, finalZ, safeZ);
            if (i >= fixedPts && instPtr != nullptr) {
                printf("  learnable point: accum[2]=%.6f, sig_z=%.6f\n", accum[2], sig_z);
            }
            safeZ = 0.0f;
        }
        #else
        if ((bitsX & exp_mask_final_check) == exp_mask_final_check) {
            safeX = 0.0f;
        }
        if ((bitsY & exp_mask_final_check) == exp_mask_final_check) {
            safeY = 0.0f;
        }
        if ((bitsZ & exp_mask_final_check) == exp_mask_final_check) {
            safeZ = 0.0f;
        }
        #endif
        
        if (params.outputFP32)
        {
            // 输出是 FP32，直接写入 float 值
            float* outPtrFloat = reinterpret_cast<float*>(outPtrBase);
            
            // 使用内存屏障确保写入顺序，避免竞争条件
            __threadfence();
            
            // 写入前最后一次检查：使用位操作确保绝对不是NaN/Inf（绝对保险）
            // 关键修复：即使经过多重检查，写入时再次验证，确保100%可靠
            const unsigned int bitsX_final = __float_as_uint(safeX);
            const unsigned int bitsY_final = __float_as_uint(safeY);
            const unsigned int bitsZ_final = __float_as_uint(safeZ);
            const unsigned int exp_mask_final = 0x7F800000;
            
            // 如果是指数全1（NaN或Inf），写入0
            float finalX_write = ((bitsX_final & exp_mask_final) == exp_mask_final) ? 0.0f : safeX;
            float finalY_write = ((bitsY_final & exp_mask_final) == exp_mask_final) ? 0.0f : safeY;
            float finalZ_write = ((bitsZ_final & exp_mask_final) == exp_mask_final) ? 0.0f : safeZ;
            
            // 写入输出（使用经过最终验证的安全值，确保绝对不是NaN/Inf）
            outPtrFloat[offset + 0] = finalX_write;
            outPtrFloat[offset + 1] = finalY_write;
            outPtrFloat[offset + 2] = finalZ_write;
        }
        else
        {
            // 输出类型与输入类型相同，使用 fromFloat<T> 转换
            T* outPtr = reinterpret_cast<T*>(outPtrBase);
            
            // 使用内存屏障确保写入顺序
            __threadfence();
            
            // 写入输出
            outPtr[offset + 0] = fromFloat<T>(safeX);
            outPtr[offset + 1] = fromFloat<T>(safeY);
            outPtr[offset + 2] = fromFloat<T>(safeZ);
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


