#include "UtilsGPU_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <math.h>
#include <device_atomic_functions.h>
#include <cstdint>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

// 确保fmaxf在设备代码中可用
#ifdef __CUDA_ARCH__
#define DEVICE_MAX(a, b) fmaxf(a, b)
#else
#define DEVICE_MAX(a, b) std::max(a, b)
#endif

// ==================== CUDA Kernel实现 ====================
// 获取前K个实例
__global__ void getTopkInstanceKernel(const half* confidence,
                                              const half* instance_feature,
                                              const half* anchor,
                                              half* output_confidence,
                                              half* output_instance_feature,
                                              half* output_anchor,
                                              int32_t* output_track_ids,
                                              uint32_t num_querys,
                                              uint32_t query_dims,
                                              uint32_t embedfeat_dims,
                                              uint32_t num_topk_querys) {
    
    __shared__ float shared_confidence[256];
    __shared__ uint32_t shared_indices[256];
    
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_querys) {
        shared_confidence[tid] = __half2float(confidence[idx]);
        shared_indices[tid] = idx;
    } else {
        shared_confidence[tid] = -INFINITY;
        shared_indices[tid] = 0xFFFFFFFF;
    }
    
    __syncthreads();
    
    // 使用warp shuffle进行并行归约找最大值
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_conf = __shfl_down_sync(0xffffffff, shared_confidence[tid], offset);
        uint32_t other_idx = __shfl_down_sync(0xffffffff, shared_indices[tid], offset);
        
        if (other_conf > shared_confidence[tid]) {
            shared_confidence[tid] = other_conf;
            shared_indices[tid] = other_idx;
        }
    }
    
    // 只有warp的第一个线程执行输出
    if (tid == 0 && blockIdx.x < num_topk_querys) {
        uint32_t selected_idx = shared_indices[0];
        
        // 逐元素复制特征数据
        for (uint32_t i = 0; i < embedfeat_dims; ++i) {
            output_instance_feature[blockIdx.x * embedfeat_dims + i] = 
                instance_feature[selected_idx * embedfeat_dims + i];
        }
        
        // 锚点数据逐元素复制
        for (uint32_t i = 0; i < query_dims; ++i) {
            output_anchor[blockIdx.x * query_dims + i] = 
                anchor[selected_idx * query_dims + i];
        }
        
        output_confidence[blockIdx.x] = __float2half_rn(shared_confidence[0]);
        output_track_ids[blockIdx.x] = selected_idx;
    }
}

// 生成TrackId
__global__ void getTrackIdKernel(const half* confidence,
                                 int32_t* output_track_ids,
                                 uint32_t num_querys,
                                 uint32_t num_topk_querys) {
    
    __shared__ float shared_confidence[256];
    __shared__ uint32_t shared_indices[256];
    
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_querys) {
        shared_confidence[tid] = __half2float(confidence[idx]);
        shared_indices[tid] = idx;
    } else {
        shared_confidence[tid] = -INFINITY;
        shared_indices[tid] = 0xFFFFFFFF;
    }
    
    __syncthreads();
    
    // 并行归约找最大值
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_conf = __shfl_down_sync(0xffffffff, shared_confidence[tid], offset);
        uint32_t other_idx = __shfl_down_sync(0xffffffff, shared_indices[tid], offset);
        
        if (other_conf > shared_confidence[tid]) {
            shared_confidence[tid] = other_conf;
            shared_indices[tid] = other_idx;
        }
    }
    
    if (tid == 0 && blockIdx.x < num_topk_querys) {
        output_track_ids[blockIdx.x] = shared_indices[0];
    }
}

// 缓存特征
__global__ void cacheFeatureKernel(const half* instance_feature,
                                           const half* anchor,
                                           const half* confidence,
                                           const int32_t* track_ids,
                                           half* cached_feature,
                                           half* cached_anchor,
                                           half* cached_confidence,
                                           int32_t* cached_track_ids,
                                           uint32_t num_querys,
                                           uint32_t query_dims,
                                           uint32_t embedfeat_dims) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_querys) return;
    
    // 逐元素复制特征数据
    for (uint32_t i = 0; i < embedfeat_dims; ++i) {
        cached_feature[idx * embedfeat_dims + i] = instance_feature[idx * embedfeat_dims + i];
    }
    
    // 逐元素复制锚点数据
    for (uint32_t i = 0; i < query_dims; ++i) {
        cached_anchor[idx * query_dims + i] = anchor[idx * query_dims + i];
    }
    
    cached_confidence[idx] = confidence[idx];
    cached_track_ids[idx] = track_ids[idx];
}

// 置信度衰减
__global__ void applyConfidenceDecayKernel(const half* input_confidence,
                                            half* output_confidence,
                                            half decay_factor,
                                            uint32_t array_size) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < array_size) {
        float x = __half2float(input_confidence[tid]);
        float d = __half2float(decay_factor);
        output_confidence[tid] = __float2half_rn(x * d);
    }
}

// 修改后的核函数
__global__ void fuseConfidenceKernel(half* output_confidence,
                                     const half* current_confidence,
                                     const half* cached_confidence,
                                     uint32_t size) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        float a = __half2float(current_confidence[tid]);
        float b = __half2float(cached_confidence[tid]);
        output_confidence[tid] = __float2half_rn(fmaxf(a, b));
    }
}

// 锚点投影
__global__ void anchorProjectionKernel(half* temp_anchor,
                                      Mat4 transform,
                                      half time_interval,
                                      uint32_t topk_querys,
                                      uint32_t query_dims) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= topk_querys) return;
    
    // 使用共享内存缓存变换矩阵（从按值参数读取）
    __shared__ float shared_matrix[16];
    if (threadIdx.x < 16) {
        shared_matrix[threadIdx.x] = transform.m[threadIdx.x];
    }
    __syncthreads();
    
    half* anchor_ptr = temp_anchor + idx * query_dims;
    
    // 提取锚点数据
    float center_x = __half2float(anchor_ptr[0]);
    float center_y = __half2float(anchor_ptr[1]);
    float center_z = __half2float(anchor_ptr[2]);
    float size_x = __half2float(anchor_ptr[3]);
    float size_y = __half2float(anchor_ptr[4]);
    float size_z = __half2float(anchor_ptr[5]);
    float yaw_x = __half2float(anchor_ptr[6]);
    float yaw_y = __half2float(anchor_ptr[7]);
    float vel_x = __half2float(anchor_ptr[8]);
    float vel_y = __half2float(anchor_ptr[9]);
    float vel_z = __half2float(anchor_ptr[10]);
    
    // 时间投影
    float dt = __half2float(time_interval);
    float time_vel_x = vel_x * dt;
    float time_vel_y = vel_y * dt;
    float time_vel_z = vel_z * dt;
    
    center_x = center_x + time_vel_x;
    center_y = center_y + time_vel_y;
    center_z = center_z + time_vel_z;
    
    // 矩阵变换 - 使用转置后的矩阵（注意索引顺序）
    float new_center_x = center_x * shared_matrix[0] + 
                         center_y * shared_matrix[4] + 
                         center_z * shared_matrix[8] + 
                         shared_matrix[12];
    
    float new_center_y = center_x * shared_matrix[1] + 
                         center_y * shared_matrix[5] + 
                         center_z * shared_matrix[9] + 
                         shared_matrix[13];
    
    float new_center_z = center_x * shared_matrix[2] + 
                         center_y * shared_matrix[6] + 
                         center_z * shared_matrix[10] + 
                         shared_matrix[14];
    
    // 速度变换 - 使用转置后的3x3矩阵
    float new_vel_x = vel_x * shared_matrix[0] + 
                      vel_y * shared_matrix[4] + 
                      vel_z * shared_matrix[8];
    
    float new_vel_y = vel_x * shared_matrix[1] + 
                      vel_y * shared_matrix[5] + 
                      vel_z * shared_matrix[9];
    
    float new_vel_z = vel_x * shared_matrix[2] + 
                      vel_y * shared_matrix[6] + 
                      vel_z * shared_matrix[10];
    
    // Yaw变换 - 先交换列，然后使用转置后的2x2矩阵
    float temp_yaw_x = yaw_x;
    float temp_yaw_y = yaw_y;
    yaw_x = temp_yaw_y;  // 交换
    yaw_y = temp_yaw_x;  // 交换
    
    float new_yaw_x = yaw_x * shared_matrix[0] + 
                      yaw_y * shared_matrix[4];
    
    float new_yaw_y = yaw_x * shared_matrix[1] + 
                      yaw_y * shared_matrix[5];
    
    // 更新锚点数据
    anchor_ptr[0] = __float2half_rn(new_center_x);
    anchor_ptr[1] = __float2half_rn(new_center_y);
    anchor_ptr[2] = __float2half_rn(new_center_z);
    anchor_ptr[3] = __float2half_rn(size_x);
    anchor_ptr[4] = __float2half_rn(size_y);
    anchor_ptr[5] = __float2half_rn(size_z);
    anchor_ptr[6] = __float2half_rn(new_yaw_x);
    anchor_ptr[7] = __float2half_rn(new_yaw_y);
    anchor_ptr[8] = __float2half_rn(new_vel_x);
    anchor_ptr[9] = __float2half_rn(new_vel_y);
    anchor_ptr[10] = __float2half_rn(new_vel_z);
}

__global__ void getMaxConfidenceScoresKernel(const half* confidence_logits,
                                             half* max_confidence_scores,
                                             uint32_t num_querys,
                                             uint32_t num_classes) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_querys) return;

    // 与CPU版本保持一致的初始化
    float max_logit = __half2float(confidence_logits[idx * num_classes]);
    for (uint32_t class_idx = 1; class_idx < num_classes; ++class_idx) {
        float logit = __half2float(confidence_logits[idx * num_classes + class_idx]);
        max_logit = fmaxf(max_logit, logit);
    }
    
    // 应用sigmoid函数（保持数值稳定版本）
    float sigmoid_value;
    if (max_logit > 0) {
        float exp_neg_x = __expf(-max_logit);
        sigmoid_value = __frcp_rn(__fadd_rn(1.0f, exp_neg_x));
    } else {
        float exp_x = __expf(max_logit);
        sigmoid_value = __fdiv_rn(exp_x, __fadd_rn(1.0f, exp_x));
    }
    
    max_confidence_scores[idx] = __float2half_rn(sigmoid_value);
}

// Sigmoid
__global__ void applySigmoidKernel(half* logits, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = __half2float(logits[idx]);
    
    // 使用快速数学函数和数值稳定性优化
    float y;
    if (x > 0) {
        float exp_neg_x = __expf(-x);
        y = __frcp_rn(__fadd_rn(1.0f, exp_neg_x));
    } else {
        float exp_x = __expf(x);
        y = __fdiv_rn(exp_x, __fadd_rn(1.0f, exp_x));
    }
    logits[idx] = __float2half_rn(y);
}

// 生成新TrackId
__global__ void generateNewTrackIdsKernel(int32_t* track_ids,
                                            const int32_t* prev_id,
                                            uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // 使用向量化操作
    track_ids[idx] = prev_id[0] + idx;
}

// 按索引选择TrackId并超出填-1
__global__ void selectTrackIdsFromIndicesKernel(int32_t* track_ids,
                                                        const int32_t* cached_indices,
                                                        uint32_t target_length,
                                                        uint32_t source_length) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < target_length) {
        if (tid < source_length) {
            // 前source_length个位置：使用缓存的索引
            int32_t index = cached_indices[tid];
            // 检查索引有效性
            if (index >= 0 ) {
                track_ids[tid] = track_ids[index];
            } else {
                track_ids[tid] = -1;  // 索引无效时设为-1
            }
        } else {
            // 超出source_length范围的位置填充-1
            track_ids[tid] = -1;
        }
    }
}

// 安全的kernel实现
__global__ void selectTrackIdsFromIndicesKernelSafe(int32_t* output_track_ids,
                                                   const int32_t* input_track_ids,
                                                   const int32_t* cached_indices,
                                                   uint32_t source_length,
                                                   uint32_t topk) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < source_length) {
        if (tid < topk) {
            // 前topk个位置：根据cached_indices中的索引重新赋值
            int32_t original_index = cached_indices[tid];
            // 检查索引有效性
            if (original_index >= 0 && original_index < source_length) {
                // 从原始数据读取，避免冲突
                output_track_ids[tid] = input_track_ids[original_index];
            } else {
                output_track_ids[tid] = -1;  // 索引无效时设为-1
            }
        } else {
            // 超出topk范围的位置填充-1
            output_track_ids[tid] = -1;
        }
    }
}

__global__ void updateNewTrackIdsKernel(int32_t* track_ids,
                                        uint32_t num_anchors_,
                                        uint32_t topk_anchors_,
                                        int32_t* prev_id) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 只有最后topk_anchors_个新目标需要赋予新的id
    if (tid < num_anchors_ && tid >= topk_anchors_) {
        track_ids[tid] = prev_id[0] + tid - topk_anchors_ + 1;
    }
}

// 数据复制kernel
__global__ void copyTopKDataFromIndicesKernel(const half* instance_feature,
                                              const half* anchor,
                                              const half* confidence,
                                              const uint32_t* topk_indices,
                                              half* output_confidence,
                                              half* output_instance_feature,
                                              half* output_anchor,
                                              int32_t* output_track_ids,
                                              uint32_t query_dims,
                                              uint32_t embedfeat_dims,
                                              uint32_t k) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    
    uint32_t selected_idx = topk_indices[idx];
    
    // 复制置信度
    output_confidence[idx] = confidence[selected_idx];
    
    // 复制track_id（索引值）
    output_track_ids[idx] = selected_idx;
    
    // 逐元素复制特征数据（half安全）
    for (uint32_t i = 0; i < embedfeat_dims; ++i) {
        output_instance_feature[idx * embedfeat_dims + i] = 
            instance_feature[selected_idx * embedfeat_dims + i];
    }
    
    // 锚点数据（query_dims=11，不能被4整除）- 使用逐元素复制
    for (uint32_t i = 0; i < query_dims; ++i) {
        output_anchor[idx * query_dims + i] = 
            anchor[selected_idx * query_dims + i];
    }
}

// half -> float 转换内核用于 Thrust 排序
__global__ void half_to_float_kernel(const half* in, float* out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// ==================== C++接口函数实现 ====================
namespace UtilsGPU {

// 注释：保留占位但不使用非 Thrust Top-K 实现
// Status getTopkInstanceOnGPU(const CudaWrapper<half>& confidence,
//                             const CudaWrapper<half>& instance_feature,
//                             const CudaWrapper<half>& anchor,
//                             const uint32_t& num_querys,
//                             const uint32_t& query_dims,
//                             const uint32_t& embedfeat_dims,
//                             const uint32_t& num_topk_querys,
//                             CudaWrapper<half>& output_confidence,
//                             CudaWrapper<half>& output_instance_feature,
//                             CudaWrapper<half>& output_anchor,
//                             CudaWrapper<int32_t>& output_track_ids,
//                             cudaStream_t stream) {
//     
//     uint32_t block_size = 256;
//     uint32_t grid_size = (num_querys + block_size - 1) / block_size;
//
//     getTopkInstanceKernel<<<grid_size, block_size, 0, stream>>>(
//         confidence.getCudaPtr(), instance_feature.getCudaPtr(), anchor.getCudaPtr(),
//         output_confidence.getCudaPtr(), output_instance_feature.getCudaPtr(),
//         output_anchor.getCudaPtr(), output_track_ids.getCudaPtr(),
//         num_querys, query_dims, embedfeat_dims, num_topk_querys
//     );
//     
//     if (!checkCudaError(cudaGetLastError())) {
//         return kInferenceErr;
//     }
//     return kSuccess;
// }

Status getTrackIdOnGPU(const CudaWrapper<half>& confidence,
                       const uint32_t& num_querys,
                       const uint32_t& num_topk_querys,
                       CudaWrapper<int32_t>& output_track_ids,
                       cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (num_querys + block_size - 1) / block_size;

    getTrackIdKernel<<<grid_size, block_size, 0, stream>>>(
        confidence.getCudaPtr(), output_track_ids.getCudaPtr(),
        num_querys, num_topk_querys
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status cacheFeatureOnGPU(const CudaWrapper<half>& instance_feature,
                         const CudaWrapper<half>& anchor,
                         const CudaWrapper<half>& confidence,
                         const CudaWrapper<int32_t>& track_ids,
                         const uint32_t& num_querys,
                         const uint32_t& query_dims,
                         const uint32_t& embedfeat_dims,
                         CudaWrapper<half>& cached_feature,
                         CudaWrapper<half>& cached_anchor,
                         CudaWrapper<half>& cached_confidence,
                         CudaWrapper<int32_t>& cached_track_ids,
                         cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (num_querys + block_size - 1) / block_size;

    cacheFeatureKernel<<<grid_size, block_size, 0, stream>>>(
        instance_feature.getCudaPtr(), anchor.getCudaPtr(), confidence.getCudaPtr(),
        track_ids.getCudaPtr(), cached_feature.getCudaPtr(), cached_anchor.getCudaPtr(),
        cached_confidence.getCudaPtr(), cached_track_ids.getCudaPtr(),
        num_querys, query_dims, embedfeat_dims
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status applyConfidenceDecayOnGPU(const CudaWrapper<half>& input_confidence,
                                 CudaWrapper<half>& output_confidence,
                                 half decay_factor,
                                 uint32_t array_size,
                                 cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (array_size + block_size - 1) / block_size;

    applyConfidenceDecayKernel<<<grid_size, block_size, 0, stream>>>(
        input_confidence.getCudaPtr(), output_confidence.getCudaPtr(), decay_factor, array_size
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

// 修改后的接口函数
Status fuseConfidenceOnGPU(const CudaWrapper<half>& current_confidence,
                           const CudaWrapper<half>& cached_confidence,
                           CudaWrapper<half>& output_confidence,
                           uint32_t fusion_length,
                           cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (fusion_length + block_size - 1) / block_size;

    fuseConfidenceKernel<<<grid_size, block_size, 0, stream>>>(
        output_confidence.getCudaPtr(),    // 输出：融合后的置信度
        current_confidence.getCudaPtr(),   // 输入：当前置信度
        cached_confidence.getCudaPtr(),    // 输入：缓存的置信度
        fusion_length
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

// Status anchorProjectionOnGPU(CudaWrapper<float>& temp_anchor,
//                              const Eigen::Matrix<float, 4, 4>& temp_to_cur_mat,
//                              float time_interval,
//                              uint32_t topk_querys,
//                              uint32_t query_dims,
//                              cudaStream_t stream) {
    
//     uint32_t block_size = 256;
//     uint32_t grid_size = (topk_querys + block_size - 1) / block_size;

//     // 将主机侧矩阵复制到设备缓冲区，避免将主机指针传入设备代码
//     float* d_transform_matrix = nullptr;
//     cudaError_t err_alloc = cudaMalloc((void**)&d_transform_matrix, 16 * sizeof(float));
//     if (err_alloc != cudaSuccess) {
//         return kInferenceErr;
//     }
//     cudaError_t err_copy = cudaMemcpyAsync(d_transform_matrix, temp_to_cur_mat.data(), 16 * sizeof(float), cudaMemcpyHostToDevice, stream);
//     if (err_copy != cudaSuccess) {
//         cudaFree(d_transform_matrix);
//         return kInferenceErr;
//     }

//     anchorProjectionKernel<<<grid_size, block_size, 0, stream>>>(
//         temp_anchor.getCudaPtr(), d_transform_matrix, time_interval, topk_querys, query_dims
//     );
    
//     // 检查并同步，确保内核完成后再释放矩阵缓冲
//     cudaError_t launch_err = cudaPeekAtLastError();
//     if (launch_err != cudaSuccess) {
//         cudaFree(d_transform_matrix);
//         return kInferenceErr;
//     }
//     cudaError_t sync_err = cudaStreamSynchronize(stream);
//     cudaFree(d_transform_matrix);
//     if (sync_err != cudaSuccess) {
//         return kInferenceErr;
//     }
    
//     if (!checkCudaError(cudaGetLastError())) {
//         return kInferenceErr;
//     }
//     return kSuccess;
// }

Status anchorProjectionOnGPU(CudaWrapper<half>& temp_anchor,
                             const Eigen::Matrix<half, 4, 4>& temp_to_cur_mat,
                             half time_interval,
                             uint32_t topk_querys,
                             uint32_t query_dims,
                             cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (topk_querys + block_size - 1) / block_size;

    // 将矩阵按值传给kernel，避免设备侧alloc/copy/sync
    Mat4 hmat;
    const half* src = temp_to_cur_mat.data();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        hmat.m[i] = __half2float(src[i]);
    }

    anchorProjectionKernel<<<grid_size, block_size, 0, stream>>>(
        temp_anchor.getCudaPtr(), hmat, time_interval, topk_querys, query_dims
    );
    
    // 仅检查启动错误，不阻塞流
    if (!checkCudaError(cudaPeekAtLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status getMaxConfidenceScoresOnGPU(const CudaWrapper<half>& confidence_logits,
                                   CudaWrapper<half>& max_confidence_scores,
                                   const uint32_t& num_querys,
                                   const uint32_t& num_classes,
                                   cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (num_querys + block_size - 1) / block_size;

    getMaxConfidenceScoresKernel<<<grid_size, block_size, 0, stream>>>(
        confidence_logits.getCudaPtr(), max_confidence_scores.getCudaPtr(), num_querys, num_classes
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status applySigmoidOnGPU(CudaWrapper<half>& logits,
                         uint32_t size,
                         cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (size + block_size - 1) / block_size;

    applySigmoidKernel<<<grid_size, block_size, 0, stream>>>(
        logits.getCudaPtr(), size
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status generateNewTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                const CudaWrapper<int32_t>& prev_id,
                                uint32_t size,
                                cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (size + block_size - 1) / block_size;

    generateNewTrackIdsKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), prev_id.getCudaPtr(), size
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

// Status selectTrackIdsFromIndicesOnGPU(CudaWrapper<int32_t>& track_ids,
//                                       const CudaWrapper<int32_t>& cached_indices,
//                                       uint32_t topk,
//                                       cudaStream_t stream) {
//     uint32_t source_length = track_ids.getSize();
    
//     // 创建临时数组存储原始数据
//     CudaWrapper<int32_t> temp_track_ids(source_length);
    
//     // 复制原始数据到临时数组
//     cudaError_t copy_error = cudaMemcpyAsync(temp_track_ids.getCudaPtr(), 
//                                             track_ids.getCudaPtr(),
//                                             source_length * sizeof(int32_t),
//                                             cudaMemcpyDeviceToDevice, stream);
//     if (copy_error != cudaSuccess) {
//         return kInferenceErr;
//     }

//     uint32_t block_size = 256;
//     uint32_t grid_size = (source_length + block_size - 1) / block_size;

//     // 修改kernel调用，使用临时数组作为源数据
//     selectTrackIdsFromIndicesKernelSafe<<<grid_size, block_size, 0, stream>>>(
//         track_ids.getCudaPtr(),            // 输出：重排序后的数组
//         temp_track_ids.getCudaPtr(),       // 输入：原始数据的副本
//         cached_indices.getCudaPtr(),       // 输入：索引数组
//         source_length,
//         topk
//     );
    
//     if (!checkCudaError(cudaGetLastError())) {
//         return kInferenceErr;
//     }
//     return kSuccess;
// }

Status selectTrackIdsFromIndicesOnGPU(CudaWrapper<int32_t>& track_ids,
                                      const CudaWrapper<int32_t>& cached_indices,
                                      uint32_t topk,
                                      cudaStream_t stream) {
    uint32_t source_length = track_ids.getSize();

    // 临时缓冲：使用流有序分配/释放，避免隐式全局同步
    int32_t* temp_track_ids = nullptr;
    if (cudaMallocAsync((void**)&temp_track_ids, source_length * sizeof(int32_t), stream) != cudaSuccess) {
        return kInferenceErr;
    }

    // D2D 异步拷贝，仍在同一 stream 上
    if (cudaMemcpyAsync(temp_track_ids,
                        track_ids.getCudaPtr(),
                        source_length * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        cudaFreeAsync(temp_track_ids, stream);
        return kInferenceErr;
    }

    uint32_t block_size = 256;
    uint32_t grid_size = (source_length + block_size - 1) / block_size;

    selectTrackIdsFromIndicesKernelSafe<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(),            // 输出
        temp_track_ids,                    // 输入：原始数据副本
        cached_indices.getCudaPtr(),       // 输入：索引数组
        source_length,
        topk
    );

    // 启动错误检查（不阻塞流）
    if (!checkCudaError(cudaPeekAtLastError())) {
        cudaFreeAsync(temp_track_ids, stream);
        return kInferenceErr;
    }

    // 释放按流顺序进行，不会提前释放
    cudaFreeAsync(temp_track_ids, stream);
    return kSuccess;
}

bool checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        return false;
    }
    return true;
}

Status updateNewTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                uint32_t num_anchors_,
                                uint32_t topk_anchors_,
                                CudaWrapper<int32_t>& prev_id,
                                cudaStream_t stream)
{
    uint32_t block_size = 256;
    uint32_t grid_size = (num_anchors_ + block_size - 1) / block_size;

    updateNewTrackIdsKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), num_anchors_, topk_anchors_, prev_id.getCudaPtr()
    );
    
    // 检查kernel启动错误
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    
    // 同步流确保kernel执行完成
    cudaError_t sync_error = cudaStreamSynchronize(stream);
    if (sync_error != cudaSuccess) {
        return kInferenceErr;
    }
    
    // 更新prev_id值：prev_id += (num_anchors_ - topk_anchors_)
    uint32_t new_ids_count = num_anchors_ - topk_anchors_;
    if (new_ids_count > 0) {
        // 从GPU读取当前prev_id值
        std::vector<int32_t> current_prev_id = prev_id.cudaMemcpyD2HResWrap();
        int32_t updated_prev_id = current_prev_id[0] + new_ids_count;
        // std::cout << "updated_prev_id : " << updated_prev_id << "current_prev_id : " << current_prev_id[0] << std::endl;
        
        // 将更新后的值写回GPU
        std::vector<int32_t> updated_prev_id_vec = {updated_prev_id};
        prev_id.cudaMemUpdateWrap(updated_prev_id_vec);
    }
    return kSuccess;
}

Status getTopkInstanceOnGPUThrust(const CudaWrapper<half>& confidence,
                                 const CudaWrapper<half>& instance_feature,
                                 const CudaWrapper<half>& anchor,
                                 const uint32_t& num_querys,
                                 const uint32_t& query_dims,
                                 const uint32_t& embedfeat_dims,
                                 const uint32_t& num_topk_querys,
                                 CudaWrapper<half>& output_confidence,
                                 CudaWrapper<half>& output_instance_feature,
                                 CudaWrapper<half>& output_anchor,
                                 CudaWrapper<int32_t>& output_track_ids,
                                 cudaStream_t stream) {
    // 创建索引-值对
    thrust::device_vector<float> d_values(num_querys);
    thrust::device_vector<uint32_t> d_indices(num_querys);
    
    // 初始化索引为原始索引
    thrust::sequence(thrust::cuda::par.on(stream), d_indices.begin(), d_indices.end());
    
    // 将half转换为float
    uint32_t bs = 256, gs = (num_querys + bs - 1) / bs;
    half_to_float_kernel<<<gs, bs, 0, stream>>>(confidence.getCudaPtr(), d_values.data().get(), num_querys);
    
    // 使用thrust进行稳定排序
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream), 
                       d_values.begin(), d_values.end(), 
                       d_indices.begin(), 
                       thrust::greater<float>());

    // 提取前k个结果
    uint32_t block_size = 256;
    uint32_t grid_size = (num_topk_querys + block_size - 1) / block_size;
    
    copyTopKDataFromIndicesKernel<<<grid_size, block_size, 0, stream>>>(
        instance_feature.getCudaPtr(),
        anchor.getCudaPtr(),
        confidence.getCudaPtr(),
        d_indices.data().get(),
        output_confidence.getCudaPtr(),
        output_instance_feature.getCudaPtr(),
        output_anchor.getCudaPtr(),
        output_track_ids.getCudaPtr(),
        query_dims,
        embedfeat_dims,
        num_topk_querys
    );
    
    if (!checkCudaError(cudaPeekAtLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

} // namespace UtilsGPU