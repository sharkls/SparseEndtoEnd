#include "UtilsGPU_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <math.h>
#include <device_atomic_functions.h>
#include <cstdint>

// 确保fmaxf在设备代码中可用
#ifdef __CUDA_ARCH__
#define DEVICE_MAX(a, b) fmaxf(a, b)
#else
#define DEVICE_MAX(a, b) std::max(a, b)
#endif

// ==================== CUDA Kernel实现 ====================
// 获取前K个实例
__global__ void getTopkInstanceKernel(const float* confidence,
                                              const float* instance_feature,
                                              const float* anchor,
                                              float* output_confidence,
                                              float* output_instance_feature,
                                              float* output_anchor,
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
        shared_confidence[tid] = confidence[idx];
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
        
        // 向量化复制特征数据（4个float一起处理）
        float4* src_feature = (float4*)&instance_feature[selected_idx * embedfeat_dims];
        float4* dst_feature = (float4*)&output_instance_feature[blockIdx.x * embedfeat_dims];
        
        for (uint32_t i = 0; i < embedfeat_dims / 4; ++i) {
            dst_feature[i] = src_feature[i];
        }
        
        // 向量化复制锚点数据
        float4* src_anchor = (float4*)&anchor[selected_idx * query_dims];
        float4* dst_anchor = (float4*)&output_anchor[blockIdx.x * query_dims];
        
        for (uint32_t i = 0; i < query_dims / 4; ++i) {
            dst_anchor[i] = src_anchor[i];
        }
        
        output_confidence[blockIdx.x] = shared_confidence[0];
        output_track_ids[blockIdx.x] = selected_idx;
    }
}

// 获取前K个分数与索引
__global__ void getTopKScoresKernel(const float* confidence,
                                            float* topk_confidence,
                                            uint32_t* topk_indices,
                                            uint32_t num_querys,
                                            uint32_t k) {
    
    __shared__ float shared_confidence[256];
    __shared__ uint32_t shared_indices[256];
    
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_querys) {
        shared_confidence[tid] = confidence[idx];
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
    
    if (tid == 0 && blockIdx.x < k) {
        topk_confidence[blockIdx.x] = shared_confidence[0];
        topk_indices[blockIdx.x] = shared_indices[0];
    }
}

// 生成TrackId
__global__ void getTrackIdKernel(const float* confidence,
                                 int32_t* output_track_ids,
                                 uint32_t num_querys,
                                 uint32_t num_topk_querys) {
    
    __shared__ float shared_confidence[256];
    __shared__ uint32_t shared_indices[256];
    
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_querys) {
        shared_confidence[tid] = confidence[idx];
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
__global__ void cacheFeatureKernel(const float* instance_feature,
                                           const float* anchor,
                                           const float* confidence,
                                           const int32_t* track_ids,
                                           float* cached_feature,
                                           float* cached_anchor,
                                           float* cached_confidence,
                                           int32_t* cached_track_ids,
                                           uint32_t num_querys,
                                           uint32_t query_dims,
                                           uint32_t embedfeat_dims) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_querys) return;
    
    // 向量化复制特征数据
    float4* src_feature = (float4*)&instance_feature[idx * embedfeat_dims];
    float4* dst_feature = (float4*)&cached_feature[idx * embedfeat_dims];
    
    for (uint32_t i = 0; i < embedfeat_dims / 4; ++i) {
        dst_feature[i] = src_feature[i];
    }
    
    // 向量化复制锚点数据
    float4* src_anchor = (float4*)&anchor[idx * query_dims];
    float4* dst_anchor = (float4*)&cached_anchor[idx * query_dims];
    
    for (uint32_t i = 0; i < query_dims / 4; ++i) {
        dst_anchor[i] = src_anchor[i];
    }
    
    cached_confidence[idx] = confidence[idx];
    cached_track_ids[idx] = track_ids[idx];
}

// 置信度衰减
__global__ void applyConfidenceDecayKernel(const float* input_confidence,
                                                   float* output_confidence,
                                                   float decay_factor,
                                                   uint32_t array_size) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < array_size) {
        // 使用快速数学函数
        output_confidence[tid] = __fmul_rn(input_confidence[tid], decay_factor);
    }
}

// 置信度融合
__global__ void fuseConfidenceKernel(float* current_confidence,
                                     const float* cached_confidence,
                                     uint32_t size) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        // 修正：使用fmaxf而不是__fmaxf
        current_confidence[tid] = fmaxf(current_confidence[tid], cached_confidence[tid]);
    }
}

// 锚点投影
__global__ void anchorProjectionKernel(float* temp_anchor,
                                               const float* transform_matrix,
                                               float time_interval,
                                               uint32_t topk_querys,
                                               uint32_t query_dims) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= topk_querys) return;
    
    // 使用共享内存缓存变换矩阵
    __shared__ float shared_matrix[16];
    if (threadIdx.x < 16) {
        shared_matrix[threadIdx.x] = transform_matrix[threadIdx.x];
    }
    __syncthreads();
    
    float* anchor_ptr = temp_anchor + idx * query_dims;
    
    // 提取锚点数据
    float center_x = anchor_ptr[0];
    float center_y = anchor_ptr[1];
    float center_z = anchor_ptr[2];
    float size_x = anchor_ptr[3];
    float size_y = anchor_ptr[4];
    float size_z = anchor_ptr[5];
    float yaw = anchor_ptr[6];
    float vel_x = anchor_ptr[7];
    float vel_y = anchor_ptr[8];
    float vel_z = anchor_ptr[9];
    
    // 时间投影优化
    float time_vel_x = __fmul_rn(vel_x, time_interval);
    float time_vel_y = __fmul_rn(vel_y, time_interval);
    float time_vel_z = __fmul_rn(vel_z, time_interval);
    
    center_x = __fsub_rn(center_x, time_vel_x);
    center_y = __fsub_rn(center_y, time_vel_y);
    center_z = __fsub_rn(center_z, time_vel_z);
    
    // 矩阵变换优化（使用共享内存）
    float new_center_x = __fmul_rn(center_x, shared_matrix[0]) + 
                         __fmul_rn(center_y, shared_matrix[1]) + 
                         __fmul_rn(center_z, shared_matrix[2]) + 
                         shared_matrix[3];
    
    float new_center_y = __fmul_rn(center_x, shared_matrix[4]) + 
                         __fmul_rn(center_y, shared_matrix[5]) + 
                         __fmul_rn(center_z, shared_matrix[6]) + 
                         shared_matrix[7];
    
    float new_center_z = __fmul_rn(center_x, shared_matrix[8]) + 
                         __fmul_rn(center_y, shared_matrix[9]) + 
                         __fmul_rn(center_z, shared_matrix[10]) + 
                         shared_matrix[11];
    
    // 更新锚点数据
    anchor_ptr[0] = new_center_x;
    anchor_ptr[1] = new_center_y;
    anchor_ptr[2] = new_center_z;
    anchor_ptr[3] = size_x;
    anchor_ptr[4] = size_y;
    anchor_ptr[5] = size_z;
    anchor_ptr[6] = yaw;
    anchor_ptr[7] = vel_x;
    anchor_ptr[8] = vel_y;
    anchor_ptr[9] = vel_z;
}

// 最大置信度
__global__ void getMaxConfidenceScoresKernel(const float* confidence_logits,
                                                     float* max_confidence_scores,
                                                     uint32_t num_querys,
                                                     uint32_t num_classes) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_querys) return;

    // 找到当前查询的最大置信度
    float max_conf = -INFINITY;
    for (uint32_t class_idx = 0; class_idx < num_classes; ++class_idx) {
        float conf = confidence_logits[idx * num_classes + class_idx];
        // 修正：使用fmaxf而不是__fmaxf
        max_conf = fmaxf(max_conf, conf);
    }
    
    max_confidence_scores[idx] = max_conf;
}

// Sigmoid
__global__ void applySigmoidKernel(float* logits, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = logits[idx];
    
    // 使用快速数学函数和数值稳定性优化
    if (x > 0) {
        float exp_neg_x = __expf(-x);
        logits[idx] = __frcp_rn(__fadd_rn(1.0f, exp_neg_x));
    } else {
        float exp_x = __expf(x);
        logits[idx] = __fdiv_rn(exp_x, __fadd_rn(1.0f, exp_x));
    }
}

// 生成新TrackId
__global__ void generateNewTrackIdsKernel(int32_t* track_ids,
                                                  const int32_t* prev_id,
                                                  uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // 使用向量化操作
    track_ids[idx] = prev_id[0] + idx + 1;
}

// 简易排序
__global__ void radixSortKernel(float* values,
                                uint32_t* indices,
                                uint32_t size) {
    
    __shared__ float shared_values[256];
    __shared__ uint32_t shared_indices[256];
    
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        shared_values[tid] = values[idx];
        shared_indices[tid] = idx;
    } else {
        shared_values[tid] = -INFINITY;
        shared_indices[tid] = 0xFFFFFFFF;
    }
    
    __syncthreads();
    
    // 使用warp shuffle进行并行排序
    for (int step = 0; step < 256; ++step) {
        bool should_swap = (tid % 2 == 0) ? 
            (shared_values[tid] < shared_values[tid + 1]) :
            (shared_values[tid - 1] < shared_values[tid]);
        
        if (should_swap && tid < 255) {
            float temp_val = shared_values[tid];
            uint32_t temp_idx = shared_indices[tid];
            
            shared_values[tid] = shared_values[tid + 1];
            shared_indices[tid] = shared_indices[tid + 1];
            
            shared_values[tid + 1] = temp_val;
            shared_indices[tid + 1] = temp_idx;
        }
        
        __syncthreads();
    }
    
    // 输出结果
    if (idx < size) {
        values[idx] = shared_values[tid];
        indices[idx] = shared_indices[tid];
    }
}

// 计数负值
__global__ void countNegativeValuesKernel(const int32_t* track_ids, 
                                                  uint32_t* negative_count,
                                                  uint32_t array_size) {
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个warp内部先归约
    uint32_t local_count = 0;
    if (idx < array_size && track_ids[idx] < 0) {
        local_count = 1;
    }
    
    // 使用warp shuffle进行归约
    for (int offset = 16; offset > 0; offset /= 2) {
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    
    // 只有warp的第一个线程执行原子操作
    if (tid == 0) {
        atomicAdd(negative_count, local_count);
    }
}

// 更新末尾N个TrackId
__global__ void updateLastNTrackIdsKernel(int32_t* track_ids,
                                                  const int32_t* new_track_ids,
                                                  uint32_t total_size,
                                                  uint32_t update_count) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 只处理最后update_count个位置
    if (tid < update_count) {
        uint32_t target_index = total_size - update_count + tid;
        track_ids[target_index] = new_track_ids[tid];
    }
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
            if (index < 0 || index >= source_length) {
                track_ids[tid] = -1;  // 索引无效时设为-1
            }
        } else {
            // 超出source_length范围的位置填充-1
            track_ids[tid] = -1;
        }
    }
}

// 添加缺失的kernel实现
__global__ void updateNegativeTrackIdsKernel(int32_t* track_ids,
                                             const int32_t* new_track_ids,
                                             uint32_t array_size) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < array_size) {
        if (track_ids[tid] < 0) {
            // 找到对应的新track ID
            uint32_t new_id_index = 0;
            for (uint32_t i = 0; i < array_size; ++i) {
                if (track_ids[i] < 0) {
                    if (i == tid) {
                        track_ids[tid] = new_track_ids[new_id_index];
                        break;
                    }
                    new_id_index++;
                }
            }
        }
    }
}

// ==================== C++接口函数实现 ====================
namespace UtilsGPU {

Status getTopkInstanceOnGPU(const CudaWrapper<float>& confidence,
                            const CudaWrapper<float>& instance_feature,
                            const CudaWrapper<float>& anchor,
                            const uint32_t& num_querys,
                            const uint32_t& query_dims,
                            const uint32_t& embedfeat_dims,
                            const uint32_t& num_topk_querys,
                            CudaWrapper<float>& output_confidence,
                            CudaWrapper<float>& output_instance_feature,
                            CudaWrapper<float>& output_anchor,
                            CudaWrapper<int32_t>& output_track_ids,
                            cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (num_querys + block_size - 1) / block_size;

    getTopkInstanceKernel<<<grid_size, block_size, 0, stream>>>(
        confidence.getCudaPtr(), instance_feature.getCudaPtr(), anchor.getCudaPtr(),
        output_confidence.getCudaPtr(), output_instance_feature.getCudaPtr(),
        output_anchor.getCudaPtr(), output_track_ids.getCudaPtr(),
        num_querys, query_dims, embedfeat_dims, num_topk_querys
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status getTopKScoresOnGPU(const CudaWrapper<float>& confidence,
                          const uint32_t& num_querys,
                          const uint32_t& k,
                          CudaWrapper<float>& topk_confidence,
                          CudaWrapper<uint32_t>& topk_indices,
                          cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (num_querys + block_size - 1) / block_size;

    getTopKScoresKernel<<<grid_size, block_size, 0, stream>>>(
        confidence.getCudaPtr(), topk_confidence.getCudaPtr(), topk_indices.getCudaPtr(),
        num_querys, k
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status getTrackIdOnGPU(const CudaWrapper<float>& confidence,
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

Status cacheFeatureOnGPU(const CudaWrapper<float>& instance_feature,
                         const CudaWrapper<float>& anchor,
                         const CudaWrapper<float>& confidence,
                         const CudaWrapper<int32_t>& track_ids,
                         const uint32_t& num_querys,
                         const uint32_t& query_dims,
                         const uint32_t& embedfeat_dims,
                         CudaWrapper<float>& cached_feature,
                         CudaWrapper<float>& cached_anchor,
                         CudaWrapper<float>& cached_confidence,
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

Status applyConfidenceDecayOnGPU(const CudaWrapper<float>& input_confidence,
                                 CudaWrapper<float>& output_confidence,
                                 float decay_factor,
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

Status fuseConfidenceOnGPU(const CudaWrapper<float>& current_confidence,
                           const CudaWrapper<float>& cached_confidence,
                           uint32_t fusion_length,
                           cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (fusion_length + block_size - 1) / block_size;

    fuseConfidenceKernel<<<grid_size, block_size, 0, stream>>>(
        current_confidence.getCudaPtr(), cached_confidence.getCudaPtr(), fusion_length
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status anchorProjectionOnGPU(CudaWrapper<float>& temp_anchor,
                             const Eigen::Matrix<float, 4, 4>& temp_to_cur_mat,
                             float time_interval,
                             uint32_t topk_querys,
                             uint32_t query_dims,
                             cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (topk_querys + block_size - 1) / block_size;

    anchorProjectionKernel<<<grid_size, block_size, 0, stream>>>(
        temp_anchor.getCudaPtr(), temp_to_cur_mat.data(), time_interval, topk_querys, query_dims
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status getMaxConfidenceScoresOnGPU(const CudaWrapper<float>& confidence_logits,
                                   CudaWrapper<float>& max_confidence_scores,
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

Status applySigmoidOnGPU(CudaWrapper<float>& logits,
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

Status sortOnGPU(CudaWrapper<float>& values,
                 CudaWrapper<uint32_t>& indices,
                 uint32_t size,
                 cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (size + block_size - 1) / block_size;

    radixSortKernel<<<grid_size, block_size, 0, stream>>>(
        values.getCudaPtr(), indices.getCudaPtr(), size
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status countNegativeValuesOnGPU(const CudaWrapper<int32_t>& track_ids,
                                uint32_t& negative_count,
                                uint32_t array_size,
                                cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (array_size + block_size - 1) / block_size;

    countNegativeValuesKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), &negative_count, array_size
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status updateNegativeTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                   const std::vector<int32_t>& new_track_ids,
                                   uint32_t array_size,
                                   cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (array_size + block_size - 1) / block_size;

    updateNegativeTrackIdsKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), new_track_ids.data(), array_size
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status updateLastNTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                const std::vector<int32_t>& new_track_ids,
                                uint32_t total_size,
                                uint32_t update_count,
                                cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (update_count + block_size - 1) / block_size;

    updateLastNTrackIdsKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), new_track_ids.data(), total_size, update_count
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status selectTrackIdsFromIndicesOnGPU(CudaWrapper<int32_t>& track_ids,
                                      const CudaWrapper<int32_t>& cached_indices,
                                      uint32_t target_length,
                                      cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (target_length + block_size - 1) / block_size;

    selectTrackIdsFromIndicesKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), cached_indices.getCudaPtr(), target_length, cached_indices.getSize()
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

bool checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        return false;
    }
    return true;
}

} // namespace UtilsGPU