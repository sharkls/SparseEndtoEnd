#ifndef __UTILS_GPU_KERNELS_CUH__
#define __UTILS_GPU_KERNELS_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

// Top-K相关kernel
__global__ void getTopkInstanceKernel(const float* confidence,
                                     const float* instance_feature,
                                     const float* anchor,
                                     float* temp_topk_confidence,
                                     float* temp_topk_instance_feature,
                                     float* temp_topk_anchors,
                                     int32_t* temp_topk_index,
                                     uint32_t num_querys,
                                     uint32_t query_dims,
                                     uint32_t embedfeat_dims,
                                     uint32_t num_topk_querys) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_topk_querys) return;
    
    // 使用共享内存进行局部排序
    __shared__ float shared_confidence[256];
    __shared__ uint32_t shared_indices[256];
    
    // 每个线程处理一个query
    if (idx < num_querys) {
        shared_confidence[threadIdx.x] = confidence[idx];
        shared_indices[threadIdx.x] = idx;
    } else {
        shared_confidence[threadIdx.x] = -1.0f;
        shared_indices[threadIdx.x] = 0;
    }
    
    __syncthreads();
    
    // 简单的冒泡排序找到前K个
    for (uint32_t i = 0; i < num_topk_querys && i < blockDim.x; ++i) {
        if (threadIdx.x == i) {
            // 找到第i大的值
            float max_val = -1.0f;
            uint32_t max_idx = 0;
            
            for (uint32_t j = 0; j < num_querys; ++j) {
                if (shared_confidence[j] > max_val) {
                    max_val = shared_confidence[j];
                    max_idx = shared_indices[j];
                }
            }
            
            // 保存结果
            temp_topk_confidence[idx] = max_val;
            temp_topk_index[idx] = max_idx;
            
            // 复制实例特征
            for (uint32_t j = 0; j < embedfeat_dims; ++j) {
                temp_topk_instance_feature[idx * embedfeat_dims + j] = 
                    instance_feature[max_idx * embedfeat_dims + j];
            }
            
            // 复制锚点
            for (uint32_t j = 0; j < query_dims; ++j) {
                temp_topk_anchors[idx * query_dims + j] = 
                    anchor[max_idx * query_dims + j];
            }
        }
        __syncthreads();
    }
}

__global__ void getTopKScoresKernel(const float* confidence,
                                    float* topk_confidence,
                                    uint32_t* topk_indices,
                                    uint32_t anchor_nums,
                                    uint32_t k) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    
    // 找到第idx大的置信度对应的索引
    float max_confidence = -1.0f;
    uint32_t max_idx = 0;
    
    for (uint32_t i = 0; i < anchor_nums; ++i) {
        if (confidence[i] > max_confidence) {
            max_confidence = confidence[i];
            max_idx = i;
        }
    }
    
    topk_confidence[idx] = max_confidence;
    topk_indices[idx] = max_idx;
}

__global__ void getTopKTrackIDKernel(const float* confidence,
                                     const int* track_ids,
                                     int* topk_track_ids,
                                     uint32_t anchor_nums,
                                     uint32_t k) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    
    // 找到第idx大的置信度对应的track_id
    float max_confidence = -1.0f;
    int max_track_id = -1;
    
    for (uint32_t i = 0; i < anchor_nums; ++i) {
        if (confidence[i] > max_confidence) {
            max_confidence = confidence[i];
            max_track_id = track_ids[i];
        }
    }
    
    topk_track_ids[idx] = max_track_id;
}

// 置信度处理相关kernel
__global__ void applyConfidenceDecayKernel(float* output_confidence,
                                          const float* input_confidence,
                                          float decay_factor,
                                          uint32_t size) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output_confidence[idx] = input_confidence[idx] * decay_factor;
}

__global__ void fuseConfidenceKernel(float* new_confidence,
                                     const float* cached_confidence,
                                     uint32_t size) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    new_confidence[idx] = max(new_confidence[idx], cached_confidence[idx]);
}

// Anchor投影相关kernel
__global__ void anchorProjectionKernel(float* temp_anchor,
                                       const float* transform_matrix,
                                       float time_interval,
                                       uint32_t topk_querys,
                                       uint32_t query_dims) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= topk_querys) return;
    
    // 获取当前anchor的起始位置
    uint32_t anchor_start = idx * query_dims;
    
    // 提取center (前3列) 和 velocity (后3列)
    float center[3] = {temp_anchor[anchor_start + 0], temp_anchor[anchor_start + 1], temp_anchor[anchor_start + 2]};
    float velocity[3] = {temp_anchor[anchor_start + 8], temp_anchor[anchor_start + 9], temp_anchor[anchor_start + 10]};
    
    // 计算translation: velocity * (-time_interval)
    float translation[3];
    for (int i = 0; i < 3; ++i) {
        translation[i] = velocity[i] * (-time_interval);
    }
    
    // 更新center: center = center - translation
    for (int i = 0; i < 3; ++i) {
        center[i] = center[i] - translation[i];
    }
    
    // 应用变换矩阵 (简化版本，实际应该使用完整的4x4矩阵变换)
    // 这里只更新center部分
    temp_anchor[anchor_start + 0] = center[0];
    temp_anchor[anchor_start + 1] = center[1];
    temp_anchor[anchor_start + 2] = center[2];
}

// 置信度计算相关kernel
__global__ void getMaxConfidenceScoresKernel(const float* confidence_logits,
                                             float* max_confidence_scores,
                                             uint32_t num_querys,
                                             uint32_t class_nums) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_querys) return;
    
    // 找到当前query的最大置信度
    float max_confidence = confidence_logits[idx * class_nums];
    for (uint32_t j = 1; j < class_nums; ++j) {
        float current_confidence = confidence_logits[idx * class_nums + j];
        if (current_confidence > max_confidence) {
            max_confidence = current_confidence;
        }
    }
    
    max_confidence_scores[idx] = max_confidence;
}

__global__ void applySigmoidKernel(float* logits, uint32_t size) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    logits[idx] = 1.0f / (1.0f + expf(-logits[idx]));
}

// 跟踪ID生成相关kernel
__global__ void generateNewTrackIdsKernel(int32_t* track_ids,
                                          int32_t* prev_id,
                                          uint32_t num_querys) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_querys) return;
    
    // 如果当前track_id为-1，生成新的ID
    if (track_ids[idx] < 0) {
        // 使用原子操作确保ID的唯一性
        int32_t new_id = atomicAdd(prev_id, 1);
        track_ids[idx] = new_id;
    }
}

// 后处理相关kernel
__global__ void topKKernel(const float* topk_cls_scores_origin,
                            const uint32_t* topk_index,
                            const float* fusioned_scores,
                            const uint8_t* cls_ids,
                            const float* box_preds,
                            const int* track_ids,
                            float* topk_cls_scores,
                            float* topk_fusioned_scores,
                            uint8_t* topk_cls_ids,
                            float* topk_box_preds,
                            int* topk_track_ids,
                            uint32_t k,
                            float threshold,
                            uint32_t kmeans_anchor_dims,
                            uint32_t* actual_topk_out) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    
    // 检查阈值
    if (topk_cls_scores_origin[idx] < threshold) {
        return;
    }
    
    // 保存结果
    topk_cls_scores[idx] = topk_cls_scores_origin[idx];
    topk_fusioned_scores[idx] = fusioned_scores[idx];
    
    uint32_t actual_idx = topk_index[idx];
    topk_cls_ids[idx] = cls_ids[actual_idx];
    topk_track_ids[idx] = track_ids[actual_idx];
    
    // 复制边界框预测
    for (uint32_t j = 0; j < kmeans_anchor_dims; ++j) {
        topk_box_preds[idx * kmeans_anchor_dims + j] = box_preds[actual_idx * kmeans_anchor_dims + j];
    }
    
    // 更新实际输出数量
    if (idx == 0) {
        *actual_topk_out = k;
    }
}

#endif  // __UTILS_GPU_KERNELS_CUH__