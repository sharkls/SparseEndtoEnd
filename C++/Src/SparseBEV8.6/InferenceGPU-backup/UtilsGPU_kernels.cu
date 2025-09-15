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
        
        // 向量化复制特征数据（embedfeat_dims=256，可以被4整除）
        if (embedfeat_dims % 4 == 0) {
            float4* src_feature = (float4*)&instance_feature[selected_idx * embedfeat_dims];
            float4* dst_feature = (float4*)&output_instance_feature[blockIdx.x * embedfeat_dims];
            
            for (uint32_t i = 0; i < embedfeat_dims / 4; ++i) {
                dst_feature[i] = src_feature[i];
            }
        } else {
            // 如果不能被4整除，使用逐元素复制
            for (uint32_t i = 0; i < embedfeat_dims; ++i) {
                output_instance_feature[blockIdx.x * embedfeat_dims + i] = 
                    instance_feature[selected_idx * embedfeat_dims + i];
            }
        }
        
        // 锚点数据（query_dims=11，不能被4整除）- 使用逐元素复制
        for (uint32_t i = 0; i < query_dims; ++i) {
            output_anchor[blockIdx.x * query_dims + i] = 
                anchor[selected_idx * query_dims + i];
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

// 修改后的核函数
__global__ void fuseConfidenceKernel(float* output_confidence,
                                     const float* current_confidence,
                                     const float* cached_confidence,
                                     uint32_t size) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        output_confidence[tid] = fmaxf(current_confidence[tid], cached_confidence[tid]);
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
    float yaw_x = anchor_ptr[6];  // 注意：这里对应yaw的第一列
    float yaw_y = anchor_ptr[7];  // 注意：这里对应yaw的第二列
    float vel_x = anchor_ptr[8];
    float vel_y = anchor_ptr[9];
    float vel_z = anchor_ptr[10];
    
    // 时间投影
    float time_vel_x = vel_x * time_interval;
    float time_vel_y = vel_y * time_interval;
    float time_vel_z = vel_z * time_interval;
    
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
    anchor_ptr[0] = new_center_x;
    anchor_ptr[1] = new_center_y;
    anchor_ptr[2] = new_center_z;
    anchor_ptr[3] = size_x;
    anchor_ptr[4] = size_y;
    anchor_ptr[5] = size_z;
    anchor_ptr[6] = new_yaw_x;
    anchor_ptr[7] = new_yaw_y;
    anchor_ptr[8] = new_vel_x;
    anchor_ptr[9] = new_vel_y;
    anchor_ptr[10] = new_vel_z;
}

// // 最大置信度
// __global__ void getMaxConfidenceScoresKernel(const float* confidence_logits,
//                                                      float* max_confidence_scores,
//                                                      uint32_t num_querys,
//                                                      uint32_t num_classes) {
    
//     uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= num_querys) return;

//     // 找到当前查询的最大置信度
//     float max_conf = -INFINITY;
//     for (uint32_t class_idx = 0; class_idx < num_classes; ++class_idx) {
//         float conf = confidence_logits[idx * num_classes + class_idx];
//         // 修正：使用fmaxf而不是__fmaxf
//         max_conf = fmaxf(max_conf, conf);
//     }
    
//     max_confidence_scores[idx] = max_conf;
// }

__global__ void getMaxConfidenceScoresKernel(const float* confidence_logits,
                                             float* max_confidence_scores,
                                             uint32_t num_querys,
                                             uint32_t num_classes) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_querys) return;

    // 与CPU版本保持一致的初始化
    float max_logit = confidence_logits[idx * num_classes];  // 初始化为第一个类别的值
    for (uint32_t class_idx = 1; class_idx < num_classes; ++class_idx) {  // 从第二个类别开始
        float logit = confidence_logits[idx * num_classes + class_idx];
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
    
    max_confidence_scores[idx] = sigmoid_value;
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
    track_ids[idx] = prev_id[0] + idx;
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

// 高效的奇偶排序算法
__global__ void efficientOddEvenSortKernel(float* values,
                                          uint32_t* indices,
                                          uint32_t num_querys) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_querys) return;
    
    // 使用全局内存进行奇偶排序
    for (uint32_t phase = 0; phase < num_querys; ++phase) {
        if (phase % 2 == 0) {
            // 偶数阶段：比较(0,1), (2,3), (4,5), ...
            if (idx % 2 == 0 && idx < num_querys - 1) {
                if (values[idx] < values[idx + 1]) {
                    // 原子交换操作
                    float temp_val = values[idx];
                    uint32_t temp_idx = indices[idx];
                    values[idx] = values[idx + 1];
                    indices[idx] = indices[idx + 1];
                    values[idx + 1] = temp_val;
                    indices[idx + 1] = temp_idx;
                }
            }
        } else {
            // 奇数阶段：比较(1,2), (3,4), (5,6), ...
            if (idx % 2 == 1 && idx < num_querys - 1) {
                if (values[idx] < values[idx + 1]) {
                    // 原子交换操作
                    float temp_val = values[idx];
                    uint32_t temp_idx = indices[idx];
                    values[idx] = values[idx + 1];
                    indices[idx] = indices[idx + 1];
                    values[idx + 1] = temp_val;
                    indices[idx + 1] = temp_idx;
                }
            }
        }
        __syncthreads();
    }
}

// 基于堆的高效Top-K选择
__global__ void heapBasedTopKSelectionKernel(const float* confidence,
                                            uint32_t* topk_indices,
                                            uint32_t num_querys,
                                            uint32_t k) {
    __shared__ float heap_values[256];
    __shared__ uint32_t heap_indices[256];
    
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 初始化最小堆
    if (tid < k) {
        heap_values[tid] = -INFINITY;
        heap_indices[tid] = 0xFFFFFFFF;
    }
    __syncthreads();
    
    // 处理输入数据
    if (idx < num_querys) {
        float current_value = confidence[idx];
        uint32_t current_index = idx;
        
        // 如果当前值大于堆的最小值，则替换
        if (current_value > heap_values[0]) {
            // 替换堆顶
            heap_values[0] = current_value;
            heap_indices[0] = current_index;
            
            // 下沉操作维护堆性质
            uint32_t pos = 0;
            while (pos < k) {
                uint32_t left = 2 * pos + 1;
                uint32_t right = 2 * pos + 2;
                uint32_t min_pos = pos;
                
                if (left < k && heap_values[left] < heap_values[min_pos]) {
                    min_pos = left;
                }
                if (right < k && heap_values[right] < heap_values[min_pos]) {
                    min_pos = right;
                }
                
                if (min_pos == pos) break;
                
                // 交换
                float temp_val = heap_values[pos];
                uint32_t temp_idx = heap_indices[pos];
                heap_values[pos] = heap_values[min_pos];
                heap_indices[pos] = heap_indices[min_pos];
                heap_values[min_pos] = temp_val;
                heap_indices[min_pos] = temp_idx;
                
                pos = min_pos;
            }
        }
    }
    __syncthreads();
    
    // 输出结果
    if (blockIdx.x == 0 && tid < k) {
        topk_indices[tid] = heap_indices[tid];
    }
}

// 数据复制kernel
__global__ void copyTopKDataFromIndicesKernel(const float* instance_feature,
                                             const float* anchor,
                                             const float* confidence,
                                             const uint32_t* topk_indices,
                                             float* output_confidence,
                                             float* output_instance_feature,
                                             float* output_anchor,
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
    
    // 向量化复制特征数据（embedfeat_dims=256，可以被4整除）
    if (embedfeat_dims % 4 == 0) {
        float4* src_feature = (float4*)&instance_feature[selected_idx * embedfeat_dims];
        float4* dst_feature = (float4*)&output_instance_feature[idx * embedfeat_dims];
        
        for (uint32_t i = 0; i < embedfeat_dims / 4; ++i) {
            dst_feature[i] = src_feature[i];
        }
    } else {
        // 如果不能被4整除，使用逐元素复制
        for (uint32_t i = 0; i < embedfeat_dims; ++i) {
            output_instance_feature[idx * embedfeat_dims + i] = 
                instance_feature[selected_idx * embedfeat_dims + i];
        }
    }
    
    // 锚点数据（query_dims=11，不能被4整除）- 使用逐元素复制
    for (uint32_t i = 0; i < query_dims; ++i) {
        output_anchor[idx * query_dims + i] = 
            anchor[selected_idx * query_dims + i];
    }
}

// 修复后的getTopkInstanceKernel - 使用正确的Top-K选择算法
__global__ void getTopkInstanceKernelFixed(const float* confidence,
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
    
    // 使用共享内存存储所有数据用于排序
    __shared__ float shared_confidence[256];
    __shared__ uint32_t shared_indices[256];
    
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    if (idx < num_querys) {
        shared_confidence[tid] = confidence[idx];
        shared_indices[tid] = idx;
    } else {
        shared_confidence[tid] = -INFINITY;
        shared_indices[tid] = 0xFFFFFFFF;
    }
    
    __syncthreads();
    
    // 使用奇偶排序算法进行完整排序
    for (int phase = 0; phase < 256; ++phase) {
        if (phase % 2 == 0) {
            // 偶数阶段：比较(0,1), (2,3), (4,5), ...
            if (tid % 2 == 0 && tid < 255) {
                if (shared_confidence[tid] < shared_confidence[tid + 1]) {
                    float temp_conf = shared_confidence[tid];
                    uint32_t temp_idx = shared_indices[tid];
                    shared_confidence[tid] = shared_confidence[tid + 1];
                    shared_indices[tid] = shared_indices[tid + 1];
                    shared_confidence[tid + 1] = temp_conf;
                    shared_indices[tid + 1] = temp_idx;
                }
            }
        } else {
            // 奇数阶段：比较(1,2), (3,4), (5,6), ...
            if (tid % 2 == 1 && tid < 255) {
                if (shared_confidence[tid] < shared_confidence[tid + 1]) {
                    float temp_conf = shared_confidence[tid];
                    uint32_t temp_idx = shared_indices[tid];
                    shared_confidence[tid] = shared_confidence[tid + 1];
                    shared_indices[tid] = shared_indices[tid + 1];
                    shared_confidence[tid + 1] = temp_conf;
                    shared_indices[tid + 1] = temp_idx;
                }
            }
        }
        __syncthreads();
    }
    
    // 只有block 0输出前k个结果
    if (blockIdx.x == 0 && tid < num_topk_querys) {
        uint32_t selected_idx = shared_indices[tid];
        
        // 向量化复制特征数据（embedfeat_dims=256，可以被4整除）
        if (embedfeat_dims % 4 == 0) {
            float4* src_feature = (float4*)&instance_feature[selected_idx * embedfeat_dims];
            float4* dst_feature = (float4*)&output_instance_feature[tid * embedfeat_dims];
            
            for (uint32_t i = 0; i < embedfeat_dims / 4; ++i) {
                dst_feature[i] = src_feature[i];
            }
        } else {
            // 如果不能被4整除，使用逐元素复制
            for (uint32_t i = 0; i < embedfeat_dims; ++i) {
                output_instance_feature[tid * embedfeat_dims + i] = 
                    instance_feature[selected_idx * embedfeat_dims + i];
            }
        }
        
        // 锚点数据（query_dims=11，不能被4整除）- 使用逐元素复制
        for (uint32_t i = 0; i < query_dims; ++i) {
            output_anchor[tid * query_dims + i] = 
                anchor[selected_idx * query_dims + i];
        }
        
        output_confidence[tid] = shared_confidence[tid];
        output_track_ids[tid] = selected_idx;
    }
}

// 创建索引-值对的kernel
__global__ void createIndexValuePairsKernel(const float* confidence,
                                           float* sorted_values,
                                           uint32_t* sorted_indices,
                                           uint32_t num_querys) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_querys) return;
    
    // 创建索引-值对
    sorted_values[idx] = confidence[idx];
    sorted_indices[idx] = idx;
}

// 简化的Top-K选择 - 使用单线程排序（稳定但较慢）
__global__ void simpleTopKSelectionKernel(const float* confidence,
                                         uint32_t* topk_indices,
                                         uint32_t num_querys,
                                         uint32_t k) {
    // 只使用第一个线程进行排序
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 创建临时数组
        float* temp_values = (float*)malloc(num_querys * sizeof(float));
        uint32_t* temp_indices = (uint32_t*)malloc(num_querys * sizeof(uint32_t));
        
        // 复制数据
        for (uint32_t i = 0; i < num_querys; ++i) {
            temp_values[i] = confidence[i];
            temp_indices[i] = i;
        }
        
        // 使用冒泡排序（简单但稳定）
        for (uint32_t i = 0; i < num_querys - 1; ++i) {
            for (uint32_t j = 0; j < num_querys - i - 1; ++j) {
                if (temp_values[j] < temp_values[j + 1]) {
                    // 交换值
                    float temp_val = temp_values[j];
                    temp_values[j] = temp_values[j + 1];
                    temp_values[j + 1] = temp_val;
                    
                    // 交换索引
                    uint32_t temp_idx = temp_indices[j];
                    temp_indices[j] = temp_indices[j + 1];
                    temp_indices[j + 1] = temp_idx;
                }
            }
        }
        
        // 输出前k个索引
        for (uint32_t i = 0; i < k; ++i) {
            topk_indices[i] = temp_indices[i];
        }
        
        free(temp_values);
        free(temp_indices);
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

// 修改后的接口函数
Status fuseConfidenceOnGPU(const CudaWrapper<float>& current_confidence,
                           const CudaWrapper<float>& cached_confidence,
                           CudaWrapper<float>& output_confidence,
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

Status anchorProjectionOnGPU(CudaWrapper<float>& temp_anchor,
                             const Eigen::Matrix<float, 4, 4>& temp_to_cur_mat,
                             float time_interval,
                             uint32_t topk_querys,
                             uint32_t query_dims,
                             cudaStream_t stream) {
    
    uint32_t block_size = 256;
    uint32_t grid_size = (topk_querys + block_size - 1) / block_size;

    // 将主机侧矩阵复制到设备缓冲区，避免将主机指针传入设备代码
    float* d_transform_matrix = nullptr;
    cudaError_t err_alloc = cudaMalloc((void**)&d_transform_matrix, 16 * sizeof(float));
    if (err_alloc != cudaSuccess) {
        return kInferenceErr;
    }
    cudaError_t err_copy = cudaMemcpyAsync(d_transform_matrix, temp_to_cur_mat.data(), 16 * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err_copy != cudaSuccess) {
        cudaFree(d_transform_matrix);
        return kInferenceErr;
    }

    anchorProjectionKernel<<<grid_size, block_size, 0, stream>>>(
        temp_anchor.getCudaPtr(), d_transform_matrix, time_interval, topk_querys, query_dims
    );
    
    // 检查并同步，确保内核完成后再释放矩阵缓冲
    cudaError_t launch_err = cudaPeekAtLastError();
    if (launch_err != cudaSuccess) {
        cudaFree(d_transform_matrix);
        return kInferenceErr;
    }
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    cudaFree(d_transform_matrix);
    if (sync_err != cudaSuccess) {
        return kInferenceErr;
    }
    
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

    // 在设备上为计数器分配内存并置零
    uint32_t* d_negative_count = nullptr;
    cudaError_t err_alloc = cudaMalloc((void**)&d_negative_count, sizeof(uint32_t));
    if (err_alloc != cudaSuccess) {
        return kInferenceErr;
    }
    cudaError_t err_set = cudaMemsetAsync(d_negative_count, 0, sizeof(uint32_t), stream);
    if (err_set != cudaSuccess) {
        cudaFree(d_negative_count);
        return kInferenceErr;
    }

    countNegativeValuesKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), d_negative_count, array_size
    );
    
    cudaError_t launch_err = cudaPeekAtLastError();
    if (launch_err != cudaSuccess) {
        cudaFree(d_negative_count);
        return kInferenceErr;
    }
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        cudaFree(d_negative_count);
        return kInferenceErr;
    }

    // 拷回到主机
    cudaError_t err_copy_back = cudaMemcpy(&negative_count, d_negative_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_negative_count);
    if (err_copy_back != cudaSuccess) {
        return kInferenceErr;
    }

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

    // 将主机侧 new_track_ids 拷贝到设备
    int32_t* d_new_track_ids = nullptr;
    size_t new_ids_bytes = new_track_ids.size() * sizeof(int32_t);
    cudaError_t err_alloc = cudaMalloc((void**)&d_new_track_ids, new_ids_bytes);
    if (err_alloc != cudaSuccess) {
        return kInferenceErr;
    }
    cudaError_t err_copy = cudaMemcpyAsync(d_new_track_ids, new_track_ids.data(), new_ids_bytes, cudaMemcpyHostToDevice, stream);
    if (err_copy != cudaSuccess) {
        cudaFree(d_new_track_ids);
        return kInferenceErr;
    }

    updateNegativeTrackIdsKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), d_new_track_ids, array_size
    );
    
    cudaError_t launch_err = cudaPeekAtLastError();
    if (launch_err != cudaSuccess) {
        cudaFree(d_new_track_ids);
        return kInferenceErr;
    }
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    cudaFree(d_new_track_ids);
    if (sync_err != cudaSuccess) {
        return kInferenceErr;
    }

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

    // 将主机侧 new_track_ids 拷贝到设备
    int32_t* d_new_track_ids = nullptr;
    size_t new_ids_bytes = new_track_ids.size() * sizeof(int32_t);
    cudaError_t err_alloc = cudaMalloc((void**)&d_new_track_ids, new_ids_bytes);
    if (err_alloc != cudaSuccess) {
        return kInferenceErr;
    }
    cudaError_t err_copy = cudaMemcpyAsync(d_new_track_ids, new_track_ids.data(), new_ids_bytes, cudaMemcpyHostToDevice, stream);
    if (err_copy != cudaSuccess) {
        cudaFree(d_new_track_ids);
        return kInferenceErr;
    }

    updateLastNTrackIdsKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), d_new_track_ids, total_size, update_count
    );
    
    cudaError_t launch_err = cudaPeekAtLastError();
    if (launch_err != cudaSuccess) {
        cudaFree(d_new_track_ids);
        return kInferenceErr;
    }
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    cudaFree(d_new_track_ids);
    if (sync_err != cudaSuccess) {
        return kInferenceErr;
    }

    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status selectTrackIdsFromIndicesOnGPU(CudaWrapper<int32_t>& track_ids,
                                      const CudaWrapper<int32_t>& cached_indices,
                                      uint32_t topk,
                                      cudaStream_t stream) {
    uint32_t source_length = track_ids.getSize();
    
    // 创建临时数组存储原始数据
    CudaWrapper<int32_t> temp_track_ids(source_length);
    
    // 复制原始数据到临时数组
    cudaError_t copy_error = cudaMemcpyAsync(temp_track_ids.getCudaPtr(), 
                                            track_ids.getCudaPtr(),
                                            source_length * sizeof(int32_t),
                                            cudaMemcpyDeviceToDevice, stream);
    if (copy_error != cudaSuccess) {
        return kInferenceErr;
    }

    uint32_t block_size = 256;
    uint32_t grid_size = (source_length + block_size - 1) / block_size;

    // 修改kernel调用，使用临时数组作为源数据
    selectTrackIdsFromIndicesKernelSafe<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(),            // 输出：重排序后的数组
        temp_track_ids.getCudaPtr(),       // 输入：原始数据的副本
        cached_indices.getCudaPtr(),       // 输入：索引数组
        source_length,
        topk
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
        std::cout << "updated_prev_id : " << updated_prev_id << "current_prev_id : " << current_prev_id[0] << std::endl;
        
        // 将更新后的值写回GPU
        std::vector<int32_t> updated_prev_id_vec = {updated_prev_id};
        prev_id.cudaMemUpdateWrap(updated_prev_id_vec);
    }
    return kSuccess;
}

// 最优的GPU Top-K实现
Status getTopkInstanceOnGPUOptimized(const CudaWrapper<float>& confidence,
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
    
    // 分配临时内存用于排序
    CudaWrapper<uint32_t> temp_indices(num_querys);
    
    uint32_t block_size = 256;
    uint32_t grid_size = 1;  // 只使用一个block进行排序
    
    // 使用简化的单线程排序
    simpleTopKSelectionKernel<<<grid_size, block_size, 0, stream>>>(
        confidence.getCudaPtr(),
        temp_indices.getCudaPtr(),
        num_querys,
        num_topk_querys
    );
    
    // 提取Top-K数据
    uint32_t topk_grid_size = (num_topk_querys + block_size - 1) / block_size;
    copyTopKDataFromIndicesKernel<<<topk_grid_size, block_size, 0, stream>>>(
        instance_feature.getCudaPtr(),
        anchor.getCudaPtr(),
        confidence.getCudaPtr(),
        temp_indices.getCudaPtr(),
        output_confidence.getCudaPtr(),
        output_instance_feature.getCudaPtr(),
        output_anchor.getCudaPtr(),
        output_track_ids.getCudaPtr(),
        query_dims,
        embedfeat_dims,
        num_topk_querys
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status getTopkInstanceOnGPUThrust(const CudaWrapper<float>& confidence,
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

    // // 将confidence数据复制到CPU
    // std::vector<float> cpu_confidence = confidence.cudaMemcpyD2HResWrap();
    //  // 将uint32_t转换为int32_t
    // std::cout << "confidence[i] : ";
    // for (size_t i = 0; i < cpu_confidence.size(); ++i) {
    //     std::cout << cpu_confidence[i] << " ";
    // }
    // std::cout << std::endl;
    
    // // 创建索引-值对
    // std::vector<std::pair<float, uint32_t>> indexed_values(num_querys);
    // for (uint32_t i = 0; i < num_querys; ++i) {
    //     indexed_values[i] = std::make_pair(cpu_confidence[i], i);
    // }
    
    // // 按confidence值降序排序
    // std::sort(indexed_values.begin(), indexed_values.end(), 
    //           [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
    //               return a.first > b.first;
    //           });
    
    // // 提取前k个索引
    // std::vector<uint32_t> topk_indices(num_topk_querys);
    // for (uint32_t i = 0; i < num_topk_querys; ++i) {
    //     topk_indices[i] = indexed_values[i].second;
    // }
    
    // // 将索引复制到GPU
    // CudaWrapper<uint32_t> gpu_indices(num_topk_querys);
    // gpu_indices.cudaMemUpdateWrap(topk_indices);
    
    
    // 创建索引-值对
    thrust::device_vector<float> d_values(num_querys);
    thrust::device_vector<uint32_t> d_indices(num_querys);
    
    // 初始化索引为原始索引
    thrust::sequence(d_indices.begin(), d_indices.end());
    
    // 复制confidence值
    cudaMemcpyAsync(d_values.data().get(), confidence.getCudaPtr(), 
                    num_querys * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    
    // 使用thrust进行稳定排序 - 关键：按confidence值排序，但保持原始索引
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream), 
                       d_values.begin(), d_values.end(), 
                       d_indices.begin(), 
                       thrust::greater<float>());

    // 在排序后添加调试代码
    // std::vector<uint32_t> debug_indices(num_topk_querys);
    // cudaMemcpy(debug_indices.data(), d_indices.data().get(), 
    //         num_topk_querys * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // std::cout << "=== Thrust排序后的前10个索引 ===" << std::endl;
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << "[" << i << "] = " << d_indices[i] << std::endl;
    // }

    // 同步流确保排序完成
    cudaStreamSynchronize(stream);

    // 打印排序后的confidence值和对应的索引
    std::vector<float> sorted_confidence(num_querys);
    std::vector<uint32_t> sorted_indices(num_querys);
    
    cudaMemcpy(sorted_confidence.data(), d_values.data().get(), 
               num_querys * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sorted_indices.data(), d_indices.data().get(), 
               num_querys * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // std::cout << "=== 排序后的confidence值和索引 ===" << std::endl;
    // std::cout << "前50个排序结果:" << std::endl;
    // std::cout << "排名 | 原始索引 | Confidence值" << std::endl;
    // std::cout << "-----|----------|-------------" << std::endl;
    
    // for (int i = 0; i < std::min(50, (int)num_querys); ++i) {
    //     std::cout << i << " | " 
    //               << sorted_indices[i] << " | " 
    //               << sorted_confidence[i] << std::endl;
    // }
    
    // // 打印前k个结果（用于Top-K选择）
    // std::cout << "\n=== 前" << num_topk_querys << "个Top-K结果 ===" << std::endl;
    // std::cout << "排名 | 原始索引 | Confidence值" << std::endl;
    // std::cout << "-----|----------|-------------" << std::endl;
    
    // for (uint32_t i = 0; i < num_topk_querys; ++i) {
    //     std::cout << i << " | " 
    //               << sorted_indices[i] << " | " 
    //               << sorted_confidence[i] << std::endl;
    // }
    
    // // 检查是否有重复的confidence值
    // std::cout << "\n=== 重复confidence值检查 ===" << std::endl;
    // int duplicate_count = 0;
    // for (uint32_t i = 1; i < num_querys; ++i) {
    //     if (std::abs(sorted_confidence[i] - sorted_confidence[i-1]) < 1e-8f) {
    //         duplicate_count++;
    //         if (duplicate_count <= 10) {  // 只打印前10个重复
    //             std::cout << "重复值: 排名" << (i-1) << "和" << i 
    //                       << ", confidence=" << sorted_confidence[i] 
    //                       << ", 索引=" << sorted_indices[i-1] << "," << sorted_indices[i] << std::endl;
    //         }
    //     }
    // }
    // std::cout << "总重复对数: " << duplicate_count << std::endl;
    
    // 现在d_indices包含的是按confidence排序后的原始索引
    // 提取前k个结果
    uint32_t block_size = 256;
    uint32_t grid_size = (num_topk_querys + block_size - 1) / block_size;
    
    copyTopKDataFromIndicesKernel<<<grid_size, block_size, 0, stream>>>(
        instance_feature.getCudaPtr(),
        anchor.getCudaPtr(),
        confidence.getCudaPtr(),
        d_indices.data().get(),  // 这里传入的是原始索引
        // gpu_indices.getCudaPtr(),
        output_confidence.getCudaPtr(),
        output_instance_feature.getCudaPtr(),
        output_anchor.getCudaPtr(),
        output_track_ids.getCudaPtr(),
        query_dims,
        embedfeat_dims,
        num_topk_querys
    );
    
    return kSuccess;
}

} // namespace UtilsGPU