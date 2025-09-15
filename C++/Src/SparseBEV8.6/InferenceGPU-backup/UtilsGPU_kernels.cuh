/*******************************************************
 文件名：UtilsGPU_kernels.cuh
 作者：sharkls
 描述：GPU版本的Utils工具函数的CUDA kernel声明和C++接口
 版本：v1.0
 日期：2025-06-18
*******************************************************/
#ifndef __UTILS_GPU_KERNELS_CUH__
#define __UTILS_GPU_KERNELS_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <memory>
#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include "GlobalContext.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"

// ==================== CUDA Kernel声明 ====================
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
                                      uint32_t num_topk_querys);

// 获取前K个分数与索引
__global__ void getTopKScoresKernel(const float* confidence,
                                    float* topk_confidence,
                                    uint32_t* topk_indices,
                                    uint32_t num_querys,
                                    uint32_t k);

// 生成TrackId
__global__ void getTrackIdKernel(const float* confidence,
                                 int32_t* output_track_ids,
                                 uint32_t num_querys,
                                 uint32_t num_topk_querys);

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
                                   uint32_t embedfeat_dims);

// 置信度衰减
__global__ void applyConfidenceDecayKernel(const float* input_confidence,
                                           float* output_confidence,
                                          float decay_factor,
                                           uint32_t size);

// 置信度融合
__global__ void fuseConfidenceKernel(float* current_confidence,
                                     const float* cached_confidence,
                                     uint32_t size);

// 锚点投影
__global__ void anchorProjectionKernel(float* temp_anchor,
                                       const float* transform_matrix,
                                       float time_interval,
                                       uint32_t topk_querys,
                                       uint32_t query_dims);

// 最大置信度
__global__ void getMaxConfidenceScoresKernel(const float* confidence_logits,
                                             float* max_confidence_scores,
                                             uint32_t num_querys,
                                             uint32_t num_classes);

// Sigmoid
__global__ void applySigmoidKernel(float* logits, uint32_t size);

// 生成新TrackId
__global__ void generateNewTrackIdsKernel(int32_t* track_ids,
                                          const int32_t* prev_id,
                                          uint32_t size);

// 简易排序
__global__ void radixSortKernel(float* values,
                                uint32_t* indices,
                                uint32_t size);

// 计数负值
__global__ void countNegativeValuesKernel(const int32_t* track_ids, 
                                          uint32_t* negative_count,
                                          uint32_t array_size);

// 更新末尾N个TrackId
__global__ void updateLastNTrackIdsKernel(int32_t* track_ids,
                                          const int32_t* new_track_ids,
                                          uint32_t total_size,
                                          uint32_t update_count);

// 按索引选择TrackId并超出填-1
__global__ void selectTrackIdsFromIndicesKernel(int32_t* track_ids,
                                                const int32_t* cached_indices,
                                                uint32_t target_length,
                                                uint32_t source_length);

// 安全的kernel实现
__global__ void selectTrackIdsFromIndicesKernelSafe(int32_t* output_track_ids,
                                                   const int32_t* input_track_ids,
                                                   const int32_t* cached_indices,
                                                   uint32_t source_length,
                                                   uint32_t topk);

// 修正：添加缺失的kernel声明
__global__ void updateNegativeTrackIdsKernel(int32_t* track_ids,
                                             const int32_t* new_track_ids,
                                             uint32_t array_size);

// 
__global__ void  copyTopKDataFromIndicesKernel(const float* instance_feature,
                                             const float* anchor,
                                             const float* confidence,
                                             const uint32_t* topk_indices,
                                             float* output_confidence,
                                             float* output_instance_feature,
                                             float* output_anchor,
                                             int32_t* output_track_ids,
                                             uint32_t query_dims,
                                             uint32_t embedfeat_dims,
                                             uint32_t k);

// 创建索引-值对
__global__ void createIndexValuePairsKernel(const float* confidence,
                                           float* sorted_values,
                                           uint32_t* sorted_indices,
                                           uint32_t num_querys);

// ==================== C++接口函数声明 ====================
namespace UtilsGPU {

// Top-K相关函数
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
                            cudaStream_t stream = 0);
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
                            cudaStream_t stream = 0);

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
                                 cudaStream_t stream);

Status getTopKScoresOnGPU(const CudaWrapper<float>& confidence,
                          const uint32_t& num_querys,
                          const uint32_t& k,
                          CudaWrapper<float>& topk_confidence,
                          CudaWrapper<uint32_t>& topk_indices,
                          cudaStream_t stream = 0);

Status getTrackIdOnGPU(const CudaWrapper<float>& confidence,
                       const uint32_t& num_querys,
                       const uint32_t& num_topk_querys,
                       CudaWrapper<int32_t>& output_track_ids,
                       cudaStream_t stream = 0);

// 缓存相关函数
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
                         cudaStream_t stream = 0);

// 置信度衰减函数
Status applyConfidenceDecayOnGPU(const CudaWrapper<float>& input_confidence,
                                 CudaWrapper<float>& output_confidence,
                                 float decay_factor,
                                 uint32_t array_size,
                                 cudaStream_t stream = 0);

// 置信度融合
Status fuseConfidenceOnGPU(const CudaWrapper<float>& current_confidence,
                           const CudaWrapper<float>& cached_confidence,
                           CudaWrapper<float>& output_confidence,
                           uint32_t fusion_length,
                           cudaStream_t stream);

// 锚点投影函数
Status anchorProjectionOnGPU(CudaWrapper<float>& temp_anchor,
                             const Eigen::Matrix<float, 4, 4>& temp_to_cur_mat,
                             float time_interval,
                             uint32_t topk_querys,
                             uint32_t query_dims,
                             cudaStream_t stream = 0);

// 置信度计算函数
Status getMaxConfidenceScoresOnGPU(const CudaWrapper<float>& confidence_logits,
                                   CudaWrapper<float>& max_confidence_scores,
                                   const uint32_t& num_querys,
                                   const uint32_t& num_classes,
                                   cudaStream_t stream = 0);

Status applySigmoidOnGPU(CudaWrapper<float>& logits,
                         uint32_t size,
                         cudaStream_t stream = 0);

// 跟踪ID生成函数
Status generateNewTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                const CudaWrapper<int32_t>& prev_id,
                                uint32_t size,
                                cudaStream_t stream = 0);

// 排序函数
Status sortOnGPU(CudaWrapper<float>& values,
                 CudaWrapper<uint32_t>& indices,
                 uint32_t size,
                 cudaStream_t stream = 0);

// 统计和更新函数
Status countNegativeValuesOnGPU(const CudaWrapper<int32_t>& track_ids,
                                uint32_t& negative_count,
                                uint32_t array_size,
                                cudaStream_t stream = 0);

Status updateNegativeTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                   const std::vector<int32_t>& new_track_ids,
                                   uint32_t array_size,
                                   cudaStream_t stream = 0);

Status updateLastNTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                const std::vector<int32_t>& new_track_ids,
                                uint32_t total_size,
                                uint32_t update_count,
                                cudaStream_t stream = 0);

Status selectTrackIdsFromIndicesOnGPU(CudaWrapper<int32_t>& track_ids,
                                      const CudaWrapper<int32_t>& cached_indices,
                                      uint32_t topk,
                                      cudaStream_t stream = 0);

Status updateNewTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                uint32_t num_anchors_,
                                uint32_t topk_anchors_,
                                CudaWrapper<int32_t>& prev_id,
                                cudaStream_t stream = 0);
// 私有辅助函数
bool checkCudaError(cudaError_t error);

} // namespace UtilsGPU

#endif // __UTILS_GPU_KERNELS_CUH__ 