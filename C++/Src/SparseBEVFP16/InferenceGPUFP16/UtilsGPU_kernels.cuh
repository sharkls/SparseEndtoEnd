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

// ==================== CUDA Kernel实现 ====================

// 用于按值传递到 kernel 的 4x4 矩阵（列主序/与原始索引一致）
struct Mat4 {
    float m[16];
};

// ==================== CUDA Kernel声明 ====================
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
                                      uint32_t num_topk_querys);

// 获取前K个分数与索引
// （未实现，删除声明）
// __global__ void getTopKScoresKernel(const half* confidence,
//                                     half* topk_confidence,
//                                     uint32_t* topk_indices,
//                                     uint32_t num_querys,
//                                     uint32_t k);

// 生成TrackId
__global__ void getTrackIdKernel(const half* confidence,
                                 int32_t* output_track_ids,
                                 uint32_t num_querys,
                                 uint32_t num_topk_querys);

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
                                   uint32_t embedfeat_dims);

// 置信度衰减
__global__ void applyConfidenceDecayKernel(const half* in_conf,
                                            half* out_conf,
                                            half decay_factor,
                                            uint32_t size);

// 置信度融合
__global__ void fuseConfidenceKernel(half* output_confidence,
                                     const half* current_confidence,
                                     const half* cached_confidence,
                                     uint32_t size);

// 锚点投影（移除未使用的transform_matrix版本声明）
__global__ void anchorProjectionKernel(half* temp_anchor,
                                       Mat4 transform,
                                       half time_interval,
                                       uint32_t topk_querys,
                                       uint32_t query_dims);

// 最大置信度
__global__ void getMaxConfidenceScoresKernel(const half* confidence_logits,
                                             half* max_confidence_scores,
                                             uint32_t num_querys,
                                             uint32_t num_classes);

// Sigmoid
__global__ void applySigmoidKernel(half* logits, uint32_t size);

// 生成新TrackId
__global__ void generateNewTrackIdsKernel(int32_t* track_ids,
                                          const int32_t* prev_id,
                                          uint32_t size);

// 简易排序（未实现，删除声明）
// __global__ void radixSortKernel(half* values,
//                                 uint32_t* indices,
//                                 uint32_t size);

// 计数负值（未实现本文件中的使用，删除声明）
// __global__ void countNegativeValuesKernel(const int32_t* track_ids, 
//                                           uint32_t* negative_count,
//                                           uint32_t array_size);

// 更新末尾N个TrackId（未实现，删除声明）
// __global__ void updateLastNTrackIdsKernel(int32_t* track_ids,
//                                           const int32_t* new_track_ids,
//                                           uint32_t total_size,
//                                           uint32_t update_count);

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
                                              uint32_t k);

// 创建索引-值对（未实现，删除声明）
// __global__ void createIndexValuePairsKernel(const half* confidence,
//                                            half* sorted_values,
//                                            uint32_t* sorted_indices,
//                                            uint32_t num_querys);

// ==================== C++接口函数声明 ====================
namespace UtilsGPU {

// Top-K相关函数
Status getTopkInstanceOnGPU(const CudaWrapper<half>& confidence,
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
                            cudaStream_t stream = 0);
Status getTopkInstanceOnGPUOptimized(const CudaWrapper<half>& confidence,
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
                            cudaStream_t stream = 0);

// 获取Top-K（Thrust实现）
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
                                  cudaStream_t stream);

Status getTopKScoresOnGPU(const CudaWrapper<half>& confidence,
                          const uint32_t& num_querys,
                          const uint32_t& k,
                          CudaWrapper<half>& topk_confidence,
                          CudaWrapper<uint32_t>& topk_indices,
                          cudaStream_t stream = 0);

Status getTrackIdOnGPU(const CudaWrapper<half>& confidence,
                       const uint32_t& num_querys,
                       const uint32_t& num_topk_querys,
                       CudaWrapper<int32_t>& output_track_ids,
                       cudaStream_t stream = 0);

// 缓存相关函数
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
                         cudaStream_t stream = 0);

// 置信度衰减/融合
Status applyConfidenceDecayOnGPU(const CudaWrapper<half>& input_confidence,
                                 CudaWrapper<half>& output_confidence,
                                 half decay_factor,
                                 uint32_t array_size,
                                 cudaStream_t stream);

// 置信度融合
Status fuseConfidenceOnGPU(const CudaWrapper<half>& current_confidence,
                           const CudaWrapper<half>& cached_confidence,
                           CudaWrapper<half>& output_confidence,
                           uint32_t fusion_length,
                           cudaStream_t stream);

// 锚点投影
Status anchorProjectionOnGPU(CudaWrapper<half>& temp_anchor,
                             const Eigen::Matrix<half, 4, 4>& temp_to_cur_mat,
                             half time_interval,
                             uint32_t topk_querys,
                             uint32_t query_dims,
                             cudaStream_t stream);

// 最大置信度
Status getMaxConfidenceScoresOnGPU(const CudaWrapper<half>& confidence_logits,
                                   CudaWrapper<half>& max_confidence_scores,
                                   const uint32_t& num_querys,
                                   const uint32_t& num_classes,
                                   cudaStream_t stream);

Status applySigmoidOnGPU(CudaWrapper<half>& logits,
                         uint32_t size,
                         cudaStream_t stream = 0);

// 跟踪ID生成函数
Status generateNewTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                const CudaWrapper<int32_t>& prev_id,
                                uint32_t size,
                                cudaStream_t stream = 0);

// 排序函数（未实现，删除声明）
// Status sortOnGPU(CudaWrapper<half>& values,
//                  CudaWrapper<uint32_t>& indices,
//                  uint32_t size,
//                  cudaStream_t stream = 0);

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