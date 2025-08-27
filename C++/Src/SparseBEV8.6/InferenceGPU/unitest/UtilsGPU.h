/*******************************************************
 文件名：UtilsGPU.h
 作者：sharkls
 描述：GPU版本的Utils工具函数，提供GPU上的计算操作
 版本：v1.0
 日期：2025-06-18
 *******************************************************/
#ifndef __UTILS_GPU_H__
#define __UTILS_GPU_H__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include "../../../Include/Common/Core/GlobalContext.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"

/// @brief GPU版本的Utils工具函数类
class UtilsGPU {
public:
    /// @brief 在GPU上获取前K个实例的特征、锚点和相关数据
    /// @param confidence 置信度数组，形状为(num_querys,)
    /// @param instance_feature 实例特征数组，形状为(num_querys, embedfeat_dims)
    /// @param anchor 锚点数组，形状为(num_querys, query_dims)
    /// @param num_querys 查询数量
    /// @param query_dims 查询维度
    /// @param embedfeat_dims 嵌入特征维度
    /// @param num_topk_querys 前K个查询数量
    /// @param output_confidence 输出置信度数组
    /// @param output_instance_feature 输出实例特征数组
    /// @param output_anchor 输出锚点数组
    /// @param output_track_ids 输出跟踪ID数组
    /// @param stream CUDA流
    /// @return 操作状态
    static Status getTopkInstanceOnGPU(const CudaWrapper<float>& confidence,
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

    /// @brief 在GPU上获取前K个最高分数和对应的索引
    /// @param confidence 置信度数组，形状为(num_querys,)
    /// @param num_querys 查询数量
    /// @param k 要获取的前K个数量
    /// @param topk_confidence 输出的前K个置信度
    /// @param topk_indices 输出的前K个索引
    /// @param stream CUDA流
    /// @return 操作状态
    static Status getTopKScoresOnGPU(const CudaWrapper<float>& confidence,
                                    const uint32_t& num_querys,
                                    const uint32_t& k,
                                    CudaWrapper<float>& topk_confidence,
                                    CudaWrapper<uint32_t>& topk_indices,
                                    cudaStream_t stream = 0);

    /// @brief 在GPU上获取跟踪ID
    /// @param confidence 置信度数组
    /// @param num_querys 查询数量
    /// @param num_topk_querys 前K个查询数量
    /// @param output_track_ids 输出跟踪ID数组
    /// @param stream CUDA流
    /// @return 操作状态
    static Status getTrackIdOnGPU(const CudaWrapper<float>& confidence,
                                 const uint32_t& num_querys,
                                 const uint32_t& num_topk_querys,
                                 CudaWrapper<int32_t>& output_track_ids,
                                 cudaStream_t stream = 0);

    /// @brief 在GPU上统计数组中负值的数量
    /// @param track_ids GPU上的跟踪ID数组
    /// @param negative_count 输出负值数量
    /// @param array_size 数组大小
    /// @param stream CUDA流
    /// @return 操作状态
    static Status countNegativeValuesOnGPU(const CudaWrapper<int32_t>& track_ids,
                                         uint32_t& negative_count,
                                         uint32_t array_size,
                                         cudaStream_t stream = 0);
    
    /// @brief 在GPU上更新负值跟踪ID
    /// @param track_ids GPU上的跟踪ID数组（将被更新）
    /// @param new_track_ids 新的跟踪ID数组
    /// @param array_size 数组大小
    /// @param stream CUDA流
    /// @return 操作状态
    static Status updateNegativeTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                             const std::vector<int32_t>& new_track_ids,
                                             uint32_t array_size,
                                             cudaStream_t stream = 0);

    /// @brief 在GPU上缓存特征
    /// @param instance_feature 实例特征数组
    /// @param anchor 锚点数组
    /// @param confidence 置信度数组
    /// @param track_ids 跟踪ID数组
    /// @param num_querys 查询数量
    /// @param query_dims 查询维度
    /// @param embedfeat_dims 嵌入特征维度
    /// @param cached_feature 缓存特征数组
    /// @param cached_anchor 缓存锚点数组
    /// @param cached_confidence 缓存置信度数组
    /// @param cached_track_ids 缓存跟踪ID数组
    /// @param stream CUDA流
    /// @return 操作状态
    static Status cacheFeatureOnGPU(const CudaWrapper<float>& instance_feature,
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

    /// @brief 在GPU上应用置信度衰减
    /// @param input_confidence 输入的置信度数组
    /// @param output_confidence 输出的衰减后置信度数组
    /// @param decay_factor 衰减因子
    /// @param array_size 数组大小
    /// @param stream CUDA流
    /// @return 操作状态
    static Status applyConfidenceDecayOnGPU(const CudaWrapper<float>& input_confidence,
                                           CudaWrapper<float>& output_confidence,
                                           float decay_factor,
                                           uint32_t array_size,
                                           cudaStream_t stream = 0);
    
    /// @brief 在GPU上融合置信度（取最大值）
    /// @param current_confidence 当前帧置信度
    /// @param cached_confidence 缓存的置信度
    /// @param fusion_length 融合长度
    /// @param stream CUDA流
    /// @return 操作状态
    static Status fuseConfidenceOnGPU(const CudaWrapper<float>& current_confidence,
                                     const CudaWrapper<float>& cached_confidence,
                                     uint32_t fusion_length,
                                     cudaStream_t stream = 0);

    /// @brief 在GPU上进行锚点投影
    /// @param temp_anchor 临时锚点
    /// @param temp2cur_mat 变换矩阵
    /// @param time_interval 时间间隔
    /// @param topk_querys 前K个查询数量
    /// @param query_dims 查询维度
    /// @param stream CUDA流
    static void anchorProjectionOnGPU(CudaWrapper<float>& temp_anchor,
                                     const Eigen::Matrix<float, 4, 4>& temp2cur_mat,
                                     float time_interval,
                                     uint32_t topk_querys,
                                     uint32_t query_dims,
                                     cudaStream_t stream = 0);

    /// @brief 在GPU上获取最大置信度分数
    /// @param confidence_logits 置信度logits
    /// @param max_confidence_scores 最大置信度分数
    /// @param num_querys 查询数量
    /// @param stream CUDA流
    static void getMaxConfidenceScoresOnGPU(const CudaWrapper<float>& confidence_logits,
                                           CudaWrapper<float>& max_confidence_scores,
                                           const uint32_t& num_querys,
                                           cudaStream_t stream = 0);

    /// @brief 在GPU上应用Sigmoid激活函数
    /// @param logits 输入logits
    /// @param stream CUDA流
    static void applySigmoidOnGPU(CudaWrapper<float>& logits, cudaStream_t stream = 0);

    /// @brief 在GPU上生成新的跟踪ID
    /// @param track_ids 跟踪ID数组
    /// @param prev_id 前一个ID
    /// @param stream CUDA流
    static void generateNewTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                        const CudaWrapper<int32_t>& prev_id,
                                        cudaStream_t stream = 0);

    /// @brief 在GPU上进行排序
    /// @param values 值数组
    /// @param indices 索引数组
    /// @param size 数组大小
    /// @param stream CUDA流
    static void sortOnGPU(CudaWrapper<float>& values, CudaWrapper<uint32_t>& indices, 
                          uint32_t size, cudaStream_t stream = 0);

    /// @brief 在GPU上更新数组最后N个位置的跟踪ID
    /// @param track_ids GPU上的跟踪ID数组（将被更新）
    /// @param new_track_ids 新的跟踪ID数组
    /// @param total_size 总数组大小
    /// @param update_count 需要更新的数量（从末尾开始）
    /// @param stream CUDA流
    /// @return 操作状态
    static Status updateLastNTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                          const std::vector<int32_t>& new_track_ids,
                                          uint32_t total_size,
                                          uint32_t update_count,
                                          cudaStream_t stream = 0);

    /// @brief 在GPU上根据缓存的索引更新跟踪ID数组
    /// @param track_ids 输入输出的跟踪ID数组（将被更新）
    /// @param cached_indices 缓存的索引数组
    /// @param target_length 目标数组长度
    /// @param stream CUDA流
    /// @return 操作状态
    static Status updateTrackIdsFromCachedIndicesOnGPU(CudaWrapper<int32_t>& track_ids,
                                                      const CudaWrapper<int32_t>& cached_indices,
                                                      uint32_t target_length,
                                                      cudaStream_t stream = 0);

    /// @brief 在GPU上根据缓存的索引更新跟踪ID数组
    /// @param track_ids 输入输出的跟踪ID数组（将被更新）
    /// @param cached_indices 缓存的索引数组
    /// @param target_length 目标数组长度
    /// @param stream CUDA流
    /// @return 操作状态
    static Status selectTrackIdsFromIndicesOnGPU(CudaWrapper<int32_t>& track_ids,
                                                const CudaWrapper<int32_t>& cached_indices,
                                                uint32_t target_length,
                                                cudaStream_t stream = 0);

private:
    /// @brief 检查CUDA错误
    /// @param error CUDA错误码
    /// @return 是否成功
    static bool checkCudaError(cudaError_t error);
};

#endif // __UTILS_GPU_H__