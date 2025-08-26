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
    /// @param temp_topk_confidence 输出的前K个置信度
    /// @param temp_topk_instance_feature 输出的前K个实例特征
    /// @param temp_topk_anchors 输出的前K个锚点
    /// @param temp_topk_index 输出的前K个索引
    /// @param stream CUDA流
    static Status getTopkInstanceOnGPU(const CudaWrapper<float>& confidence,
                                      const CudaWrapper<float>& instance_feature,
                                      const CudaWrapper<float>& anchor,
                                      const uint32_t& num_querys,
                                      const uint32_t& query_dims,
                                      const uint32_t& embedfeat_dims,
                                      const uint32_t& num_topk_querys,
                                      CudaWrapper<float>& temp_topk_confidence,
                                      CudaWrapper<float>& temp_topk_instance_feature,
                                      CudaWrapper<float>& temp_topk_anchors,
                                      CudaWrapper<int32_t>& temp_topk_index,
                                      cudaStream_t stream);

    /// @brief 在GPU上获取前K个最高分数和对应的索引
    /// @param confidence 置信度数组，形状为(nums_anchors,)
    /// @param anchor_nums 锚点数量
    /// @param k 要获取的前K个数量
    /// @param topk_confidence 输出的前K个置信度
    /// @param topk_indices 输出的前K个索引
    /// @param stream CUDA流
    static Status getTopKScoresOnGPU(const CudaWrapper<float>& confidence,
                                    const uint32_t& anchor_nums,
                                    const uint32_t& k,
                                    CudaWrapper<float>& topk_confidence,
                                    CudaWrapper<uint32_t>& topk_indices,
                                    cudaStream_t stream);

    /// @brief 在GPU上获取前K个跟踪ID
    /// @param confidence 置信度数组，形状为(nums_anchors,)
    /// @param anchor_nums 锚点数量
    /// @param k 要获取的前K个数量
    /// @param track_ids 原始跟踪ID数组
    /// @param topk_track_ids 输出的前K个跟踪ID
    /// @param stream CUDA流
    static Status getTopKTrackIDOnGPU(const CudaWrapper<float>& confidence,
                                     const uint32_t& anchor_nums,
                                     const uint32_t& k,
                                     const CudaWrapper<int>& track_ids,
                                     CudaWrapper<int>& topk_track_ids,
                                     cudaStream_t stream);

    /// @brief 在GPU上应用置信度衰减
    /// @param input_confidence 输入置信度
    /// @param output_confidence 输出衰减后的置信度
    /// @param decay_factor 衰减因子
    /// @param stream CUDA流
    static void applyConfidenceDecayOnGPU(const CudaWrapper<float>& input_confidence,
                                         CudaWrapper<float>& output_confidence,
                                         float decay_factor,
                                         cudaStream_t stream);

    /// @brief 在GPU上进行置信度融合
    /// @param new_confidence 新的置信度
    /// @param cached_confidence 缓存的置信度
    /// @param stream CUDA流
    static void fuseConfidenceOnGPU(CudaWrapper<float>& new_confidence,
                                   const CudaWrapper<float>& cached_confidence,
                                   cudaStream_t stream);

    /// @brief 在GPU上进行anchor投影计算
    /// @param temp_anchor 临时锚点，形状为(temp_num_querys, query_dims_)
    /// @param temp2cur_mat t-1到t时刻的变换矩阵
    /// @param time_interval 时间间隔
    /// @param topk_querys 前K个查询数量
    /// @param query_dims 查询维度
    /// @param stream CUDA流
    static void anchorProjectionOnGPU(CudaWrapper<float>& temp_anchor,
                                     const Eigen::Matrix<float, 4, 4>& temp2cur_mat,
                                     float time_interval,
                                     uint32_t topk_querys,
                                     uint32_t query_dims,
                                     cudaStream_t stream);

    /// @brief 在GPU上计算最大置信度分数
    /// @param confidence_logits 置信度logits，形状为(num_querys, class_nums)
    /// @param max_confidence_scores 输出的最大置信度分数
    /// @param num_querys 查询数量
    /// @param stream CUDA流
    static void getMaxConfidenceScoresOnGPU(const CudaWrapper<float>& confidence_logits,
                                           CudaWrapper<float>& max_confidence_scores,
                                           const uint32_t& num_querys,
                                           cudaStream_t stream);

    /// @brief 在GPU上应用Sigmoid激活函数
    /// @param logits 输入logits
    /// @param stream CUDA流
    static void applySigmoidOnGPU(CudaWrapper<float>& logits, cudaStream_t stream);

    /// @brief 在GPU上生成新的跟踪ID
    /// @param track_ids 跟踪ID数组
    /// @param prev_id 前一个ID
    /// @param stream CUDA流
    static void generateNewTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                        const CudaWrapper<int32_t>& prev_id,
                                        cudaStream_t stream);

    /// @brief 在GPU上进行后处理：根据阈值筛选并更新前K个分数
    /// @param topk_cls_scores_origin 原始前K个分类分数
    /// @param topk_index 前K个索引
    /// @param fusioned_scores 融合分数
    /// @param cls_ids 分类ID数组
    /// @param box_preds 边界框预测数组
    /// @param track_ids 跟踪ID数组
    /// @param threshold 阈值
    /// @param kmeans_anchor_dims K-means锚点维度
    /// @param k 前K个数量
    /// @param topk_cls_scores 输出的前K个分类分数
    /// @param topk_fusioned_scores 输出的前K个融合分数
    /// @param topk_cls_ids 输出的前K个分类ID
    /// @param topk_box_preds 输出的前K个边界框预测
    /// @param topk_track_ids 输出的前K个跟踪ID
    /// @param actual_topk_out 实际输出的前K个数量
    /// @param stream CUDA流
    static Status topKOnGPU(const CudaWrapper<float>& topk_cls_scores_origin,
                            const CudaWrapper<uint32_t>& topk_index,
                            const CudaWrapper<float>& fusioned_scores,
                            const CudaWrapper<uint8_t>& cls_ids,
                            const CudaWrapper<float>& box_preds,
                            const CudaWrapper<int>& track_ids,
                            const float& threshold,
                            const uint32_t& kmeans_anchor_dims,
                            const uint32_t& k,
                            CudaWrapper<float>& topk_cls_scores,
                            CudaWrapper<float>& topk_fusioned_scores,
                            CudaWrapper<uint8_t>& topk_cls_ids,
                            CudaWrapper<float>& topk_box_preds,
                            CudaWrapper<int>& topk_track_ids,
                            uint32_t& actual_topk_out,
                            cudaStream_t stream);

private:
    /// @brief 在GPU上进行排序的辅助函数
    /// @param values 要排序的值
    /// @param indices 对应的索引
    /// @param size 数组大小
    /// @param stream CUDA流
    static void sortOnGPU(CudaWrapper<float>& values, CudaWrapper<uint32_t>& indices, 
                          uint32_t size, cudaStream_t stream);

    /// @brief 在GPU上计算Sigmoid的辅助函数
    /// @param x 输入值
    /// @return Sigmoid结果
    __device__ static float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    /// @brief 在GPU上计算最大值的辅助函数
    /// @param values 输入值数组
    /// @param size 数组大小
    /// @return 最大值
    __device__ static float maxValue(const float* values, uint32_t size) {
        float max_val = values[0];
        for (uint32_t i = 1; i < size; ++i) {
            if (values[i] > max_val) {
                max_val = values[i];
            }
        }
        return max_val;
    }
};

#endif  // __UTILS_GPU_H__