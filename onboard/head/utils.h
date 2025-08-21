// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_HEAD_UTILS_H
#define ONBOARD_HEAD_UTILS_H

#include <vector>

#include "../common/common.h"

namespace sparse_end2end {
namespace head {

/// @brief 根据置信度获取前K个跟踪ID
/// @param confidence 置信度数组，形状为(nums_anchors,)
/// @param anchor_nums 锚点数量
/// @param k 要获取的前K个数量
/// @param track_ids 原始跟踪ID数组
/// @param topk_track_ids 输出的前K个跟踪ID
common::Status getTopKTrackID(const std::vector<float>& confidence,  // Shape is (nums_anchors, )
                              const std::uint32_t& anchor_nums,      // confidence size = nums_anchors
                              const std::uint32_t& k,                // Size =  temp_anchor_nums
                              const std::vector<int>& track_ids,
                              std::vector<int>& topk_track_ids);

/// @brief 根据置信度获取前K个实例的特征、锚点和相关数据
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
common::Status getTopkInstance(const std::vector<float>& confidence,        // Shape is (num_querys, )
                               const std::vector<float>& instance_feature,  // Shape is (num_querys, embedfeat_dims)
                               const std::vector<float>& anchor,            // Shape is (num_querys, query_dims)
                               const std::uint32_t& num_querys,             // Ssize = num_querys
                               const std::uint32_t& query_dims,             // Size = num_querys
                               const std::uint32_t& embedfeat_dims,         // Size = embedfeat_dims
                               const std::uint32_t& num_topk_querys,
                               std::vector<float>& temp_topk_confidence,
                               std::vector<float>& temp_topk_instance_feature,
                               std::vector<float>& temp_topk_anchors,
                               std::vector<std::uint32_t>& temp_topk_index);

/// @brief 获取前K个最高分数和对应的索引
/// @param confidence 置信度数组，形状为(nums_anchors,)
/// @param anchor_nums 锚点数量
/// @param k 要获取的前K个数量
/// @param topk_confidence 输出的前K个置信度
/// @param topk_indices 输出的前K个索引
common::Status getTopKScores(const std::vector<float>& confidence,  // Shape is (nums_anchors, )
                             const std::uint32_t& anchor_nums,      // Size = anchor_nums
                             const std::uint32_t& k,                // default = post_process_out_nums
                             std::vector<float>& topk_confidence,
                             std::vector<std::uint32_t>& topk_indices);

/// @brief 后处理：根据阈值筛选并更新前K个分数、质量分数等
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
common::Status topK(const std::vector<float>& topk_cls_scores_origin,  // Shape is (topk, )
                    const std::vector<std::uint32_t>& topk_index,      // Shape is (topk, )
                    const std::vector<float>& fusioned_scores,         // Shape is (topk, )
                    const std::vector<std::uint8_t>& cls_ids,          // Shape is (nums_anchors, )
                    const std::vector<float>& box_preds,               // Shape is (nums_anchors, )
                    const std::vector<int>& track_ids,                 // Shape is (nums_anchors, )
                    const float& threshold,
                    const std::uint32_t& kmeans_anchor_dims,
                    const std::uint32_t& k,
                    std::vector<float>& topk_cls_scores,
                    std::vector<float>& topk_fusioned_scores,
                    std::vector<std::uint8_t>& topk_cls_ids,
                    std::vector<float>& topk_box_preds,
                    std::vector<int>& topk_track_ids,
                    std::uint32_t& actual_topk_out);

}  // namespace head
}  // namespace sparse_end2end
#endif  // ONBOARD_HEAD_UTILS_H