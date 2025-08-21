// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#ifndef ONBOARD_HEAD_INSTANCE_BANK_H
#define ONBOARD_HEAD_INSTANCE_BANK_H

#include <Eigen/Dense>
#include <vector>

#include "../common/common.h"
#include "../common/parameter.h"

namespace sparse_end2end {
namespace head {

/// @brief 实例银行类：管理多帧检测实例的缓存和更新
class InstanceBank {
 public:
  /// @brief 构造函数：使用参数初始化实例银行
  InstanceBank(const common::E2EParams& params);
  InstanceBank() = delete;
  ~InstanceBank() = default;

  /// @brief 重置实例银行状态
  common::Status reset();

  /// @brief 获取指定时间戳的实例数据
  /// @param timestamp 时间戳
  /// @param global_to_lidar_mat 全局坐标系到激光雷达坐标系的变换矩阵
  /// @return 返回缓存的实例特征、锚点、置信度等数据
  std::tuple<const std::vector<float>&,
             const std::vector<float>&,
             const std::vector<float>&,
             const std::vector<float>&,
             const float&,
             const std::int32_t&,
             const std::vector<std::int32_t>&>
  get(const double& timestamp, const Eigen::Matrix<double, 4, 4>& global_to_lidar_mat);

  /// @brief 缓存当前帧的前K个实例特征、锚点和置信度
  /// @param instance_feature 实例特征，形状为(num_querys, embedfeat_dims)
  /// @param anchor 锚点，形状为(num_querys, query_dims)
  /// @param confidence_logits 置信度logits，形状为(num_querys, class_nums)
  common::Status cache(const std::vector<float>& instance_feature,
                       const std::vector<float>& anchor,
                       const std::vector<float>& confidence_logits);

  /// @brief 获取跟踪ID
  /// @param refined_track_ids 精炼后的跟踪ID数组，形状为(num_querys,)
  /// @return 返回跟踪ID数组
  std::vector<std::int32_t> getTrackId(const std::vector<std::int32_t>& refined_track_ids);

  /// @brief 获取临时置信度数组
  std::vector<float> getTempConfidence() const;
  /// @brief 获取临时前K个置信度数组
  std::vector<float> getTempTopKConfidence() const;
  /// @brief 获取缓存的实例特征
  std::vector<float> getCachedFeature() const;
  /// @brief 获取缓存的锚点
  std::vector<float> getCachedAnchor() const;
  /// @brief 获取前一个ID
  std::int32_t getPrevId() const;
  /// @brief 获取缓存的跟踪ID
  std::vector<std::int32_t> getCachedTrackIds() const;

 private:
  /// @brief 时空对齐：将t-1时刻的锚点投影到t时刻
  /// @param temp_anchor 临时锚点，形状为(temp_num_querys, query_dims_)
  /// @param temp_to_cur_mat t-1到t时刻的变换矩阵
  /// @param time_interval 时间间隔
  void anchorProjection(std::vector<float>& temp_anchor,
                        const Eigen::Matrix<float, 4, 4>& temp_to_cur_mat,
                        const float& time_interval);

  /// @brief 计算tensor.max(-1).values.sigmoid，tensor形状为(x, y)
  /// @param confidence_logits 置信度logits，形状为(num_querys,)
  /// @param num_querys 查询数量
  /// @return 每个查询的最大置信度分数
  template <typename T>
  static std::vector<T> getMaxConfidenceScores(const std::vector<T>& confidence_logits,
                                               const std::uint32_t& num_querys);

  /// @brief Sigmoid激活函数
  template <typename T>
  static T sigmoid(const T& logits);

  /// @brief 更新跟踪ID
  /// @param track_ids 跟踪ID数组，形状为(num_querys,)
  void updateTrackId(const std::vector<std::int32_t>& track_ids);

  common::E2EParams params_;

  /// @brief 实例银行初始化参数
  std::uint32_t num_querys_;           // 默认900，查询数量
  std::uint32_t num_topk_querys_;      // 默认600，前K个查询数量
  std::vector<float> kmeans_anchors_;  // K-means锚点，形状为(num_querys, query_dims)
  float max_time_interval_;            // 最大时间间隔（秒）
  float default_time_interval_;        // 默认时间间隔（秒）
  uint32_t query_dims_;                // 查询维度：X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ
  uint32_t embedfeat_dims_;            // 默认256，嵌入特征维度
  float confidence_decay_;             // 默认0.5，置信度衰减因子

  std::int32_t mask_;                  // 掩码
  float time_interval_;                // 时间间隔（秒）
  double history_time_;                // 历史时间
  std::uint32_t track_size_;           // 默认900*10，跟踪大小
  Eigen::Matrix<double, 4, 4> temp_lidar_to_global_mat_;  // 临时激光雷达到全局坐标变换矩阵
  std::vector<float> instance_feature_{};            // 实例特征，形状为(num_querys_, embedfeat_dims)
  std::vector<float> temp_topk_instance_feature_{};  // 临时前K个实例特征，形状为(topk, embedfeat_dims)
  std::vector<float> temp_topk_anchors_{};           // 临时前K个锚点，形状为(topk, query_dims)
  std::vector<uint32_t> temp_topk_index_{};          // 临时前K个索引，形状为(topk,)
  std::vector<std::int32_t> temp_track_ids_{};       // 临时跟踪ID，形状为(num_querys_,)
  std::vector<float> temp_confidence_{};             // 临时置信度，形状为(num_querys_,)
  std::vector<float> temp_topk_confidence_{};        // 临时前K个置信度，形状为(topk,)

  std::int32_t prev_id_;               // 前一个ID
};

}  // namespace head
}  // namespace sparse_end2end

#endif  // ONBOARD_HEAD_INSTANCE_BANK_H