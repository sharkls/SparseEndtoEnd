/*******************************************************
 文件名：InstanceBank.h
 作者：sharkls
 描述：实例银行类，负责管理多帧检测实例的缓存和更新
 版本：v1.0
 日期：2025-06-18
 *******************************************************/
#ifndef __INSTANCE_BANK_H__
#define __INSTANCE_BANK_H__

#include <vector>
#include <cstdint>

// 使用条件编译避免重复包含Eigen
#ifndef EIGEN_DENSE_INCLUDED
#define EIGEN_DENSE_INCLUDED
#include <Eigen/Dense>
#endif

#include "GlobalContext.h"
#include "SparseEnd2EndConfig_conf.pb.h"

/// @brief 实例银行类：管理多帧检测实例的缓存和更新
class InstanceBank {
 public:
  /// @brief 构造函数：使用参数初始化实例银行
  InstanceBank(const sparsebev::TaskConfig& params);
  ~InstanceBank() = default;

  /// @brief 重置实例银行状态
  Status reset();

  /// @brief 更新实例银行状态
  /// @param instance_feature 实例特征，形状为(num_querys, embedfeat_dims)
  /// @param anchor 锚点，形状为(num_querys, query_dims)
  /// @param confidence 置信度，形状为(num_querys,)
  /// @param track_ids 跟踪ID数组，形状为(num_querys,)
  /// @param time_interval 时间间隔（秒）
  Status update(const std::vector<float>& instance_feature,
                const std::vector<float>& anchor,
                const std::vector<float>& confidence,
                const std::vector<std::int32_t>& track_ids,
                const float& time_interval);

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

  /// @brief 获取实例特征
  const std::vector<float>& getInstanceFeature() const { return instance_feature_; }

  /// @brief 获取锚点
  const std::vector<float>& getAnchor() const { return kmeans_anchors_; }

  /// @brief 获取置信度
  const std::vector<float>& getConfidence() const { return temp_confidence_; }

  /// @brief 获取跟踪ID
  const std::vector<std::int32_t>& getTrackIds() const { return temp_track_ids_; }

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

  /// @brief 缓存当前帧的前K个实例特征、锚点和置信度
  /// @param instance_feature 实例特征，形状为(num_querys, embedfeat_dims)
  /// @param anchor 锚点，形状为(num_querys, query_dims)
  /// @param confidence_logits 置信度logits，形状为(num_querys, class_nums)
  Status cache(const std::vector<float>& instance_feature,
               const std::vector<float>& anchor,
               const std::vector<float>& confidence_logits);

 private:
  /// @brief 初始化实例银行
  void initializeInstanceBank();

  /// @brief 从文件加载锚点数据
  /// @param file_path 锚点文件路径
  /// @return 是否加载成功
  bool loadAnchorsFromFile(const std::string& file_path);

  /// @brief 从文件加载实例特征数据
  /// @param file_path 实例特征文件路径
  /// @return 是否加载成功
  bool loadInstanceFeatureFromFile(const std::string& file_path);

  /// @brief 更新跟踪ID
  /// @param track_ids 跟踪ID数组，形状为(num_querys,)
  void updateTrackId(const std::vector<std::int32_t>& track_ids);

  /// @brief 更新置信度
  /// @param confidence 置信度，形状为(num_querys,)
  void updateConfidence(const std::vector<float>& confidence);

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

  // 配置参数
  sparsebev::TaskConfig params_;
  uint32_t num_querys_;
  uint32_t query_dims_;
  std::vector<float> kmeans_anchors_;
  uint32_t topk_querys_;
  float max_time_interval_;
  float default_time_interval_;
  float confidence_decay_;

  // 实例库状态
  uint32_t mask_;
  float history_time_;
  float time_interval_;
  Eigen::Matrix<double, 4, 4> temp_lidar_to_global_mat_;

  // 实例数据
  uint32_t track_size_;
  std::vector<float> instance_feature_;
  std::vector<float> temp_confidence_;
  std::vector<std::int32_t> temp_track_ids_;
  std::vector<std::int32_t> track_ids_;
  std::vector<float> confidence_;
  std::vector<float> anchor_;
  std::vector<float> query_;
  std::vector<float> query_anchor_;
  std::vector<float> query_confidence_;
  std::vector<std::int32_t> query_track_ids_;

  std::int32_t prev_id_;               // 前一个ID
};

#endif  // __INSTANCE_BANK_H__