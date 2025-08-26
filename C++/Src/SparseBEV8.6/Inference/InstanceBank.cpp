#include "InstanceBank.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <numeric>
#include <limits>
#include <algorithm>

#include "Utils.h"

InstanceBank::InstanceBank(const sparsebev::TaskConfig& params)
    : params_(params),
      num_querys_(params.instance_bank_params().num_querys()),
      query_dims_(params.instance_bank_params().query_dims()),
      topk_querys_(params.instance_bank_params().topk_querys()),
      max_time_interval_(params.instance_bank_params().max_time_interval()),
      default_time_interval_(params.instance_bank_params().default_time_interval()),
      confidence_decay_(params.instance_bank_params().confidence_decay())
{
    // 验证参数的有效性
    if (num_querys_ == 0 || num_querys_ > 10000) {
        std::cout << "[ERROR] Invalid num_querys: " << num_querys_ << std::endl;
        throw std::runtime_error("Invalid num_querys value");
    }
    
    if (query_dims_ == 0 || query_dims_ > 100) {
        std::cout << "[ERROR] Invalid query_dims: " << query_dims_ << std::endl;
        throw std::runtime_error("Invalid query_dims value");
    }
    
    if (topk_querys_ == 0 || topk_querys_ > num_querys_) {
        std::cout << "[ERROR] Invalid topk_querys: " << topk_querys_ << " (should be <= " << num_querys_ << ")" << std::endl;
        throw std::runtime_error("Invalid topk_querys value");
    }
    
    // 加载锚点数据
    bool anchors_loaded = false;
    
    // 首先尝试从proto配置中的kmeans_anchors字段加载
    if (params.instance_bank_params().kmeans_anchors_size() > 0) {
        kmeans_anchors_.assign(params.instance_bank_params().kmeans_anchors().begin(),
                              params.instance_bank_params().kmeans_anchors().end());
        anchors_loaded = true;
        std::cout << "[INFO] Loaded anchors from proto config, size: " << kmeans_anchors_.size() << std::endl;
    }
    
    // 如果proto中没有锚点数据，尝试从文件加载
    if (!anchors_loaded && !params.instance_bank_params().instance_bank_anchor_path().empty()) {
        anchors_loaded = loadAnchorsFromFile(params.instance_bank_params().instance_bank_anchor_path());
    }
    
    // 如果都没有加载成功，使用默认值
    if (!anchors_loaded) {
        std::cout << "[WARNING] No anchors loaded, using default values" << std::endl;
        kmeans_anchors_.resize(num_querys_ * query_dims_, 0.0f);
    }
    
    // 验证锚点数据
    if (kmeans_anchors_.size() != num_querys_ * query_dims_) {
        std::cout << "[ERROR] Invalid kmeans_anchors size: " 
                  << kmeans_anchors_.size() 
                  << " (expected: " << num_querys_ * query_dims_ << ")" << std::endl;
        throw std::runtime_error("Invalid kmeans_anchors size");
    }
    
    // 打印锚点数据信息
    std::cout << "[INFO] InstanceBank initialized successfully:" << std::endl;
    std::cout << "  - num_querys: " << num_querys_ << std::endl;
    std::cout << "  - query_dims: " << query_dims_ << std::endl;
    std::cout << "  - kmeans_anchors size: " << kmeans_anchors_.size() << std::endl;
    std::cout << "  - topk_querys: " << topk_querys_ << std::endl;
    
    // 初始化实例库
    initializeInstanceBank();
}

bool InstanceBank::loadAnchorsFromFile(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[ERROR] Failed to open anchor file: " << file_path << std::endl;
        return false;
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 计算float数量
    size_t num_floats = file_size / sizeof(float);
    
    // 检查数据大小是否匹配预期
    size_t expected_size = num_querys_ * query_dims_;
    if (num_floats != expected_size) {
        std::cout << "[ERROR] Anchor file size mismatch. Expected: " 
                  << expected_size << " floats, got: " << num_floats << " floats" << std::endl;
        return false;
    }
    
    // 读取锚点数据
    kmeans_anchors_.resize(num_floats);
    file.read(reinterpret_cast<char*>(kmeans_anchors_.data()), file_size);
    
    if (file.fail()) {
        std::cout << "[ERROR] Failed to read anchor data from file: " << file_path << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Successfully loaded anchors from file: " << file_path << std::endl;
    std::cout << "[INFO] Anchor data size: " << num_floats << " floats" << std::endl;
    
    return true;
}

void InstanceBank::initializeInstanceBank()
{
    // 初始化成员变量
    mask_ = 0;
    history_time_ = 0.0F;
    time_interval_ = 0.0F;
    temp_lidar_to_global_mat_ = Eigen::Matrix<double, 4, 4>::Zero();
    is_first_frame_ = true;
    
    // 初始化向量大小
    track_size_ = static_cast<std::uint32_t>(num_querys_ * 10);
    
    // 初始化实例数据向量
    instance_feature_.resize(num_querys_ * params_.model_cfg_params().embedfeat_dims());
    temp_confidence_.resize(num_querys_);
    temp_track_ids_.resize(num_querys_);
    track_ids_.resize(num_querys_);
    confidence_.resize(num_querys_);
    anchor_.resize(num_querys_ * query_dims_);
    query_.resize(topk_querys_ * params_.model_cfg_params().embedfeat_dims());
    query_anchor_.resize(topk_querys_ * query_dims_);
    query_confidence_.resize(topk_querys_);
    query_track_ids_.resize(topk_querys_);
    
    // 初始化所有向量为0
    std::fill(instance_feature_.begin(), instance_feature_.end(), 0.0f);
    std::fill(temp_confidence_.begin(), temp_confidence_.end(), 0.0f);
    std::fill(temp_track_ids_.begin(), temp_track_ids_.end(), -1);
    std::fill(track_ids_.begin(), track_ids_.end(), -1);
    std::fill(confidence_.begin(), confidence_.end(), 0.0f);
    std::fill(anchor_.begin(), anchor_.end(), 0.0f);
    std::fill(query_.begin(), query_.end(), 0.0f);
    std::fill(query_anchor_.begin(), query_anchor_.end(), 0.0f);
    std::fill(query_confidence_.begin(), query_confidence_.end(), 0.0f);
    std::fill(query_track_ids_.begin(), query_track_ids_.end(), -1);
}

Status InstanceBank::reset()
{
    // 重置所有状态
    mask_ = 0;
    history_time_ = 0.0F;
    time_interval_ = 0.0F;
    temp_lidar_to_global_mat_ = Eigen::Matrix<double, 4, 4>::Zero();
    is_first_frame_ = true;
    
    // 从bin文件加载instance_feature初始值
    std::string instance_feature_path = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_instance_feature_1*900*256_float32.bin";
    
    if (loadInstanceFeatureFromFile(instance_feature_path)) {
        std::cout << "[INFO] Successfully loaded instance_feature from: " << instance_feature_path << std::endl;
    } else {
        std::cout << "[WARNING] Failed to load instance_feature from: " << instance_feature_path << ", using zeros" << std::endl;
        // 如果加载失败，使用零值初始化
        std::fill(instance_feature_.begin(), instance_feature_.end(), 0.0f);
    }
    
    // 清空其他向量
    std::fill(temp_confidence_.begin(), temp_confidence_.end(), 0.0f);
    std::fill(temp_track_ids_.begin(), temp_track_ids_.end(), -1);
    std::fill(track_ids_.begin(), track_ids_.end(), -1);
    std::fill(confidence_.begin(), confidence_.end(), 0.0f);
    std::fill(anchor_.begin(), anchor_.end(), 0.0f);
    std::fill(query_.begin(), query_.end(), 0.0f);
    std::fill(query_anchor_.begin(), query_anchor_.end(), 0.0f);
    std::fill(query_confidence_.begin(), query_confidence_.end(), 0.0f);
    std::fill(query_track_ids_.begin(), query_track_ids_.end(), -1);
    
    return Status::kSuccess;
}

Status InstanceBank::update(const std::vector<float>& instance_feature,
                           const std::vector<float>& anchor,
                           const std::vector<float>& confidence,
                           const std::vector<std::int32_t>& track_ids,
                           const float& time_interval)
{
    // 验证输入数据
    if (instance_feature.size() != num_querys_ * params_.model_cfg_params().embedfeat_dims() ||
        anchor.size() != num_querys_ * query_dims_ ||
        confidence.size() != num_querys_ ||
        track_ids.size() != num_querys_) {
        return Status::kInvalidInput;
    }
    
    // 更新时间间隔
    time_interval_ = time_interval;
    
    // 更新实例特征
    instance_feature_ = instance_feature;
    
    // 更新锚点
    anchor_ = anchor;
    
    // 更新置信度
    updateConfidence(confidence);
    
    // 更新跟踪ID
    updateTrackId(track_ids);
    
    return Status::kSuccess;
}

void InstanceBank::updateTrackId(const std::vector<std::int32_t>& track_ids)
{
    if (track_ids.size() != num_querys_) {
        return;
    }
    
    // 对应Python版本的逻辑：
    // track_id = track_id.flatten(end_dim=1)[self.temp_topk_indice].reshape(bs, -1)
    // self.track_id = F.pad(track_id, (0, self.num_anchor - self.num_temp_instances), value=-1)
    
    // 根据query_track_ids_中存储的索引，从输入的track_ids中选择对应位置的值
    std::vector<std::int32_t> selected_track_ids;
    selected_track_ids.reserve(query_track_ids_.size());
    
    for (size_t i = 0; i < query_track_ids_.size(); ++i) {
        // query_track_ids_[i]存储的是索引，用于从track_ids中选择值
        std::int32_t index = query_track_ids_[i];
        if (index >= 0 && static_cast<size_t>(index) < track_ids.size()) {
            selected_track_ids.push_back(track_ids[index]);
        } else {
            selected_track_ids.push_back(-1);  // 索引无效时设为-1
        }
    }
    
    // 创建最终结果数组，先复制选中的track_ids，然后用-1填充到num_querys_长度
    std::vector<std::int32_t> padded_track_ids;
    padded_track_ids.reserve(num_querys_);
    
    // 先添加选中的track_ids
    padded_track_ids.insert(padded_track_ids.end(), selected_track_ids.begin(), selected_track_ids.end());
    
    // 在数组后面添加-1，使其满足长度为num_querys_
    while (padded_track_ids.size() < static_cast<size_t>(num_querys_)) {
        padded_track_ids.push_back(-1);
    }
    
    // 更新成员变量
    temp_track_ids_ = padded_track_ids;
    track_ids_ = padded_track_ids;
    
    // LOG(INFO) << "[INFO] updateTrackId: selected " << selected_track_ids.size() 
    //           << " track_ids from indices, padded to " << padded_track_ids.size() << " with -1";
}

void InstanceBank::updateConfidence(const std::vector<float>& confidence)
{
    if (confidence.size() != num_querys_) {
        return;
    }
    
    temp_confidence_ = confidence;
    confidence_ = confidence;
}

void InstanceBank::anchorProjection(std::vector<float>& temp_topk_anchors,
                                    const Eigen::Matrix<float, 4, 4>& temp2cur_mat,
                                    const float& time_interval) {
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> temp_topk_anchors_mat(
      temp_topk_anchors.data(), topk_querys_, query_dims_);

  // t-1 center to t center
  auto center = temp_topk_anchors_mat.leftCols<3>();  // topk_querys_ * 3
  auto vel = temp_topk_anchors_mat.rightCols<3>();    // topk_querys_ * 3
  auto translation = vel * (-time_interval);          // topk_querys_ * 3
  center = center - translation;                      // topk_querys_ × 3

  Eigen::MatrixXf center_homo(topk_querys_, 4);
  center_homo.block(0, 0, topk_querys_, 3) = center;
  center_homo.col(3).setOnes();
  center_homo = center_homo * temp2cur_mat.transpose();

  // t-1 vel to t vel
  vel = vel * temp2cur_mat.block(0, 0, 3, 3).transpose();

  // t-1 yaw to t yaw
  auto yaw = temp_topk_anchors_mat.block(0, 6, topk_querys_, 2);  // topk_querys_ × 2
  yaw.col(0).swap(yaw.col(1));
  yaw = yaw * temp2cur_mat.block(0, 0, 2, 2).transpose();  // topk_querys_ × 2

  auto size = temp_topk_anchors_mat.block(0, 3, topk_querys_, 3);
  Eigen::MatrixXf temp2cur_anchor_m(topk_querys_, query_dims_);
  temp2cur_anchor_m << center_homo.leftCols<3>(), size, yaw, vel;

  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> temp2cur_anchor_map(
      temp_topk_anchors.data(), topk_querys_, query_dims_);
  temp2cur_anchor_map = temp2cur_anchor_m;
}

std::tuple<const std::vector<float>&,
           const std::vector<float>&,
           const std::vector<float>&,
           const std::vector<float>&,
           const float&,
           const std::int32_t&,
           const std::vector<std::int32_t>&>
InstanceBank::get(const double& timestamp, const Eigen::Matrix<double, 4, 4>& global_to_lidar_mat, const bool& is_first_frame) 
{ 
  
  // if (!query_anchor_.empty()) {   // 初始化实例时query_anchor_就不为空了
  if (!is_first_frame) {      // 第二帧
    time_interval_ = static_cast<float>(std::fabs(timestamp - history_time_) / 1000.0f);
    float epsilon = std::numeric_limits<float>::epsilon();
    mask_ = (time_interval_ < max_time_interval_ || std::fabs(time_interval_ - max_time_interval_) < epsilon) ? 1 : 0;
    time_interval_ = (static_cast<bool>(mask_) && time_interval_ > epsilon) ? time_interval_ : default_time_interval_;

    Eigen::Matrix<double, 4, 4> temp2cur_mat_double = global_to_lidar_mat * temp_lidar_to_global_mat_;
    Eigen::Matrix<float, 4, 4> temp2cur_mat = temp2cur_mat_double.cast<float>();
    anchorProjection(query_anchor_, temp2cur_mat, time_interval_);
  } else{       // 第一帧
    reset();
    time_interval_ = default_time_interval_;
    std::cout << "resrt() instance_bank~ : " << time_interval_;
  }

  history_time_ = timestamp;
  temp_lidar_to_global_mat_ = global_to_lidar_mat.inverse();

  // 调试：赋值后打印
//   std::cout << "[InstanceBank][DEBUG] After update: history_time_=" << history_time_ << std::endl;
//   std::cout << "[InstanceBank][DEBUG] temp_lidar_to_global_mat_:" << std::endl;
  std::cout << " time_interval_ :  " << time_interval_ << " , mask_ :" << mask_ << std::endl;
//   for (int r = 0; r < 4; ++r) {
//     std::cout << "  ";
//     for (int c = 0; c < 4; ++c) {
//       std::cout << temp_lidar_to_global_mat_(r, c) << (c == 3 ? "" : " ");
//     }
//     std::cout << std::endl;
//   }

  return std::make_tuple(std::cref(instance_feature_), std::cref(kmeans_anchors_),
                         std::cref(query_), std::cref(query_anchor_),
                         std::cref(time_interval_), std::cref(mask_), std::cref(query_track_ids_));
}

Status InstanceBank::cache(const std::vector<float>& instance_feature,
                                   const std::vector<float>& anchor,
                                   const std::vector<float>& confidence_logits,
                                   const bool& is_first_frame) {
  std::vector<float> confidence = InstanceBank::getMaxConfidenceScores(confidence_logits, num_querys_);

  // 应用置信度衰减和融合
  // if (!query_confidence_.empty()) {
  if(!is_first_frame){
    std::vector<float> temp_query_confidence_with_decay(query_confidence_.size());
    std::transform(query_confidence_.begin(), query_confidence_.end(), temp_query_confidence_with_decay.begin(),
                   [this](float element) { return element * confidence_decay_; });
    std::transform(confidence.begin(), confidence.begin() + topk_querys_, temp_query_confidence_with_decay.begin(),
                   confidence.begin(), [](float a, float b) { return std::max(a, b); });
  }
  
  // 保存当前置信度用于下次融合
  query_confidence_ = confidence;
  // updateConfidence(confidence);
  
  // 清空之前的缓存数据
  query_.clear();
  query_anchor_.clear();
  query_track_ids_.clear();
  
  // 创建临时uint32_t向量用于getTopkInstance函数
  std::vector<std::uint32_t> temp_topk_index;
  getTopkInstance(confidence, instance_feature, anchor, num_querys_, query_dims_, params_.model_cfg_params().embedfeat_dims(),
                  topk_querys_, query_confidence_, query_, query_anchor_, temp_topk_index);
  
  // 将uint32_t转换为int32_t
  query_track_ids_.resize(temp_topk_index.size());
  for (size_t i = 0; i < temp_topk_index.size(); ++i) {
    query_track_ids_[i] = static_cast<std::int32_t>(temp_topk_index[i]);
  }

//   LOG(INFO) << "[INFO] InstanceBank cache completed successfully";
//   LOG(INFO) << "[INFO] Cached feature size: " << query_.size();
//   LOG(INFO) << "[INFO] Cached anchor size: " << query_anchor_.size();
//   LOG(INFO) << "[INFO] Cached confidence size: " << query_confidence_.size();
//   LOG(INFO) << "[INFO] Cached track IDs size: " << query_track_ids_.size();

  return Status::kSuccess;
}

std::vector<std::int32_t> InstanceBank::getTrackId(const std::vector<std::int32_t>& refined_track_ids) {
  std::vector<std::int32_t> track_ids(num_querys_, -1);
  
  // std::cout << "[InstanceBank][DEBUG] getTrackId called with refined_track_ids size: " << refined_track_ids.size() << std::endl;
  // std::cout << "[InstanceBank][DEBUG] num_querys_: " << num_querys_ << std::endl;
  
  if (!refined_track_ids.empty()) {
    if (refined_track_ids.size() != num_querys_) {
      throw "[ERROR] refined_track_ids size is mismatch !";
    }
    std::copy(refined_track_ids.begin(), refined_track_ids.end(), track_ids.begin());
    std::cout << "[InstanceBank][DEBUG] Copied refined_track_ids to track_ids" << std::endl;
  } else {
    std::cout << "[InstanceBank][DEBUG] refined_track_ids is empty, will generate new track_ids" << std::endl;
  }

  auto nums_new_anchor = static_cast<size_t>(
      std::count_if(track_ids.begin(), track_ids.end(), [](std::int32_t track_id) { return track_id < 0; }));
  
  std::vector<std::int32_t> new_track_ids(nums_new_anchor);
  for (size_t k = 0; k < nums_new_anchor; ++k) {
    new_track_ids[k] = k + prev_id_;
  }

  for (size_t i = num_querys_ - nums_new_anchor, j = 0; i < num_querys_ && j < nums_new_anchor; ++i, ++j) {
    track_ids[i] = new_track_ids[j];
  }

  prev_id_ += nums_new_anchor;
  updateTrackId(track_ids);

//   if (!track_ids_.empty()) {
//         LOG(INFO) << "[DEBUG] All track_ids_ values:";
//         for (size_t i = 0; i < track_ids_.size(); ++i) {
//             std::cout << " " << track_ids_[i];
//         }
//     } else {
//         LOG(INFO) << "[DEBUG] track_ids_ is empty";
//     }
//     LOG(INFO) << "[DEBUG] =================================================";
  
  return track_ids;     // track_ids_
}

std::vector<std::int32_t> InstanceBank::getTrackId(const bool& is_first_frame) {
  std::vector<std::int32_t> track_ids(num_querys_, -1);
  
  // std::cout << "[InstanceBank][DEBUG] getTrackId called with track_ids_ size: " << track_ids_.size() << std::endl;
  // std::cout << "[InstanceBank][DEBUG] num_querys_: " << num_querys_ << std::endl;
  
  if (!is_first_frame) {
    if (track_ids_.size() != num_querys_) {
      throw "[ERROR] track_ids_ size is mismatch !";
    }
    std::copy(track_ids_.begin(), track_ids_.end(), track_ids.begin());
    std::cout << "[InstanceBank][DEBUG] Copied track_ids_ to track_ids" << std::endl;
  } else {
    std::cout << "[InstanceBank][DEBUG] track_ids_ is empty, will generate new track_ids" << std::endl;
  }

  auto nums_new_anchor = static_cast<size_t>(
      std::count_if(track_ids.begin(), track_ids.end(), [](std::int32_t track_id) { return track_id < 0; }));
  
  std::vector<std::int32_t> new_track_ids(nums_new_anchor);
  for (size_t k = 0; k < nums_new_anchor; ++k) {
    new_track_ids[k] = k + prev_id_;
  }

  for (size_t i = num_querys_ - nums_new_anchor, j = 0; i < num_querys_ && j < nums_new_anchor; ++i, ++j) {
    track_ids[i] = new_track_ids[j];
  }

  prev_id_ += nums_new_anchor;
  updateTrackId(track_ids);

//   if (!track_ids_.empty()) {
//         LOG(INFO) << "[DEBUG] All track_ids_ values:";
//         for (size_t i = 0; i < track_ids_.size(); ++i) {
//             std::cout << " " << track_ids_[i];
//         }
//     } else {
//         LOG(INFO) << "[DEBUG] track_ids_ is empty";
//     }
//     LOG(INFO) << "[DEBUG] =================================================";
  
  return track_ids;     // track_ids_
}

std::vector<float> InstanceBank::getTempConfidence() const { return temp_confidence_; }

std::vector<float> InstanceBank::getTempTopKConfidence() const { return query_confidence_; }

std::vector<float> InstanceBank::getCachedFeature() const { return query_; }

std::vector<float> InstanceBank::getCachedAnchor() const { return query_anchor_; }

std::int32_t InstanceBank::getPrevId() const { return prev_id_; }

std::vector<std::int32_t> InstanceBank::getCachedTrackIds() const { return query_track_ids_; }

template <typename T>
std::vector<T> InstanceBank::getMaxConfidenceScores(const std::vector<T>& confidence_logits,
                                                    const std::uint32_t& num_querys) {
  std::vector<T> max_confidence_scores;
  for (std::uint32_t i = 0; i < num_querys; ++i) {
    T max_confidence_logit = confidence_logits[i * static_cast<std::uint32_t>(ObstacleType::OBSTACLETYPE_Max)];
    for (std::uint32_t j = 0; j < static_cast<std::uint32_t>(ObstacleType::OBSTACLETYPE_Max); ++j) {
      std::uint32_t index = i * static_cast<std::uint32_t>(ObstacleType::OBSTACLETYPE_Max) + j;
      if (confidence_logits[index] > max_confidence_logit) {
        max_confidence_logit = confidence_logits[index];
      }
    }
    T max_confidence_score = sigmoid(max_confidence_logit);
    max_confidence_scores.emplace_back(max_confidence_score);
  }
  return max_confidence_scores;
}

template <typename T>
T InstanceBank::sigmoid(const T& score_logits) {
  T scores = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-score_logits));

  return scores;
}

bool InstanceBank::loadInstanceFeatureFromFile(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[ERROR] Failed to open instance_feature file: " << file_path << std::endl;
        return false;
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 计算float数量
    size_t num_floats = file_size / sizeof(float);
    
    // 检查数据大小是否匹配预期
    size_t expected_size = num_querys_ * params_.model_cfg_params().embedfeat_dims();
    if (num_floats != expected_size) {
        std::cout << "[ERROR] Instance feature file size mismatch. Expected: " 
                  << expected_size << " floats, got: " << num_floats << " floats" << std::endl;
        std::cout << "[ERROR] Expected shape: [" << num_querys_ << ", " 
                  << params_.model_cfg_params().embedfeat_dims() << "]" << std::endl;
        return false;
    }
    
    // 读取实例特征数据
    instance_feature_.resize(num_floats);
    file.read(reinterpret_cast<char*>(instance_feature_.data()), file_size);
    
    if (file.fail()) {
        std::cout << "[ERROR] Failed to read instance feature data from file: " << file_path << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Successfully loaded instance feature from file: " << file_path << std::endl;
    std::cout << "[INFO] Instance feature data size: " << num_floats << " floats" << std::endl;
    std::cout << "[INFO] Instance feature shape: [" << num_querys_ << ", " 
              << params_.model_cfg_params().embedfeat_dims() << "]" << std::endl;
    
    // 打印前几个值的统计信息
    if (!instance_feature_.empty()) {
        float min_val = instance_feature_[0];
        float max_val = instance_feature_[0];
        float sum = 0.0f;
        
        for (size_t i = 0; i < std::min(instance_feature_.size(), size_t(1000)); ++i) {
            float val = instance_feature_[i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
        }
        
        std::cout << "[INFO] Instance feature stats (first 1000 elements):" << std::endl;
        std::cout << "  - Min value: " << min_val << std::endl;
        std::cout << "  - Max value: " << max_val << std::endl;
        std::cout << "  - Average: " << (sum / std::min(instance_feature_.size(), size_t(1000))) << std::endl;
    }
    
    return true;
}

bool InstanceBank::saveInstanceBankData(const int sample_id) {
    std::string output_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
    
    // 保存query_anchor_ (1*600*11)
    std::string anchor_path = output_dir + "sample_" + std::to_string(sample_id) + "_ibank_cached_anchor_1*600*11_float32.bin";
    std::ofstream anchor_file(anchor_path, std::ios::binary);
    if (anchor_file.is_open()) {
        anchor_file.write(reinterpret_cast<const char*>(query_anchor_.data()), 
                         query_anchor_.size() * sizeof(float));
        anchor_file.close();
        std::cout << "[INFO] Saved query_anchor_ to: " << anchor_path << std::endl;
    } else {
        std::cout << "[ERROR] Failed to open file for writing: " << anchor_path << std::endl;
        return false;
    }
    
    // 保存query_ (1*600*256)
    std::string feature_path = output_dir + "sample_" + std::to_string(sample_id) + "_ibank_cached_feature_1*600*256_float32.bin";
    std::ofstream feature_file(feature_path, std::ios::binary);
    if (feature_file.is_open()) {
        feature_file.write(reinterpret_cast<const char*>(query_.data()), 
                          query_.size() * sizeof(float));
        feature_file.close();
        std::cout << "[INFO] Saved query_ to: " << feature_path << std::endl;
    } else {
        std::cout << "[ERROR] Failed to open file for writing: " << feature_path << std::endl;
        return false;
    }
    
    // 保存query_confidence_ (1*600)
    std::string confidence_path = output_dir + "sample_" + std::to_string(sample_id) + "_ibank_confidence_1*600_float32.bin";
    std::ofstream confidence_file(confidence_path, std::ios::binary);
    if (confidence_file.is_open()) {
        confidence_file.write(reinterpret_cast<const char*>(query_confidence_.data()), 
                             query_confidence_.size() * sizeof(float));
        confidence_file.close();
        std::cout << "[INFO] Saved query_confidence_ to: " << confidence_path << std::endl;
    } else {
        std::cout << "[ERROR] Failed to open file for writing: " << confidence_path << std::endl;
        return false;
    }
    
    // 保存track_ids_ (1*900)
    std::string track_id_path = output_dir + "sample_" + std::to_string(sample_id) + "_ibank_updated_temp_track_id_1*900_int32.bin";
    std::ofstream track_id_file(track_id_path, std::ios::binary);
    if (track_id_file.is_open()) {
        track_id_file.write(reinterpret_cast<const char*>(track_ids_.data()), 
                           track_ids_.size() * sizeof(std::int32_t));
        track_id_file.close();
        std::cout << "[INFO] Saved track_ids_ to: " << track_id_path << std::endl;
    } else {
        std::cout << "[ERROR] Failed to open file for writing: " << track_id_path << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Successfully saved all InstanceBank data for sample " << sample_id << std::endl;
    return true;
}
