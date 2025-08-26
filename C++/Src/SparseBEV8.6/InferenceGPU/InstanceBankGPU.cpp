#include "InstanceBankGPU.h"
#include "UtilsGPU.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <numeric>
#include <limits>

InstanceBankGPU::InstanceBankGPU(const sparsebev::TaskConfig& params)
    : params_(params),
      num_querys_(params.instance_bank_params().num_querys()),
      query_dims_(params.instance_bank_params().query_dims()),
      topk_querys_(params.instance_bank_params().topk_querys()),
      max_time_interval_(params.instance_bank_params().max_time_interval()),
      default_time_interval_(params.instance_bank_params().default_time_interval()),
      confidence_decay_(params.instance_bank_params().confidence_decay()),
      mask_(0),
      history_time_(0.0f),
      time_interval_(0.0f),
      temp_lidar_to_global_mat_(Eigen::Matrix<double, 4, 4>::Zero()),
      is_first_frame_(true),
      prev_id_(0) {
    
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
    
    // 初始化GPU内存
    if (!initializeGPUMemory()) {
        throw std::runtime_error("Failed to initialize GPU memory for InstanceBank");
    }
    
    std::cout << "[INFO] GPU InstanceBank initialized successfully:" << std::endl;
    std::cout << "  - num_querys: " << num_querys_ << std::endl;
    std::cout << "  - query_dims: " << query_dims_ << std::endl;
    std::cout << "  - kmeans_anchors size: " << kmeans_anchors_.size() << std::endl;
    std::cout << "  - topk_querys: " << topk_querys_ << std::endl;
}

InstanceBankGPU::~InstanceBankGPU() {
    cleanupGPUMemory();
}

bool InstanceBankGPU::initializeGPUMemory() {
    try {
        // 分配GPU内存
        gpu_instance_feature_ = std::make_unique<CudaWrapper<float>>();
        gpu_anchor_ = std::make_unique<CudaWrapper<float>>();
        gpu_confidence_ = std::make_unique<CudaWrapper<float>>();
        gpu_track_ids_ = std::make_unique<CudaWrapper<int32_t>>();
        gpu_cached_feature_ = std::make_unique<CudaWrapper<float>>();
        gpu_cached_anchor_ = std::make_unique<CudaWrapper<float>>();
        gpu_kmeans_anchors_ = std::make_unique<CudaWrapper<float>>();
        gpu_query_ = std::make_unique<CudaWrapper<float>>();
        gpu_query_anchor_ = std::make_unique<CudaWrapper<float>>();
        gpu_query_confidence_ = std::make_unique<CudaWrapper<float>>();
        gpu_query_track_ids_ = std::make_unique<CudaWrapper<int32_t>>();
        gpu_mask_ = std::make_unique<CudaWrapper<int32_t>>();
        gpu_time_interval_ = std::make_unique<CudaWrapper<float>>();
        gpu_prev_id_ = std::make_unique<CudaWrapper<int32_t>>();
        gpu_history_time_ = std::make_unique<CudaWrapper<float>>();
        gpu_temp_lidar_to_global_mat_ = std::make_unique<CudaWrapper<float>>();

        // 分配内存空间
        size_t feature_size = num_querys_ * params_.model_cfg_params().embedfeat_dims();
        size_t anchor_size = num_querys_ * query_dims_;
        size_t topk_feature_size = topk_querys_ * params_.model_cfg_params().embedfeat_dims();
        size_t topk_anchor_size = topk_querys_ * query_dims_;
        
        if (!gpu_instance_feature_->allocate(feature_size) ||
            !gpu_anchor_->allocate(anchor_size) ||
            !gpu_confidence_->allocate(num_querys_) ||
            !gpu_track_ids_->allocate(num_querys_) ||
            !gpu_cached_feature_->allocate(topk_feature_size) ||
            !gpu_cached_anchor_->allocate(topk_anchor_size) ||
            !gpu_kmeans_anchors_->allocate(kmeans_anchors_.size()) ||
            !gpu_query_->allocate(topk_feature_size) ||
            !gpu_query_anchor_->allocate(topk_anchor_size) ||
            !gpu_query_confidence_->allocate(topk_querys_) ||
            !gpu_query_track_ids_->allocate(topk_querys_) ||
            !gpu_mask_->allocate(1) ||
            !gpu_time_interval_->allocate(1) ||
            !gpu_prev_id_->allocate(1) ||
            !gpu_history_time_->allocate(1) ||
            !gpu_temp_lidar_to_global_mat_->allocate(16)) {  // 4x4 matrix
            return false;
        }

        // 初始化GPU内存
        gpu_mask_->cudaMemSetWrap(0);
        gpu_time_interval_->cudaMemSetWrap(0.0f);
        gpu_prev_id_->cudaMemSetWrap(0);
        gpu_history_time_->cudaMemSetWrap(0.0f);
        
        // 将锚点数据复制到GPU
        gpu_kmeans_anchors_->cudaMemUpdateWrap(kmeans_anchors_);
        
        // 初始化其他GPU内存为0
        gpu_instance_feature_->cudaMemSetWrap(0.0f);
        gpu_anchor_->cudaMemSetWrap(0.0f);
        gpu_confidence_->cudaMemSetWrap(0.0f);
        gpu_track_ids_->cudaMemSetWrap(-1);
        gpu_cached_feature_->cudaMemSetWrap(0.0f);
        gpu_cached_anchor_->cudaMemSetWrap(0.0f);
        gpu_query_->cudaMemSetWrap(0.0f);
        gpu_query_anchor_->cudaMemSetWrap(0.0f);
        gpu_query_confidence_->cudaMemSetWrap(0.0f);
        gpu_query_track_ids_->cudaMemSetWrap(-1);

        return true;
    } catch (...) {
        cleanupGPUMemory();
        return false;
    }
}

void InstanceBankGPU::cleanupGPUMemory() {
    gpu_instance_feature_.reset();
    gpu_anchor_.reset();
    gpu_confidence_.reset();
    gpu_track_ids_.reset();
    gpu_cached_feature_.reset();
    gpu_cached_anchor_.reset();
    gpu_kmeans_anchors_.reset();
    gpu_query_.reset();
    gpu_query_anchor_.reset();
    gpu_query_confidence_.reset();
    gpu_query_track_ids_.reset();
    gpu_mask_.reset();
    gpu_time_interval_.reset();
    gpu_prev_id_.reset();
    gpu_history_time_.reset();
    gpu_temp_lidar_to_global_mat_.reset();
}

Status InstanceBankGPU::reset() {
    // 重置CPU状态
    mask_ = 0;
    history_time_ = 0.0f;
    time_interval_ = 0.0f;
    temp_lidar_to_global_mat_ = Eigen::Matrix<double, 4, 4>::Zero();
    is_first_frame_ = true;
    prev_id_ = 0;
    
    // 重置GPU状态
    gpu_mask_->cudaMemSetWrap(0);
    gpu_time_interval_->cudaMemSetWrap(0.0f);
    gpu_prev_id_->cudaMemSetWrap(0);
    gpu_history_time_->cudaMemSetWrap(0.0f);
    
    // 从bin文件加载instance_feature初始值
    std::string instance_feature_path = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_instance_feature_1*900*256_float32.bin";
    
    if (loadInstanceFeatureFromFile(instance_feature_path)) {
        std::cout << "[INFO] Successfully loaded instance_feature from: " << instance_feature_path << std::endl;
        // 将加载的数据复制到GPU
        gpu_instance_feature_->cudaMemUpdateWrap(instance_feature_);
    } else {
        std::cout << "[WARNING] Failed to load instance_feature from: " << instance_feature_path << ", using zeros" << std::endl;
        gpu_instance_feature_->cudaMemSetWrap(0.0f);
    }
    
    // 重置其他GPU内存
    gpu_confidence_->cudaMemSetWrap(0.0f);
    gpu_track_ids_->cudaMemSetWrap(-1);
    gpu_anchor_->cudaMemSetWrap(0.0f);
    gpu_cached_feature_->cudaMemSetWrap(0.0f);
    gpu_cached_anchor_->cudaMemSetWrap(0.0f);
    gpu_query_->cudaMemSetWrap(0.0f);
    gpu_query_anchor_->cudaMemSetWrap(0.0f);
    gpu_query_confidence_->cudaMemSetWrap(0.0f);
    gpu_query_track_ids_->cudaMemSetWrap(-1);
    
    return Status::kSuccess;
}

Status InstanceBankGPU::updateOnGPU(const CudaWrapper<float>& instance_feature,
                                   const CudaWrapper<float>& anchor,
                                   const CudaWrapper<float>& confidence,
                                   const CudaWrapper<int32_t>& track_ids,
                                   float time_interval,
                                   cudaStream_t stream) {
    
    // 在GPU上更新数据
    cudaMemcpyAsync(gpu_instance_feature_->getCudaPtr(),
                    instance_feature.getCudaPtr(),
                    instance_feature.getSizeBytes(),
                    cudaMemcpyDeviceToDevice,
                    stream);
    
    cudaMemcpyAsync(gpu_anchor_->getCudaPtr(),
                    anchor.getCudaPtr(),
                    anchor.getSizeBytes(),
                    cudaMemcpyDeviceToDevice,
                    stream);
    
    cudaMemcpyAsync(gpu_confidence_->getCudaPtr(),
                    confidence.getCudaPtr(),
                    confidence.getSizeBytes(),
                    cudaMemcpyDeviceToDevice,
                    stream);
    
    cudaMemcpyAsync(gpu_track_ids_->getCudaPtr(),
                    track_ids.getCudaPtr(),
                    track_ids.getSizeBytes(),
                    cudaMemcpyDeviceToDevice,
                    stream);

    // 更新GPU上的状态
    std::vector<float> time_vec = {time_interval};
    gpu_time_interval_->cudaMemUpdateWrapAsync(time_vec, stream);

    return Status::kSuccess;
}

std::tuple<const CudaWrapper<float>&,
           const CudaWrapper<float>&,
           const CudaWrapper<float>&,
           const CudaWrapper<float>&,
           float,
           int32_t,
           const CudaWrapper<int32_t>&>
InstanceBankGPU::getOnGPU(const double& timestamp, 
                          const Eigen::Matrix<double, 4, 4>& global_to_lidar_mat, 
                          const bool& is_first_frame,
                          cudaStream_t stream) {
    
    if (!is_first_frame) {  // 第二帧
        time_interval_ = static_cast<float>(std::fabs(timestamp - history_time_) / 1000.0f);
        float epsilon = std::numeric_limits<float>::epsilon();
        mask_ = (time_interval_ < max_time_interval_ || std::fabs(time_interval_ - max_time_interval_) < epsilon) ? 1 : 0;
        time_interval_ = (static_cast<bool>(mask_) && time_interval_ > epsilon) ? time_interval_ : default_time_interval_;

        Eigen::Matrix<double, 4, 4> temp2cur_mat_double = global_to_lidar_mat * temp_lidar_to_global_mat_;
        Eigen::Matrix<float, 4, 4> temp2cur_mat = temp2cur_mat_double.cast<float>();
        
        // 在GPU上进行anchor投影
        anchorProjectionOnGPU(*gpu_query_anchor_, temp2cur_mat, time_interval_, stream);
        
        // 更新GPU上的状态
        std::vector<float> time_vec = {time_interval_};
        gpu_time_interval_->cudaMemUpdateWrapAsync(time_vec, stream);
        
        std::vector<int32_t> mask_vec = {mask_};
        gpu_mask_->cudaMemUpdateWrapAsync(mask_vec, stream);
    } else {  // 第一帧
        reset();
        time_interval_ = default_time_interval_;
        std::cout << "reset() GPU instance_bank~ : " << time_interval_ << std::endl;
    }

    history_time_ = timestamp;
    temp_lidar_to_global_mat_ = global_to_lidar_mat.inverse();
    
    // 将变换矩阵复制到GPU
    std::vector<float> transform_vec(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            transform_vec[i * 4 + j] = static_cast<float>(temp_lidar_to_global_mat_(i, j));
        }
    }
    gpu_temp_lidar_to_global_mat_->cudaMemUpdateWrapAsync(transform_vec, stream);

    std::cout << " time_interval_ : " << time_interval_ << " , mask_ :" << mask_ << std::endl;

    return std::make_tuple(std::cref(*gpu_instance_feature_), std::cref(*gpu_kmeans_anchors_),
                          std::cref(*gpu_query_), std::cref(*gpu_query_anchor_),
                          time_interval_, mask_, std::cref(*gpu_query_track_ids_));
}

Status InstanceBankGPU::cacheOnGPU(const CudaWrapper<float>& instance_feature,
                                  const CudaWrapper<float>& anchor,
                                  const CudaWrapper<float>& confidence_logits,
                                  const bool& is_first_frame,
                                  cudaStream_t stream) {
    
    // 在GPU上计算最大置信度分数
    auto gpu_confidence = getMaxConfidenceScoresOnGPU(confidence_logits, num_querys_, stream);
    
    // 应用置信度衰减和融合（在GPU上）
    if (!is_first_frame) {
        // 创建临时GPU内存用于置信度融合
        CudaWrapper<float> temp_query_confidence_with_decay;
        temp_query_confidence_with_decay.allocate(gpu_query_confidence_->getSize());
        
        // 在GPU上应用置信度衰减
        UtilsGPU::applyConfidenceDecayOnGPU(*gpu_query_confidence_, temp_query_confidence_with_decay, 
                                           confidence_decay_, stream);
        
        // 在GPU上进行置信度融合
        UtilsGPU::fuseConfidenceOnGPU(gpu_confidence, temp_query_confidence_with_decay, stream);
    }
    
    // 在GPU上获取top-k实例
    UtilsGPU::getTopkInstanceOnGPU(gpu_confidence, instance_feature, anchor,
                                   num_querys_, query_dims_, params_.model_cfg_params().embedfeat_dims(),
                                   topk_querys_, *gpu_query_confidence_, *gpu_query_, 
                                   *gpu_query_anchor_, *gpu_query_track_ids_, stream);

    return Status::kSuccess;
}

CudaWrapper<int32_t> InstanceBankGPU::getTrackIdOnGPU(const CudaWrapper<int32_t>& refined_track_ids, cudaStream_t stream) {
    // 在GPU上创建结果数组
    CudaWrapper<int32_t> track_ids;
    track_ids.allocate(num_querys_);
    track_ids.cudaMemSetWrap(-1);
    
    if (refined_track_ids.getSize() > 0) {
        if (refined_track_ids.getSize() != num_querys_) {
            throw "[ERROR] refined_track_ids size is mismatch !";
        }
        
        // 在GPU上复制refined_track_ids
        cudaMemcpyAsync(track_ids.getCudaPtr(), refined_track_ids.getCudaPtr(),
                        refined_track_ids.getSizeBytes(), cudaMemcpyDeviceToDevice, stream);
    }
    
    // 在GPU上计算新的track IDs
    UtilsGPU::generateNewTrackIdsOnGPU(track_ids, *gpu_prev_id_, stream);
    
    // 更新GPU上的prev_id
    std::vector<int32_t> prev_id_vec = {prev_id_};
    gpu_prev_id_->cudaMemUpdateWrapAsync(prev_id_vec, stream);
    
    // 更新GPU上的track_ids
    cudaMemcpyAsync(gpu_track_ids_->getCudaPtr(), track_ids.getCudaPtr(),
                    track_ids.getSizeBytes(), cudaMemcpyDeviceToDevice, stream);
    
    return track_ids;
}

CudaWrapper<int32_t> InstanceBankGPU::getTrackIdOnGPU(const bool& is_first_frame, cudaStream_t stream) {
    // 在GPU上创建结果数组
    CudaWrapper<int32_t> track_ids;
    track_ids.allocate(num_querys_);
    track_ids.cudaMemSetWrap(-1);
    
    if (!is_first_frame) {
        if (gpu_track_ids_->getSize() != num_querys_) {
            throw "[ERROR] track_ids_ size is mismatch !";
        }
        
        // 在GPU上复制现有的track_ids
        cudaMemcpyAsync(track_ids.getCudaPtr(), gpu_track_ids_->getCudaPtr(),
                        gpu_track_ids_->getSizeBytes(), cudaMemcpyDeviceToDevice, stream);
    }
    
    // 在GPU上计算新的track IDs
    UtilsGPU::generateNewTrackIdsOnGPU(track_ids, *gpu_prev_id_, stream);
    
    // 更新GPU上的prev_id
    std::vector<int32_t> prev_id_vec = {prev_id_};
    gpu_prev_id_->cudaMemUpdateWrapAsync(prev_id_vec, stream);
    
    // 更新GPU上的track_ids
    cudaMemcpyAsync(gpu_track_ids_->getCudaPtr(), track_ids.getCudaPtr(),
                    track_ids.getSizeBytes(), cudaMemcpyDeviceToDevice, stream);
    
    return track_ids;
}

void InstanceBankGPU::updateTrackIdOnGPU(const CudaWrapper<int32_t>& track_ids, cudaStream_t stream) {
    // 在GPU上更新track IDs
    cudaMemcpyAsync(gpu_track_ids_->getCudaPtr(), track_ids.getCudaPtr(),
                    track_ids.getSizeBytes(), cudaMemcpyDeviceToDevice, stream);
}

void InstanceBankGPU::updateConfidenceOnGPU(const CudaWrapper<float>& confidence, cudaStream_t stream) {
    // 在GPU上更新置信度
    cudaMemcpyAsync(gpu_confidence_->getCudaPtr(), confidence.getCudaPtr(),
                    confidence.getSizeBytes(), cudaMemcpyDeviceToDevice, stream);
}

void InstanceBankGPU::anchorProjectionOnGPU(CudaWrapper<float>& temp_anchor,
                                           const Eigen::Matrix<float, 4, 4>& temp2cur_mat,
                                           float time_interval,
                                           cudaStream_t stream) {
    // 在GPU上进行anchor投影计算
    UtilsGPU::anchorProjectionOnGPU(temp_anchor, temp2cur_mat, time_interval, topk_querys_, query_dims_, stream);
}

CudaWrapper<float> InstanceBankGPU::getMaxConfidenceScoresOnGPU(const CudaWrapper<float>& confidence_logits,
                                                                const uint32_t& num_querys,
                                                                cudaStream_t stream) {
    // 在GPU上计算最大置信度分数
    CudaWrapper<float> max_confidence_scores;
    max_confidence_scores.allocate(num_querys);
    
    UtilsGPU::getMaxConfidenceScoresOnGPU(confidence_logits, max_confidence_scores, num_querys, stream);
    
    // 应用Sigmoid激活函数
    applySigmoidOnGPU(max_confidence_scores, stream);
    
    return max_confidence_scores;
}

void InstanceBankGPU::applySigmoidOnGPU(CudaWrapper<float>& logits, cudaStream_t stream) {
    // 在GPU上应用Sigmoid激活函数
    UtilsGPU::applySigmoidOnGPU(logits, stream);
}

bool InstanceBankGPU::loadAnchorsFromFile(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[ERROR] Failed to open anchor file: " << file_path << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t num_floats = file_size / sizeof(float);
    size_t expected_size = num_querys_ * query_dims_;
    
    if (num_floats != expected_size) {
        std::cout << "[ERROR] Anchor file size mismatch. Expected: " 
                  << expected_size << " floats, got: " << num_floats << " floats" << std::endl;
        return false;
    }
    
    kmeans_anchors_.resize(num_floats);
    file.read(reinterpret_cast<char*>(kmeans_anchors_.data()), file_size);
    
    if (file.fail()) {
        std::cout << "[ERROR] Failed to read anchor data from file: " << file_path << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Successfully loaded anchors from file: " << file_path << std::endl;
    return true;
}

bool InstanceBankGPU::loadInstanceFeatureFromFile(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[ERROR] Failed to open instance_feature file: " << file_path << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t num_floats = file_size / sizeof(float);
    size_t expected_size = num_querys_ * params_.model_cfg_params().embedfeat_dims();
    
    if (num_floats != expected_size) {
        std::cout << "[ERROR] Instance feature file size mismatch. Expected: " 
                  << expected_size << " floats, got: " << num_floats << " floats" << std::endl;
        return false;
    }
    
    instance_feature_.resize(num_floats);
    file.read(reinterpret_cast<char*>(instance_feature_.data()), file_size);
    
    if (file.fail()) {
        std::cout << "[ERROR] Failed to read instance feature data from file: " << file_path << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Successfully loaded instance feature from file: " << file_path << std::endl;
    return true;
}

bool InstanceBankGPU::saveInstanceBankData(const int sample_id) {
    std::string output_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
    
    // 从GPU拷贝数据到CPU
    auto query_anchor_cpu = gpu_query_anchor_->cudaMemcpyD2HResWrap();
    auto query_cpu = gpu_query_->cudaMemcpyD2HResWrap();
    auto query_confidence_cpu = gpu_query_confidence_->cudaMemcpyD2HResWrap();
    auto track_ids_cpu = gpu_track_ids_->cudaMemcpyD2HResWrap();
    
    // 保存query_anchor_ (1*600*11)
    std::string anchor_path = output_dir + "sample_" + std::to_string(sample_id) + "_ibank_cached_anchor_1*600*11_float32.bin";
    std::ofstream anchor_file(anchor_path, std::ios::binary);
    if (anchor_file.is_open()) {
        anchor_file.write(reinterpret_cast<const char*>(query_anchor_cpu.data()), 
                         query_anchor_cpu.size() * sizeof(float));
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
        feature_file.write(reinterpret_cast<const char*>(query_cpu.data()), 
                          query_cpu.size() * sizeof(float));
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
        confidence_file.write(reinterpret_cast<const char*>(query_confidence_cpu.data()), 
                             query_confidence_cpu.size() * sizeof(float));
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
        track_id_file.write(reinterpret_cast<const char*>(track_ids_cpu.data()), 
                           track_ids_cpu.size() * sizeof(std::int32_t));
        track_id_file.close();
        std::cout << "[INFO] Saved track_ids_ to: " << track_id_path << std::endl;
    } else {
        std::cout << "[ERROR] Failed to open file for writing: " << track_id_path << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Successfully saved all GPU InstanceBank data for sample " << sample_id << std::endl;
    return true;
}