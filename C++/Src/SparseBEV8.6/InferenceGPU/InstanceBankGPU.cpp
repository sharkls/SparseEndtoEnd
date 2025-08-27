#include "InstanceBankGPU.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <numeric>
#include <limits>
#include <algorithm>

#include "UtilsGPU_kernels.cuh"

InstanceBankGPU::InstanceBankGPU(const sparsebev::TaskConfig& params)
    : params_(params) {
    
    // 获取参数值
    num_anchors_ = params.instance_bank_params().num_querys();
    anchor_dims_ = params.instance_bank_params().query_dims();
    topk_anchors_ = params.instance_bank_params().topk_querys();
    max_time_interval_ = params.instance_bank_params().max_time_interval();
    default_time_interval_ = params.instance_bank_params().default_time_interval();
    confidence_decay_ = params.instance_bank_params().confidence_decay();
    embedfeat_dims_ = params.model_cfg_params().embedfeat_dims();
    
    // 添加详细的参数验证
    if (num_anchors_ == 0 || num_anchors_ > 10000) {
        std::cout << "[ERROR] Invalid num_anchors: " << num_anchors_ << std::endl;
        throw std::runtime_error("Invalid num_anchors value");
    }
    
    if (anchor_dims_ == 0 || anchor_dims_ > 100) {
        std::cout << "[ERROR] Invalid anchor_dims: " << anchor_dims_ << std::endl;
        throw std::runtime_error("Invalid anchor_dims value");
    }
    
    if (embedfeat_dims_ == 0 || embedfeat_dims_ > 10000) {
        std::cout << "[ERROR] Invalid embedfeat_dims: " << embedfeat_dims_ << std::endl;
        throw std::runtime_error("Invalid embedfeat_dims value");
    }
    
    if (topk_anchors_ == 0 || topk_anchors_ > num_anchors_) {
        std::cout << "[ERROR] Invalid topk_anchors: " << topk_anchors_ << " (should be <= " << num_anchors_ << ")" << std::endl;
        throw std::runtime_error("Invalid topk_anchors value");
    }
    
    // 检查内存大小计算是否会导致溢出
    size_t instance_feature_size = static_cast<size_t>(num_anchors_) * static_cast<size_t>(embedfeat_dims_);
    size_t anchor_size = static_cast<size_t>(num_anchors_) * static_cast<size_t>(anchor_dims_);
    size_t cached_feature_size = static_cast<size_t>(topk_anchors_) * static_cast<size_t>(embedfeat_dims_);
    size_t cached_anchor_size = static_cast<size_t>(topk_anchors_) * static_cast<size_t>(anchor_dims_);
    
    // 检查是否超出vector的最大大小
    if (instance_feature_size > std::vector<float>().max_size()) {
        std::cout << "[ERROR] Instance feature size too large: " << instance_feature_size << std::endl;
        throw std::runtime_error("Instance feature size too large");
    }
    
    if (anchor_size > std::vector<float>().max_size()) {
        std::cout << "[ERROR] Anchor size too large: " << anchor_size << std::endl;
        throw std::runtime_error("Anchor size too large");
    }
    
    std::cout << "[INFO] Memory sizes:" << std::endl;
    std::cout << "  - Instance feature: " << instance_feature_size << " floats (" << (instance_feature_size * sizeof(float)) << " bytes)" << std::endl;
    std::cout << "  - Anchor: " << anchor_size << " floats (" << (anchor_size * sizeof(float)) << " bytes)" << std::endl;
    std::cout << "  - Cached feature: " << cached_feature_size << " floats (" << (cached_feature_size * sizeof(float)) << " bytes)" << std::endl;
    std::cout << "  - Cached anchor: " << cached_anchor_size << " floats (" << (cached_anchor_size * sizeof(float)) << " bytes)" << std::endl;
    
    // 初始化实例库
    initializeInstanceBank();
}

void InstanceBankGPU::initializeInstanceBank() {
    try {
        // 使用size_t避免整数溢出
        size_t instance_feature_size = static_cast<size_t>(num_anchors_) * static_cast<size_t>(embedfeat_dims_);
        size_t anchor_size = static_cast<size_t>(num_anchors_) * static_cast<size_t>(anchor_dims_);
        size_t cached_feature_size = static_cast<size_t>(topk_anchors_) * static_cast<size_t>(embedfeat_dims_);
        size_t cached_anchor_size = static_cast<size_t>(topk_anchors_) * static_cast<size_t>(anchor_dims_);
        
        // 确保内存对齐（4字节对齐）
        if (instance_feature_size % 4 != 0) {
            instance_feature_size = ((instance_feature_size + 3) / 4) * 4;
        }
        if (anchor_size % 4 != 0) {
            anchor_size = ((anchor_size + 3) / 4) * 4;
        }
        if (cached_feature_size % 4 != 0) {
            cached_feature_size = ((cached_feature_size + 3) / 4) * 4;
        }
        if (cached_anchor_size % 4 != 0) {
            cached_anchor_size = ((cached_anchor_size + 3) / 4) * 4;
        }
        
        std::cout << "[INFO] Allocating GPU memory with alignment..." << std::endl;
        
        // 分配GPU内存
        m_gpu_instance_feature_ = CudaWrapper<float>(instance_feature_size);
        m_gpu_anchor_ = CudaWrapper<float>(anchor_size);
        m_gpu_confidence_ = CudaWrapper<float>(num_anchors_);
        m_gpu_track_ids_ = CudaWrapper<int32_t>(num_anchors_);
        m_gpu_cached_feature_ = CudaWrapper<float>(cached_feature_size);
        m_gpu_cached_anchor_ = CudaWrapper<float>(cached_anchor_size);
        m_gpu_cached_confidence_ = CudaWrapper<float>(topk_anchors_);
        m_gpu_cached_track_ids_ = CudaWrapper<int32_t>(topk_anchors_);
        
        // 初始化新的成员变量
        m_gpu_prev_id_wrapper_.allocate(1);
        std::vector<int32_t> prev_id_vec = {prev_id_};
        m_gpu_prev_id_wrapper_.cudaMemUpdateWrap(prev_id_vec);
        
        std::cout << "[INFO] GPU memory allocation completed successfully with alignment" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[ERROR] Failed to initialize GPU memory: " << e.what() << std::endl;
        throw;
    }
}

Status InstanceBankGPU::reset() {
    // 重置所有状态变量
    mask_ = 0;
    history_time_ = 0.0f;
    time_interval_ = 0.0f;
    temp_lidar_to_global_mat_ = Eigen::Matrix<double, 4, 4>::Identity();
    is_first_frame_ = true;
    prev_id_ = 0;
    
    // 清空GPU内存
    m_gpu_instance_feature_.cudaMemSetWrap();
    m_gpu_anchor_.cudaMemSetWrap();
    m_gpu_confidence_.cudaMemSetWrap();
    m_gpu_track_ids_.cudaMemSetWrap();
    m_gpu_cached_feature_.cudaMemSetWrap();
    m_gpu_cached_anchor_.cudaMemSetWrap();
    m_gpu_cached_confidence_.cudaMemSetWrap();
    m_gpu_cached_track_ids_.cudaMemSetWrap();
    
    // 修正：使用正确的cudaMemUpdateWrap调用方式
    std::vector<int32_t> prev_id_vec = {prev_id_};
    m_gpu_prev_id_wrapper_.cudaMemUpdateWrap(prev_id_vec);
    
    // 重新加载锚点数据和实例特征到GPU
    if(loadDataFromFile(params_.instance_bank_params().instance_bank_anchor_path(), num_anchors_ * anchor_dims_, m_gpu_anchor_)) {
        LOG(INFO) << "[INFO] Successfully loaded anchors from file and copied to GPU memory";
    } else {
        LOG(ERROR) << "[ERROR] Failed to load anchor data from file";
        return kInvalidInput;
    }
    
    std::string instance_feature_path = "/share/Code/Sparse4d/script/tutorial/asset/sample_0_instance_feature_1*900*256_float32.bin";
    if(loadDataFromFile(instance_feature_path, num_anchors_ * embedfeat_dims_, m_gpu_instance_feature_)) {
        LOG(INFO) << "[INFO] Successfully loaded instance feature data from file";
    } else {
        LOG(ERROR) << "[ERROR] Failed to load instance feature data from file";
        return kInvalidInput;
    }
    
    std::cout << "reset() instance_bank~ : " << time_interval_ << std::endl;
    return kSuccess;
}

std::tuple<const CudaWrapper<float>&, const CudaWrapper<float>&, 
           const CudaWrapper<float>&, const CudaWrapper<float>&, 
           const float&, const std::int32_t&, const CudaWrapper<std::int32_t>&>
InstanceBankGPU::get(const double& timestamp, const Eigen::Matrix<double, 4, 4>& global_to_lidar_mat, 
                     const bool& is_first_frame, cudaStream_t stream) {
    // 更新状态
    if (!is_first_frame) {
        time_interval_ = static_cast<float>(std::fabs(timestamp - history_time_) / 1000.0f);
        float epsilon = std::numeric_limits<float>::epsilon();
        mask_ = (time_interval_ < max_time_interval_ || std::fabs(time_interval_ - max_time_interval_) < epsilon) ? 1 : 0;
        time_interval_ = (static_cast<bool>(mask_) && time_interval_ > epsilon) ? time_interval_ : default_time_interval_;
        Eigen::Matrix<double, 4, 4> temp2cur_mat_double = global_to_lidar_mat * temp_lidar_to_global_mat_;
        Eigen::Matrix<float, 4, 4> temp2cur_mat = temp2cur_mat_double.cast<float>();
        anchorProjection(m_gpu_cached_anchor_, temp2cur_mat, time_interval_, stream);
        updateTrackId(stream);    // 修正：传入stream参数
    } else {
        reset();
        time_interval_ = default_time_interval_;
        std::cout << "reset() instance_bank~ : " << time_interval_;
    }
    
    history_time_ = static_cast<float>(timestamp);
    temp_lidar_to_global_mat_ = global_to_lidar_mat.inverse();  // 修正：删除多余分号
    is_first_frame_ = is_first_frame;
    
    std::cout << " time_interval_ :  " << time_interval_ << " , mask_ :" << mask_ << std::endl;

    // 返回缓存的数据
    return std::make_tuple(
        std::ref(m_gpu_instance_feature_),
        std::ref(m_gpu_anchor_),
        std::ref(m_gpu_cached_feature_),
        std::ref(m_gpu_cached_anchor_),  // 使用置信度作为质量分数
        time_interval_,
        mask_,
        std::ref(m_gpu_cached_track_ids_)
    );
}

Status InstanceBankGPU::cache(const CudaWrapper<float>& instance_feature,
                                  const CudaWrapper<float>& anchor,
                                  const CudaWrapper<float>& confidence_logits,
                             const bool& is_first_frame, cudaStream_t stream) {
    
    // 添加输入参数验证
    if (!instance_feature.isValid() || !anchor.isValid() || !confidence_logits.isValid()) {
        LOG(ERROR) << "[ERROR] Invalid input parameters in cache method";
        return kInvalidInput;
    }
    
    // 验证数据大小
    if (instance_feature.getSize() != num_anchors_ * embedfeat_dims_ ||
        anchor.getSize() != num_anchors_ * anchor_dims_ ||
        confidence_logits.getSize() != num_anchors_ * params_.model_cfg_params().num_classes()) {
        LOG(ERROR) << "[ERROR] Input data size mismatch in cache method";
        return kInvalidInput;
    }
    
    // 1. 获取最大置信度分数
    CudaWrapper<float> confidence = InstanceBankGPU::getMaxConfidenceScores(confidence_logits, num_anchors_);
    
    // 2. 应用置信度衰减和融合（非第一帧时）
    if (!is_first_frame) {
        // 创建临时GPU数组存储衰减后的置信度
        CudaWrapper<float> temp_anchor_confidence_with_decay(topk_anchors_);
        
        // 在GPU上应用置信度衰减
        Status decay_status = UtilsGPU::applyConfidenceDecayOnGPU(
            m_gpu_cached_confidence_,           // 输入：缓存的置信度
            temp_anchor_confidence_with_decay,  // 输出：衰减后的置信度
            confidence_decay_,                  // 衰减因子
            topk_anchors_,                      // 修正：使用topk_anchors_而不是getSize()
            stream
        );
        
        if (decay_status != kSuccess) {
            LOG(ERROR) << "[ERROR] Failed to apply confidence decay on GPU";
            return kInferenceErr;
        }
        
        // 在GPU上融合置信度（取最大值）
        Status fusion_status = UtilsGPU::fuseConfidenceOnGPU(
            temp_anchor_confidence_with_decay,  // 融合后的置信度
            m_gpu_cached_confidence_,          // 缓存的置信度
            topk_anchors_,                     // 修正：使用topk_anchors_
            stream
        );
        
        if (fusion_status != kSuccess) {
            LOG(ERROR) << "[ERROR] Failed to fuse confidence on GPU";
            return kInferenceErr;
        }
    }
    
    // 3. 保存当前置信度用于下次融合
    // 修正：使用正确的内存大小
    if (confidence.getSize() > 0) {
        cudaMemcpyAsync(m_gpu_confidence_.getCudaPtr(), 
                       confidence.getCudaPtr(),
                       std::min(confidence.getSize(), m_gpu_confidence_.getSize()) * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
        
        // 检查CUDA错误
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to copy confidence data: " << cudaGetErrorString(cuda_error);
            return kInferenceErr;
        }
    }
    
    // 4. 清空之前的缓存数据
    m_gpu_cached_feature_.cudaMemSetWrap();
    m_gpu_cached_anchor_.cudaMemSetWrap();
    m_gpu_cached_track_ids_.cudaMemSetWrap();
    
    // 5. 获取Top-K实例
    Status status = UtilsGPU::getTopkInstanceOnGPU(
        confidence, instance_feature, anchor,
        num_anchors_, anchor_dims_, embedfeat_dims_,
        topk_anchors_, m_gpu_cached_confidence_, 
        m_gpu_cached_feature_, m_gpu_cached_anchor_,
        m_gpu_cached_track_ids_,
        stream
    );
    
    if (status != kSuccess) {
        LOG(ERROR) << "[ERROR] Failed to get top-k instances";
        return kInferenceErr;
    }
    
    // 6. 同步CUDA流确保所有操作完成
    cudaStreamSynchronize(stream);
    
    return kSuccess;
}

 Status InstanceBankGPU::getTrackId(const bool& is_first_frame, CudaWrapper<int32_t>& track_ids, cudaStream_t stream) {
    if (is_first_frame) {
        // 生成新的跟踪ID
        Status status = UtilsGPU::generateNewTrackIdsOnGPU(
            m_gpu_track_ids_, m_gpu_prev_id_wrapper_, num_anchors_, stream);
        
        if (status == kSuccess) {
            // 将结果复制到输出参数
            cudaMemcpyAsync(track_ids.getCudaPtr(), m_gpu_track_ids_.getCudaPtr(),
                           num_anchors_ * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        }
        prev_id_ += num_anchors_;
        return status;
    } else {
        // 统计负值数量
        uint32_t negative_count = 0;
        Status status = UtilsGPU::countNegativeValuesOnGPU(
            m_gpu_track_ids_, negative_count, num_anchors_, stream);
        
        if (status != kSuccess) {
            LOG(ERROR) << "[ERROR] Failed to count negative values on GPU";
            return status;
        }
        
        if (negative_count > 0) {
            // 生成新的跟踪ID
            std::vector<int32_t> new_track_ids(negative_count);
            for (uint32_t i = 0; i < negative_count; ++i) {
                new_track_ids[i] = prev_id_ + i + 1;
            }
            prev_id_ += negative_count;
            
            // 修正：使用正确的cudaMemUpdateWrap调用方式
            std::vector<int32_t> prev_id_vec = {prev_id_};
            m_gpu_prev_id_wrapper_.cudaMemUpdateWrap(prev_id_vec);
            
            // 更新负值跟踪ID
            status = UtilsGPU::updateNegativeTrackIdsOnGPU(
                m_gpu_track_ids_, new_track_ids, num_anchors_, stream);
            
            if (status != kSuccess) {
                LOG(ERROR) << "[ERROR] Failed to update negative track IDs on GPU";
                return status;
            }
        }
        
        // 将结果复制到输出参数
        cudaMemcpyAsync(track_ids.getCudaPtr(), m_gpu_track_ids_.getCudaPtr(),
                       num_anchors_ * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        return kSuccess;
    }
}

void InstanceBankGPU::updateTrackId(cudaStream_t stream) {
    // 直接在GPU上完成索引选择和填充操作
    Status status = UtilsGPU::selectTrackIdsFromIndicesOnGPU(
        m_gpu_track_ids_,           // 输出：跟踪ID数组(topk个id + 900-k个-1)
        m_gpu_cached_track_ids_,    // 输入：索引数组（topk个索引值）
        num_anchors_,               // 目标长度（900）
        stream                       // 修正：传入stream参数
    );
    
    if (status != kSuccess) {
        LOG(ERROR) << "[ERROR] Failed to select track IDs from indices on GPU";
    } else {
        LOG(INFO) << "[INFO] Successfully updated track IDs in place";
    }
}

void InstanceBankGPU::updateConfidence(const CudaWrapper<float>& confidence) {
    m_gpu_confidence_.cudaMemcpyD2DWrap(confidence.getCudaPtr(), confidence.getSize());
}

void InstanceBankGPU::anchorProjection(CudaWrapper<float>& temp_anchor,
                                       const Eigen::Matrix<float, 4, 4>& temp_to_cur_mat,
                                       const float& time_interval,
                                       cudaStream_t stream) {
    UtilsGPU::anchorProjectionOnGPU(temp_anchor, temp_to_cur_mat, time_interval, 
                                   topk_anchors_, anchor_dims_, stream);
}

template <typename T>
CudaWrapper<T> InstanceBankGPU::getMaxConfidenceScores(const CudaWrapper<T>& confidence_logits,
                                                       const std::uint32_t& num_anchors) {
    // 验证输入参数
    if (!confidence_logits.isValid() || confidence_logits.getSize() == 0) {
        LOG(ERROR) << "[ERROR] Invalid confidence_logits in getMaxConfidenceScores";
        return CudaWrapper<T>(0);
    }
    
    // 计算正确的输出大小：每个anchor的最大置信度
    uint32_t output_size = num_anchors;
    
    CudaWrapper<T> max_scores(output_size);
    
    // 调用GPU函数
    Status status = UtilsGPU::getMaxConfidenceScoresOnGPU(
        confidence_logits, max_scores, 
        confidence_logits.getSize() / num_anchors,  // 每个anchor的类别数
        num_anchors,                                // anchor数量
        0                                           // stream
    );
    
    if (status != kSuccess) {
        LOG(ERROR) << "[ERROR] Failed to get max confidence scores on GPU";
        return CudaWrapper<T>(0);
    }
    
    return max_scores;
}

template <typename T>
T InstanceBankGPU::sigmoid(const T& logits) {
    if (logits > 0) {
        return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-logits));
    } else {
        T exp_x = std::exp(logits);
        return exp_x / (static_cast<T>(1.0) + exp_x);
    }
}

// 添加loadDataFromFile模板函数的实现
template<typename T>
bool InstanceBankGPU::loadDataFromFile(const std::string& file_path, size_t expected_size, CudaWrapper<T>& data) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << "[ERROR] Failed to open file: " << file_path;
        return false;
    }
    
    // 读取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 检查文件大小是否匹配
    if (file_size != expected_size * sizeof(T)) {
        LOG(ERROR) << "[ERROR] File size mismatch. Expected: " << expected_size * sizeof(T) 
                   << ", Actual: " << file_size;
        file.close();
        return false;
    }
    
    // 分配GPU内存
    data.allocate(expected_size);
    
    // 读取数据到临时缓冲区
    std::vector<T> temp_buffer(expected_size);
    file.read(reinterpret_cast<char*>(temp_buffer.data()), file_size);
    file.close();
    
    // 修正：cudaMemUpdateWrap返回void，不能用于if条件判断
    data.cudaMemUpdateWrap(temp_buffer);
    
    // 检查CUDA错误
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        LOG(ERROR) << "[ERROR] Failed to copy data to GPU: " << cudaGetErrorString(cuda_error);
        return false;
    }
    
    LOG(INFO) << "[INFO] Successfully loaded " << expected_size << " elements from " << file_path;
    return true;
}