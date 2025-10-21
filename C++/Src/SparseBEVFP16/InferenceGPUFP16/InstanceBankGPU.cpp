#include "InstanceBankGPU.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <numeric>
#include <limits>
#include <algorithm>
#include <cuda_runtime.h>

#include "UtilsGPU_kernels.cuh"

InstanceBankGPU::InstanceBankGPU(const sparsebev::TaskConfig& params)
    : params_(params) {
    
    // 获取参数值
    num_anchors_ = params.instance_bank_params().num_querys();
    anchor_dims_ = params.instance_bank_params().query_dims();
    topk_anchors_ = params.instance_bank_params().topk_querys();
    max_time_interval_ = __float2half(params.instance_bank_params().max_time_interval());
    default_time_interval_ = __float2half(params.instance_bank_params().default_time_interval());
    confidence_decay_ = __float2half(params.instance_bank_params().confidence_decay());
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
    if (instance_feature_size > std::vector<half>().max_size()) {
        std::cout << "[ERROR] Instance feature size too large: " << instance_feature_size << std::endl;
        throw std::runtime_error("Instance feature size too large");
    }
    
    if (anchor_size > std::vector<half>().max_size()) {
        std::cout << "[ERROR] Anchor size too large: " << anchor_size << std::endl;
        throw std::runtime_error("Anchor size too large");
    }
    
    // std::cout << "[INFO] Memory sizes:" << std::endl;
    // std::cout << "  - Instance feature: " << instance_feature_size << " floats (" << (instance_feature_size * sizeof(float)) << " bytes)" << std::endl;
    // std::cout << "  - Anchor: " << anchor_size << " floats (" << (anchor_size * sizeof(float)) << " bytes)" << std::endl;
    // std::cout << "  - Cached feature: " << cached_feature_size << " floats (" << (cached_feature_size * sizeof(float)) << " bytes)" << std::endl;
    // std::cout << "  - Cached anchor: " << cached_anchor_size << " floats (" << (cached_anchor_size * sizeof(float)) << " bytes)" << std::endl;
    
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
        
        // 保存原始大小用于验证
        size_t original_cached_feature_size = cached_feature_size;
        size_t original_cached_anchor_size = cached_anchor_size;
        
        // 确保内存对齐（16字节对齐，这是GPU内存的标准对齐要求，但是half需要用8字节对齐）
        const size_t alignment = 8;
        if (instance_feature_size % alignment != 0) {
            instance_feature_size = ((instance_feature_size + alignment - 1) / alignment) * alignment;
        }
        if (anchor_size % alignment != 0) {
            anchor_size = ((anchor_size + alignment - 1) / alignment) * alignment;
        }
        if (cached_feature_size % alignment != 0) {
            cached_feature_size = ((cached_feature_size + alignment - 1) / alignment) * alignment;
        }
        if (cached_anchor_size % alignment != 0) {
            cached_anchor_size = ((cached_anchor_size + alignment - 1) / alignment) * alignment;
        }
        
        // std::cout << "[INFO] Allocating GPU memory with 16-byte alignment..." << std::endl;
        // std::cout << "[INFO] Original vs Aligned sizes:" << std::endl;
        // std::cout << "  - Instance feature: " << (num_anchors_ * embedfeat_dims_) << " -> " << instance_feature_size << " floats" << std::endl;
        // std::cout << "  - Anchor: " << (num_anchors_ * anchor_dims_) << " -> " << anchor_size << " floats" << std::endl;
        // std::cout << "  - Cached feature: " << original_cached_feature_size << " -> " << cached_feature_size << " floats" << std::endl;
        // std::cout << "  - Cached anchor: " << original_cached_anchor_size << " -> " << cached_anchor_size << " floats" << std::endl;
        
        // 分配GPU内存
        m_gpu_instance_feature_ = CudaWrapper<half>(instance_feature_size);
        m_gpu_anchor_ = CudaWrapper<half>(anchor_size);
        m_gpu_confidence_ = CudaWrapper<half>(num_anchors_);
        m_gpu_track_ids_ = CudaWrapper<int32_t>(num_anchors_);
        m_gpu_cached_feature_ = CudaWrapper<half>(cached_feature_size);
        m_gpu_cached_anchor_ = CudaWrapper<half>(cached_anchor_size);
        m_gpu_cached_confidence_ = CudaWrapper<half>(topk_anchors_);
        m_gpu_cached_track_ids_ = CudaWrapper<int32_t>(topk_anchors_);
        m_gpu_cached_track_id_index_ = CudaWrapper<int32_t>(topk_anchors_);
        
        // 验证所有GPU内存对象是否有效
        if (!m_gpu_instance_feature_.isValid() || !m_gpu_anchor_.isValid() || 
            !m_gpu_confidence_.isValid() || !m_gpu_track_ids_.isValid() ||
            !m_gpu_cached_feature_.isValid() || !m_gpu_cached_anchor_.isValid() ||
            !m_gpu_cached_confidence_.isValid() || !m_gpu_cached_track_ids_.isValid() ||
            !m_gpu_cached_track_id_index_.isValid()) {
            throw std::runtime_error("Failed to allocate valid GPU memory objects");
        }
        
        // 初始化新的成员变量
        m_gpu_prev_id_wrapper_.allocate(1);
        std::vector<int32_t> prev_id_vec = {prev_id_};
        m_gpu_prev_id_wrapper_.cudaMemUpdateWrap(prev_id_vec);
        
        // std::cout << "[INFO] GPU memory allocation completed successfully with 16-byte alignment" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[ERROR] Failed to initialize GPU memory: " << e.what() << std::endl;
        throw;
    }
}

Status InstanceBankGPU::reset() {
    // 重置所有状态变量
    mask_ = 0;
    history_time_ = 0;
    time_interval_ = __float2half(0.0f);
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
    m_gpu_cached_track_id_index_.cudaMemSetWrap();
    
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

std::tuple<const CudaWrapper<half>&, const CudaWrapper<half>&, 
           const CudaWrapper<half>&, const CudaWrapper<half>&, 
           const half, const std::int32_t, const CudaWrapper<std::int32_t>&>
InstanceBankGPU::get(const int64_t& timestamp, const Eigen::Matrix<double, 4, 4>& global_to_lidar_mat, 
                     const bool& is_first_frame, cudaStream_t stream) 
{   
    auto begin_time = GetTimeStamp();
    // 更新状态
    if (!is_first_frame) {
        // Eigen::Matrix<float, 4, 4> global_to_lidar_mat_float = global_to_lidar_mat.cast<float>();
        // std::vector<float> global_to_lidar_vec(global_to_lidar_mat_float.data(), 
        //                                     global_to_lidar_mat_float.data() + global_to_lidar_mat_float.size());
        // if(saveCpuDataToFile(global_to_lidar_vec, 16, "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_global_to_lidar_mat_4*4_float32.bin"))
        // {
        //     LOG(INFO) << "[INFO] cache global_to_lidar_mat saved successfully";
        // }
        time_interval_ = __float2half(std::fabs(timestamp - history_time_) / 1000.0f);
        half epsilon = std::numeric_limits<half>::epsilon();
        mask_ = (time_interval_ < max_time_interval_ || std::fabs(time_interval_ - max_time_interval_) < epsilon) ? 1 : 0;
        time_interval_ = (static_cast<bool>(mask_) && time_interval_ > epsilon) ? time_interval_ : default_time_interval_;
        Eigen::Matrix<double, 4, 4> temp2cur_mat_double = global_to_lidar_mat * temp_lidar_to_global_mat_;
        Eigen::Matrix<half, 4, 4> temp2cur_mat = temp2cur_mat_double.cast<half>();
        anchorProjection(m_gpu_cached_anchor_, temp2cur_mat, time_interval_, stream);
        // if(savePartialFast(m_gpu_cached_anchor_, 6600, "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_output_anchor_1*600*11_float32.bin"))
        // {
        //     LOG(INFO) << "[INFO] cache anchor saved successfully";
        // }
        // {
        //     cudaError_t err = cudaPeekAtLastError();
        //     if (err != cudaSuccess) { LOG(ERROR) << "[ERROR] anchorProjection launch: " << cudaGetErrorString(err); return std::make_tuple(std::ref(m_gpu_instance_feature_), std::ref(m_gpu_anchor_), std::ref(m_gpu_cached_feature_), std::ref(m_gpu_cached_anchor_), time_interval_, mask_, std::ref(m_gpu_cached_track_ids_)); }
        //     err = cudaStreamSynchronize(stream);
        //     if (err != cudaSuccess) { LOG(ERROR) << "[ERROR] anchorProjection exec: " << cudaGetErrorString(err); return std::make_tuple(std::ref(m_gpu_instance_feature_), std::ref(m_gpu_anchor_), std::ref(m_gpu_cached_feature_), std::ref(m_gpu_cached_anchor_), time_interval_, mask_, std::ref(m_gpu_cached_track_ids_)); }
        // }

        updateTrackId(stream); 
    } else {
        reset();
        time_interval_ = default_time_interval_;
        std::cout << "reset() instance_bank~ : " << time_interval_;
    }
    
    history_time_ = timestamp;
    temp_lidar_to_global_mat_ = global_to_lidar_mat.inverse(); 
    is_first_frame_ = is_first_frame;
    
    // std::cout << " time_interval_ :  " << time_interval_ << " , mask_ :" << mask_ << std::endl;
    auto end_time = GetTimeStamp();
    auto total_time = end_time - begin_time;
    LOG(INFO) << "[INFO] get completed in " << total_time << "ms";
    // 返回缓存的数据
    return std::make_tuple(
        std::ref(m_gpu_instance_feature_),
        std::ref(m_gpu_anchor_),
        std::ref(m_gpu_cached_feature_),
        std::ref(m_gpu_cached_anchor_),  // 使用置信度作为质量分数
        time_interval_,
        mask_,
        std::ref(m_gpu_track_ids_)
    );
}

Status InstanceBankGPU::cache(const CudaWrapper<half>& instance_feature,
                                const CudaWrapper<half>& anchor,
                                const CudaWrapper<half>& confidence_logits,
                                const bool& is_first_frame, cudaStream_t stream) 
{
    // 检查指针是否为空
    if (!m_gpu_cached_feature_.getCudaPtr() || !m_gpu_cached_anchor_.getCudaPtr() ||
        !m_gpu_cached_track_ids_.getCudaPtr() || !m_gpu_cached_confidence_.getCudaPtr()) {
        LOG(ERROR) << "[ERROR] GPU memory pointers are null in cache method";
        return kInvalidInput;
    }
    
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
    // std::cout << "[DEBUG] Getting max confidence scores..." << std::endl;
    CudaWrapper<half> confidence = InstanceBankGPU::getMaxConfidenceScores(confidence_logits, num_anchors_);

    // 验证confidence对象
    if (!confidence.isValid()) {
        LOG(ERROR) << "[ERROR] Invalid confidence object after getMaxConfidenceScores";
        return kInferenceErr;
    }
    
    // 2. 应用置信度衰减和融合（非第一帧时）
    if (!is_first_frame) {
        // 创建临时GPU数组存储衰减后的置信度
        CudaWrapper<half> temp_anchor_confidence_with_decay(topk_anchors_);
        
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
            confidence,                         // 输入：当前置信度
            temp_anchor_confidence_with_decay,  // 输入：缓存的置信度经过衰减
            confidence,                         // 输出：输出的置信度
            topk_anchors_,                     // 修正：使用topk_anchors_
            stream
        );
        
        if (fusion_status != kSuccess) {
            LOG(ERROR) << "[ERROR] Failed to fuse confidence on GPU";
            return kInferenceErr;
        }
    }
    
    // 4. 清空之前的缓存数据
    // std::cout << "[DEBUG] Clearing previous cache data..." << std::endl;
    m_gpu_cached_feature_.cudaMemSetWrap();
    m_gpu_cached_anchor_.cudaMemSetWrap();
    m_gpu_cached_track_ids_.cudaMemSetWrap();
    m_gpu_cached_track_id_index_.cudaMemSetWrap();
    
    // 5. 获取Top-K实例
    // std::cout << "[DEBUG] Getting top-k instances..." << std::endl;
    // std::cout << "[DEBUG] Calling getTopkInstanceOnGPU with parameters:" << std::endl;
    // std::cout << "  - confidence size: " << confidence.getSize() << std::endl;
    // std::cout << "  - instance_feature size: " << instance_feature.getSize() << std::endl;
    // std::cout << "  - anchor size: " << anchor.getSize() << std::endl;
    // std::cout << "  - num_anchors: " << num_anchors_ << std::endl;
    // std::cout << "  - anchor_dims: " << anchor_dims_ << std::endl;
    // std::cout << "  - embedfeat_dims: " << embedfeat_dims_ << std::endl;
    // std::cout << "  - topk_anchors: " << topk_anchors_ << std::endl;

    // 验证输入参数的有效性
    if (!confidence.isValid() || !instance_feature.isValid() || !anchor.isValid()) {
        LOG(ERROR) << "[ERROR] Invalid input parameters for getTopkInstanceOnGPU";
        return kInvalidInput;
    }

    Status status = UtilsGPU::getTopkInstanceOnGPUThrust(     // getTopkInstanceOnGPU / getTopkInstanceOnGPUThrust / getTopkInstanceOnGPUOptimized
        confidence, instance_feature, anchor,
        num_anchors_, anchor_dims_, embedfeat_dims_,
        topk_anchors_, m_gpu_cached_confidence_, 
        m_gpu_cached_feature_, m_gpu_cached_anchor_,
        m_gpu_cached_track_id_index_,
        stream
    );

    // {
    //     cudaError_t err = cudaPeekAtLastError();
    //     if (err != cudaSuccess) { LOG(ERROR) << "[ERROR] getTopkInstanceOnGPU launch: " << cudaGetErrorString(err); return kInferenceErr; }
    //     // err = cudaStreamSynchronize(stream);
    //     if (err != cudaSuccess) { LOG(ERROR) << "[ERROR] getTopkInstanceOnGPU exec: " << cudaGetErrorString(err); return kInferenceErr; }
    // }
    // 只检查启动错误，不同步
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) { 
        LOG(ERROR) << "[ERROR] getTopkInstanceOnGPU launch: " << cudaGetErrorString(err); 
        return kInferenceErr; 
    }

    if (status != kSuccess) {
        LOG(ERROR) << "[ERROR] Failed to get top-k instances";
        return kInferenceErr;
    }
    
    // 6. 在同步之前检查CUDA错误
    // std::cout << "[DEBUG] Checking CUDA errors before synchronization..." << std::endl;
    
    // 检查每个GPU内存对象的有效性
    // std::cout << "[DEBUG] GPU memory validation:" << std::endl;
    // std::cout << "  - m_gpu_cached_feature_: valid=" << m_gpu_cached_feature_.isValid() 
    //           << ", size=" << m_gpu_cached_feature_.getSize() 
    //           << ", ptr=" << m_gpu_cached_feature_.getCudaPtr() << std::endl;
    // std::cout << "  - m_gpu_cached_anchor_: valid=" << m_gpu_cached_anchor_.isValid() 
    //           << ", size=" << m_gpu_cached_anchor_.getSize() 
    //           << ", ptr=" << m_gpu_cached_anchor_.getCudaPtr() << std::endl;
    // std::cout << "  - m_gpu_cached_track_ids_: valid=" << m_gpu_cached_track_ids_.isValid() 
    //           << ", size=" << m_gpu_cached_track_ids_.getSize() 
    //           << ", ptr=" << m_gpu_cached_track_ids_.getCudaPtr() << std::endl;
    // std::cout << "  - m_gpu_cached_confidence_: valid=" << m_gpu_cached_confidence_.isValid() 
            //   << ", size=" << m_gpu_cached_confidence_.getSize() 
            //   << ", ptr=" << m_gpu_cached_confidence_.getCudaPtr() << std::endl;
    
    // // 检查指针对齐
    // if (m_gpu_cached_feature_.getCudaPtr()) {
    //     uintptr_t ptr_val = reinterpret_cast<uintptr_t>(m_gpu_cached_feature_.getCudaPtr());
    //     std::cout << "[DEBUG] m_gpu_cached_feature_ alignment: " << (ptr_val % 16) << std::endl;
    // }
    // if (m_gpu_cached_anchor_.getCudaPtr()) {
    //     uintptr_t ptr_val = reinterpret_cast<uintptr_t>(m_gpu_cached_anchor_.getCudaPtr());
    //     std::cout << "[DEBUG] m_gpu_cached_anchor_ alignment: " << (ptr_val % 16) << std::endl;
    // }
    // if (m_gpu_cached_track_ids_.getCudaPtr()) {
    //     uintptr_t ptr_val = reinterpret_cast<uintptr_t>(m_gpu_cached_track_ids_.getCudaPtr());
    //     std::cout << "[DEBUG] m_gpu_cached_track_ids_ alignment: " << (ptr_val % 16) << std::endl;
    // }
    // if (m_gpu_cached_confidence_.getCudaPtr()) {
    //     uintptr_t ptr_val = reinterpret_cast<uintptr_t>(m_gpu_cached_confidence_.getCudaPtr());
    //     std::cout << "[DEBUG] m_gpu_cached_confidence_ alignment: " << (ptr_val % 16) << std::endl;
    // }
    
    // 检查CUDA错误
    cudaError_t pre_sync_error = cudaGetLastError();
    if (pre_sync_error != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA error before stream synchronization: " << cudaGetErrorString(pre_sync_error);
        return kInferenceErr;
    }
    
    // 7. 同步CUDA流确保所有操作完成
    // std::cout << "[DEBUG] Synchronizing CUDA device..." << std::endl;
    
    // // 使用cudaDeviceSynchronize()来获取更详细的错误信息
    // cudaError_t sync_error = cudaDeviceSynchronize();
    // if (sync_error != cudaSuccess) {
    //     LOG(ERROR) << "[ERROR] CUDA device synchronization failed: " << cudaGetErrorString(sync_error);
        
    //     // 获取更多错误信息
    //     cudaDeviceProp prop;
    //     cudaGetDeviceProperties(&prop, 0);
    //     std::cout << "[ERROR] GPU device: " << prop.name << std::endl;
    //     std::cout << "[ERROR] CUDA capability: " << prop.major << "." << prop.minor << std::endl;
        
    //     return kInferenceErr;
    // }
    
    // 检查CUDA错误
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA error after device synchronization: " << cudaGetErrorString(cuda_error);
        return kInferenceErr;
    }
    
    // 验证所有GPU内存对象仍然有效
    if (!m_gpu_cached_feature_.isValid() || !m_gpu_cached_anchor_.isValid() || 
        !m_gpu_cached_track_ids_.isValid() || !m_gpu_cached_confidence_.isValid()) {
        LOG(ERROR) << "[ERROR] GPU memory objects became invalid after operations";
        return kInferenceErr;
    }
    
    // std::cout << "[DEBUG] cache() - Completed successfully" << std::endl;
    return kSuccess;
}

 Status InstanceBankGPU::getTrackId(const bool& is_first_frame, CudaWrapper<int32_t>& track_ids, cudaStream_t stream) {
    if (is_first_frame) 
    {
        // 生成新的跟踪ID
        Status status = UtilsGPU::generateNewTrackIdsOnGPU(
            m_gpu_track_ids_, m_gpu_prev_id_wrapper_, num_anchors_, stream);
        
        if (status == kSuccess) {
            // 将结果复制到输出参数
            cudaMemcpyAsync(track_ids.getCudaPtr(), m_gpu_track_ids_.getCudaPtr(),
                           num_anchors_ * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        }
        // prev_id_ += num_anchors_ - 1;

        std::vector<int32_t> prev_id_vec = {num_anchors_ - 1};
        m_gpu_prev_id_wrapper_.cudaMemUpdateWrap(prev_id_vec);
        return status;
    } else 
    {
        // 更新新的track_ids
        Status status = UtilsGPU::updateNewTrackIdsOnGPU(
            m_gpu_track_ids_, num_anchors_, topk_anchors_, m_gpu_prev_id_wrapper_,stream);

        // 将结果复制到输出参数
        cudaMemcpyAsync(track_ids.getCudaPtr(), m_gpu_track_ids_.getCudaPtr(),
                       num_anchors_ * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        // updateTrackId(stream);
        return kSuccess;
    }
}

void InstanceBankGPU::updateTrackId(cudaStream_t stream) {
    // 根据topk的索引值从m_gpu_track_ids_中按topk顺序更新m_gpu_track_ids_并将后300个位置填充-1
    Status status = UtilsGPU::selectTrackIdsFromIndicesOnGPU(
        m_gpu_track_ids_,                  // 输入输出：当前目标的track_ids(900)， 输出排序后的track_dis(600) + -1(300)
        m_gpu_cached_track_id_index_,      // 输入：索引数组（topk个索引值）
        topk_anchors_,                     // topk个索引值
        stream                             // 修正：传入stream参数
    );
}

void InstanceBankGPU::updateConfidence(const CudaWrapper<half>& confidence) {
    m_gpu_confidence_.cudaMemcpyD2DWrap(confidence.getCudaPtr(), confidence.getSize());
}

void InstanceBankGPU::anchorProjection(CudaWrapper<half>& temp_anchor,
                                       const Eigen::Matrix<half, 4, 4>& temp_to_cur_mat,
                                       const half& time_interval,
                                       cudaStream_t stream) {
    UtilsGPU::anchorProjectionOnGPU(temp_anchor, temp_to_cur_mat, time_interval, 
                                   topk_anchors_, anchor_dims_, stream);
}

template <typename T>
CudaWrapper<T> InstanceBankGPU::getMaxConfidenceScores(const CudaWrapper<T>& confidence_logits,
                                                       const std::uint32_t& num_anchors) {
    // std::cout << "[DEBUG] getMaxConfidenceScores() - Start" << std::endl;
    // 验证输入参数
    if (!confidence_logits.isValid() || confidence_logits.getSize() == 0) {
        LOG(ERROR) << "[ERROR] Invalid confidence_logits in getMaxConfidenceScores";
        return CudaWrapper<T>(0);
    }
    
    // 计算正确的输出大小：每个anchor的最大置信度
    uint32_t output_size = num_anchors;
    uint32_t num_classes = confidence_logits.getSize() / num_anchors;
    
    // std::cout << "[DEBUG] Calculated num_classes: " << num_classes << ", output_size: " << output_size << std::endl;
    
    CudaWrapper<T> max_scores(output_size);
    // std::cout << "[DEBUG] Created max_scores wrapper with size: " << max_scores.getSize() << std::endl;
    
    // 验证max_scores对象是否有效
    if (!max_scores.isValid()) {
        LOG(ERROR) << "[ERROR] Failed to create valid max_scores wrapper";
        return CudaWrapper<T>(0);
    }
    
    // 调用GPU函数
    Status status = UtilsGPU::getMaxConfidenceScoresOnGPU(
        confidence_logits,
        max_scores, 
        num_anchors,                    // anchor数量
        num_classes,                    // 每个anchor的类别数
        0                               // stream
    );
    
    if (status != kSuccess) {
        LOG(ERROR) << "[ERROR] Failed to get max confidence scores on GPU";
        return CudaWrapper<T>(0);
    }
    
    // 再次验证max_scores对象是否有效
    if (!max_scores.isValid()) {
        LOG(ERROR) << "[ERROR] max_scores wrapper became invalid after GPU operation";
        return CudaWrapper<T>(0);
    }
    
    // std::cout << "[DEBUG] getMaxConfidenceScores() - Completed successfully" << std::endl;
    return max_scores;
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
    
    // 检查文件大小是否匹配（文件是float格式）
    size_t expected_file_size = expected_size * sizeof(float);
    if (file_size != expected_file_size) {
        LOG(ERROR) << "[ERROR] File size mismatch. Expected: " << expected_file_size 
                   << " bytes (float format), Actual: " << file_size << " bytes";
        file.close();
        return false;
    }
    
    // 分配GPU内存
    data.allocate(expected_size);
    
    // 读取float数据到临时缓冲区
    std::vector<float> temp_float_buffer(expected_size);
    file.read(reinterpret_cast<char*>(temp_float_buffer.data()), file_size);
    file.close();
    
    // 根据目标类型T进行转换
    std::vector<T> temp_buffer(expected_size);
    
    if constexpr (std::is_same_v<T, float>) {
        // 目标类型是float，直接复制
        std::copy(temp_float_buffer.begin(), temp_float_buffer.end(), temp_buffer.begin());
    } else if constexpr (std::is_same_v<T, half>) {
        // 目标类型是half，进行类型转换
        for (size_t i = 0; i < expected_size; ++i) {
            temp_buffer[i] = __float2half(temp_float_buffer[i]);
        }
    } else if constexpr (std::is_same_v<T, int8_t>) {
        // 目标类型是int8，进行量化转换
        for (size_t i = 0; i < expected_size; ++i) {
            // 简单的线性量化：将float映射到int8范围[-128, 127]
            float clamped_value = std::max(-128.0f, std::min(127.0f, temp_float_buffer[i]));
            temp_buffer[i] = static_cast<int8_t>(std::round(clamped_value));
        }
    } else {
        // 不支持的类型
        LOG(ERROR) << "[ERROR] Unsupported target type for conversion from float";
        return false;
    }
    
    // 将转换后的数据复制到GPU
    data.cudaMemUpdateWrap(temp_buffer);
    
    // 检查CUDA错误
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        LOG(ERROR) << "[ERROR] Failed to copy data to GPU: " << cudaGetErrorString(cuda_error);
        return false;
    }
    
    LOG(INFO) << "[INFO] Successfully loaded and converted " << expected_size << " elements from " << file_path;
    return true;
}

/**
 * @brief 将CudaWrapper中的数据保存到文件
 * @param gpu 要保存的GPU数据
 * @param effective_elems 有效元素数量
 * @param path 保存文件路径
 * @return 保存是否成功
 */
template<typename T>
bool InstanceBankGPU::savePartialFast(const CudaWrapper<T>& gpu, size_t effective_elems, const std::string& path) {
    if (!gpu.isValid() || effective_elems == 0 || effective_elems > gpu.getSize()) return false;
    std::vector<T> host(effective_elems);
    cudaError_t err = cudaMemcpy(host.data(), gpu.getCudaPtr(),
                                 effective_elems * sizeof(T),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.write(reinterpret_cast<const char*>(host.data()), effective_elems * sizeof(T));
    return f.good();
}

// 在InstanceBankGPU.cpp中添加打印函数
void InstanceBankGPU::printCachedTrackIds() {
    // 直接复制数据，不需要同步stream
    std::vector<int32_t> cpu_track_ids = m_gpu_cached_track_ids_.cudaMemcpyD2HResWrap();
    
    std::cout << "=== m_gpu_cached_track_ids_ values ===" << std::endl;
    std::cout << "Size: " << cpu_track_ids.size() << " elements" << std::endl;
    
    // 逐元素打印
    for (size_t i = 0; i < cpu_track_ids.size(); ++i) {
        std::cout << "[" << i << "] = " << cpu_track_ids[i];
        
        // 每行打印10个元素，便于阅读
        if ((i + 1) % 10 == 0) {
            std::cout << std::endl;
        } else {
            std::cout << " ";
        }
    }
    
    // 如果最后一行没有满10个元素，换行
    if (cpu_track_ids.size() % 10 != 0) {
        std::cout << std::endl;
    }
    
    std::cout << "=== End of m_gpu_cached_track_ids_ ===" << std::endl;
}

/**
 * @brief 将CPU数据保存到文件
 * @param cpu_data 要保存的CPU数据
 * @param effective_elems 有效元素数量
 * @param path 保存文件路径
 * @return 保存是否成功
 */
template<typename T>
bool InstanceBankGPU::saveCpuDataToFile(const std::vector<T>& cpu_data, size_t effective_elems, const std::string& path) {
    if (effective_elems == 0 || effective_elems > cpu_data.size()) {
        std::cout << "[ERROR] Invalid effective_elems: " << effective_elems 
                  << ", cpu_data size: " << cpu_data.size() << std::endl;
        return false;
    }
    
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cout << "[ERROR] Failed to open file: " << path << std::endl;
        return false;
    }
    
    f.write(reinterpret_cast<const char*>(cpu_data.data()), effective_elems * sizeof(T));
    bool success = f.good();
    f.close();
    
    if (success) {
        std::cout << "[INFO] Successfully saved " << effective_elems 
                  << " elements to file: " << path << std::endl;
    } else {
        std::cout << "[ERROR] Failed to write data to file: " << path << std::endl;
    }
    
    return success;
}