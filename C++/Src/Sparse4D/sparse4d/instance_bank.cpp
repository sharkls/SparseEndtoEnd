#include "instance_bank.hpp"
#include "utils_kernels.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <cmath>
#include "../../../Include/Common/Core/GlobalContext.h"
#include "../../Common/Core/FunctionHub.h"
#include "../common/functionhub.hpp"

namespace sparse4d{
namespace instance_bank{

// // 显式定义虚析构函数，确保typeinfo符号被正确生成
// InstanceBank::~InstanceBank() = default;

class InstanceBankImplement : public InstanceBank{
public:
    ~InstanceBankImplement()  override = default;

    Status init(const TaskConfig &param) override;

    /// @brief 重置实例银行状态
    Status reset() override;

    Status get(const CTimeMatchSrcData *raw_data, const bool& is_first_frame, cudaStream_t stream, common::PipelineContext& pipeline_context) override;
    
    Status cache(common::HeadOutput& head_output, const bool& is_first_frame, cudaStream_t stream = 0) override;

    Status getTrackId(common::HeadOutput& head_output, const bool& is_first_frame, cudaStream_t stream = 0) override;

private:
    /// @brief 从文件加载锚点数据
    /// @param file_path 锚点文件路径
    /// @return 是否加载成功
    bool loadAnchorsFromFile(const std::string& file_path);

    /// @brief 从文件加载实例特征数据
    /// @param file_path 实例特征文件路径
    /// @return 是否加载成功
    bool loadInstanceFeatureFromFile(const std::string& file_path);

    /// @brief 从文件加载数据到GPU的通用函数
    /// @param file_path 文件路径
    /// @param expected_size 期望的数据大小
    /// @param data GPU数据包装器
    /// @return 是否加载成功
    template<typename T>
    bool loadDataFromFile(const std::string& file_path, size_t expected_size, CudaWrapper<T>& data);

    /// @brief 更新跟踪ID
    /// @param stream CUDA流
    void updateTrackId(cudaStream_t stream = 0);

    /// @brief 更新置信度
    /// @param confidence 置信度，形状为(num_querys,)
    void updateConfidence(const CudaWrapper<float>& confidence);

    /// @brief 时空对齐：将t-1时刻的锚点投影到t时刻
    /// @param temp_anchor 临时锚点，形状为(temp_num_querys, query_dims_)
    /// @param temp_to_cur_mat t-1到t时刻的变换矩阵
    /// @param time_interval 时间间隔
    /// @param stream CUDA流
    void anchorProjection(CudaWrapper<float>& temp_anchor,
                            const Eigen::Matrix<float, 4, 4>& temp_to_cur_mat,
                            const float& time_interval,
                            cudaStream_t stream = 0);

    /// @brief 计算tensor.max(-1).values.sigmoid，tensor形状为(x, y)
    /// @param confidence_logits 置信度logits，形状为(num_querys,)
    /// @param num_querys 查询数量
    /// @return 每个查询的最大置信度分数
    template <typename T>
    static CudaWrapper<T> getMaxConfidenceScores(const CudaWrapper<T>& confidence_logits,
                                                const std::uint32_t& num_anchors);
    
    TaskConfig m_taskConfig;            // 任务配置参数
    uint32_t num_anchors_;                      // 锚点数量 900
    uint32_t anchor_dims_;                      // 锚点特征维度 11
    std::vector<float> kmeans_anchors_; 
    uint32_t topk_anchors_;                     // 前k个锚点数量 600
    float max_time_interval_;                   // 最大时间间隔 2.0
    float default_time_interval_;               // 默认时间间隔 0.5
    float confidence_decay_;                    // 置信度衰减因子
    uint32_t embedfeat_dims_;                   // 实例特征维度 256

    // 实例库状态
    std::int32_t mask_;
    double history_time_;                       // 上一帧时间戳（毫秒），使用double避免精度丢失
    float time_interval_;                       // 时间间隔
    Eigen::Matrix<double, 4, 4> temp_lidar_to_global_mat_; // 上一帧激光雷达到全局坐标系的变换矩阵
    bool is_first_frame_;                       // 是否为第一帧
    std::int32_t prev_id_;                      // 前一个ID，用于生成新的跟踪ID

    // GPU内存包装器
    CudaWrapper<float> m_gpu_instance_feature_;     // 实例特征（900， 256）
    CudaWrapper<float> m_gpu_anchor_;               // 锚点(900, 11)
    CudaWrapper<float> m_gpu_confidence_;           // 置信度(900)
    CudaWrapper<int32_t> m_gpu_track_ids_;          // 跟踪ID(900)
    CudaWrapper<float> m_gpu_cached_feature_;       // 缓存的特征(600, 256)
    CudaWrapper<float> m_gpu_cached_anchor_;        // 缓存的锚点(600, 11)
    CudaWrapper<float> m_gpu_cached_confidence_;    // 缓存的置信度(600)
    CudaWrapper<int32_t> m_gpu_cached_track_ids_;   // 缓存的跟踪ID(600)
    CudaWrapper<int32_t> m_gpu_cached_track_id_index_; // 缓存的跟踪ID索引(600)
    CudaWrapper<int32_t> m_gpu_prev_id_wrapper_;    // 前一帧目标最大ID值的GPU包装器
};

Status InstanceBankImplement::init(const TaskConfig &param)
{
    LOG(INFO) << "[INFO] Sparse4D::InstanceBankImplement::init start";

    m_taskConfig = param;

    // 获取参数值
    num_anchors_ = param.instance_bank_params().num_querys();
    anchor_dims_ = param.instance_bank_params().query_dims();
    topk_anchors_ = param.instance_bank_params().topk_querys();
    max_time_interval_ = param.instance_bank_params().max_time_interval();
    default_time_interval_ = param.instance_bank_params().default_time_interval();
    confidence_decay_ = param.instance_bank_params().confidence_decay();
    embedfeat_dims_ = param.model_cfg_params().embedfeat_dims();

    // 使用size_t避免整数溢出
    size_t instance_feature_size = static_cast<size_t>(num_anchors_) * static_cast<size_t>(embedfeat_dims_);
    size_t anchor_size = static_cast<size_t>(num_anchors_) * static_cast<size_t>(anchor_dims_);
    size_t cached_feature_size = static_cast<size_t>(topk_anchors_) * static_cast<size_t>(embedfeat_dims_);
    size_t cached_anchor_size = static_cast<size_t>(topk_anchors_) * static_cast<size_t>(anchor_dims_);
    
    // 确保内存对齐（16字节对齐，这是GPU内存的标准对齐要求）
    const size_t alignment = 16;
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
    
    // 分配GPU内存
    m_gpu_instance_feature_ = CudaWrapper<float>(instance_feature_size);
    m_gpu_anchor_ = CudaWrapper<float>(anchor_size);
    m_gpu_confidence_ = CudaWrapper<float>(num_anchors_);
    m_gpu_track_ids_ = CudaWrapper<int32_t>(num_anchors_);
    m_gpu_cached_feature_ = CudaWrapper<float>(cached_feature_size);
    m_gpu_cached_anchor_ = CudaWrapper<float>(cached_anchor_size);
    m_gpu_cached_confidence_ = CudaWrapper<float>(topk_anchors_);
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

    Status status = reset();
    if (status != kSuccess) {
        LOG(ERROR) << "[ERROR] Failed to reset instance bank";
        return kInvalidInput;
    }
    
    // std::cout << "[INFO] GPU memory allocation completed successfully with 16-byte alignment" << std::endl;
    LOG(INFO) << "[INFO] InstanceBankImplement init completed";
    return Status::kSuccess;
}

Status InstanceBankImplement::reset()
{
    // 重置所有状态变量
    mask_ = 0;
    history_time_ = 0.0;
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
    m_gpu_cached_track_id_index_.cudaMemSetWrap();
    
    // 修正：使用正确的cudaMemUpdateWrap调用方式
    std::vector<int32_t> prev_id_vec = {prev_id_};
    m_gpu_prev_id_wrapper_.cudaMemUpdateWrap(prev_id_vec);
    
    // 重新加载锚点数据和实例特征到GPU
    if(loadDataFromFile(m_taskConfig.instance_bank_params().instance_bank_anchor_path(), num_anchors_ * anchor_dims_, m_gpu_anchor_)) {
        LOG(INFO) << "[INFO] Successfully loaded anchors from file and copied to GPU memory";
    } else {
        LOG(ERROR) << "[ERROR] Failed to load anchor data from file";
        return kInvalidInput;
    }
    
    if(loadDataFromFile(m_taskConfig.instance_bank_params().instance_bank_feature_path(), num_anchors_ * embedfeat_dims_, m_gpu_instance_feature_)) {
        LOG(INFO) << "[INFO] Successfully loaded instance feature data from file";
    } else {
        LOG(ERROR) << "[ERROR] Failed to load instance feature data from file";
        return kInvalidInput;
    }
    
    LOG(INFO) << "[INFO] reset() instance_bank~ : " << time_interval_;
    return kSuccess;
}

Status InstanceBankImplement::get(const CTimeMatchSrcData *raw_data, const bool& is_first_frame, cudaStream_t stream, common::PipelineContext& pipeline_context) 
{   
    const auto &g2l = raw_data->transform_info().global2lidar_matrix();
    Eigen::Matrix<double, 4, 4> global_to_lidar_mat;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            global_to_lidar_mat(r, c) = static_cast<double>(g2l[r * 4 + c]);
        }
    }

    auto begin_time = GetTimeStamp();
    // 更新状态
    if (!is_first_frame) {
        time_interval_ = static_cast<float>(std::fabs(static_cast<double>(raw_data->lTimeStamp()) - static_cast<double>(history_time_)) / 1000.0);
        float epsilon = std::numeric_limits<float>::epsilon();
        mask_ = (time_interval_ < max_time_interval_ || std::fabs(time_interval_ - max_time_interval_) < epsilon) ? 1 : 0;
        time_interval_ = (static_cast<bool>(mask_) && time_interval_ > epsilon) ? time_interval_ : default_time_interval_;
        Eigen::Matrix<double, 4, 4> temp2cur_mat_double = global_to_lidar_mat * temp_lidar_to_global_mat_;
        Eigen::Matrix<float, 4, 4> temp2cur_mat = temp2cur_mat_double.cast<float>();
        anchorProjection(m_gpu_cached_anchor_, temp2cur_mat, time_interval_, stream);

        updateTrackId(stream); 
    } else {
        // reset();
        time_interval_ = default_time_interval_;
        // std::cout << "default_time_interval_ : " << default_time_interval_ << std::endl;
        // std::cout << "reset() instance_bank~ : " << time_interval_;
    }
    
    history_time_ = static_cast<double>(raw_data->lTimeStamp());
    temp_lidar_to_global_mat_ = global_to_lidar_mat.inverse(); 
    is_first_frame_ = is_first_frame;
    
    // std::cout << " time_interval_ :  " << time_interval_ << " , mask_ :" << mask_ << std::endl;
    // auto end_time = GetTimeStamp();
    // auto total_time = end_time - begin_time;
    // LOG(INFO) << "[INFO] get completed in " << total_time << "ms";

    // 使用 GPU D2D 异步拷贝将数据复制到 pipeline_context（带大小限制）
    // 使用std::min限制拷贝大小，避免越界
    // LOG(INFO) << "[INFO] instance_feature_copy_size: " << pipeline_context.instance_feature.getSize() << " , m_gpu_instance_feature_.getSize(): " << m_gpu_instance_feature_.getSize();
    const std::uint64_t instance_feature_copy_size = std::min(pipeline_context.instance_feature.getSize(), m_gpu_instance_feature_.getSize());
    cudaError_t err = cudaMemcpyAsync(
        pipeline_context.instance_feature.getCudaPtr(),
        m_gpu_instance_feature_.getCudaPtr(),
        instance_feature_copy_size * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream
    );
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] Failed to copy instance_feature: " << cudaGetErrorString(err);
        return kInferenceErr;
    }

    // 拷贝锚点数据
    // LOG(INFO) << "[INFO] anchor_copy_size: " << pipeline_context.anchor.getSize() << " , m_gpu_anchor_.getSize(): " << m_gpu_anchor_.getSize();
    const std::uint64_t anchor_copy_size = std::min(pipeline_context.anchor.getSize(), m_gpu_anchor_.getSize());
    err = cudaMemcpyAsync(
        pipeline_context.anchor.getCudaPtr(),
        m_gpu_anchor_.getCudaPtr(),
        anchor_copy_size * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream
    );
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] Failed to copy anchor: " << cudaGetErrorString(err);
        return kInferenceErr;
    }

    // 第二帧时，将缓存的实例特征和锚点复制到 temp_instance_feature 和 temp_anchor（异步）
    if (!is_first_frame) {
        // 拷贝 temp_instance_feature（带大小限制）
        // LOG(INFO) << "[INFO] temp_feature_copy_size: " << pipeline_context.temp_instance_feature.getSize() << " , m_gpu_cached_feature_.getSize(): " << m_gpu_cached_feature_.getSize();
        const std::uint64_t temp_feature_copy_size = std::min(
            pipeline_context.temp_instance_feature.getSize(), 
            m_gpu_cached_feature_.getSize()
        );
        err = cudaMemcpyAsync(
            pipeline_context.temp_instance_feature.getCudaPtr(),
            m_gpu_cached_feature_.getCudaPtr(),
            temp_feature_copy_size * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        );
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to copy temp_instance_feature: " << cudaGetErrorString(err);
            return kInferenceErr;
        }

        // 拷贝 temp_anchor（带大小限制）
        // LOG(INFO) << "[INFO] temp_anchor_copy_size: " << pipeline_context.temp_anchor.getSize() << " , m_gpu_cached_anchor_.getSize(): " << m_gpu_cached_anchor_.getSize();
        const std::uint64_t temp_anchor_copy_size = std::min(
            pipeline_context.temp_anchor.getSize(), 
            m_gpu_cached_anchor_.getSize()
        );
        err = cudaMemcpyAsync(
            pipeline_context.temp_anchor.getCudaPtr(),
            m_gpu_cached_anchor_.getCudaPtr(),
            temp_anchor_copy_size * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        );
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to copy temp_anchor: " << cudaGetErrorString(err);
            return kInferenceErr;
        }
    }

    // 更新时间间隔（标量转 CudaWrapper，异步）
    std::vector<float> time_interval_vec = {time_interval_};
    pipeline_context.time_interval.cudaMemUpdateWrapAsync(time_interval_vec, stream);

    // 更新 mask（标量转 CudaWrapper，异步）
    std::vector<int32_t> mask_vec = {mask_};
    pipeline_context.mask.cudaMemUpdateWrapAsync(mask_vec, stream);
    // std::cout << "mask_ : " << mask_ << " , time_interval_ : " << time_interval_  << ", max_time_interval_:"<< max_time_interval_ << std::endl;

    // // 更新 track_ids（异步，带大小限制）
    // LOG(INFO) << "[INFO] track_ids_copy_size: " << pipeline_context.track_ids.getSize() << " , m_gpu_track_ids_.getSize(): " << m_gpu_track_ids_.getSize();
    // const std::uint64_t track_ids_copy_size = std::min(
    //     pipeline_context.track_ids.getSize(), 
    //     m_gpu_track_ids_.getSize()
    // );
    // err = cudaMemcpyAsync(
    //     pipeline_context.track_ids.getCudaPtr(),
    //     m_gpu_track_ids_.getCudaPtr(),
    //     track_ids_copy_size * sizeof(int32_t),
    //     cudaMemcpyDeviceToDevice,
    //     stream
    // );
    // if (err != cudaSuccess) {
    //     LOG(ERROR) << "[ERROR] Failed to copy track_ids: " << cudaGetErrorString(err);
    //     return kInferenceErr;
    // }

    return kSuccess;
}

Status InstanceBankImplement::cache(common::HeadOutput& head_output, const bool& is_first_frame, cudaStream_t stream) 
{
    // 检查指针是否为空
    if (!m_gpu_cached_feature_.getCudaPtr() || !m_gpu_cached_anchor_.getCudaPtr() ||
        !m_gpu_cached_track_ids_.getCudaPtr() || !m_gpu_cached_confidence_.getCudaPtr()) {
        LOG(ERROR) << "[ERROR] GPU memory pointers are null in cache method";
        return kInvalidInput;
    }

    // 添加输入参数验证
    if (!head_output.pred_instance_feature.isValid() || !head_output.pred_anchor.isValid() || !head_output.pred_class_score.isValid()) {
        LOG(ERROR) << "[ERROR] Invalid input parameters in cache method";
        return kInvalidInput;
    }

    // 验证数据大小
    if (head_output.pred_instance_feature.getSize() != num_anchors_ * embedfeat_dims_ ||
        head_output.pred_anchor.getSize() != num_anchors_ * anchor_dims_ ||
        head_output.pred_class_score.getSize() != num_anchors_ * m_taskConfig.model_cfg_params().num_classes()) {
        LOG(ERROR) << "[ERROR] Input data size mismatch in cache method";
        return kInvalidInput;
    }

    // 1. 获取最大置信度分数
    CudaWrapper<float> confidence = InstanceBankImplement::getMaxConfidenceScores(head_output.pred_class_score, num_anchors_);

    // 验证confidence对象
    if (!confidence.isValid()) {
        LOG(ERROR) << "[ERROR] Invalid confidence object after getMaxConfidenceScores";
        return kInferenceErr;
    }

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
                                        stream);

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
                                        stream);

        if (fusion_status != kSuccess) {
            LOG(ERROR) << "[ERROR] Failed to fuse confidence on GPU";
            return kInferenceErr;
        }
    }

    // 4. 清空之前的缓存数据
    m_gpu_cached_feature_.cudaMemSetWrap();
    m_gpu_cached_anchor_.cudaMemSetWrap();
    m_gpu_cached_track_ids_.cudaMemSetWrap();
    m_gpu_cached_track_id_index_.cudaMemSetWrap();

    // 验证输入参数的有效性
    if (!confidence.isValid() || !head_output.pred_instance_feature.isValid() || !head_output.pred_anchor.isValid()) {
        LOG(ERROR) << "[ERROR] Invalid input parameters for getTopkInstanceOnGPU";
        return kInvalidInput;
    }

    Status status = UtilsGPU::getTopkInstanceOnGPUThrust(     // getTopkInstanceOnGPU / getTopkInstanceOnGPUThrust / getTopkInstanceOnGPUOptimized
        confidence, head_output.pred_instance_feature, head_output.pred_anchor,
        num_anchors_, anchor_dims_, embedfeat_dims_,
        topk_anchors_, m_gpu_cached_confidence_, 
        m_gpu_cached_feature_, m_gpu_cached_anchor_,
        m_gpu_cached_track_id_index_,
        stream);

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

    // 检查CUDA错误
    cudaError_t pre_sync_error = cudaGetLastError();
    if (pre_sync_error != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA error before stream synchronization: " << cudaGetErrorString(pre_sync_error);
        return kInferenceErr;
    }

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

    return kSuccess;
}

Status InstanceBankImplement::getTrackId(common::HeadOutput& head_output, const bool& is_first_frame, cudaStream_t stream) 
{
    if (is_first_frame) 
    {
        // 生成新的跟踪ID
        Status status = UtilsGPU::generateNewTrackIdsOnGPU(
            m_gpu_track_ids_, m_gpu_prev_id_wrapper_, num_anchors_, stream);

        if (status == kSuccess) {
            // 将结果复制到输出参数
            cudaMemcpyAsync(head_output.pred_track_ids.getCudaPtr(), m_gpu_track_ids_.getCudaPtr(),
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
        cudaMemcpyAsync(head_output.pred_track_ids.getCudaPtr(), m_gpu_track_ids_.getCudaPtr(),
                       num_anchors_ * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        // updateTrackId(stream);
        return kSuccess;
    }
}

void InstanceBankImplement::updateTrackId(cudaStream_t stream) {
    // 根据topk的索引值从m_gpu_track_ids_中按topk顺序更新m_gpu_track_ids_并将后300个位置填充-1
    Status status = UtilsGPU::selectTrackIdsFromIndicesOnGPU(
        m_gpu_track_ids_,                  // 输入输出：当前目标的track_ids(900)， 输出排序后的track_dis(600) + -1(300)
        m_gpu_cached_track_id_index_,      // 输入：索引数组（topk个索引值）
        topk_anchors_,                     // topk个索引值
        stream                             // 修正：传入stream参数
    );
}

void InstanceBankImplement::anchorProjection(CudaWrapper<float>& temp_anchor,
                                       const Eigen::Matrix<float, 4, 4>& temp_to_cur_mat,
                                       const float& time_interval,
                                       cudaStream_t stream) {
    UtilsGPU::anchorProjectionOnGPU(temp_anchor, temp_to_cur_mat, time_interval, 
                                   topk_anchors_, anchor_dims_, stream);
}

template <typename T>
CudaWrapper<T> InstanceBankImplement::getMaxConfidenceScores(const CudaWrapper<T>& confidence_logits,
                                                       const std::uint32_t& num_anchors) 
{
    // 验证输入参数
    if (!confidence_logits.isValid() || confidence_logits.getSize() == 0) {
        LOG(ERROR) << "[ERROR] Invalid confidence_logits in getMaxConfidenceScores";
        return CudaWrapper<T>(0);
    }
    
    // 计算正确的输出大小：每个anchor的最大置信度
    uint32_t output_size = num_anchors;
    uint32_t num_classes = confidence_logits.getSize() / num_anchors;

    CudaWrapper<T> max_scores(output_size);
    
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
    
    return max_scores;
}

// 添加loadDataFromFile模板函数的实现
template<typename T>
bool InstanceBankImplement::loadDataFromFile(const std::string& file_path, size_t expected_size, CudaWrapper<T>& data) 
{
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

std::shared_ptr<InstanceBank> create_instance_bank(const TaskConfig &param) {
    auto instance = std::make_shared<InstanceBankImplement>();
    Status status = instance->init(param);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Failed to create instance bank";
        return nullptr;
    }
    return instance;
}

}// namespace instance_bank
}// namespace sparse4d