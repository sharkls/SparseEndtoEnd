/*******************************************************
 文件名：InstanceBankGPU.h
 作者：sharkls
 描述：GPU版本的InstanceBank类，提供GPU上的实例管理功能
 版本：v1.0
 日期：2025-06-18
 *******************************************************/
#ifndef __INSTANCE_BANK_GPU_H__
#define __INSTANCE_BANK_GPU_H__

#include <memory>
#include <tuple>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>



#include <vector>
#include <cstdint>
// 使用条件编译避免重复包含Eigen
#ifndef EIGEN_DENSE_INCLUDED
#define EIGEN_DENSE_INCLUDED
#include <Eigen/Dense>
#endif

#include "GlobalContext.h"
#include "SparseEnd2EndConfig_conf.pb.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"

/// @brief GPU版本的实例银行类
class InstanceBankGPU {
  public:
    /// @brief 构造函数：使用参数初始化实例银行
    InstanceBankGPU(const sparsebev::TaskConfig& params);

    /// @brief 析构函数
    ~InstanceBankGPU() = default;

    /// @brief 重置实例银行状态
    Status reset();

    /// @brief 获取指定时间戳的实例数据
    /// @param timestamp 时间戳
    /// @param global_to_lidar_mat 全局坐标系到激光雷达坐标系的变换矩阵
    /// @param is_first_frame 是否为第一帧
    /// @return 返回缓存的实例特征、锚点、置信度等数据
    std::tuple<const CudaWrapper<float>&,
                const CudaWrapper<float>&,
                const CudaWrapper<float>&,
                const CudaWrapper<float>&,
                const float&,
                const std::int32_t&,
                const CudaWrapper<std::int32_t>&>
    get(const double& timestamp, const Eigen::Matrix<double, 4, 4>& global_to_lidar_mat, const bool& is_first_frame, cudaStream_t stream = 0);

    /// @brief 获取跟踪ID
    /// @param is_first_frame 是否为第一帧
    /// @param track_ids 输出参数：跟踪ID数组(900个目标)
    /// @param stream CUDA流
    /// @return 操作状态
    Status getTrackId(const bool& is_first_frame, CudaWrapper<int32_t>& track_ids, cudaStream_t stream = 0);

    /// @brief 缓存当前帧的前K个实例特征、锚点和置信度
    /// @param instance_feature 实例特征，形状为(num_querys, embedfeat_dims)
    /// @param anchor 锚点，形状为(num_querys, query_dims)
    /// @param confidence_logits 置信度logits，形状为(num_querys, class_nums)
    /// @param is_first_frame 是否为第一帧
    Status cache(const CudaWrapper<float>& instance_feature,
                const CudaWrapper<float>& anchor,
                const CudaWrapper<float>& confidence_logits,
                const bool& is_first_frame, cudaStream_t stream = 0);
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

    /// @brief Sigmoid激活函数
    template <typename T>
    static T sigmoid(const T& logits);

    // 配置参数
    sparsebev::TaskConfig params_;
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
    float history_time_;                        // 上一帧时间戳
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

    CudaWrapper<int32_t> m_gpu_prev_id_wrapper_;    // 前一帧目标最大ID值的GPU包装器
};
#endif // __INSTANCE_BANK_GPU_H__