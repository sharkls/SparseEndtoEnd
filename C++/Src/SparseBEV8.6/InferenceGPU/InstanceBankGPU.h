/*******************************************************
 文件名：InstanceBankGPU.h
 作者：sharkls
 描述：GPU版本的实例银行类，负责管理多帧检测实例的缓存和更新
 版本：v1.0
 日期：2025-06-18
 *******************************************************/
#ifndef __INSTANCE_BANK_GPU_H__
#define __INSTANCE_BANK_GPU_H__

#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>

// 使用条件编译避免重复包含Eigen
#ifndef EIGEN_DENSE_INCLUDED
#define EIGEN_DENSE_INCLUDED
#include <Eigen/Dense>
#endif

#include "../../../Include/Common/Utils/CudaWrapper.h"
#include "GlobalContext.h"
#include "SparseEnd2EndConfig_conf.pb.h"

/// @brief GPU版本的实例银行类：管理多帧检测实例的缓存和更新
class InstanceBankGPU {
public:
    /// @brief 构造函数：使用参数初始化GPU实例银行
    InstanceBankGPU(const sparsebev::TaskConfig& params);
    ~InstanceBankGPU();

    /// @brief 重置实例银行状态
    Status reset();

    /// @brief 更新实例银行状态（在GPU上）
    /// @param instance_feature 实例特征，形状为(num_querys, embedfeat_dims)
    /// @param anchor 锚点，形状为(num_querys, query_dims)
    /// @param confidence 置信度，形状为(num_querys,)
    /// @param track_ids 跟踪ID数组，形状为(num_querys,)
    /// @param time_interval 时间间隔（秒）
    Status updateOnGPU(const CudaWrapper<float>& instance_feature,
                       const CudaWrapper<float>& anchor,
                       const CudaWrapper<float>& confidence,
                       const CudaWrapper<int32_t>& track_ids,
                       float time_interval,
                       cudaStream_t stream);

    /// @brief 获取指定时间戳的实例数据（从GPU）
    /// @param timestamp 时间戳
    /// @param global_to_lidar_mat 全局坐标系到激光雷达坐标系的变换矩阵
    /// @param is_first_frame 是否为第一帧
    /// @param stream CUDA流
    /// @return 返回GPU上的实例特征、锚点、置信度等数据
    std::tuple<const CudaWrapper<float>&,
               const CudaWrapper<float>&,
               const CudaWrapper<float>&,
               const CudaWrapper<float>&,
               float,
               int32_t,
               const CudaWrapper<int32_t>&>
    getOnGPU(const double& timestamp, 
             const Eigen::Matrix<double, 4, 4>& global_to_lidar_mat, 
             const bool& is_first_frame,
             cudaStream_t stream);

    /// @brief 缓存当前帧的前K个实例特征、锚点和置信度（在GPU上）
    /// @param instance_feature 实例特征，形状为(num_querys, embedfeat_dims)
    /// @param anchor 锚点，形状为(num_querys, query_dims)
    /// @param confidence_logits 置信度logits，形状为(num_querys, class_nums)
    /// @param is_first_frame 是否为第一帧
    /// @param stream CUDA流
    Status cacheOnGPU(const CudaWrapper<float>& instance_feature,
                      const CudaWrapper<float>& anchor,
                      const CudaWrapper<float>& confidence_logits,
                      const bool& is_first_frame,
                      cudaStream_t stream);

    /// @brief 获取跟踪ID（在GPU上）
    /// @param refined_track_ids 精炼后的跟踪ID数组，形状为(num_querys,)
    /// @param stream CUDA流
    /// @return 返回跟踪ID数组
    CudaWrapper<int32_t> getTrackIdOnGPU(const CudaWrapper<int32_t>& refined_track_ids, cudaStream_t stream);
    
    /// @brief 获取跟踪ID（在GPU上）
    /// @param is_first_frame 是否为第一帧
    /// @param stream CUDA流
    /// @return 返回跟踪ID数组
    CudaWrapper<int32_t> getTrackIdOnGPU(const bool& is_first_frame, cudaStream_t stream);

    /// @brief 获取GPU上的实例特征
    const CudaWrapper<float>& getGPUInstanceFeature() const { return *gpu_instance_feature_; }
    
    /// @brief 获取GPU上的锚点
    const CudaWrapper<float>& getGPUAnchor() const { return *gpu_anchor_; }
    
    /// @brief 获取GPU上的置信度
    const CudaWrapper<float>& getGPUConfidence() const { return *gpu_confidence_; }
    
    /// @brief 获取GPU上的跟踪ID
    const CudaWrapper<int32_t>& getGPUTrackIds() const { return *gpu_track_ids_; }
    
    /// @brief 获取GPU上的缓存特征
    const CudaWrapper<float>& getGPUCachedFeature() const { return *gpu_cached_feature_; }
    
    /// @brief 获取GPU上的缓存锚点
    const CudaWrapper<float>& getGPUCachedAnchor() const { return *gpu_cached_anchor_; }

    /// @brief 保存InstanceBank缓存数据到文件（从GPU拷贝到CPU后保存）
    /// @param sample_id 样本ID
    /// @return 是否保存成功
    bool saveInstanceBankData(const int sample_id);

private:
    /// @brief 初始化GPU内存
    bool initializeGPUMemory();
    
    /// @brief 清理GPU内存
    void cleanupGPUMemory();

    /// @brief 在GPU上更新跟踪ID
    /// @param track_ids 跟踪ID数组，形状为(num_querys,)
    /// @param stream CUDA流
    void updateTrackIdOnGPU(const CudaWrapper<int32_t>& track_ids, cudaStream_t stream);

    /// @brief 在GPU上更新置信度
    /// @param confidence 置信度，形状为(num_querys,)
    /// @param stream CUDA流
    void updateConfidenceOnGPU(const CudaWrapper<float>& confidence, cudaStream_t stream);

    /// @brief GPU上的时空对齐：将t-1时刻的锚点投影到t时刻
    /// @param temp_anchor 临时锚点，形状为(temp_num_querys, query_dims_)
    /// @param temp2cur_mat t-1到t时刻的变换矩阵
    /// @param time_interval 时间间隔
    /// @param stream CUDA流
    void anchorProjectionOnGPU(CudaWrapper<float>& temp_anchor,
                               const Eigen::Matrix<float, 4, 4>& temp2cur_mat,
                               float time_interval,
                               cudaStream_t stream);

    /// @brief 在GPU上计算tensor.max(-1).values.sigmoid
    /// @param confidence_logits 置信度logits，形状为(num_querys, class_nums)
    /// @param num_querys 查询数量
    /// @param stream CUDA流
    /// @return 每个查询的最大置信度分数
    CudaWrapper<float> getMaxConfidenceScoresOnGPU(const CudaWrapper<float>& confidence_logits,
                                                   const uint32_t& num_querys,
                                                   cudaStream_t stream);

    /// @brief 在GPU上应用Sigmoid激活函数
    /// @param logits 输入logits
    /// @param stream CUDA流
    void applySigmoidOnGPU(CudaWrapper<float>& logits, cudaStream_t stream);

    /// @brief 从文件加载锚点数据
    /// @param file_path 锚点文件路径
    /// @return 是否加载成功
    bool loadAnchorsFromFile(const std::string& file_path);

    /// @brief 从文件加载实例特征数据
    /// @param file_path 实例特征文件路径
    /// @return 是否加载成功
    bool loadInstanceFeatureFromFile(const std::string& file_path);

    // 配置参数
    sparsebev::TaskConfig params_;
    uint32_t num_querys_;
    uint32_t query_dims_;
    uint32_t topk_querys_;
    float max_time_interval_;
    float default_time_interval_;
    float confidence_decay_;

    // GPU内存
    std::unique_ptr<CudaWrapper<float>> gpu_instance_feature_;
    std::unique_ptr<CudaWrapper<float>> gpu_anchor_;
    std::unique_ptr<CudaWrapper<float>> gpu_confidence_;
    std::unique_ptr<CudaWrapper<int32_t>> gpu_track_ids_;
    std::unique_ptr<CudaWrapper<float>> gpu_cached_feature_;
    std::unique_ptr<CudaWrapper<float>> gpu_cached_anchor_;
    std::unique_ptr<CudaWrapper<float>> gpu_kmeans_anchors_;
    std::unique_ptr<CudaWrapper<float>> gpu_query_;
    std::unique_ptr<CudaWrapper<float>> gpu_query_anchor_;
    std::unique_ptr<CudaWrapper<float>> gpu_query_confidence_;
    std::unique_ptr<CudaWrapper<int32_t>> gpu_query_track_ids_;
    
    // GPU上的状态变量
    std::unique_ptr<CudaWrapper<int32_t>> gpu_mask_;
    std::unique_ptr<CudaWrapper<float>> gpu_time_interval_;
    std::unique_ptr<CudaWrapper<int32_t>> gpu_prev_id_;
    std::unique_ptr<CudaWrapper<float>> gpu_history_time_;
    std::unique_ptr<CudaWrapper<float>> gpu_temp_lidar_to_global_mat_;

    // CPU上的状态变量（用于调试和保存）
    int32_t mask_;
    float history_time_;
    float time_interval_;
    Eigen::Matrix<double, 4, 4> temp_lidar_to_global_mat_;
    bool is_first_frame_;
    int32_t prev_id_;

    // 临时CPU向量（用于文件操作）
    std::vector<float> kmeans_anchors_;
    std::vector<float> instance_feature_;
};

#endif  // __INSTANCE_BANK_GPU_H__