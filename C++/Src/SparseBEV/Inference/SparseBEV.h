/*******************************************************
 文件名：SparseBEV.h
 作者：sharkls
 描述：SparseBEV推理类，负责将图像预处理为模型输入格式
 版本：v1.0
 日期：2025-06-18
 *******************************************************/
#ifndef __SPARSEBEV_H__
#define __SPARSEBEV_H__

#include <iostream>
#include <cuda_fp16.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <limits>
#include "log.h"

#include "../../../Include/Common/GlobalContext.h"
#include "../../../Include/Common/CudaWrapper.h"
#include "../../Src/Common/IBaseModule.h"
#include "../../../Src/Common/ModuleFactory.h"
#include "../../Src/Common/TensorRT/TensorRT.h"
#include "Eigen/Dense"

#include "CTimeMatchSrcData.h"
#include "SparseEnd2EndConfig_conf.pb.h"
#include "CAlgResult.h"
#include "RawInferenceResult.h"
#include "../data/SparseBEVInputData.h"
#include "./InstanceBank.h"

class SparseBEV : public IBaseModule {
public:
    SparseBEV(const std::string& exe_path) : IBaseModule(exe_path) {}
    ~SparseBEV() override;

    // 实现基类接口
    std::string getModuleName() const override { return "SparseBEV"; }
    ModuleType getModuleType() const override { return ModuleType::INFERENCE; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

    // 加载辅助数据
    void loadAuxiliaryData();
    
    // 计算变换后的lidar2img矩阵
    void computeTransformedLidar2img();
    
    // 加载单应性标定数据
    bool loadHomographyCalibration(const std::string& calib_path);

    // 获取原始推理结果（避免数据转换）
    RawInferenceResult getRawInferenceResult() const;

    // 从输入包装器设置输入数据
    bool setInputDataFromWrapper(const sparsebev::SparseBEVInputData& input_data);

    // 将推理结果转换为输出格式（不进行后处理，只做基本格式转换）
    void convertToOutputFormat(const std::vector<float>& pred_instance_feature,
                              const std::vector<float>& pred_anchor,
                              const std::vector<float>& pred_class_score,
                              const std::vector<float>& pred_quality_score,
                              const std::vector<float>& pred_track_id,
                              RawInferenceResult& raw_result);

    void convertToOutputFormat(const std::vector<half>& pred_instance_feature,
                              const std::vector<half>& pred_anchor,
                              const std::vector<half>& pred_class_score,
                              const std::vector<half>& pred_quality_score,
                              const std::vector<half>& pred_track_id,
                              RawInferenceResult& raw_result);

private:
    // 设置各种输入数据的方法
    bool setImageFeature(const sparsebev::ImageFeature& image_feature);
    bool setSpatialShapes(const sparsebev::SpatialShapes& spatial_shapes);
    bool setInstanceData(const sparsebev::InstanceData& instance_data);
    bool setTimeInterval(const sparsebev::TimeInterval& time_interval);
    bool setImageCalibration(const sparsebev::ImageCalibration& image_calibration);
    bool setTemporalData(const sparsebev::TemporalData& temporal_data);
    bool setLidarData(const sparsebev::LidarPointCloud& lidar_data);

    // 特征提取推理
    Status extractFeatures(const CudaWrapper<float>& input_imgs, 
                          const cudaStream_t& stream,
                          CudaWrapper<float>& output_features);
    
    Status extractFeatures(const CudaWrapper<half>& input_imgs, 
                          const cudaStream_t& stream,
                          CudaWrapper<half>& output_features);

    // 第一帧头部推理
    Status headFirstFrame(const CudaWrapper<float>& features,
                         const cudaStream_t& stream,
                         CudaWrapper<float>& pred_instance_feature,
                         CudaWrapper<float>& pred_anchor,
                         CudaWrapper<float>& pred_class_score,
                         CudaWrapper<float>& pred_quality_score);
    
    Status headFirstFrame(const CudaWrapper<half>& features,
                         const cudaStream_t& stream,
                         CudaWrapper<half>& pred_instance_feature,
                         CudaWrapper<half>& pred_anchor,
                         CudaWrapper<half>& pred_class_score,
                         CudaWrapper<half>& pred_quality_score);

    // 第二帧头部推理
    Status headSecondFrame(const CudaWrapper<float>& features,
                          const cudaStream_t& stream,
                          CudaWrapper<float>& pred_instance_feature,
                          CudaWrapper<float>& pred_anchor,
                          CudaWrapper<float>& pred_class_score,
                          CudaWrapper<float>& pred_quality_score,
                          CudaWrapper<float>& pred_track_id);
    
    Status headSecondFrame(const CudaWrapper<half>& features,
                          const cudaStream_t& stream,
                          CudaWrapper<half>& pred_instance_feature,
                          CudaWrapper<half>& pred_anchor,
                          CudaWrapper<half>& pred_class_score,
                          CudaWrapper<half>& pred_quality_score,
                          CudaWrapper<half>& pred_track_id);

private:
    sparsebev::TaskConfig m_taskConfig;            // 任务配置参数
    
    // TensorRT引擎
    std::shared_ptr<TensorRT> m_extract_feat_engine;    // 特征提取引擎
    std::shared_ptr<TensorRT> m_head1st_engine;         // 第一帧头部引擎
    std::shared_ptr<TensorRT> m_head2nd_engine;         // 第二帧头部引擎

    // 输入数据
    CudaWrapper<float> m_float_input_wrapper;      // float精度输入GPU内存包装器
    CudaWrapper<half> m_half_input_wrapper;        // half精度输入GPU内存包装器

    // 中间特征数据
    CudaWrapper<float> m_float_features_wrapper;   // float精度特征GPU内存包装器
    CudaWrapper<half> m_half_features_wrapper;     // half精度特征GPU内存包装器

    // 推理结果数据
    CudaWrapper<float> m_float_pred_instance_feature_wrapper;
    CudaWrapper<float> m_float_pred_anchor_wrapper;
    CudaWrapper<float> m_float_pred_class_score_wrapper;
    CudaWrapper<float> m_float_pred_quality_score_wrapper;
    CudaWrapper<float> m_float_pred_track_id_wrapper;

    CudaWrapper<half> m_half_pred_instance_feature_wrapper;
    CudaWrapper<half> m_half_pred_anchor_wrapper;
    CudaWrapper<half> m_half_pred_class_score_wrapper;
    CudaWrapper<half> m_half_pred_quality_score_wrapper;
    CudaWrapper<half> m_half_pred_track_id_wrapper;

    // 时序数据（第二帧推理需要）
    CudaWrapper<float> m_float_temp_instance_feature_wrapper;
    CudaWrapper<float> m_float_temp_anchor_wrapper;
    CudaWrapper<int32_t> m_float_mask_wrapper;
    CudaWrapper<int32_t> m_float_track_id_wrapper;

    CudaWrapper<half> m_half_temp_instance_feature_wrapper;
    CudaWrapper<half> m_half_temp_anchor_wrapper;
    CudaWrapper<int32_t> m_half_mask_wrapper;
    CudaWrapper<int32_t> m_half_track_id_wrapper;

    // 辅助数据（从配置文件加载）
    std::vector<int32_t> m_spatial_shapes;         // 空间形状
    std::vector<int32_t> m_level_start_index;      // 层级起始索引
    std::vector<float> m_instance_feature;         // 实例特征
    std::vector<float> m_anchor;                   // 锚点
    std::vector<float> m_time_interval;            // 时间间隔
    std::vector<float> m_image_wh;                 // 图像宽高
    std::vector<float> m_lidar2img;                // 激光雷达到图像变换矩阵

    // 输出结果
    CAlgResult m_alg_result;                       // 算法结果
    RawInferenceResult m_raw_result;               // 原始推理结果

    // 运行状态
    bool status_ = false;
    bool use_half_precision_ = false;
    bool is_first_frame_ = true;                   // 是否为第一帧

    // 添加缺失的成员变量
    // 时间戳信息
    double current_timestamp_;
    double previous_timestamp_;
    
    // 坐标变换矩阵
    Eigen::Matrix<double, 4, 4> current_global_to_lidar_mat_;
    Eigen::Matrix<double, 4, 4> previous_global_to_lidar_mat_;
    
    // InstanceBank状态管理
    std::unique_ptr<InstanceBank> instance_bank_;
    bool has_previous_frame_;
    
    // 缓存上一帧的推理结果
    std::vector<float> cached_instance_feature_;
    std::vector<float> cached_anchor_;
    std::vector<float> cached_confidence_;
    std::vector<int32_t> cached_track_ids_;
};

#endif  // __SPARSEBEV_H__