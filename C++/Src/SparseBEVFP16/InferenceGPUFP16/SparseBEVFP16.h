/*******************************************************
 文件名：SparseBEVFP16.h
 作者：sharkls
 描述：SparseBEVFP16推理类，适配GPU版本的InstanceBank
 版本：v2.0
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

#include "../../../Include/Common/Core/GlobalContext.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"
#include "../../../Include/Common/Interface/IBaseModule.h"
#include "../../../Include/Common/Factory/ModuleFactory.h"
#include "../../Common/TensorRT/TensorRT.h"
#include "../../Common/Core/FunctionHub.h"

// 使用条件编译避免重复包含Eigen
#ifndef EIGEN_DENSE_INCLUDED
#define EIGEN_DENSE_INCLUDED
#include <Eigen/Dense>
#endif

#include "CTimeMatchSrcData.h"
#include "SparseEnd2EndConfig_conf.pb.h"
#include "CAlgResult.h"
#include "RawInferenceResult.h"
#include "../data/SparseBEVInputData.h"
#include "InstanceBankGPU.h"  // 使用GPU版本的InstanceBank

class SparseBEVFP16 : public IBaseModule {
public:
    SparseBEVFP16(const std::string& exe_path) : IBaseModule(exe_path) {}
    ~SparseBEVFP16() override;

    // 实现基类接口
    std::string getModuleName() const override { return "SparseBEVFP16"; }
    ModuleType getModuleType() const override { return ModuleType::INFERENCE; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;
    void setCurrentSampleIndex(int sample_index) override;

    // 加载辅助数据
    void loadAuxiliaryData();

    // 获取原始推理结果（避免数据转换）
    RawInferenceResult getRawInferenceResult() const;

    // 从输入包装器设置输入数据
    bool setInputDataFromWrapper(const sparsebev::SparseBEVInputData& input_data);

    // 将推理结果转换为输出格式（不进行后处理，只做基本格式转换）
    void convertToOutputFormat(const std::vector<half>& pred_instance_feature,
                              const std::vector<half>& pred_anchor,
                              const std::vector<half>& pred_class_score,
                              const std::vector<half>& pred_quality_score,
                              const std::vector<int32_t>& track_ids,
                              RawInferenceResult& raw_result);

     // 数据保存函数
    template<typename T>
    bool saveDataToFile(const CudaWrapper<T>& data, const std::string& file_path);

    template<typename T>
    bool savePartialFast(const CudaWrapper<T>& gpu, size_t effective_elems, const std::string& path);

private:
    // 设置各种输入数据的方法
    bool setImageFeature(const sparsebev::ImageFeature& image_data);
    bool setTimeInterval(const sparsebev::TimeInterval& time_interval);
    bool setImageCalibration(const sparsebev::ImageCalibration& image_calibration);
    
    // 特征提取推理
    Status extractFeatures(const CudaWrapper<half>& input_imgs, 
                          const cudaStream_t& stream,
                          CudaWrapper<half>& output_features);
    
    // 第一帧头部推理
    Status headFirstFrame(const CudaWrapper<half>& features,
                         const cudaStream_t& stream,
                         CudaWrapper<half>& pred_instance_feature,
                         CudaWrapper<half>& pred_anchor,
                         CudaWrapper<half>& pred_class_score,
                         CudaWrapper<half>& pred_quality_score);
    
    Status headSecondFrame(const CudaWrapper<half>& features,
                          const cudaStream_t& stream,
                          CudaWrapper<half>& pred_instance_feature,
                          CudaWrapper<half>& pred_anchor,
                          CudaWrapper<half>& pred_class_score,
                          CudaWrapper<half>& pred_quality_score,
                          CudaWrapper<int32_t>& pred_track_id);

private:
    sparsebev::TaskConfig m_taskConfig;            // 任务配置参数
    
    // TensorRT引擎
    std::shared_ptr<TensorRT> m_extract_feat_engine;    // 特征提取引擎
    std::shared_ptr<TensorRT> m_head1st_engine;         // 第一帧头部引擎
    std::shared_ptr<TensorRT> m_head2nd_engine;         // 第二帧头部引擎

    // 输入数据
    CudaWrapper<half> m_half_input_wrapper;        // half精度输入GPU内存包装器


    // 中间特征数据
    CudaWrapper<half> m_half_features_wrapper;     // half精度特征GPU内存包装器

    // 推理结果数据
    CudaWrapper<half> m_half_pred_instance_feature_wrapper;
    CudaWrapper<half> m_half_pred_anchor_wrapper;
    CudaWrapper<half> m_half_pred_class_score_wrapper;
    CudaWrapper<half> m_half_pred_quality_score_wrapper;
    CudaWrapper<int32_t> m_int32_pred_track_id_wrapper;

    // 时序数据（第二帧推理需要）
    CudaWrapper<half> m_half_temp_instance_feature_wrapper; 
    CudaWrapper<half> m_half_temp_anchor_wrapper;
    CudaWrapper<int32_t> m_int_mask_wrapper;
    CudaWrapper<int32_t> m_int32_temp_track_id_wrapper;
    

    // TensorRT引擎临时输出缓冲区
    CudaWrapper<half> m_half_tmp_outs0_wrapper;
    CudaWrapper<half> m_half_tmp_outs1_wrapper;
    CudaWrapper<half> m_half_tmp_outs2_wrapper;
    CudaWrapper<half> m_half_tmp_outs3_wrapper;
    CudaWrapper<half> m_half_tmp_outs4_wrapper;
    CudaWrapper<half> m_half_tmp_outs5_wrapper;

    // 辅助数据（从配置文件加载）
    std::vector<int32_t> m_spatial_shapes;         // 空间形状
    std::vector<int32_t> m_level_start_index;      // 层级起始索引
    std::vector<half> m_instance_feature;         // 实例特征
    std::vector<half> m_anchor;                   // 锚点
    std::vector<half> m_time_interval;            // 时间间隔
    std::vector<half> m_image_wh;                 // 图像宽高
    std::vector<half> m_lidar2img;                // 激光雷达到图像变换矩阵

    // GPU内存包装器用于辅助数据
    CudaWrapper<int32_t> m_gpu_spatial_shapes_wrapper;
    CudaWrapper<int32_t> m_gpu_level_start_index_wrapper;
    CudaWrapper<half> m_gpu_instance_feature_wrapper;
    CudaWrapper<half> m_gpu_anchor_wrapper;
    CudaWrapper<half> m_gpu_time_interval_wrapper;
    CudaWrapper<half> m_gpu_image_wh_wrapper;
    CudaWrapper<half> m_gpu_lidar2img_wrapper;

    // 输出结果
    CAlgResult m_alg_result;                       // 算法结果
    RawInferenceResult m_raw_result;               // 原始推理结果

    // 运行状态
    bool status_ = false;
    bool is_first_frame_ = true;                   // 是否为第一帧

    // 时间戳相关
    double current_timestamp_;
    double previous_timestamp_;
    
    // 坐标变换矩阵
    Eigen::Matrix<double, 4, 4> current_global_to_lidar_mat_;
    Eigen::Matrix<double, 4, 4> previous_global_to_lidar_mat_;
    
    // GPU版本的InstanceBank状态管理
    std::unique_ptr<InstanceBankGPU> instance_bank_gpu_;
    bool has_previous_frame_;

    // 添加当前样本索引
    int current_sample_index_;
};

#endif  // __SPARSEBEV_H__