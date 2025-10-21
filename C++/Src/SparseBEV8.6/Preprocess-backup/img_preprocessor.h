/*******************************************************
 文件名：img_preprocessor.h
 作者：sharkls
 描述：图像预处理类，负责将图像预处理为模型输入格式
 版本：v1.0
 日期：2025-06-18
 *******************************************************/
#ifndef __IMG_PREPROCESSOR_H__
#define __IMG_PREPROCESSOR_H__

#include <iostream>
#include <cuda_fp16.h>
#include <cstdint>
#include <memory>
#include <cmath>
#include "log.h"

#include "../../../Include/Common/Core/GlobalContext.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"
#include "../../../Include/Common/Interface/IBaseModule.h"
#include "../../../Include/Common/Factory/ModuleFactory.h"
#include "./img_aug_with_bilinearinterpolation_kernel.h"

// 使用条件编译避免重复包含Eigen
#ifndef EIGEN_DENSE_INCLUDED
#define EIGEN_DENSE_INCLUDED
#include "../../../Submodules/ThirdParty/eigen/Eigen/Dense"
#endif

#include "CTimeMatchSrcData.h"
#include "SparseEnd2EndConfig_conf.pb.h"
#include "../data/SparseBEVInputData.h"


class ImagePreprocessor : public IBaseModule {
public:
    ImagePreprocessor(const std::string& exe_path) : IBaseModule(exe_path) {}
    ~ImagePreprocessor() override;

    // 实现基类接口
    std::string getModuleName() const override { return "ImagePreprocessor"; }
    ModuleType getModuleType() const override { return ModuleType::PRE_PROCESS; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;
    void setCurrentSampleIndex(int sample_index) override;

private:
    Status forward(const CudaWrapper<std::uint8_t>& input_imgs, 
                         const cudaStream_t& stream,
                  CudaWrapper<float>& output_imgs);
    
    // 新增方法：创建完整的输入数据结构
    std::shared_ptr<sparsebev::SparseBEVInputWrapper> createInputWrapper();
    
    /**
     * @brief 保存预处理数据为bin文件
     */
    void savePreprocessedDataToBin();

    /**
     * @brief 保存原始图像数据为bin文件
     */
    void saveOriginalDataToBin();

private:
    sparsebev::TaskConfig m_taskConfig;            // 任务配置参数
    CTimeMatchSrcData m_inputImage;                // 预处理输入数据
    std::vector<float> m_outputImage;              // 模型输入数据缓存区

    // GPU内存包装器
    CudaWrapper<float> m_float_output_wrapper;     // float精度输出GPU内存包装器
    
    // 新增成员：完整的输入数据包装器
    std::shared_ptr<sparsebev::SparseBEVInputWrapper> m_output_wrapper;

    // 运行状态
    bool status_ = false;
    bool use_half_precision_ = false;

    // 添加当前样本索引
    int current_sample_index_;
};

#endif  // __IMG_PREPROCESSOR_H__