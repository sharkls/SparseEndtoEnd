/*******************************************************
 文件名：PostProcessor.h
 作者：sharkls
 描述：后处理类，负责对SparseBEV推理结果进行去重和优化
 版本：v1.0
 日期：2025-06-18
 *******************************************************/
#ifndef __POSTPROCESSOR_H__
#define __POSTPROCESSOR_H__

#include <iostream>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "log.h"

#include "../../../Include/Common/Core/GlobalContext.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"
#include "../../../Include/Common/Interface/IBaseModule.h"
#include "../../../Include/Common/Factory/ModuleFactory.h"

#include "CTimeMatchSrcData.h"
#include "SparseEnd2EndConfig_conf.pb.h"
#include "CAlgResult.h"
#include "gpu_nms.h"  // 添加GPU NMS支持
#include "../data/RawInferenceResult.h"  // 添加原始推理结果支持

class PostProcessor : public IBaseModule {
public:
    PostProcessor(const std::string& exe_path) : IBaseModule(exe_path) {}
    ~PostProcessor() override;

    // 实现基类接口
    std::string getModuleName() const override { return "PostProcessor"; }
    ModuleType getModuleType() const override { return ModuleType::POST_PROCESS; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;
    void setCurrentSampleIndex(int sample_index) override;

private:
    // CPU NMS相关方法
    std::vector<BoundingBox3D> cpuNMS(const std::vector<BoundingBox3D>& boxes, 
                                     float iou_threshold, 
                                     int max_output_boxes);
    float calculate3DIoU(const BoundingBox3D& box1, const BoundingBox3D& box2);
    
    // 数据转换方法
    std::vector<CObjectResult> convertRawToObjectResult(const RawInferenceResult& raw_result);
    std::vector<CObjectResult> convertRawToObjectResult(const std::vector<BoundingBox3D>& boxes);
    
    // 结果优化方法
    void optimizeResults(std::vector<CObjectResult>& objects);
    void removeLowConfidenceObjects(std::vector<CObjectResult>& objects, float confidence_threshold);
    void sortByConfidence(std::vector<CObjectResult>& objects);

private:
    sparsebev::TaskConfig m_taskConfig;            // 任务配置参数
    CAlgResult m_output_result;                    // 输出结果
    
    // GPU推理结果输入
    RawInferenceResult m_raw_input_result;         // GPU推理结果输入
    
    // GPU NMS相关
    GPUNMS m_gpuNMS;                               // GPU NMS处理器
    bool m_useGPU;                                 // 是否使用GPU NMS
    float m_gpuNMSThreshold;                       // GPU NMS阈值
    int m_maxOutputBoxes;                          // 最大输出框数量
    float m_confidenceThreshold;                   // 置信度阈值
    
    // 运行状态
    bool status_ = false;
    
    // 当前样本索引
    int current_sample_index_;
};

#endif  // __POSTPROCESSOR_H__