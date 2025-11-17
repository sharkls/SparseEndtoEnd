/*******************************************************
文件名：postprocessor.hpp
作者：sharkls
描述：后处理类，负责将模型输出后处理为最终结果
版本：v1.0
日期：2025-10-22
*******************************************************/
#ifndef __POSTPROCESSOR_H__
#define __POSTPROCESSOR_H__

#include <iostream>
#include <cuda_fp16.h>
#include <cstdint>
#include <memory>
#include <cmath>
#include "log.h"

#include "../../../Include/Common/Core/GlobalContext.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"
#include "Sparse4D_conf.pb.h"
#include "gpu_nms.hpp"
#include "../common/context.hpp"

// 使用条件编译避免重复包含Eigen
#ifndef EIGEN_DENSE_INCLUDED
#define EIGEN_DENSE_INCLUDED
#include "../../../Submodules/ThirdParty/eigen/Eigen/Dense"
#endif



namespace sparse4d{
namespace postprocessor{

struct PostprocessorParams{
    bool use_gpu_nms;
    float nms_threshold;
    int max_output_boxes;
    float confidence_threshold;
};

// 前向声明
struct BoundingBox3D;
 
class Postprocessor
{
    public: 
        virtual ~Postprocessor() = default;

        virtual Status init(const TaskConfig &param) = 0;

        virtual Status forward(const common::HeadOutput& head_output,
                             const cudaStream_t& stream,
                             CAlgResult& output_result) = 0;
};

std::shared_ptr<Postprocessor> create_postprocessor(const TaskConfig &param);
}//namespace postprocessor
}//namespace sparse4d
 
 #endif  // __POSTPROCESSOR_H__