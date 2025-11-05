/*******************************************************
 文件名：img_preprocessor.h
 作者：sharkls
 描述：图像预处理类，负责将图像预处理为模型输入格式
 版本：v1.0
 日期：2025-10-22
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
#include "./img_aug_with_bilinearinterpolation_kernel.h"
#include "Sparse4D_conf.pb.h"
#include "CTimeMatchSrcData.h"
#include "../common/context.hpp"

// 使用条件编译避免重复包含Eigen
#ifndef EIGEN_DENSE_INCLUDED
#define EIGEN_DENSE_INCLUDED
#include "../../../Submodules/ThirdParty/eigen/Eigen/Dense"
#endif



namespace sparse4d{
namespace preprocessor{

class Preprocessor
{
    public: 
        virtual ~Preprocessor() = default;

        virtual Status init(const TaskConfig &param) = 0;

        virtual Status forward(const CTimeMatchSrcData *raw_data, 
                                const cudaStream_t& stream,
                                common::PipelineContext& pipeline_context) = 0;
};

std::shared_ptr<Preprocessor> create_preprocessor(const TaskConfig &param);
}//namespace preprocessor
}//namespace sparse4d

#endif  // __IMG_PREPROCESSOR_H__