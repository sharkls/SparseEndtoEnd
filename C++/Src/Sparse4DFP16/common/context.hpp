#ifndef __CONTEXT_HPP__
#define __CONTEXT_HPP__

#include "../../../Include/Common/Utils/CudaWrapper.h"
#include <cuda_fp16.h>

namespace sparse4d{
namespace common{

struct PipelineContext
{   
    // 预处理输入
    CudaWrapper<half> input_images;
    // 骨干输出
    CudaWrapper<half> features;
    // 第一帧独有的输入
    CudaWrapper<int32_t> spatial_shapes;
    CudaWrapper<int32_t> level_start_index;
    CudaWrapper<half> instance_feature;
    CudaWrapper<half> anchor;
    CudaWrapper<half> time_interval;
    CudaWrapper<half> image_wh;
    CudaWrapper<half> lidar2img;
    // 第二帧独有的输入
    CudaWrapper<half> temp_instance_feature;
    CudaWrapper<half> temp_anchor;
    CudaWrapper<int32_t> mask;
    CudaWrapper<int32_t> track_ids;
};

struct HeadOutput{
    CudaWrapper<half> pred_instance_feature;
    CudaWrapper<half> pred_anchor;
    CudaWrapper<half> pred_class_score;
    CudaWrapper<half> pred_quality_score;
    CudaWrapper<int32_t> pred_track_ids;

    // 临时输出
    CudaWrapper<half> tmp_outs0;
    CudaWrapper<half> tmp_outs1;
    CudaWrapper<half> tmp_outs2;
    CudaWrapper<half> tmp_outs3;
    CudaWrapper<half> tmp_outs4;
    CudaWrapper<half> tmp_outs5;
};

}//namespace common
}//namespace sparse4d
#endif //__CONTEXT_HPP__