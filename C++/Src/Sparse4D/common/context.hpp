#ifndef __CONTEXT_HPP__
#define __CONTEXT_HPP__

#include "../../../Include/Common/Utils/CudaWrapper.h"

namespace sparse4d{
namespace common{

struct PipelineContext
{   
    // 预处理输入
    CudaWrapper<float> input_images;
    // 骨干输出
    CudaWrapper<float> features;
    // 第一帧独有的输入
    CudaWrapper<int32_t> spatial_shapes;
    CudaWrapper<int32_t> level_start_index;
    CudaWrapper<float> instance_feature;
    CudaWrapper<float> anchor;
    CudaWrapper<float> time_interval;
    CudaWrapper<float> image_wh;
    CudaWrapper<float> lidar2img;
    // 第二帧独有的输入
    CudaWrapper<float> temp_instance_feature;
    CudaWrapper<float> temp_anchor;
    CudaWrapper<int32_t> mask;
    CudaWrapper<int32_t> track_ids;
};

struct HeadOutput{
    CudaWrapper<float> pred_instance_feature;
    CudaWrapper<float> pred_anchor;
    CudaWrapper<float> pred_class_score;
    CudaWrapper<float> pred_quality_score;
    CudaWrapper<int32_t> pred_track_ids;

    // 临时输出
    CudaWrapper<float> tmp_outs0;
    CudaWrapper<float> tmp_outs1;
    CudaWrapper<float> tmp_outs2;
    CudaWrapper<float> tmp_outs3;
    CudaWrapper<float> tmp_outs4;
    CudaWrapper<float> tmp_outs5;
};

}//namespace common
}//namespace sparse4d
#endif //__CONTEXT_HPP__