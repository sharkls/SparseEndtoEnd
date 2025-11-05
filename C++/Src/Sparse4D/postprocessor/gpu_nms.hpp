#ifndef __GPU_NMS_H__
#define __GPU_NMS_H__

#include <vector>
#include <cuda_runtime.h>

namespace sparse4d {
namespace postprocessor {

// 3D边界框结构体
struct BoundingBox3D {
    float x, y, z;        // 中心点坐标
    float l, w, h;        // 长度、宽度、高度
    float yaw;            // 偏航角
    float confidence;     // 置信度
    int label;            // 类别标签
    int index;            // 原始索引
    int track_id;         // 跟踪ID
    
    __host__ __device__
    BoundingBox3D() : x(0), y(0), z(0), l(0), w(0), h(0), yaw(0), confidence(0), label(0), index(0), track_id(-1) {}
    
    __host__ __device__
    BoundingBox3D(float x_, float y_, float z_, float l_, float w_, float h_, 
                  float yaw_, float conf_, int label_, int idx_, int track_id_ = -1)
        : x(x_), y(y_), z(z_), l(l_), w(w_), h(h_), yaw(yaw_), confidence(conf_), label(label_), index(idx_), track_id(track_id_) {}
};

/**
 * @brief GPU NMS函数 - 直接处理原始输入数据
 * @param pred_anchor GPU上的锚点数据
 * @param pred_class_score GPU上的类别分数数据
 * @param pred_quality_score GPU上的质量分数数据
 * @param pred_track_ids GPU上的跟踪ID数据
 * @param num_objects 目标数量
 * @param num_classes 类别数量
 * @param d_output_boxes GPU上的输出边界框数据
 * @param output_count 输出框数量（GPU上的int指针）
 * @param confidence_threshold 置信度阈值
 * @param iou_threshold IoU阈值
 * @param max_output_boxes 最大输出框数
 * @param stream CUDA流
 * @return 实际输出的框数量，-1表示失败
 */
int gpu_nms_direct(const float* pred_anchor,
                   const float* pred_class_score,
                   const float* pred_quality_score,
                   const int32_t* pred_track_ids,
                   int num_objects,
                   int num_classes,
                   BoundingBox3D* d_output_boxes,
                   int* output_count,
                   float confidence_threshold,
                   float iou_threshold = 0.5f,
                   int max_output_boxes = 1000,
                   cudaStream_t stream = 0);

/**
 * @brief 检查GPU是否可用
 * @return true表示GPU可用，false表示不可用
 */
bool is_gpu_nms_available();

} // namespace postprocessor
} // namespace sparse4d

#endif // __GPU_NMS_H__ 