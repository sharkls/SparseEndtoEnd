#ifndef GPU_NMS_H
#define GPU_NMS_H

#include <vector>
#include <string>
#include <cuda_runtime.h>  // 添加CUDA头文件

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

// GPU NMS函数声明
extern "C" {
    // 单类别NMS
    int gpuNMS(const BoundingBox3D* host_boxes, 
               BoundingBox3D* host_output_boxes,
               int num_boxes, 
               float iou_threshold,
               int max_output_boxes);
    
    // 多类别批量NMS
    int gpuBatchNMS(const BoundingBox3D* host_boxes, 
                    BoundingBox3D* host_output_boxes,
                    int num_boxes, 
                    float iou_threshold,
                    int max_output_boxes);
}

// GPU NMS包装类
class GPUNMS {
public:
    GPUNMS();
    ~GPUNMS();
    
    // 初始化GPU NMS
    bool initialize();
    
    // 执行NMS
    int process(const std::vector<BoundingBox3D>& input_boxes,
                std::vector<BoundingBox3D>& output_boxes,
                float iou_threshold = 0.5f,
                int max_output_boxes = 1000);
    
    // 批量处理多个类别
    int processBatch(const std::vector<BoundingBox3D>& input_boxes,
                     std::vector<BoundingBox3D>& output_boxes,
                     float iou_threshold = 0.5f,
                     int max_output_boxes = 1000);
    
    // 检查GPU是否可用
    bool isGPUAvaliable() const;
    
    // 获取GPU信息
    void getGPUInfo(int& device_count, std::string& device_name) const;

private:
    bool m_initialized;
    int m_device_count;
    std::string m_device_name;
};

#endif // GPU_NMS_H 