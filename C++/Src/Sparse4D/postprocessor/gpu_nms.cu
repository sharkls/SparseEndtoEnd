#include "gpu_nms.hpp"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include "log.h"

namespace sparse4d {
namespace postprocessor {

// 设备端比较函数（用于排序）
struct CompareBoxes {
    __host__ __device__
    bool operator()(const BoundingBox3D& a, const BoundingBox3D& b) const {
        return a.confidence > b.confidence;
    }
};

// 计算两个3D边界框的IoU（GPU版本）
__device__ float calculate_3d_iou(const BoundingBox3D& box1, const BoundingBox3D& box2) {
    // 简化的3D IoU计算（基于2D投影）
    // 计算中心点距离
    float dx = box1.x - box2.x;
    float dy = box1.y - box2.y;
    float distance = sqrtf(dx * dx + dy * dy);
    
    // 计算边界框对角线长度的一半作为阈值
    float threshold1 = sqrtf(box1.l * box1.l + box1.w * box1.w) / 2.0f;
    float threshold2 = sqrtf(box2.l * box2.l + box2.w * box2.w) / 2.0f;
    float overlap_threshold = (threshold1 + threshold2) * 0.5f;
    
    // 如果距离太远，IoU为0
    if (distance > overlap_threshold) {
        return 0.0f;
    }
    
    // 简化的IoU计算
    float overlap_ratio = 1.0f - (distance / overlap_threshold);
    return fmaxf(0.0f, overlap_ratio);
}


// Sigmoid函数（设备端）
__device__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// 第一步：转换和过滤内核（只进行置信度过滤）
__global__ void convert_and_filter_kernel(const float* pred_anchor,
                                         const float* pred_class_score,
                                         const float* pred_quality_score,
                                         const int32_t* pred_track_ids,
                                         BoundingBox3D* temp_boxes,
                                         int* valid_count,
                                         int num_objects,
                                         int num_classes,
                                         int anchor_dim,
                                         float confidence_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_objects) return;
    
    // 计算置信度
    // 1. 对class_score应用sigmoid并找到最大值
    int class_offset = idx * num_classes;
    float max_score = -1e10f;  // 初始化为很小的值
    int best_class = 0;
    
    for (int c = 0; c < num_classes; ++c) {
        float sigmoid_score = sigmoid_f(pred_class_score[class_offset + c]);
        if (sigmoid_score > max_score) {
            max_score = sigmoid_score;
            best_class = c;
        }
    }
    
    // 2. 获取quality_score（格式为[1, 900, 2]，取第一个维度作为centerness）
    // quality_score的实际格式：每个对象有2个值，索引为 idx * 2 + 0 和 idx * 2 + 1
    int quality_offset = idx * 2;  // quality_dims = 2
    float centerness_raw = pred_quality_score[quality_offset + 0];  // 取第一个维度（centerness）
    float centerness = sigmoid_f(centerness_raw);  // 应用sigmoid
    
    // 3. 计算最终置信度：centerness * max_class_score
    float confidence = centerness * max_score;
    
    if (confidence <= confidence_threshold) return;
    
    // 构建边界框
    int anchor_offset = idx * anchor_dim;
    BoundingBox3D box;
    
    // 位置信息
    box.x = pred_anchor[anchor_offset + 0];
    box.y = pred_anchor[anchor_offset + 1];
    box.z = pred_anchor[anchor_offset + 2];
    
    // 尺寸信息：模型输出的是log值，需要应用exp变换
    box.w = expf(pred_anchor[anchor_offset + 3]);  // exp(log(w)) = w
    box.l = expf(pred_anchor[anchor_offset + 4]);  // exp(log(l)) = l
    box.h = expf(pred_anchor[anchor_offset + 5]);  // exp(log(h)) = h
    
    // 航向角：从anchor[6]读取sin(yaw)，从anchor[7]读取cos(yaw)
    const float sin_yaw = pred_anchor[anchor_offset + 6];
    const float cos_yaw = pred_anchor[anchor_offset + 7];
    box.yaw = atan2f(sin_yaw, cos_yaw);
    
    box.confidence = confidence;
    box.label = best_class;
    box.index = idx;
    // 读取track_id（如果pred_track_ids不为空）
    if (pred_track_ids != nullptr) {
        box.track_id = static_cast<int>(pred_track_ids[idx]);
    } else {
        box.track_id = -1;
    }

    
    // 添加到临时数组
    int output_idx = atomicAdd(valid_count, 1);
    temp_boxes[output_idx] = box;
}

// 第二步：并行NMS内核（基于已排序的数据）
__global__ void parallel_nms_kernel(const BoundingBox3D* boxes,
                                   int* suppressed,
                                   int num_boxes,
                                   float iou_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_boxes) return;
    
    // 每个线程检查是否被前面的框抑制
    for (int i = 0; i < idx; ++i) {
        if (!suppressed[i] && boxes[i].label == boxes[idx].label) {
            float iou = calculate_3d_iou(boxes[i], boxes[idx]);
            if (iou > iou_threshold) {
                suppressed[idx] = 1;
                break;
            }
        }
    }
}

// 第三步：压缩结果内核
__global__ void compact_results_kernel(const BoundingBox3D* input_boxes,
                                     const int* suppressed,
                                     BoundingBox3D* output_boxes,
                                     int* output_count,
                                     int num_boxes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_boxes && !suppressed[idx]) {
        int output_idx = atomicAdd(output_count, 1);
        output_boxes[output_idx] = input_boxes[idx];
    }
}



// GPU NMS直接处理原始输入数据
int gpu_nms_direct(const float* pred_anchor,
                   const float* pred_class_score,
                   const float* pred_quality_score,
                   const int32_t* pred_track_ids,
                   int num_objects,
                   int num_classes,
                   BoundingBox3D* d_output_boxes,
                   int* output_count,
                   float confidence_threshold,
                   float iou_threshold,
                   int max_output_boxes,
                   cudaStream_t stream) {
    
    if (num_objects <= 0) {
        cudaMemset(output_count, 0, sizeof(int));
        return 0;
    }
    
    try {
        // 分配临时内存
        BoundingBox3D* d_all_boxes = nullptr;
        BoundingBox3D* d_temp_boxes = nullptr;
        int* d_valid_count = nullptr;
        int* d_suppressed = nullptr;
        
        // 分配足够的内存存储所有可能的框
        cudaError_t err = cudaMalloc(&d_all_boxes, num_objects * sizeof(BoundingBox3D));
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for all boxes: " << cudaGetErrorString(err);
            return -1;
        }
        
        err = cudaMalloc(&d_temp_boxes, max_output_boxes * sizeof(BoundingBox3D));
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for temp boxes: " << cudaGetErrorString(err);
            cudaFree(d_all_boxes);
            return -1;
        }
        
        err = cudaMalloc(&d_valid_count, sizeof(int));
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for valid count: " << cudaGetErrorString(err);
            cudaFree(d_all_boxes);
            cudaFree(d_temp_boxes);
            return -1;
        }
        
        err = cudaMalloc(&d_suppressed, max_output_boxes * sizeof(int));
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for suppressed: " << cudaGetErrorString(err);
            cudaFree(d_all_boxes);
            cudaFree(d_temp_boxes);
            cudaFree(d_valid_count);
            return -1;
        }
        
        // 初始化计数
        cudaMemset(d_valid_count, 0, sizeof(int));
        cudaMemset(output_count, 0, sizeof(int));
        cudaMemset(d_suppressed, 0, max_output_boxes * sizeof(int));
        
        // 设置CUDA内核参数
        int block_size = 256;
        int grid_size = (num_objects + block_size - 1) / block_size;
        
        // 第一步：转换和过滤（只进行置信度过滤）
        convert_and_filter_kernel<<<grid_size, block_size, 0, stream>>>(
            pred_anchor,
            pred_class_score,
            pred_quality_score,
            pred_track_ids,
            d_all_boxes,
            d_valid_count,
            num_objects,
            num_classes,
            11,  // anchor_dim
            confidence_threshold
        );
        
        // 检查CUDA错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] CUDA error in convert kernel: " << cudaGetErrorString(err);
            cudaFree(d_all_boxes);
            cudaFree(d_temp_boxes);
            cudaFree(d_valid_count);
            cudaFree(d_suppressed);
            return -1;
        }
        
        // 获取有效框数量
        int valid_boxes;
        cudaMemcpy(&valid_boxes, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (valid_boxes <= 0) {
            cudaFree(d_all_boxes);
            cudaFree(d_temp_boxes);
            cudaFree(d_valid_count);
            cudaFree(d_suppressed);
            return 0;
        }
        
        // 按置信度排序
        thrust::device_ptr<BoundingBox3D> d_all_ptr(d_all_boxes);
        thrust::sort(thrust::cuda::par.on(stream),
                     d_all_ptr, d_all_ptr + valid_boxes,
                     CompareBoxes());
        
        // 第二步：选择前max_output_boxes个框进行NMS
        int nms_boxes = std::min(valid_boxes, max_output_boxes);
        cudaMemcpyAsync(d_temp_boxes, d_all_boxes, 
                       nms_boxes * sizeof(BoundingBox3D),
                       cudaMemcpyDeviceToDevice, stream);
        
        // 第三步：并行NMS
        int nms_grid_size = (nms_boxes + block_size - 1) / block_size;
        parallel_nms_kernel<<<nms_grid_size, block_size, 0, stream>>>(
            d_temp_boxes,
            d_suppressed,
            nms_boxes,
            iou_threshold
        );
        
        // 检查CUDA错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] CUDA error in NMS kernel: " << cudaGetErrorString(err);
            cudaFree(d_all_boxes);
            cudaFree(d_temp_boxes);
            cudaFree(d_valid_count);
            cudaFree(d_suppressed);
            return -1;
        }
        
        // 第四步：压缩结果
        compact_results_kernel<<<nms_grid_size, block_size, 0, stream>>>(
            d_temp_boxes,
            d_suppressed,
            d_output_boxes,
            output_count,
            nms_boxes
        );
        
        // 检查CUDA错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] CUDA error in compact kernel: " << cudaGetErrorString(err);
            cudaFree(d_all_boxes);
            cudaFree(d_temp_boxes);
            cudaFree(d_valid_count);
            cudaFree(d_suppressed);
            return -1;
        }
        
        // 同步流
        cudaStreamSynchronize(stream);
        
        // 清理临时内存
        cudaFree(d_all_boxes);
        cudaFree(d_temp_boxes);
        cudaFree(d_valid_count);
        cudaFree(d_suppressed);
        
        // 获取输出计数
        int host_output_count;
        cudaMemcpy(&host_output_count, output_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        return host_output_count;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] Exception in GPU NMS direct: " << e.what();
        return -1;
    }
}

// 检查GPU是否可用
bool is_gpu_nms_available() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

} // namespace postprocessor
} // namespace sparse4d 