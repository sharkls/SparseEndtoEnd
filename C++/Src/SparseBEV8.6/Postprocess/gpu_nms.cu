#include "gpu_nms.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

// 设备端比较函数（用于排序）
struct CompareBoxes {
    __host__ __device__
    bool operator()(const BoundingBox3D& a, const BoundingBox3D& b) const {
        return a.confidence > b.confidence;
    }
};

// 计算两个3D边界框的IoU（GPU版本）
__device__ float calculate3DIoU(const BoundingBox3D& box1, const BoundingBox3D& box2) {
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
    return max(0.0f, overlap_ratio);
}

// 检查是否应该抑制当前检测框
__device__ bool shouldSuppress(const BoundingBox3D* boxes, int current_idx, int* suppressed, 
                              int num_boxes, float iou_threshold) {
    if (suppressed[current_idx]) {
        return true;
    }
    
    for (int i = 0; i < current_idx; ++i) {
        if (!suppressed[i] && boxes[i].label == boxes[current_idx].label) {
            float iou = calculate3DIoU(boxes[i], boxes[current_idx]);
            if (iou > iou_threshold) {
                return true;
            }
        }
    }
    
    return false;
}

// GPU NMS内核
__global__ void gpuNMSKernel(const BoundingBox3D* boxes, int* suppressed, 
                            int num_boxes, float iou_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_boxes) {
        // 检查当前检测框是否应该被抑制
        if (shouldSuppress(boxes, idx, suppressed, num_boxes, iou_threshold)) {
            suppressed[idx] = 1;
        }
    }
}

// 并行NMS实现
__global__ void parallelNMSKernel(const BoundingBox3D* boxes, int* suppressed, 
                                 int num_boxes, float iou_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_boxes) return;
    
    // 每个线程处理一个检测框
    for (int i = 0; i < idx; ++i) {
        if (!suppressed[i] && boxes[i].label == boxes[idx].label) {
            float iou = calculate3DIoU(boxes[i], boxes[idx]);
            if (iou > iou_threshold) {
                suppressed[idx] = 1;
                break;
            }
        }
    }
}

// 压缩结果内核
__global__ void compactResultsKernel(const BoundingBox3D* input_boxes, 
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

// GPU NMS主函数
extern "C" int gpuNMS(const BoundingBox3D* host_boxes, 
                     BoundingBox3D* host_output_boxes,
                     int num_boxes, 
                     float iou_threshold,
                     int max_output_boxes) {
    
    // 分配GPU内存
    thrust::device_vector<BoundingBox3D> d_boxes(host_boxes, host_boxes + num_boxes);
    thrust::device_vector<int> d_suppressed(num_boxes, 0);
    thrust::device_vector<BoundingBox3D> d_output_boxes(max_output_boxes);
    thrust::device_vector<int> d_output_count(1, 0);
    
    // 按置信度排序（降序）- 使用设备函数而不是lambda
    thrust::sort(d_boxes.begin(), d_boxes.end(), CompareBoxes());
    
    // 设置CUDA内核参数
    int block_size = 256;
    int grid_size = (num_boxes + block_size - 1) / block_size;
    
    // 执行NMS
    parallelNMSKernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(d_boxes.data()),
        thrust::raw_pointer_cast(d_suppressed.data()),
        num_boxes,
        iou_threshold
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in NMS kernel: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // 压缩结果
    compactResultsKernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(d_boxes.data()),
        thrust::raw_pointer_cast(d_suppressed.data()),
        thrust::raw_pointer_cast(d_output_boxes.data()),
        thrust::raw_pointer_cast(d_output_count.data()),
        num_boxes
    );
    
    // 检查CUDA错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in compact kernel: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // 获取输出数量
    int output_count = d_output_count[0];
    output_count = min(output_count, max_output_boxes);
    
    // 复制结果到主机
    thrust::copy(d_output_boxes.begin(), d_output_boxes.begin() + output_count, host_output_boxes);
    
    return output_count;
}

// 批量NMS处理（处理多个类别）
extern "C" int gpuBatchNMS(const BoundingBox3D* host_boxes, 
                          BoundingBox3D* host_output_boxes,
                          int num_boxes, 
                          float iou_threshold,
                          int max_output_boxes) {
    
    // 按类别分组处理
    std::vector<std::vector<BoundingBox3D>> class_groups;
    std::vector<int> class_labels;
    
    // 收集所有类别
    for (int i = 0; i < num_boxes; ++i) {
        bool found = false;
        for (int j = 0; j < class_labels.size(); ++j) {
            if (class_labels[j] == host_boxes[i].label) {
                class_groups[j].push_back(host_boxes[i]);
                found = true;
                break;
            }
        }
        if (!found) {
            class_labels.push_back(host_boxes[i].label);
            class_groups.push_back({host_boxes[i]});
        }
    }
    
    // 对每个类别执行NMS
    int total_output = 0;
    for (int c = 0; c < class_groups.size(); ++c) {
        if (class_groups[c].empty()) continue;
        
        int remaining_boxes = max_output_boxes - total_output;
        if (remaining_boxes <= 0) break;
        
        int class_output = gpuNMS(class_groups[c].data(), 
                                 host_output_boxes + total_output,
                                 class_groups[c].size(), 
                                 iou_threshold,
                                 remaining_boxes);
        
        total_output += class_output;
    }
    
    return total_output;
} 