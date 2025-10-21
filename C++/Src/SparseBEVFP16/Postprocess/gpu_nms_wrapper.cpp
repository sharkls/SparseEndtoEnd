#include "gpu_nms.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

GPUNMS::GPUNMS() : m_initialized(false), m_device_count(0) {
}

GPUNMS::~GPUNMS() {
    if (m_initialized) {
        cudaDeviceReset();
    }
}

bool GPUNMS::initialize() {
    // 检查CUDA设备
    cudaError_t err = cudaGetDeviceCount(&m_device_count);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    if (m_device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    // 设置设备
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // 获取设备信息
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    m_device_name = prop.name;
    m_initialized = true;
    
    std::cout << "GPU NMS initialized successfully on device: " << m_device_name << std::endl;
    return true;
}

int GPUNMS::process(const std::vector<BoundingBox3D>& input_boxes,
                    std::vector<BoundingBox3D>& output_boxes,
                    float iou_threshold,
                    int max_output_boxes) {
    
    if (!m_initialized) {
        std::cerr << "GPU NMS not initialized" << std::endl;
        return -1;
    }
    
    if (input_boxes.empty()) {
        output_boxes.clear();
        return 0;
    }
    
    // 准备输出缓冲区
    output_boxes.resize(max_output_boxes);
    
    // 调用GPU NMS
    int output_count = gpuNMS(input_boxes.data(), 
                              output_boxes.data(),
                              input_boxes.size(), 
                              iou_threshold,
                              max_output_boxes);
    
    if (output_count < 0) {
        std::cerr << "GPU NMS failed" << std::endl;
        return -1;
    }
    
    // 调整输出大小
    output_boxes.resize(output_count);
    
    return output_count;
}

int GPUNMS::processBatch(const std::vector<BoundingBox3D>& input_boxes,
                         std::vector<BoundingBox3D>& output_boxes,
                         float iou_threshold,
                         int max_output_boxes) {
    
    if (!m_initialized) {
        std::cerr << "GPU NMS not initialized" << std::endl;
        return -1;
    }
    
    if (input_boxes.empty()) {
        output_boxes.clear();
        return 0;
    }
    
    // 准备输出缓冲区
    output_boxes.resize(max_output_boxes);
    
    // 调用批量GPU NMS
    int output_count = gpuBatchNMS(input_boxes.data(), 
                                   output_boxes.data(),
                                   input_boxes.size(), 
                                   iou_threshold,
                                   max_output_boxes);
    
    if (output_count < 0) {
        std::cerr << "GPU Batch NMS failed" << std::endl;
        return -1;
    }
    
    // 调整输出大小
    output_boxes.resize(output_count);
    
    return output_count;
}

bool GPUNMS::isGPUAvaliable() const {
    return m_initialized;
}

void GPUNMS::getGPUInfo(int& device_count, std::string& device_name) const {
    device_count = m_device_count;
    device_name = m_device_name;
} 