#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include "../../gpu_nms.h"

// 生成随机3D边界框
std::vector<BoundingBox3D> generateRandomBoxes(int num_boxes) {
    std::vector<BoundingBox3D> boxes;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_real_distribution<float> pos_dist(-50.0f, 50.0f);
    std::uniform_real_distribution<float> size_dist(1.0f, 10.0f);
    std::uniform_real_distribution<float> angle_dist(-M_PI, M_PI);
    std::uniform_real_distribution<float> conf_dist(0.1f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, 9);
    
    for (int i = 0; i < num_boxes; ++i) {
        BoundingBox3D box;
        box.x = pos_dist(gen);
        box.y = pos_dist(gen);
        box.z = pos_dist(gen);
        box.l = size_dist(gen);
        box.w = size_dist(gen);
        box.h = size_dist(gen);
        box.yaw = angle_dist(gen);
        box.confidence = conf_dist(gen);
        box.label = label_dist(gen);
        box.index = i;
        
        boxes.push_back(box);
    }
    
    return boxes;
}

// 生成重叠的边界框用于测试NMS
std::vector<BoundingBox3D> generateOverlappingBoxes() {
    std::vector<BoundingBox3D> boxes;
    
    // 创建一组重叠的边界框
    for (int i = 0; i < 10; ++i) {
        BoundingBox3D box;
        box.x = 10.0f + i * 0.5f;  // 逐渐偏移
        box.y = 10.0f + i * 0.3f;
        box.z = 0.0f;
        box.l = 4.0f;
        box.w = 2.0f;
        box.h = 1.5f;
        box.yaw = 0.0f;
        box.confidence = 0.9f - i * 0.05f;  // 递减的置信度
        box.label = 0;  // 同一类别
        box.index = i;
        
        boxes.push_back(box);
    }
    
    return boxes;
}

// 测试GPU NMS基本功能
void testBasicGPUNMS() {
    std::cout << "=== Testing Basic GPU NMS ===" << std::endl;
    
    GPUNMS gpuNMS;
    if (!gpuNMS.initialize()) {
        std::cout << "Failed to initialize GPU NMS, skipping test" << std::endl;
        return;
    }
    
    // 生成测试数据
    auto input_boxes = generateOverlappingBoxes();
    std::vector<BoundingBox3D> output_boxes;
    
    std::cout << "Input boxes: " << input_boxes.size() << std::endl;
    
    // 执行GPU NMS
    auto start = std::chrono::high_resolution_clock::now();
    int output_count = gpuNMS.process(input_boxes, output_boxes, 0.5f, 100);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Output boxes: " << output_count << std::endl;
    std::cout << "GPU NMS time: " << duration.count() << " microseconds" << std::endl;
    
    // 验证结果
    assert(output_count > 0);
    assert(output_count <= input_boxes.size());
    
    std::cout << "Basic GPU NMS test passed!" << std::endl;
}

// 测试批量GPU NMS
void testBatchGPUNMS() {
    std::cout << "=== Testing Batch GPU NMS ===" << std::endl;
    
    GPUNMS gpuNMS;
    if (!gpuNMS.initialize()) {
        std::cout << "Failed to initialize GPU NMS, skipping test" << std::endl;
        return;
    }
    
    // 生成多类别测试数据
    auto input_boxes = generateRandomBoxes(100);
    std::vector<BoundingBox3D> output_boxes;
    
    std::cout << "Input boxes: " << input_boxes.size() << std::endl;
    
    // 执行批量GPU NMS
    auto start = std::chrono::high_resolution_clock::now();
    int output_count = gpuNMS.processBatch(input_boxes, output_boxes, 0.5f, 50);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Output boxes: " << output_count << std::endl;
    std::cout << "Batch GPU NMS time: " << duration.count() << " microseconds" << std::endl;
    
    // 验证结果
    assert(output_count >= 0);
    assert(output_count <= 50);
    
    std::cout << "Batch GPU NMS test passed!" << std::endl;
}

// 性能对比测试
void testPerformanceComparison() {
    std::cout << "=== Performance Comparison Test ===" << std::endl;
    
    GPUNMS gpuNMS;
    if (!gpuNMS.initialize()) {
        std::cout << "Failed to initialize GPU NMS, skipping test" << std::endl;
        return;
    }
    
    // 生成大量测试数据
    auto input_boxes = generateRandomBoxes(1000);
    std::vector<BoundingBox3D> output_boxes;
    
    std::cout << "Testing with " << input_boxes.size() << " boxes" << std::endl;
    
    // GPU NMS性能测试
    auto start = std::chrono::high_resolution_clock::now();
    int gpu_output_count = gpuNMS.processBatch(input_boxes, output_boxes, 0.5f, 200);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "GPU NMS: " << gpu_output_count << " boxes in " 
              << gpu_duration.count() << " microseconds" << std::endl;
    
    // 简单的CPU NMS性能测试（用于对比）
    start = std::chrono::high_resolution_clock::now();
    
    // 简化的CPU NMS实现
    std::vector<BoundingBox3D> cpu_output;
    std::vector<bool> suppressed(input_boxes.size(), false);
    
    // 按置信度排序
    std::sort(input_boxes.begin(), input_boxes.end(), 
              [](const BoundingBox3D& a, const BoundingBox3D& b) {
                  return a.confidence > b.confidence;
              });
    
    for (size_t i = 0; i < input_boxes.size(); ++i) {
        if (suppressed[i]) continue;
        
        cpu_output.push_back(input_boxes[i]);
        
        for (size_t j = i + 1; j < input_boxes.size(); ++j) {
            if (suppressed[j] || input_boxes[i].label != input_boxes[j].label) continue;
            
            // 简化的IoU计算
            float dx = input_boxes[i].x - input_boxes[j].x;
            float dy = input_boxes[i].y - input_boxes[j].y;
            float distance = sqrtf(dx * dx + dy * dy);
            float threshold = 2.0f;  // 简化的阈值
            
            if (distance < threshold) {
                suppressed[j] = true;
            }
        }
        
        if (cpu_output.size() >= 200) break;
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "CPU NMS: " << cpu_output.size() << " boxes in " 
              << cpu_duration.count() << " microseconds" << std::endl;
    
    if (cpu_duration.count() > 0) {
        float speedup = static_cast<float>(cpu_duration.count()) / gpu_duration.count();
        std::cout << "GPU speedup: " << speedup << "x" << std::endl;
    }
    
    std::cout << "Performance comparison test completed!" << std::endl;
}

// 测试GPU NMS错误处理
void testErrorHandling() {
    std::cout << "=== Testing Error Handling ===" << std::endl;
    
    GPUNMS gpuNMS;
    
    // 测试未初始化的情况
    std::vector<BoundingBox3D> input_boxes = generateRandomBoxes(10);
    std::vector<BoundingBox3D> output_boxes;
    
    int result = gpuNMS.process(input_boxes, output_boxes, 0.5f, 100);
    assert(result == -1);  // 应该返回错误
    
    std::cout << "Error handling test passed!" << std::endl;
}

int main() {
    std::cout << "Starting GPU NMS Unit Tests..." << std::endl;
    
    try {
        testErrorHandling();
        testBasicGPUNMS();
        testBatchGPUNMS();
        testPerformanceComparison();
        
        std::cout << "\nAll GPU NMS tests passed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 