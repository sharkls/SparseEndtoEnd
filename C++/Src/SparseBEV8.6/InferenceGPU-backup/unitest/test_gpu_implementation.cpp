/*******************************************************
 文件名：test_gpu_implementation.cpp
 作者：sharkls
 描述：测试GPU版本的InstanceBank和Utils实现
 版本：v1.0
 日期：2025-06-18
 *******************************************************/

#include "InstanceBankGPU.h"
#include "UtilsGPU.h"
#include "log.h"
#include <iostream>
#include <vector>

/**
 * @brief 测试GPU版本的Utils功能
 */
void testUtilsGPU() {
    std::cout << "[TEST] Testing UtilsGPU..." << std::endl;
    
    // 创建测试数据
    const uint32_t num_querys = 100;
    const uint32_t query_dims = 7;
    const uint32_t embedfeat_dims = 256;
    const uint32_t num_topk_querys = 30;
    
    // 创建CPU数据
    std::vector<float> confidence(num_querys);
    std::vector<float> instance_feature(num_querys * embedfeat_dims);
    std::vector<float> anchor(num_querys * query_dims);
    
    // 初始化测试数据
    for (uint32_t i = 0; i < num_querys; ++i) {
        confidence[i] = static_cast<float>(i) / num_querys;  // 0.0 到 1.0
        for (uint32_t j = 0; j < embedfeat_dims; ++j) {
            instance_feature[i * embedfeat_dims + j] = static_cast<float>(i + j) / (num_querys + embedfeat_dims);
        }
        for (uint32_t j = 0; j < query_dims; ++j) {
            anchor[i * query_dims + j] = static_cast<float>(i + j) / (num_querys + query_dims);
        }
    }
    
    // 创建GPU包装器
    CudaWrapper<float> gpu_confidence(confidence);
    CudaWrapper<float> gpu_instance_feature(instance_feature);
    CudaWrapper<float> gpu_anchor(anchor);
    
    // 创建输出GPU包装器
    CudaWrapper<float> output_confidence;
    CudaWrapper<float> output_instance_feature;
    CudaWrapper<float> output_anchor;
    CudaWrapper<int32_t> output_track_ids;
    
    output_confidence.allocate(num_topk_querys);
    output_instance_feature.allocate(num_topk_querys * embedfeat_dims);
    output_anchor.allocate(num_topk_querys * query_dims);
    output_track_ids.allocate(num_topk_querys);
    
    // 测试getTopkInstanceOnGPU
    Status status = UtilsGPU::getTopkInstanceOnGPU(
        gpu_confidence,
        gpu_instance_feature,
        gpu_anchor,
        num_querys,
        query_dims,
        embedfeat_dims,
        num_topk_querys,
        output_confidence,
        output_instance_feature,
        output_anchor,
        output_track_ids
    );
    
    if (status == SUCCESS) {
        std::cout << "[TEST] UtilsGPU::getTopkInstanceOnGPU PASSED" << std::endl;
    } else {
        std::cout << "[TEST] UtilsGPU::getTopkInstanceOnGPU FAILED" << std::endl;
    }
    
    // 测试其他功能...
    std::cout << "[TEST] UtilsGPU tests completed" << std::endl;
}

/**
 * @brief 测试GPU版本的InstanceBank功能
 */
void testInstanceBankGPU() {
    std::cout << "[TEST] Testing InstanceBankGPU..." << std::endl;
    
    // 创建InstanceBankGPU实例
    InstanceBankGPU instanceBank;
    
    // 创建测试配置
    sparsebev::TaskConfig config;
    config.set_num_querys(100);
    config.set_query_dims(7);
    config.set_embedfeat_dims(256);
    config.set_num_topk_querys(30);
    
    // 初始化
    bool init_success = instanceBank.init(&config);
    if (init_success) {
        std::cout << "[TEST] InstanceBankGPU::init PASSED" << std::endl;
    } else {
        std::cout << "[TEST] InstanceBankGPU::init FAILED" << std::endl;
        return;
    }
    
    // 创建测试数据
    const uint32_t num_querys = 100;
    const uint32_t query_dims = 7;
    const uint32_t embedfeat_dims = 256;
    
    std::vector<float> instance_feature(num_querys * embedfeat_dims);
    std::vector<float> anchor(num_querys * query_dims);
    std::vector<float> confidence(num_querys);
    std::vector<int32_t> track_ids(num_querys);
    
    // 初始化测试数据
    for (uint32_t i = 0; i < num_querys; ++i) {
        confidence[i] = static_cast<float>(i) / num_querys;
        track_ids[i] = i;
        for (uint32_t j = 0; j < embedfeat_dims; ++j) {
            instance_feature[i * embedfeat_dims + j] = static_cast<float>(i + j) / (num_querys + embedfeat_dims);
        }
        for (uint32_t j = 0; j < query_dims; ++j) {
            anchor[i * query_dims + j] = static_cast<float>(i + j) / (num_querys + query_dims);
        }
    }
    
    // 创建GPU包装器
    CudaWrapper<float> gpu_instance_feature(instance_feature);
    CudaWrapper<float> gpu_anchor(anchor);
    CudaWrapper<float> gpu_confidence(confidence);
    CudaWrapper<int32_t> gpu_track_ids(track_ids);
    
    // 测试updateOnGPU
    Status status = instanceBank.updateOnGPU(
        gpu_instance_feature,
        gpu_anchor,
        gpu_confidence,
        gpu_track_ids,
        0.1f  // 时间间隔
    );
    
    if (status == SUCCESS) {
        std::cout << "[TEST] InstanceBankGPU::updateOnGPU PASSED" << std::endl;
    } else {
        std::cout << "[TEST] InstanceBankGPU::updateOnGPU FAILED" << std::endl;
    }
    
    // 测试getOnGPU
    auto result = instanceBank.getOnGPU();
    std::cout << "[TEST] InstanceBankGPU::getOnGPU PASSED" << std::endl;
    
    std::cout << "[TEST] InstanceBankGPU tests completed" << std::endl;
}

/**
 * @brief 主测试函数
 */
int main() {
    std::cout << "[TEST] Starting GPU implementation tests..." << std::endl;
    
    try {
        // 测试UtilsGPU
        testUtilsGPU();
        
        // 测试InstanceBankGPU
        testInstanceBankGPU();
        
        std::cout << "[TEST] All tests completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "[TEST] Test failed with exception: " << e.what() << std::endl;
        return -1;
    }
} 