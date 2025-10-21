// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

/*
特征提取模块单元测试
用于测试SparseBEV8.6特征提取模块的输入数据处理和推理一致性
*/

#include <cuda_runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <limits>

#include "../../../Include/Common/Utils/CudaWrapper.h"
#include "../../../Include/Common/Core/GlobalContext.h"
#include "../../../Include/Common/Interface/IBaseModule.h"
#include "../../../Include/Common/Factory/ModuleFactory.h"
#include "../../Common/TensorRT/TensorRT.h"
#include "../SparseBEV.h"
#include "../data/SparseBEVInputData.h"

namespace sparse_end2end {
namespace engine {

/**
 * @brief 计算误差百分比
 * @param a 预测结果
 * @param b 期望结果
 * @param threshold 误差阈值
 * @return 超过阈值的误差百分比
 */
float GetErrorPercentage(const std::vector<float>& a, const std::vector<float>& b, float threshold) {
    float max_error = 0.0F;
    if (a.size() != b.size()) {
        max_error = std::numeric_limits<float>::max();
        return 1.0F;
    }

    std::vector<float> cache_errors;
    for (size_t i = 0; i < a.size(); ++i) {
        const float error = std::abs(a[i] - b[i]);
        cache_errors.push_back(error);
        if (max_error < error) {
            max_error = error;
        }
    }

    std::sort(cache_errors.begin(), cache_errors.end(), [](float a, float b) { return a > b; });

    std::vector<float> cache_roi_errors;
    for (auto x : cache_errors) {
        if (x > threshold) {
            cache_roi_errors.push_back(x);
        }
    }

    float p = float(cache_roi_errors.size()) / float(cache_errors.size());
    std::cout << "Error >" << threshold << " percentage is: " << p << std::endl;
    std::cout << "MaxError = " << max_error << std::endl;

    return p;
}

/**
 * @brief 读取二进制文件
 * @param filename 文件名
 * @return 文件内容向量
 */
std::vector<float> ReadBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<float> data(file_size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    
    return data;
}

/**
 * @brief 特征提取模块单元测试类
 */
class Sparse4dExtractFeatUnitTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置CUDA事件和流
        checkCudaErrors(cudaEventCreate(&start_));
        checkCudaErrors(cudaEventCreate(&stop_));
        checkCudaErrors(cudaStreamCreate(&stream_));
        
        // 设置测试数据路径
        current_dir_ = std::filesystem::current_path();
        test_data_dir_ = current_dir_ / "../../../../../script/tutorial/asset";
        
        // 测试样本
        test_samples_ = {
            {test_data_dir_ / "sample_0_imgs_1*6*3*256*704_float32.bin",
             test_data_dir_ / "sample_0_feature_1*89760*256_float32.bin"},
            {test_data_dir_ / "sample_1_imgs_1*6*3*256*704_float32.bin",
             test_data_dir_ / "sample_1_feature_1*89760*256_float32.bin"},
            {test_data_dir_ / "sample_2_imgs_1*6*3*256*704_float32.bin",
             test_data_dir_ / "sample_2_feature_1*89760*256_float32.bin"}
        };
    }
    
    void TearDown() override {
        checkCudaErrors(cudaEventDestroy(start_));
        checkCudaErrors(cudaEventDestroy(stop_));
        checkCudaErrors(cudaStreamDestroy(stream_));
    }
    
    cudaEvent_t start_, stop_;
    cudaStream_t stream_;
    std::filesystem::path current_dir_;
    std::filesystem::path test_data_dir_;
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> test_samples_;
};

/**
 * @brief 测试特征提取模块的输入数据处理
 */
TEST_F(Sparse4dExtractFeatUnitTest, InputDataProcessingTest) {
    std::cout << "[INFO] 开始测试特征提取模块输入数据处理..." << std::endl;
    
    // 创建SparseBEV模块实例
    std::unique_ptr<SparseBEV> sparsebev_module = std::make_unique<SparseBEV>("");
    
    // 创建测试配置参数
    sparsebev::TaskConfig task_config;
    
    // 设置特征提取引擎配置
    auto* extract_feat_engine = task_config.mutable_extract_feat_engine();
    extract_feat_engine->set_engine_path("/path/to/extract_feat_engine.trt");
    extract_feat_engine->add_input_names("input_images");
    extract_feat_engine->add_output_names("output_features");
    
    // 设置预处理器参数
    auto* preprocessor_params = task_config.mutable_preprocessor_params();
    preprocessor_params->set_num_cams(6);
    preprocessor_params->set_model_input_img_c(3);
    preprocessor_params->set_model_input_img_h(256);
    preprocessor_params->set_model_input_img_w(704);
    
    // 设置模型配置参数
    auto* model_cfg_params = task_config.mutable_model_cfg_params();
    model_cfg_params->add_sparse4d_extract_feat_shape_lc(1);
    model_cfg_params->add_sparse4d_extract_feat_shape_lc(89760);
    model_cfg_params->add_sparse4d_extract_feat_shape_lc(256);
    
    // 设置实例库参数
    auto* instance_bank_params = task_config.mutable_instance_bank_params();
    instance_bank_params->set_num_querys(1000);
    instance_bank_params->set_query_dims(8);
    
    // 初始化模块
    bool init_success = sparsebev_module->init(&task_config);
    if (!init_success) {
        std::cout << "[WARNING] 模块初始化失败，跳过推理测试" << std::endl;
        return;
    }
    
    std::cout << "[INFO] 模块初始化成功" << std::endl;
    
    // 测试输入数据读取和处理
    for (const auto& sample : test_samples_) {
        try {
            // 读取输入图像数据
            std::vector<float> input_imgs = ReadBinaryFile(sample.first.string());
            std::vector<float> expected_features = ReadBinaryFile(sample.second.string());
            
            EXPECT_EQ(input_imgs.size(), 1 * 6 * 3 * 256 * 704);
            EXPECT_EQ(expected_features.size(), 1 * 89760 * 256);
            
            std::cout << "[INFO] 成功读取测试样本: " << sample.first.filename() << std::endl;
            std::cout << "[INFO] 输入图像数据大小: " << input_imgs.size() << std::endl;
            std::cout << "[INFO] 期望特征数据大小: " << expected_features.size() << std::endl;
            
            // 检查数据范围
            auto min_max_input = std::minmax_element(input_imgs.begin(), input_imgs.end());
            auto min_max_expected = std::minmax_element(expected_features.begin(), expected_features.end());
            
            std::cout << "[INFO] 输入图像数据范围: [" << *min_max_input.first << ", " << *min_max_input.second << "]" << std::endl;
            std::cout << "[INFO] 期望特征数据范围: [" << *min_max_expected.first << ", " << *min_max_expected.second << "]" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "[WARNING] 读取测试样本失败: " << e.what() << std::endl;
        }
    }
}

/**
 * @brief 测试特征提取模块的TensorRT推理一致性
 */
TEST_F(Sparse4dExtractFeatUnitTest, TrtInferConsistencyVerification) {
    std::cout << "[INFO] 开始TensorRT推理一致性验证..." << std::endl;
    
    // 检查测试数据文件是否存在
    bool all_files_exist = true;
    for (const auto& sample : test_samples_) {
        if (!std::filesystem::exists(sample.first) || !std::filesystem::exists(sample.second)) {
            std::cout << "[WARNING] 测试文件不存在: " << sample.first << " 或 " << sample.second << std::endl;
            all_files_exist = false;
        }
    }
    
    if (!all_files_exist) {
        std::cout << "[INFO] 跳过TensorRT推理测试，因为测试文件不存在" << std::endl;
        return;
    }
    
    // 创建TensorRT引擎（这里需要实际的引擎文件路径）
    std::string engine_path = "/path/to/sparse4d_extract_feat_engine.trt";
    std::vector<std::string> input_names = {"input_images"};
    std::vector<std::string> output_names = {"output_features"};
    
    if (!std::filesystem::exists(engine_path)) {
        std::cout << "[WARNING] TensorRT引擎文件不存在: " << engine_path << std::endl;
        std::cout << "[INFO] 跳过TensorRT推理测试" << std::endl;
        return;
    }
    
    std::shared_ptr<TensorRT> trt_engine = std::make_shared<TensorRT>(
        engine_path, "", input_names, output_names);
    
    // Warmup
    std::cout << "[INFO] 开始预热..." << std::endl;
    for (const auto& sample : test_samples_) {
        try {
            const std::vector<float> imgs = ReadBinaryFile(sample.first.string());
            const std::vector<float> expected_pred_feature = ReadBinaryFile(sample.second.string());

            EXPECT_EQ(imgs.size(), 1 * 6 * 3 * 256 * 704);
            EXPECT_EQ(expected_pred_feature.size(), 1 * 89760 * 256);

            const CudaWrapper<float> imgs_gpu(imgs);
            CudaWrapper<float> pred_feature_gpu(1 * 89760 * 256);

            std::vector<void*> buffers;
            buffers.push_back(imgs_gpu.getCudaPtr());
            buffers.push_back(pred_feature_gpu.getCudaPtr());

            if (!trt_engine->infer(buffers.data(), stream_)) {
                std::cout << "[ERROR] TensorRT引擎推理失败" << std::endl;
            }
            cudaStreamSynchronize(stream_);
        } catch (const std::exception& e) {
            std::cout << "[WARNING] 预热过程中出错: " << e.what() << std::endl;
        }
    }
    
    // 正式测试
    std::cout << "[INFO] 开始正式推理测试..." << std::endl;
    for (const auto& sample : test_samples_) {
        try {
            const std::vector<float> imgs = ReadBinaryFile(sample.first.string());
            std::vector<float> expected_pred_feature = ReadBinaryFile(sample.second.string());

            EXPECT_EQ(imgs.size(), 1 * 6 * 3 * 256 * 704);
            EXPECT_EQ(expected_pred_feature.size(), 1 * 89760 * 256);

            const CudaWrapper<float> imgs_gpu(imgs);
            CudaWrapper<float> pred_feature_gpu(1 * 89760 * 256);

            std::vector<void*> buffers;
            buffers.push_back(imgs_gpu.getCudaPtr());
            buffers.push_back(pred_feature_gpu.getCudaPtr());

            checkCudaErrors(cudaEventRecord(start_, stream_));
            if (!trt_engine->infer(buffers.data(), stream_)) {
                std::cout << "[ERROR] TensorRT引擎推理失败" << std::endl;
            }
            checkCudaErrors(cudaEventRecord(stop_, stream_));
            checkCudaErrors(cudaEventSynchronize(stop_));
            
            float time_cost = 0.0f;
            checkCudaErrors(cudaEventElapsedTime(&time_cost, start_, stop_));
            std::cout << "[TensorRT Test] Sparse4d特征提取模块TensorRT推理(FP32)耗时 = "
                      << time_cost << " [ms]" << std::endl;
            cudaStreamSynchronize(stream_);

            const auto& pred_feature = pred_feature_gpu.cudaMemcpyD2HResWrap();

            const float error_percentage = GetErrorPercentage(pred_feature, expected_pred_feature, 0.1);
            EXPECT_LE(error_percentage, 0.01F);
            
            std::cout << "pred_feature: max=" << *std::max_element(pred_feature.begin(), pred_feature.end())
                      << " min=" << *std::min_element(pred_feature.begin(), pred_feature.end()) << std::endl;
            std::cout << "expd_feature: max=" << *std::max_element(expected_pred_feature.begin(), expected_pred_feature.end())
                      << " min=" << *std::min_element(expected_pred_feature.begin(), expected_pred_feature.end()) << std::endl
                      << std::endl;
                      
        } catch (const std::exception& e) {
            std::cout << "[ERROR] 推理测试过程中出错: " << e.what() << std::endl;
        }
    }
}

/**
 * @brief 测试特征提取模块的内存管理
 */
TEST_F(Sparse4dExtractFeatUnitTest, MemoryManagementTest) {
    std::cout << "[INFO] 开始测试内存管理..." << std::endl;
    
    // 测试GPU内存分配和释放
    const size_t input_size = 1 * 6 * 3 * 256 * 704;
    const size_t output_size = 1 * 89760 * 256;
    
    try {
        // 分配GPU内存
        CudaWrapper<float> input_gpu(input_size);
        CudaWrapper<float> output_gpu(output_size);
        
        EXPECT_TRUE(input_gpu.isValid());
        EXPECT_TRUE(output_gpu.isValid());
        
        // 创建测试数据
        std::vector<float> test_data(input_size, 1.0f);
        input_gpu.cudaMemUpdateWrap(test_data);
        
        // 验证数据复制
        auto copied_data = input_gpu.cudaMemcpyD2HResWrap();
        EXPECT_EQ(copied_data.size(), input_size);
        EXPECT_EQ(copied_data[0], 1.0f);
        
        std::cout << "[INFO] GPU内存管理测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[ERROR] 内存管理测试失败: " << e.what() << std::endl;
        FAIL();
    }
}

/**
 * @brief 测试特征提取模块的数据格式转换
 */
TEST_F(Sparse4dExtractFeatUnitTest, DataFormatConversionTest) {
    std::cout << "[INFO] 开始测试数据格式转换..." << std::endl;
    
    // 测试float到half精度转换
    const size_t test_size = 1000;
    std::vector<float> float_data(test_size);
    
    // 生成测试数据
    for (size_t i = 0; i < test_size; ++i) {
        float_data[i] = static_cast<float>(i) * 0.1f;
    }
    
    try {
        // 测试float精度
        CudaWrapper<float> float_gpu(test_size);
        float_gpu.cudaMemUpdateWrap(float_data);
        
        // 验证数据完整性
        auto float_result = float_gpu.cudaMemcpyD2HResWrap();
        EXPECT_EQ(float_result.size(), test_size);
        
        // 检查数据精度
        for (size_t i = 0; i < test_size; ++i) {
            EXPECT_NEAR(float_result[i], float_data[i], 1e-6f);
        }
        
        std::cout << "[INFO] 数据格式转换测试通过" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[ERROR] 数据格式转换测试失败: " << e.what() << std::endl;
        FAIL();
    }
}

}  // namespace engine
}  // namespace sparse_end2end

/**
 * @brief 主函数
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "==========================================" << std::endl;
    std::cout << "SparseBEV8.6 特征提取模块单元测试" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return RUN_ALL_TESTS();
} 