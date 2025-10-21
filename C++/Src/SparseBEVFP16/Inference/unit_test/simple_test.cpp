// 简单的单元测试，不依赖CUDA
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

namespace sparse_end2end {
namespace engine {

/**
 * @brief 简单的数据验证测试
 */
class SimpleDataTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化测试数据
        test_data_ = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    }
    
    std::vector<float> test_data_;
};

/**
 * @brief 测试数据读取功能
 */
TEST_F(SimpleDataTest, DataReadingTest) {
    EXPECT_EQ(test_data_.size(), 5);
    EXPECT_EQ(test_data_[0], 1.0f);
    EXPECT_EQ(test_data_[4], 5.0f);
    
    std::cout << "[INFO] 数据读取测试通过" << std::endl;
}

/**
 * @brief 测试数据计算功能
 */
TEST_F(SimpleDataTest, DataComputationTest) {
    float sum = std::accumulate(test_data_.begin(), test_data_.end(), 0.0f);
    float expected_sum = 15.0f;
    
    EXPECT_NEAR(sum, expected_sum, 1e-6f);
    
    auto max_element = *std::max_element(test_data_.begin(), test_data_.end());
    EXPECT_EQ(max_element, 5.0f);
    
    std::cout << "[INFO] 数据计算测试通过" << std::endl;
}

/**
 * @brief 测试误差计算功能
 */
TEST_F(SimpleDataTest, ErrorCalculationTest) {
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> actual = {1.1f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    float max_error = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        float error = std::abs(expected[i] - actual[i]);
        max_error = std::max(max_error, error);
    }
    
    EXPECT_NEAR(max_error, 0.1f, 1e-6f);
    
    std::cout << "[INFO] 误差计算测试通过" << std::endl;
}

/**
 * @brief 测试数据格式验证
 */
TEST_F(SimpleDataTest, DataFormatValidationTest) {
    // 模拟输入图像数据大小
    size_t num_cams = 6;
    size_t channels = 3;
    size_t height = 256;
    size_t width = 704;
    size_t expected_size = 1 * num_cams * channels * height * width;
    
    EXPECT_EQ(expected_size, 3244032);
    
    // 模拟输出特征数据大小
    size_t feature_size = 1 * 89760 * 256;
    EXPECT_EQ(feature_size, 22978560);
    
    std::cout << "[INFO] 数据格式验证测试通过" << std::endl;
    std::cout << "[INFO] 输入图像数据大小: " << expected_size << std::endl;
    std::cout << "[INFO] 输出特征数据大小: " << feature_size << std::endl;
}

}  // namespace engine
}  // namespace sparse_end2end

/**
 * @brief 主函数
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "==========================================" << std::endl;
    std::cout << "SparseBEV8.6 简单单元测试" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return RUN_ALL_TESTS();
} 