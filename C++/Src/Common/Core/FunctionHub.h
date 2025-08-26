/*******************************************************
 文件名：FunctionHub.h
 作者：sharkls
 描述：函数集，用于基础模块的运行及结果数据处理
 版本：v1.0
 日期：2025-06-18
 *******************************************************/

#ifndef __FUNCTIONHUB_H__
#define __FUNCTIONHUB_H__

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

// 保存 float 数组为二进制文件
inline void save_bin(const std::vector<float>& input_data, const std::string& filename) 
{
    // 自动创建父目录
    std::filesystem::path file_path(filename);
    std::filesystem::create_directories(file_path.parent_path());

    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(input_data.data()), input_data.size() * sizeof(float));
    ofs.close();
}

inline void save_bin(const std::vector<std::vector<float>>& input_data, const std::string& filename) 
{   
    // 自动创建父目录
    std::filesystem::path file_path(filename);
    std::filesystem::create_directories(file_path.parent_path());

    std::ofstream ofs(filename, std::ios::binary);
    for (const auto& v : input_data) {
        ofs.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    }
    ofs.close();
}

/**
 * 获取当前ms UTC时间
 * 参数：
 * 返回值：ms UTC时间
 */
inline int64_t GetTimeStamp()
{
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp =
        std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());

    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    return tmp.count();
}

/**
 * 从CTestSrcData中提取预处理图像数据
 * @param preprocessed_imgs 预处理图像数据
 * @param num_cameras 相机数量
 * @param img_channels 图像通道数
 * @param img_height 图像高度
 * @param img_width 图像宽度
 * @param camera_id 相机ID (0-5)
 * @param channel 通道 (0-2 for RGB)
 * @return 提取的图像数据
 */
inline std::vector<float> extract_preprocessed_image_data(
    const std::vector<float>& preprocessed_imgs,
    unsigned long num_cameras,
    unsigned long img_channels,
    unsigned long img_height,
    unsigned long img_width,
    int camera_id = -1,
    int channel = -1)
{
    if (preprocessed_imgs.empty()) {
        return std::vector<float>();
    }
    
    // 验证数据大小
    size_t expected_size = num_cameras * img_channels * img_height * img_width;
    if (preprocessed_imgs.size() != expected_size) {
        return std::vector<float>();
    }
    
    // 如果指定了相机ID，提取该相机的数据
    if (camera_id >= 0 && camera_id < static_cast<int>(num_cameras)) {
        size_t camera_offset = camera_id * img_channels * img_height * img_width;
        size_t camera_size = img_channels * img_height * img_width;
        
        if (channel >= 0 && channel < static_cast<int>(img_channels)) {
            // 提取指定通道的数据
            size_t channel_offset = camera_offset + channel * img_height * img_width;
            size_t channel_size = img_height * img_width;
            
            return std::vector<float>(
                preprocessed_imgs.begin() + channel_offset,
                preprocessed_imgs.begin() + channel_offset + channel_size
            );
        } else {
            // 提取整个相机的数据
            return std::vector<float>(
                preprocessed_imgs.begin() + camera_offset,
                preprocessed_imgs.begin() + camera_offset + camera_size
            );
        }
    }
    
    // 返回所有数据
    return preprocessed_imgs;
}

/**
 * 验证预处理图像数据的有效性
 * @param preprocessed_imgs 预处理图像数据
 * @param num_cameras 相机数量
 * @param img_channels 图像通道数
 * @param img_height 图像高度
 * @param img_width 图像宽度
 * @return 是否有效
 */
inline bool validate_preprocessed_image_data(
    const std::vector<float>& preprocessed_imgs,
    unsigned long num_cameras,
    unsigned long img_channels,
    unsigned long img_height,
    unsigned long img_width)
{
    if (preprocessed_imgs.empty()) {
        return false;
    }
    
    size_t expected_size = num_cameras * img_channels * img_height * img_width;
    return preprocessed_imgs.size() == expected_size;
}

/**
 * 分析两个数据数组的差异
 * @param current_data 当前数据
 * @param expected_data 期望数据
 * @param data_name 数据名称（用于日志）
 */
inline void analyze_data_difference(
    const std::vector<float>& current_data,
    const std::vector<float>& expected_data,
    const std::string& data_name = "data")
{
    if (current_data.size() != expected_data.size()) {
        std::cout << "[ERROR] " << data_name << " size mismatch: current=" << current_data.size() 
                  << ", expected=" << expected_data.size() << std::endl;
        return;
    }
    
    // 基本统计信息
    float current_min = *std::min_element(current_data.begin(), current_data.end());
    float current_max = *std::max_element(current_data.begin(), current_data.end());
    float current_sum = std::accumulate(current_data.begin(), current_data.end(), 0.0f);
    float current_mean = current_sum / current_data.size();
    
    float expected_min = *std::min_element(expected_data.begin(), expected_data.end());
    float expected_max = *std::max_element(expected_data.begin(), expected_data.end());
    float expected_sum = std::accumulate(expected_data.begin(), expected_data.end(), 0.0f);
    float expected_mean = expected_sum / expected_data.size();
    
    std::cout << "[INFO] " << data_name << " statistics comparison:" << std::endl;
    std::cout << "[INFO] Current:  min=" << current_min << ", max=" << current_max << ", mean=" << current_mean << std::endl;
    std::cout << "[INFO] Expected: min=" << expected_min << ", max=" << expected_max << ", mean=" << expected_mean << std::endl;
    
    // 差异分析
    float mean_diff = current_mean - expected_mean;
    float range_diff = (current_max - current_min) - (expected_max - expected_min);
    
    std::cout << "[INFO] Difference analysis:" << std::endl;
    std::cout << "[INFO] - Mean difference: " << mean_diff << std::endl;
    std::cout << "[INFO] - Range difference: " << range_diff << std::endl;
    
    // 检查是否可能是归一化差异
    if (std::abs(mean_diff) > 0.1f) {
        std::cout << "[WARNING] Large mean difference detected - possible normalization issue" << std::endl;
    }
    
    if (std::abs(range_diff) > 0.1f) {
        std::cout << "[WARNING] Large range difference detected - possible scaling issue" << std::endl;
    }
}

/**
 * 检查数据是否可能是不同的归一化方式
 * @param current_data 当前数据
 * @param expected_data 期望数据
 * @return 归一化差异的可能性
 */
inline std::string check_normalization_difference(
    const std::vector<float>& current_data,
    const std::vector<float>& expected_data)
{
    if (current_data.size() != expected_data.size()) {
        return "Size mismatch";
    }
    
    float current_mean = std::accumulate(current_data.begin(), current_data.end(), 0.0f) / current_data.size();
    float expected_mean = std::accumulate(expected_data.begin(), expected_data.end(), 0.0f) / expected_data.size();
    
    float current_std = 0.0f;
    float expected_std = 0.0f;
    
    for (float val : current_data) {
        current_std += (val - current_mean) * (val - current_mean);
    }
    current_std = std::sqrt(current_std / current_data.size());
    
    for (float val : expected_data) {
        expected_std += (val - expected_mean) * (val - expected_mean);
    }
    expected_std = std::sqrt(expected_std / expected_data.size());
    
    // 检查是否可能是ImageNet归一化 vs 其他归一化
    if (std::abs(current_mean) < 0.1f && std::abs(expected_mean) > 0.1f) {
        return "Current data appears to be ImageNet normalized (mean~0), expected data is not";
    }
    
    if (std::abs(expected_mean) < 0.1f && std::abs(current_mean) > 0.1f) {
        return "Expected data appears to be ImageNet normalized (mean~0), current data is not";
    }
    
    // 检查标准差差异
    if (std::abs(current_std - expected_std) > 0.1f) {
        return "Different standard deviations detected - possible different normalization";
    }
    
    return "No obvious normalization difference detected";
}

#endif // __FUNCTIONHUB_H__