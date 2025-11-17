#ifndef __FUNCTIONHUB_HPP__
#define __FUNCTIONHUB_HPP__

#include <fstream>
#include <vector>
#include <string>
#include <typeinfo>
#include <cuda_runtime.h>
#include <filesystem>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "../../../Include/Common/Utils/CudaWrapper.h"
#include "log.h"

namespace sparse4d{
namespace common{

/**
 * @brief 将CudaWrapper中的数据保存到文件
 * @param data 要保存的GPU数据
 * @param file_path 保存文件路径
 * @return 保存是否成功
 */
template<typename T>
bool saveDataToFile(const CudaWrapper<T>& data, const std::string& file_path) {
    if (!data.isValid()) {
        LOG(ERROR) << "[ERROR] Invalid data wrapper for saving to file: " << file_path;
        return false;
    }
    
    if (data.getSize() == 0) {
        LOG(ERROR) << "[ERROR] Data wrapper is empty, nothing to save: " << file_path;
        return false;
    }
    
    try {
        // 自动创建父目录
        std::filesystem::path file_path_obj(file_path);
        std::filesystem::create_directories(file_path_obj.parent_path());
        
        // 创建输出文件流
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            LOG(ERROR) << "[ERROR] Failed to open file for writing: " << file_path;
            return false;
        }
        
        // 从GPU复制数据到CPU
        std::vector<T> cpu_data = data.cudaMemcpyD2HResWrap();
        
        // 检查数据大小
        if (cpu_data.size() != data.getSize()) {
            LOG(ERROR) << "[ERROR] Data size mismatch after GPU to CPU copy. "
                       << "Expected: " << data.getSize() << ", Got: " << cpu_data.size();
            file.close();
            return false;
        }
        
        // 写入数据到文件
        file.write(reinterpret_cast<const char*>(cpu_data.data()), 
                   cpu_data.size() * sizeof(T));
        
        if (file.fail()) {
            LOG(ERROR) << "[ERROR] Failed to write data to file: " << file_path;
            file.close();
            return false;
        }
        
        file.close();
        
        LOG(INFO) << "[INFO] Successfully saved " << cpu_data.size() 
                  << " elements of type " << typeid(T).name() 
                  << " to file: " << file_path;
        LOG(INFO) << "[INFO] File size: " << (cpu_data.size() * sizeof(T)) << " bytes";
        
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] Exception during file saving: " << e.what();
        return false;
    }
}
 
/**
 * @brief 将CudaWrapper中的部分数据保存到文件（快速版本，不进行详细错误检查）
 * @param gpu 要保存的GPU数据
 * @param effective_elems 有效元素数量
 * @param path 保存文件路径
 * @return 保存是否成功
 */
template<typename T>
bool savePartialFast(const CudaWrapper<T>& gpu, size_t effective_elems, const std::string& path) {
    if (!gpu.isValid() || effective_elems == 0 || effective_elems > gpu.getSize()) {
        LOG(ERROR) << "[ERROR] Invalid parameters for savePartialFast";
        return false;
    }
    
    try {
        // 自动创建父目录
        std::filesystem::path file_path_obj(path);
        std::filesystem::create_directories(file_path_obj.parent_path());
        
        std::vector<T> host(effective_elems);
        cudaError_t err = cudaMemcpy(host.data(), gpu.getCudaPtr(),
                                     effective_elems * sizeof(T),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] CUDA memcpy failed: " << cudaGetErrorString(err);
            return false;
        }
        
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) {
            LOG(ERROR) << "[ERROR] Failed to open file: " << path;
            return false;
        }
        
        f.write(reinterpret_cast<const char*>(host.data()), effective_elems * sizeof(T));
        bool success = f.good();
        f.close();
        
        if (success) {
            LOG(INFO) << "[INFO] Successfully saved " << effective_elems 
                      << " elements to file: " << path;
        }
        
        return success;
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] Exception in savePartialFast: " << e.what();
        return false;
    }
}

/**
 * @brief 从文件加载数据到CudaWrapper
 * @param data 目标GPU数据包装器
 * @param file_path 文件路径
 * @param expected_size 期望的数据大小（如果为0，则从文件大小推断）
 * @return 加载是否成功
 */
template<typename T>
bool loadDataFromFile(CudaWrapper<T>& data, const std::string& file_path, size_t expected_size = 0) {
    try {
        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG(ERROR) << "[ERROR] Failed to open file for reading: " << file_path;
            return false;
        }
        
        // 获取文件大小
        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // 确定期望的大小
        size_t target_size = expected_size;
        if (target_size == 0) {
            target_size = file_size / sizeof(T);
            if (file_size % sizeof(T) != 0) {
                LOG(ERROR) << "[ERROR] File size is not a multiple of element size: " 
                           << file_path << " (size: " << file_size << ", element size: " << sizeof(T) << ")";
                file.close();
                return false;
            }
        }
        
        // 检查文件大小是否匹配
        size_t expected_file_size = target_size * sizeof(T);
        if (static_cast<size_t>(file_size) != expected_file_size) {
            LOG(ERROR) << "[ERROR] File size mismatch. Expected: " << expected_file_size 
                       << " bytes, Got: " << file_size << " bytes";
            file.close();
            return false;
        }
        
        // 读取数据到CPU
        std::vector<T> cpu_data(target_size);
        file.read(reinterpret_cast<char*>(cpu_data.data()), expected_file_size);
        
        if (file.fail()) {
            LOG(ERROR) << "[ERROR] Failed to read data from file: " << file_path;
            file.close();
            return false;
        }
        
        file.close();
        
        // 更新GPU数据
        data.cudaMemUpdateWrap(cpu_data);
        
        LOG(INFO) << "[INFO] Successfully loaded " << target_size 
                  << " elements from file: " << file_path;
        
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] Exception during file loading: " << e.what();
        return false;
    }
}

/**
 * @brief 从文件加载数据到std::vector（CPU版本）
 * @param file_path 文件路径
 * @param expected_size 期望的数据大小（如果为0，则从文件大小推断）
 * @return 加载的数据向量，失败返回空向量
 */
template<typename T>
std::vector<T> loadDataFromFileToVector(const std::string& file_path, size_t expected_size = 0) {
    std::vector<T> result;
    
    try {
        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG(ERROR) << "[ERROR] Failed to open file for reading: " << file_path;
            return result;
        }
        
        // 获取文件大小
        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // 确定期望的大小
        size_t target_size = expected_size;
        if (target_size == 0) {
            target_size = file_size / sizeof(T);
            if (file_size % sizeof(T) != 0) {
                LOG(ERROR) << "[ERROR] File size is not a multiple of element size: " 
                           << file_path << " (size: " << file_size << ", element size: " << sizeof(T) << ")";
                file.close();
                return result;
            }
        }
        
        // 检查文件大小是否匹配
        size_t expected_file_size = target_size * sizeof(T);
        if (static_cast<size_t>(file_size) != expected_file_size) {
            LOG(ERROR) << "[ERROR] File size mismatch. Expected: " << expected_file_size 
                       << " bytes, Got: " << file_size << " bytes";
            file.close();
            return result;
        }
        
        // 读取数据
        result.resize(target_size);
        file.read(reinterpret_cast<char*>(result.data()), expected_file_size);
        
        if (file.fail()) {
            LOG(ERROR) << "[ERROR] Failed to read data from file: " << file_path;
            result.clear();
        } else {
            LOG(INFO) << "[INFO] Successfully loaded " << target_size 
                      << " elements from file: " << file_path;
        }
        
        file.close();
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] Exception during file loading: " << e.what();
        result.clear();
    }
    
    return result;
}

/**
 * @brief 验证两个CudaWrapper的数据是否相等（用于调试）
 * @param data1 第一个GPU数据
 * @param data2 第二个GPU数据
 * @param tolerance 允许的误差范围（对于浮点类型）
 * @return 是否相等
 */
template<typename T>
bool validateDataEquality(const CudaWrapper<T>& data1, const CudaWrapper<T>& data2, 
                          double tolerance = 1e-6) {
    if (data1.getSize() != data2.getSize()) {
        LOG(WARNING) << "[WARNING] Data size mismatch: " << data1.getSize() 
                     << " vs " << data2.getSize();
        return false;
    }
    
    std::vector<T> cpu_data1 = data1.cudaMemcpyD2HResWrap();
    std::vector<T> cpu_data2 = data2.cudaMemcpyD2HResWrap();
    
    if (cpu_data1.size() != cpu_data2.size()) {
        return false;
    }
    
    for (size_t i = 0; i < cpu_data1.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::abs(static_cast<double>(cpu_data1[i]) - static_cast<double>(cpu_data2[i])) > tolerance) {
                LOG(WARNING) << "[WARNING] Data mismatch at index " << i << ": " 
                             << cpu_data1[i] << " vs " << cpu_data2[i];
                return false;
            }
        } else {
            if (cpu_data1[i] != cpu_data2[i]) {
                LOG(WARNING) << "[WARNING] Data mismatch at index " << i << ": " 
                             << cpu_data1[i] << " vs " << cpu_data2[i];
                return false;
            }
        }
    }
    
    return true;
}

/**
 * @brief 打印CudaWrapper的统计信息（用于调试）
 * @param data GPU数据
 * @param name 数据名称（用于日志）
 * @param max_print_elems 最多打印的元素数量（0表示全部）
 */
template<typename T>
void printDataInfo(const CudaWrapper<T>& data, const std::string& name = "Data", 
                   size_t max_print_elems = 10) {
    if (!data.isValid()) {
        LOG(INFO) << "[INFO] " << name << ": Invalid or empty";
        return;
    }
    
    std::vector<T> cpu_data = data.cudaMemcpyD2HResWrap();
    
    LOG(INFO) << "[INFO] " << name << " Statistics:";
    LOG(INFO) << "  - Size: " << data.getSize() << " elements";
    LOG(INFO) << "  - Total bytes: " << (data.getSize() * sizeof(T)) << " bytes";
    LOG(INFO) << "  - Type: " << typeid(T).name();
    
    if (cpu_data.empty()) {
        return;
    }
    
    // 计算统计信息
    if constexpr (std::is_arithmetic_v<T>) {
        T min_val = cpu_data[0];
        T max_val = cpu_data[0];
        double sum = 0.0;
        
        for (const auto& val : cpu_data) {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum += static_cast<double>(val);
        }
        
        double mean = sum / cpu_data.size();
        LOG(INFO) << "  - Min: " << min_val;
        LOG(INFO) << "  - Max: " << max_val;
        LOG(INFO) << "  - Mean: " << mean;
    }
    
    // 打印前几个元素
    size_t print_count = (max_print_elems == 0) ? cpu_data.size() : 
                         std::min(max_print_elems, cpu_data.size());
    LOG(INFO) << "  - First " << print_count << " elements:";
    for (size_t i = 0; i < print_count; ++i) {
        LOG(INFO) << "    [" << i << "] = " << cpu_data[i];
    }
}

} // namespace common
} // namespace sparse4d

#endif // __FUNCTIONHUB_HPP__