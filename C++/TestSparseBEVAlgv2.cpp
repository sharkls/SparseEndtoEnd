#pragma once

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <regex>
#include "log.h"
#include <chrono>

#include "./Include/Interface/ExportSparseBEVAlgLib.h"

#include "CAlgResult.h"
#include "CTimeMatchSrcData.h"

// 点云数据结构
struct Point3D {
    float x, y, z;
    float intensity;
    
    Point3D() : x(0), y(0), z(0), intensity(0) {}
    Point3D(float x_, float y_, float z_, float intensity_ = 0) 
        : x(x_), y(y_), z(z_), intensity(intensity_) {}
};

// 3D边界框结构（用于标签和预测结果）
struct BBox3D {
    float x, y, z;        // 中心点
    float l, w, h;        // 长度、宽度、高度
    float yaw;            // 偏航角
    std::string label;    // 类别标签
    float confidence;     // 置信度
    int track_id;         // 跟踪ID
    bool is_gt;           // 是否为真值标签
    
    BBox3D() : x(0), y(0), z(0), l(0), w(0), h(0), yaw(0), confidence(0), track_id(-1), is_gt(false) {}
};

namespace sparse_bev_v2 {

// 全局变量
std::string g_asset_path;
std::string g_save_dir;
int g_current_frame_index = 0;

// 文件名解析结构
struct BinFileInfo {
    int sample_idx;
    std::string data_name;
    std::vector<int> shape;
    std::string dtype;
    std::string filepath;
    
    BinFileInfo() : sample_idx(-1) {}
};

// 函数声明
CTimeMatchSrcData loadAssetData(std::string asset_path, int index);
void testSparseBEVAlg(const CAlgResult& alg_result, void* p_handle);
std::vector<Point3D> loadPointCloud(const std::string& file_path);
std::vector<BBox3D> loadGroundTruthLabels(const std::string& file_path);
std::vector<BBox3D> convertDetectionsToBBox3D(const std::vector<CObjectResult>& detections);
void visualizeBEV(const std::vector<Point3D>& points, 
                  const std::vector<BBox3D>& gt_boxes,
                  const std::vector<BBox3D>& pred_boxes,
                  const std::string& save_path);
BinFileInfo parseBinFilename(const std::string& filename);
std::vector<float> loadBinFile(const std::string& filepath, const std::vector<int>& expected_shape, const std::string& dtype);
std::vector<int32_t> loadBinFileInt32(const std::string& filepath, const std::vector<int>& expected_shape);

/**
 * @brief 解析二进制文件名
 * @param filename 文件名
 * @return 解析结果
 */
BinFileInfo parseBinFilename(const std::string& filename) {
    BinFileInfo info;
    
    // 示例: sample_0_imgs_1*6*3*256*704_float32.bin
    std::regex pattern(R"(sample_(\d+)_(.+?)_(\d+(?:\*\d+)*)_(.+)\.bin)");
    std::smatch match;
    
    if (std::regex_match(filename, match, pattern)) {
        info.sample_idx = std::stoi(match[1]);
        info.data_name = match[2];
        std::string shape_str = match[3];
        info.dtype = match[4];
        
        // 解析形状
        std::istringstream iss(shape_str);
        std::string dim;
        while (std::getline(iss, dim, '*')) {
            info.shape.push_back(std::stoi(dim));
        }
    }
    
    return info;
}

/**
 * @brief 加载float32二进制文件
 * @param filepath 文件路径
 * @param expected_shape 期望的形状
 * @param dtype 数据类型
 * @return 数据向量
 */
std::vector<float> loadBinFile(const std::string& filepath, const std::vector<int>& expected_shape, const std::string& dtype) {
    std::vector<float> data;
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file: " << filepath;
        return data;
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 计算元素大小
    size_t element_size = 4; // float32
    if (dtype == "float64") element_size = 8;
    else if (dtype == "int32") element_size = 4;
    else if (dtype == "uint8") element_size = 1;
    
    // 计算期望的元素数量
    size_t expected_elements = 1;
    for (int dim : expected_shape) {
        expected_elements *= dim;
    }
    
    // 验证文件大小
    size_t actual_elements = file_size / element_size;
    if (actual_elements != expected_elements) {
        LOG(ERROR) << "File size mismatch for " << filepath 
                   << ": expected " << expected_elements << " elements, got " << actual_elements;
        return data;
    }
    
    // 读取数据
    if (dtype == "float32") {
        data.resize(actual_elements);
        file.read(reinterpret_cast<char*>(data.data()), file_size);
    } else if (dtype == "uint8") {
        std::vector<uint8_t> uint8_data(actual_elements);
        file.read(reinterpret_cast<char*>(uint8_data.data()), file_size);
        data.assign(uint8_data.begin(), uint8_data.end());
    } else {
        LOG(ERROR) << "Unsupported dtype: " << dtype;
        return data;
    }
    
    LOG(INFO) << "Loaded " << data.size() << " elements from " << filepath;
    return data;
}

/**
 * @brief 加载int32二进制文件
 * @param filepath 文件路径
 * @param expected_shape 期望的形状
 * @return 数据向量
 */
std::vector<int32_t> loadBinFileInt32(const std::string& filepath, const std::vector<int>& expected_shape) {
    std::vector<int32_t> data;
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open file: " << filepath;
        return data;
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 计算期望的元素数量
    size_t expected_elements = 1;
    for (int dim : expected_shape) {
        expected_elements *= dim;
    }
    
    // 验证文件大小
    size_t actual_elements = file_size / 4; // int32 = 4 bytes
    if (actual_elements != expected_elements) {
        LOG(ERROR) << "File size mismatch for " << filepath 
                   << ": expected " << expected_elements << " elements, got " << actual_elements;
        return data;
    }
    
    // 读取数据
    data.resize(actual_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    
    LOG(INFO) << "Loaded " << data.size() << " int32 elements from " << filepath;
    return data;
}

/**
 * @brief 加载asset数据
 * @param asset_path asset目录路径
 * @param index 样本索引
 * @return CTimeMatchSrcData
 */
CTimeMatchSrcData loadAssetData(std::string asset_path, int index) 
{   
    CTimeMatchSrcData data;
    
    LOG(INFO) << "Loading asset data for sample " << index << " from " << asset_path;
    
    // 1. 加载图像数据 (ori_imgs)
    std::string ori_imgs_path = asset_path + "sample_" + std::to_string(index) + "_ori_imgs_1*6*3*900*1600_uint8.bin";
    std::vector<float> ori_imgs = loadBinFile(ori_imgs_path, {1, 6, 3, 900, 1600}, "uint8");
    
    if (ori_imgs.empty()) {
        LOG(ERROR) << "Failed to load original images";
        return data;
    }
    
    // 2. 创建视频源数据
    std::vector<CVideoSrcData> video_data;
    for (int cam_idx = 0; cam_idx < 6; ++cam_idx) {
        CVideoSrcData video_src_data;
        video_src_data.ucCameraId(cam_idx);
        video_src_data.usBmpWidth(1600);
        video_src_data.usBmpLength(900);
        video_src_data.unBmpBytes(900 * 1600 * 3);
        
        // 提取当前相机的图像数据
        std::vector<uint8_t> img_vec;
        size_t offset = cam_idx * 900 * 1600 * 3;
        for (size_t i = 0; i < 900 * 1600 * 3; ++i) {
            img_vec.push_back(static_cast<uint8_t>(ori_imgs[offset + i]));
        }
        video_src_data.vecImageBuf(img_vec);
        
        video_data.push_back(video_src_data);
        LOG(INFO) << "Loaded image for camera " << cam_idx << " (size: 1600x900x3)";
    }
    
    data.vecVideoSrcData(video_data);
    
    // 3. 设置时间戳信息
    data.lTimeStamp(static_cast<unsigned long long>(index * 100));  // 转换为毫秒时间戳
    data.unFrameId(index);
    
    // 4. 加载lidar2img变换矩阵
    std::string lidar2img_path = asset_path + "sample_" + std::to_string(index) + "_lidar2img_1*6*4*4_float32.bin";
    std::vector<float> lidar2img_matrices = loadBinFile(lidar2img_path, {1, 6, 4, 4}, "float32");
    
    // 5. 加载预处理后的图像数据，用于子模块验证
    std::string preprocessed_imgs_path = asset_path + "sample_" + std::to_string(index) + "_imgs_1*6*3*256*704_float32.bin";
    std::vector<float> preprocessed_imgs = loadBinFile(preprocessed_imgs_path, {1, 6, 3, 256, 704}, "float32");
    
    if (!lidar2img_matrices.empty()) {
        // 重新排列数据：从 [1, 6, 4, 4] 到 [6, 16]
        std::vector<float> reshaped_matrices;
        for (int cam = 0; cam < 6; ++cam) {
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    size_t idx = cam * 4 * 4 + row * 4 + col;
                    reshaped_matrices.push_back(lidar2img_matrices[idx]);
                }
            }
        }
        
        CCalibrationData calib_data;
        calib_data.lidar2img_matrices(reshaped_matrices);
        calib_data.num_cameras(6);
        calib_data.matrix_size(16);
        data.calibration_data(calib_data);
        
        LOG(INFO) << "Loaded lidar2img matrices: " << reshaped_matrices.size() << " floats";
    } else {
        LOG(WARNING) << "Failed to load lidar2img matrices, using default values";
        // 使用默认的单位矩阵
        CCalibrationData calib_data;
        std::vector<float> default_matrices(6 * 16, 0.0f);
        for (int i = 0; i < 6; ++i) {
            default_matrices[i * 16 + 0] = 1.0f;  // [0,0]
            default_matrices[i * 16 + 5] = 1.0f;  // [1,1]
            default_matrices[i * 16 + 10] = 1.0f; // [2,2]
            default_matrices[i * 16 + 15] = 1.0f; // [3,3]
        }
        calib_data.lidar2img_matrices(default_matrices);
        calib_data.num_cameras(6);
        calib_data.matrix_size(16);
        data.calibration_data(calib_data);
    }
    
    // 6. 设置测试数据，包含预处理后的图像数据
    CTestSrcData test_data;
    if (!preprocessed_imgs.empty()) {
        test_data.preprocessed_imgs(preprocessed_imgs);
        test_data.img_channels(3);
        test_data.img_height(256);
        test_data.img_width(704);
        test_data.test_data_path(asset_path);
        test_data.test_data_type(0); // 0表示asset数据
        LOG(INFO) << "Loaded preprocessed images to test data: " << preprocessed_imgs.size() << " floats (6*3*256*704)";
    } else {
        LOG(WARNING) << "Failed to load preprocessed images, using empty test data";
        test_data.preprocessed_imgs(std::vector<float>());
        test_data.img_channels(0);
        test_data.img_height(0);
        test_data.img_width(0);
        test_data.test_data_path("");
        test_data.test_data_type(0);
    }
    data.test_data(test_data);
    
    // 7. 设置ego pose信息（使用默认值）
    CEgoPoseInfo ego_pose_info;
    std::vector<double> ego_translation = {index * 10.0, 0.0, 0.0};
    std::vector<double> ego_rotation = {1.0, 0.0, 0.0, 0.0};
    ego_pose_info.ego2global_translation(ego_translation);
    ego_pose_info.ego2global_rotation(ego_rotation);
    ego_pose_info.ego_pose_token("ego_pose_" + std::to_string(index));
    data.ego_pose_info(ego_pose_info);
    
    // 8. 设置lidar2ego信息（使用默认值）
    CLidar2EgoInfo lidar2ego_info;
    std::vector<double> lidar2ego_translation = {0.0, 0.0, 1.8};
    std::vector<double> lidar2ego_rotation = {1.0, 0.0, 0.0, 0.0};
    lidar2ego_info.lidar2ego_translation(lidar2ego_translation);
    lidar2ego_info.lidar2ego_rotation(lidar2ego_rotation);
    lidar2ego_info.calibrated_sensor_token("calib_" + std::to_string(index));
    data.lidar2ego_info(lidar2ego_info);
    
    // 8. 设置变换矩阵信息（使用默认值）
    CTransformInfo transform_info;
    std::vector<double> lidar2ego_matrix(16, 0.0);
    std::vector<double> ego2global_matrix(16, 0.0);
    std::vector<double> lidar2global_matrix(16, 0.0);
    std::vector<double> global2lidar_matrix(16, 0.0);
    
    for (int i = 0; i < 4; ++i) {
        lidar2ego_matrix[i * 4 + i] = 1.0;
        ego2global_matrix[i * 4 + i] = 1.0;
        lidar2global_matrix[i * 4 + i] = 1.0;
        global2lidar_matrix[i * 4 + i] = 1.0;
    }
    
    lidar2ego_matrix[3] = lidar2ego_translation[0];
    lidar2ego_matrix[7] = lidar2ego_translation[1];
    lidar2ego_matrix[11] = lidar2ego_translation[2];
    
    ego2global_matrix[3] = ego_translation[0];
    ego2global_matrix[7] = ego_translation[1];
    ego2global_matrix[11] = ego_translation[2];
    
    transform_info.lidar2ego_matrix(lidar2ego_matrix);
    transform_info.ego2global_matrix(ego2global_matrix);
    transform_info.lidar2global_matrix(lidar2global_matrix);
    transform_info.global2lidar_matrix(global2lidar_matrix);
    data.transform_info(transform_info);
    
    // 8. 设置空的激光雷达数据（asset中没有点云数据）
    CLidarData lidar_data;
    lidar_data.num_points(0);
    data.lidar_data(lidar_data);
    
    LOG(INFO) << "Successfully loaded asset data for sample " << index 
              << " with " << video_data.size() << " cameras";
    
    return data;
}

void testSparseBEVAlg(const CAlgResult& alg_result, void* p_handle)
{
    // 获取检测结果
    const auto& detections = alg_result.vecFrameResult().at(0).vecObjectResult();
    
    LOG(INFO) << "检测到的目标数量: " << detections.size();
    
    // 打印详细的检测结果
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        // LOG(INFO) << "目标 " << i << ": " << det.strClass() 
        //           << " 置信度: " << std::fixed << std::setprecision(3) << det.confidence()
        //           << " 位置: (" << det.x() << ", " << det.y() << ", " << det.z() << ")"
        //           << " 尺寸: (" << det.w() << ", " << det.l() << ", " << det.h() << ")"
        //           << " 偏航角: " << det.yaw()
        //           << " 跟踪ID: " << det.trackid();
    }
    
    // 保存结果到文件
    std::string result_file = g_save_dir + "sample_" + std::to_string(g_current_frame_index) + "_results.txt";
    std::ofstream outfile(result_file);
    if (outfile.is_open()) {
        outfile << "Sample " << g_current_frame_index << " Detection Results:" << std::endl;
        outfile << "Total detections: " << detections.size() << std::endl;
        outfile << std::endl;
        
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            outfile << "Detection " << i << ":" << std::endl;
            outfile << "  Class: " << det.strClass() << std::endl;
            outfile << "  Confidence: " << std::fixed << std::setprecision(3) << det.confidence() << std::endl;
            outfile << "  Position: (" << det.x() << ", " << det.y() << ", " << det.z() << ")" << std::endl;
            outfile << "  Size: (" << det.w() << ", " << det.l() << ", " << det.h() << ")" << std::endl;
            outfile << "  Yaw: " << det.yaw() << std::endl;
            outfile << "  Track ID: " << det.trackid() << std::endl;
            outfile << std::endl;
        }
        outfile.close();
        LOG(INFO) << "Results saved to: " << result_file;
    }
    
    LOG(INFO) << "Asset data inference completed for sample " << g_current_frame_index;
}

// 其他函数保持与原始版本相同
std::vector<Point3D> loadPointCloud(const std::string& file_path) {
    std::vector<Point3D> points;
    LOG(INFO) << "Point cloud loading not implemented for asset data";
    return points;
}

std::vector<BBox3D> loadGroundTruthLabels(const std::string& file_path) {
    std::vector<BBox3D> gt_boxes;
    LOG(INFO) << "Ground truth loading not implemented for asset data";
    return gt_boxes;
}

std::vector<BBox3D> convertDetectionsToBBox3D(const std::vector<CObjectResult>& detections) {
    std::vector<BBox3D> pred_boxes;
    pred_boxes.reserve(detections.size());
    
    for (const auto& det : detections) {
        BBox3D box;
        box.x = det.x();
        box.y = det.y();
        box.z = det.z();
        box.l = det.l();
        box.w = det.w();
        box.h = det.h();
        box.yaw = det.yaw();
        box.label = det.strClass();
        box.confidence = det.confidence();
        box.track_id = det.trackid();
        box.is_gt = false;
        
        pred_boxes.push_back(box);
    }
    
    return pred_boxes;
}

void visualizeBEV(const std::vector<Point3D>& points, 
                  const std::vector<BBox3D>& gt_boxes,
                  const std::vector<BBox3D>& pred_boxes,
                  const std::string& save_path) {
    LOG(INFO) << "BEV visualization not implemented for asset data";
}

/**
 * @brief 保存时间间隔数据为bin文件
 */
void saveTimeIntervalToBin(float time_interval, const std::string& save_dir) {
    try {
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_0_time_interval_1_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(&time_interval), sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 时间间隔数据已保存到: " << filename;
            
            // 保存信息文件
            std::string info_filename = filename + ".info";
            std::ofstream info_file(info_filename);
            if (info_file.is_open()) {
                info_file << "时间间隔数据信息:" << std::endl;
                info_file << "  文件名: " << filename << std::endl;
                info_file << "  形状: 1" << std::endl;
                info_file << "  数据类型: float32" << std::endl;
                info_file << "  时间间隔值: " << time_interval << std::endl;
                info_file << "  文件大小: " << sizeof(float) << " 字节" << std::endl;
                info_file.close();
                LOG(INFO) << "[INFO] 时间间隔信息已保存到: " << info_filename;
            }
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存时间间隔数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存lidar2img变换矩阵为bin文件
 */
void saveLidar2ImgToBin(const std::vector<float>& lidar2img_data, const std::string& save_dir) {
    try {
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_0_lidar2img_6*4*4_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(lidar2img_data.data()), 
                         lidar2img_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] Lidar2img变换矩阵已保存到: " << filename;
            
            // 保存信息文件
            std::string info_filename = filename + ".info";
            std::ofstream info_file(info_filename);
            if (info_file.is_open()) {
                info_file << "Lidar2img变换矩阵信息:" << std::endl;
                info_file << "  文件名: " << filename << std::endl;
                info_file << "  形状: 6*4*4" << std::endl;
                info_file << "  数据类型: float32" << std::endl;
                info_file << "  元素数量: " << lidar2img_data.size() << std::endl;
                info_file << "  文件大小: " << (lidar2img_data.size() * sizeof(float)) << " 字节" << std::endl;
                
                // 显示前几个矩阵元素
                info_file << "  前10个元素: ";
                for (int i = 0; i < std::min(10, static_cast<int>(lidar2img_data.size())); ++i) {
                    if (i > 0) info_file << ", ";
                    info_file << lidar2img_data[i];
                }
                info_file << std::endl;
                
                info_file.close();
                LOG(INFO) << "[INFO] Lidar2img信息已保存到: " << info_filename;
            }
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存Lidar2img数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存图像宽高信息为bin文件
 */
void saveImageWhToBin(int width, int height, const std::string& save_dir) {
    try {
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_0_image_wh_2_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            float wh_data[2] = {static_cast<float>(width), static_cast<float>(height)};
            outfile.write(reinterpret_cast<const char*>(wh_data), 2 * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 图像宽高信息已保存到: " << filename;
            
            // 保存信息文件
            std::string info_filename = filename + ".info";
            std::ofstream info_file(info_filename);
            if (info_file.is_open()) {
                info_file << "图像宽高信息:" << std::endl;
                info_file << "  文件名: " << filename << std::endl;
                info_file << "  形状: 2" << std::endl;
                info_file << "  数据类型: float32" << std::endl;
                info_file << "  宽度: " << width << std::endl;
                info_file << "  高度: " << height << std::endl;
                info_file << "  文件大小: " << (2 * sizeof(float)) << " 字节" << std::endl;
                info_file.close();
                LOG(INFO) << "[INFO] 图像宽高信息已保存到: " << info_filename;
            }
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存图像宽高信息时发生异常: " << e.what();
    }
}

/**
 * @brief 获取当前时间戳（毫秒）
 */
int64_t getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    return timestamp;
}

/**
 * @brief 格式化时间戳为可读字符串
 */
std::string formatTimestamp(int64_t timestamp) {
    auto time_point = std::chrono::system_clock::from_time_t(timestamp / 1000);
    auto time_t = std::chrono::system_clock::to_time_t(time_point);
    auto ms = timestamp % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms;
    return ss.str();
}

} // namespace sparse_bev_v2

// 导出给main.cpp使用的主函数
void main_sparse_bev_v2() {
    try {
        // 设置路径
        std::string deploy_path = "/share/Code/Sparse4d/C++/Output/";
        sparse_bev_v2::g_save_dir = deploy_path + "vis/";
        std::string asset_path = "/share/Code/Sparse4d/script/tutorial/asset/";
        std::string save_dir = "/share/Code/Sparse4d/C++/Output/val_bin/";

        // 创建保存目录
        if (!std::filesystem::exists(sparse_bev_v2::g_save_dir)) {
            std::filesystem::create_directories(sparse_bev_v2::g_save_dir);
            LOG(INFO) << "创建保存目录: " << sparse_bev_v2::g_save_dir;
        }
        
        // 创建bin文件保存目录
        if (!std::filesystem::exists(save_dir)) {
            std::filesystem::create_directories(save_dir);
            LOG(INFO) << "创建bin文件保存目录: " << save_dir;
        }

        // 记录开始时间戳
        int64_t start_timestamp = sparse_bev_v2::getCurrentTimestamp();
        LOG(INFO) << "[INFO] 开始时间戳: " << start_timestamp << " (" << sparse_bev_v2::formatTimestamp(start_timestamp) << ")";

        // 算法接口调用流程
        ISparseBEVAlg* l_pObj = CreateSparseBEVAlgObj(deploy_path);
        
        // 初始化算法接口对象
        l_pObj->initAlgorithm(deploy_path, sparse_bev_v2::testSparseBEVAlg, nullptr);

        // 处理多个样本的数据
        int num_samples = 1;  // asset中有3个样本 (0, 1, 2)
        for (int i = 0; i < num_samples; i++)
        {
            LOG(INFO) << "Processing sample " << i;
            
            // 设置当前样本索引
            sparse_bev_v2::g_current_frame_index = i;
            
            // 加载asset数据
            CTimeMatchSrcData multi_modal_data = sparse_bev_v2::loadAssetData(asset_path, i);
            
            // 检查数据是否加载成功
            if (multi_modal_data.vecVideoSrcData().empty()) {
                LOG(WARNING) << "No video data loaded for sample " << i << ", skipping...";
                continue;
            }
            
            // 更新全局变量用于可视化
            sparse_bev_v2::g_asset_path = asset_path;
            
            // 计算时间间隔（这里使用固定的时间间隔，实际应用中应该从数据中获取）
            float time_interval = 1.0f;  // 1秒间隔
            LOG(INFO) << "[INFO] 使用的时间间隔: " << time_interval << " 秒";
            sparse_bev_v2::saveTimeIntervalToBin(time_interval, save_dir);
            
            // 保存图像宽高信息
            int image_width = 704;
            int image_height = 256;
            sparse_bev_v2::saveImageWhToBin(image_width, image_height, save_dir);
            
            // 保存lidar2img变换矩阵（这里使用示例数据，实际应用中应该从标定数据中获取）
            std::vector<float> lidar2img_data(6 * 4 * 4, 0.0f);
            // 设置单位矩阵作为示例
            for (int cam = 0; cam < 6; ++cam) {
                for (int j = 0; j < 4; ++j) {
                    lidar2img_data[cam * 16 + j * 4 + j] = 1.0f;
                }
            }
            sparse_bev_v2::saveLidar2ImgToBin(lidar2img_data, save_dir);
            
            // 运行算法
            l_pObj->runAlgorithm(&multi_modal_data);
            
            LOG(INFO) << "Sample " << i << " processed successfully";
        }
        
        // 记录结束时间戳
        int64_t end_timestamp = sparse_bev_v2::getCurrentTimestamp();
        LOG(INFO) << "[INFO] 结束时间戳: " << end_timestamp << " (" << sparse_bev_v2::formatTimestamp(end_timestamp) << ")";
        
        // 计算实际执行时间
        int64_t execution_time = end_timestamp - start_timestamp;
        LOG(INFO) << "[INFO] 总执行时间: " << execution_time << " 毫秒";
        
        // 时间戳差异分析
        LOG(INFO) << "[INFO] ========== 时间戳差异分析 ==========";
        LOG(INFO) << "[INFO] 开始时间戳: " << start_timestamp << " (" << sparse_bev_v2::formatTimestamp(start_timestamp) << ")";
        LOG(INFO) << "[INFO] 结束时间戳: " << end_timestamp << " (" << sparse_bev_v2::formatTimestamp(end_timestamp) << ")";
        LOG(INFO) << "[INFO] 实际执行时间: " << execution_time << " 毫秒";
        LOG(INFO) << "[INFO] 使用的时间间隔: " << 1.0f << " 秒 (" << (1.0f * 1000) << " 毫秒)";
        LOG(INFO) << "[INFO] 时间间隔与实际执行时间的差异: " << std::abs(1.0f * 1000 - execution_time) << " 毫秒";
        
        // 保存时间戳信息到文件
        std::string timestamp_info_file = save_dir + "timestamp_info.txt";
        std::ofstream timestamp_file(timestamp_info_file);
        if (timestamp_file.is_open()) {
            timestamp_file << "时间戳信息分析:" << std::endl;
            timestamp_file << "  开始时间戳: " << start_timestamp << " (" << sparse_bev_v2::formatTimestamp(start_timestamp) << ")" << std::endl;
            timestamp_file << "  结束时间戳: " << end_timestamp << " (" << sparse_bev_v2::formatTimestamp(end_timestamp) << ")" << std::endl;
            timestamp_file << "  实际执行时间: " << execution_time << " 毫秒" << std::endl;
            timestamp_file << "  使用的时间间隔: " << 1.0f << " 秒 (" << (1.0f * 1000) << " 毫秒)" << std::endl;
            timestamp_file << "  时间间隔与实际执行时间的差异: " << std::abs(1.0f * 1000 - execution_time) << " 毫秒" << std::endl;
            timestamp_file << std::endl;
            timestamp_file << "说明:" << std::endl;
            timestamp_file << "  - 实际执行时间是指从开始到结束的总耗时" << std::endl;
            timestamp_file << "  - 时间间隔是指相邻帧之间的时间差，用于时序推理" << std::endl;
            timestamp_file << "  - 两者概念不同，不应该直接比较" << std::endl;
            timestamp_file << "  - 时间间隔应该从数据的时间戳中获取，而不是从执行时间计算" << std::endl;
            timestamp_file.close();
            LOG(INFO) << "[INFO] 时间戳信息已保存到: " << timestamp_info_file;
        }
        
        LOG(INFO) << "All samples processed successfully";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "错误: " << e.what();
        throw;
    }
} 