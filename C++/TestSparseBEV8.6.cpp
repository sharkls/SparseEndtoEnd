// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <regex>
#include <algorithm>
#include <limits>
#include "log.h"

#include "./Include/Interface/ExportSparseBEVAlgLib.h"
#include "./Src/Common/Core/FunctionHub.h"

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

namespace sparse_bev_8_6 {

// 全局变量
std::string g_asset_path;
std::string g_save_dir;
int g_current_frame_index = 0;
int64_t g_time_stamp = GetTimeStamp();  // 直接初始化为当前系统时间戳（毫秒）

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
void testSparseBEV8_6Alg(const CAlgResult& alg_result, void* p_handle);
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
std::vector<double> loadBinFileFloat64(const std::string& filepath, const std::vector<int>& expected_shape);
std::vector<double> convertFloat64ToFloat32Precision(const std::vector<double>& float64_data);

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
 * @brief 加载float64二进制文件
 * @param filepath 文件路径
 * @param expected_shape 期望的形状
 * @return 数据向量
 */
std::vector<double> loadBinFileFloat64(const std::string& filepath, const std::vector<int>& expected_shape) {
    std::vector<double> data;
    
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
    size_t actual_elements = file_size / 8; // float64 = 8 bytes
    if (actual_elements != expected_elements) {
        LOG(ERROR) << "File size mismatch for " << filepath 
                   << ": expected " << expected_elements << " elements, got " << actual_elements;
        return data;
    }
    
    // 读取数据
    data.resize(actual_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    
    LOG(INFO) << "Loaded " << data.size() << " float64 elements from " << filepath;
    return data;
}

/**
 * @brief 将float64精度转换为float32精度，保持double类型
 * @param float64_data 原始float64数据
 * @return 转换后的float32精度数据（double类型）
 */
std::vector<double> convertFloat64ToFloat32Precision(const std::vector<double>& float64_data) {
    std::vector<double> float32_precision_data;
    float32_precision_data.reserve(float64_data.size());
    
    for (const auto& value : float64_data) {
        // 先转换为float32，再转换回double，实现精度降低
        float float32_value = static_cast<float>(value);
        double converted_value = static_cast<double>(float32_value);
        float32_precision_data.push_back(converted_value);
    }
    
    return float32_precision_data;
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
    std::string time_interval_path = asset_path + "sample_" + std::to_string(index) + "_time_interval_1_float32.bin";
    std::vector<float> time_interval = loadBinFile(time_interval_path, {1}, "float32");
    
    if (!time_interval.empty()) {
        // 将时间间隔转换为毫秒并加到全局时间戳上
        int64_t time_interval_ms = static_cast<int64_t>(time_interval[0] * 1000.0f);  // 秒转毫秒
        g_time_stamp += time_interval_ms;  // 增量更新全局时间戳
        
        // 设置时间戳（毫秒级）
        data.lTimeStamp(g_time_stamp);
        
        LOG(INFO) << "Time interval: " << time_interval[0] << " seconds (" << time_interval_ms << " ms)";
        LOG(INFO) << "Updated global timestamp: " << g_time_stamp << " ms";
        LOG(INFO) << "data.lTimeStamp() after setting: " << data.lTimeStamp();
    } else {
        LOG(WARNING) << "Failed to load time interval, using current global timestamp";
        data.lTimeStamp(g_time_stamp);
        LOG(INFO) << "Using current global timestamp: " << g_time_stamp << " ms";
    }
    
    data.unFrameId(index);
    
    // 4. 加载lidar2img变换矩阵
    std::string lidar2img_path = asset_path + "sample_" + std::to_string(index) + "_lidar2img_1*6*4*4_float32.bin";
    std::vector<float> lidar2img_matrices = loadBinFile(lidar2img_path, {1, 6, 4, 4}, "float32");
    
    // 5. 加载global2lidar变换矩阵
    std::string global2lidar_path = asset_path + "sample_" + std::to_string(index) + "_ibank_global2lidar_4*4_float64.bin";
    std::vector<double> global2lidar_matrix_double = loadBinFileFloat64(global2lidar_path, {4, 4});
    
    // 转换为float32精度，符合IDL定义
    std::vector<double> global2lidar_matrix;
    if (global2lidar_matrix_double.empty()) {
        LOG(WARNING) << "Failed to load global2lidar matrix, using default identity matrix";
        // 使用默认的单位矩阵
        global2lidar_matrix = {1.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 1.0};
    } else {
        LOG(INFO) << "Loaded global2lidar matrix from " << global2lidar_path << " (converting from float64 to float32 precision)";
        // 使用辅助函数将float64转换为float32精度，保持double类型以符合IDL定义
        global2lidar_matrix = convertFloat64ToFloat32Precision(global2lidar_matrix_double);
        
        // 打印转换信息
        LOG(INFO) << "Matrix conversion: float64 -> float32 -> double (for IDL compatibility)";
        LOG(INFO) << "Matrix size: " << global2lidar_matrix.size() << " elements";
    }
    
    // 6. 设置标定数据
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
    
    // 7. 加载预处理后的图像数据，用于子模块验证
    std::string preprocessed_imgs_path = asset_path + "sample_" + std::to_string(index) + "_imgs_1*6*3*256*704_float32.bin";
    std::vector<float> preprocessed_imgs = loadBinFile(preprocessed_imgs_path, {1, 6, 3, 256, 704}, "float32");
    
    // 8. 设置测试数据，包含预处理后的图像数据
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
    
    // 9. 设置变换矩阵信息 - 只设置global2lidar_matrix
    CTransformInfo transform_info;
    // 其他矩阵设置为空或默认值
    std::vector<double> empty_matrix(16, 0.0);
    transform_info.lidar2ego_matrix(empty_matrix);
    transform_info.ego2global_matrix(empty_matrix);
    transform_info.lidar2global_matrix(empty_matrix);
    transform_info.global2lidar_matrix(global2lidar_matrix);
    data.transform_info(transform_info);
    
    // 10. 设置空的激光雷达数据（asset中没有点云数据）
    CLidarData lidar_data;
    lidar_data.num_points(0);
    data.lidar_data(lidar_data);
    
    LOG(INFO) << "Successfully loaded asset data for sample " << index 
              << " with " << video_data.size() << " cameras";
    
    return data;
}

void testSparseBEV8_6Alg(const CAlgResult& alg_result, void* p_handle)
{
    // 获取检测结果
    const auto& detections = alg_result.vecFrameResult().at(0).vecObjectResult();
    
    LOG(INFO) << "检测到的目标数量: " << detections.size();
    
    // 打印详细的检测结果
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        LOG(INFO) << "目标 " << i << ": " << det.strClass() 
                  << " 置信度: " << std::fixed << std::setprecision(3) << det.confidence()
                  << " 位置: (" << det.x() << ", " << det.y() << ", " << det.z() << ")"
                  << " 尺寸: (" << det.w() << ", " << det.l() << ", " << det.h() << ")"
                  << " 偏航角: " << det.yaw()
                  << " 跟踪ID: " << det.trackid();
    }
    
    // 保存结果到文件
    std::string result_file = g_save_dir + "sample_" + std::to_string(g_current_frame_index) + "_results_8_6.txt";
    std::ofstream outfile(result_file);
    if (outfile.is_open()) {
        outfile << "Sample " << g_current_frame_index << " Detection Results (SparseBEV8.6):" << std::endl;
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
    
    // 转换检测结果为BBox3D格式
    std::vector<BBox3D> pred_boxes = convertDetectionsToBBox3D(detections);
    
    // 尝试加载点云数据（如果存在）
    std::vector<Point3D> points;
    std::string pointcloud_file = g_asset_path + "sample_" + std::to_string(g_current_frame_index) + "_points.bin";
    if (std::filesystem::exists(pointcloud_file)) {
        points = loadPointCloud(pointcloud_file);
    } else {
        LOG(INFO) << "No point cloud data found, creating synthetic points for visualization";
        // 创建一些合成点云数据用于可视化
        for (float x = -50.0f; x <= 50.0f; x += 2.0f) {
            for (float y = -50.0f; y <= 50.0f; y += 2.0f) {
                float z = 0.0f;
                float intensity = 0.5f + 0.5f * sin(x * 0.1f) * cos(y * 0.1f);
                points.emplace_back(x, y, z, intensity);
            }
        }
    }
    
    // 尝试加载真值标签（如果存在）
    std::vector<BBox3D> gt_boxes;
    std::string gt_file = g_asset_path + "sample_" + std::to_string(g_current_frame_index) + "_gt.txt";
    if (std::filesystem::exists(gt_file)) {
        gt_boxes = loadGroundTruthLabels(gt_file);
    }
    
    // 生成BEV可视化
    std::string bev_file = g_save_dir + "sample_" + std::to_string(g_current_frame_index) + "_bev.jpg";
    visualizeBEV(points, gt_boxes, pred_boxes, bev_file);
    
    LOG(INFO) << "Asset data inference completed for sample " << g_current_frame_index << " (SparseBEV8.6)";
}

// 其他函数保持与原始版本相同
std::vector<Point3D> loadPointCloud(const std::string& file_path) {
    std::vector<Point3D> points;
    
    // 检查文件是否存在
    if (!std::filesystem::exists(file_path)) {
        LOG(WARNING) << "Point cloud file not found: " << file_path;
        return points;
    }
    
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open point cloud file: " << file_path;
        return points;
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 假设点云数据格式为 N x 4 (x, y, z, intensity)
    size_t num_points = file_size / (4 * sizeof(float));
    points.reserve(num_points);
    
    std::vector<float> buffer(num_points * 4);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);
    
    for (size_t i = 0; i < num_points; ++i) {
        Point3D point;
        point.x = buffer[i * 4 + 0];
        point.y = buffer[i * 4 + 1];
        point.z = buffer[i * 4 + 2];
        point.intensity = buffer[i * 4 + 3];
        points.push_back(point);
    }
    
    LOG(INFO) << "Loaded " << points.size() << " points from " << file_path;
    return points;
}

std::vector<BBox3D> loadGroundTruthLabels(const std::string& file_path) {
    std::vector<BBox3D> gt_boxes;
    
    // 检查文件是否存在
    if (!std::filesystem::exists(file_path)) {
        LOG(WARNING) << "Ground truth file not found: " << file_path;
        return gt_boxes;
    }
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open ground truth file: " << file_path;
        return gt_boxes;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        BBox3D box;
        box.is_gt = true;
        
        // 假设格式: class x y z l w h yaw track_id
        if (iss >> box.label >> box.x >> box.y >> box.z >> box.l >> box.w >> box.h >> box.yaw >> box.track_id) {
            box.confidence = 1.0f; // 真值置信度为1.0
            gt_boxes.push_back(box);
        }
    }
    
    LOG(INFO) << "Loaded " << gt_boxes.size() << " ground truth boxes from " << file_path;
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
    
    // BEV可视化参数
    const int img_width = 800;
    const int img_height = 800;
    const float range_x = 100.0f;  // 前后100米
    const float range_y = 100.0f;  // 左右100米
    const float scale_x = img_width / (2.0f * range_x);
    const float scale_y = img_height / (2.0f * range_y);
    
    // 创建BEV图像
    cv::Mat bev_img = cv::Mat::zeros(img_height, img_width, CV_8UC3);
    
    // 绘制网格
    cv::Scalar grid_color(50, 50, 50);
    for (int i = 0; i <= 20; ++i) {
        int x = i * img_width / 20;
        cv::line(bev_img, cv::Point(x, 0), cv::Point(x, img_height), grid_color, 1);
    }
    for (int i = 0; i <= 20; ++i) {
        int y = i * img_height / 20;
        cv::line(bev_img, cv::Point(0, y), cv::Point(img_width, y), grid_color, 1);
    }
    
    // 绘制坐标轴
    cv::line(bev_img, cv::Point(img_width/2, 0), cv::Point(img_width/2, img_height), cv::Scalar(0, 255, 0), 2);
    cv::line(bev_img, cv::Point(0, img_height/2), cv::Point(img_width, img_height/2), cv::Scalar(0, 255, 0), 2);
    
    // 绘制点云（如果有点云数据）
    if (!points.empty()) {
        for (const auto& point : points) {
            // 过滤超出范围的点
            if (std::abs(point.x) > range_x || std::abs(point.y) > range_y) {
                continue;
            }
            
            // 转换到图像坐标
            int img_x = static_cast<int>((point.x + range_x) * scale_x);
            int img_y = static_cast<int>((point.y + range_y) * scale_y);
            
            // 确保坐标在图像范围内
            if (img_x >= 0 && img_x < img_width && img_y >= 0 && img_y < img_height) {
                // 根据强度值设置颜色
                int intensity = static_cast<int>(std::min(255.0f, point.intensity * 255.0f));
                cv::circle(bev_img, cv::Point(img_x, img_y), 1, cv::Scalar(intensity, intensity, intensity), -1);
            }
        }
    }
    
    // 绘制预测框
    for (const auto& box : pred_boxes) {
        // 过滤超出范围的目标
        if (std::abs(box.x) > range_x || std::abs(box.y) > range_y) {
            continue;
        }
        
        // 转换到图像坐标
        int center_x = static_cast<int>((box.x + range_x) * scale_x);
        int center_y = static_cast<int>((box.y + range_y) * scale_y);
        
        // 计算框的尺寸
        int box_width = static_cast<int>(box.w * scale_x);
        int box_length = static_cast<int>(box.l * scale_y);
        
        // 设置颜色（根据类别）
        cv::Scalar color;
        if (box.label == "car") {
            color = cv::Scalar(0, 0, 255);  // 红色
        } else if (box.label == "truck") {
            color = cv::Scalar(0, 255, 255);  // 黄色
        } else if (box.label == "bus") {
            color = cv::Scalar(255, 0, 255);  // 紫色
        } else if (box.label == "pedestrian") {
            color = cv::Scalar(255, 0, 0);  // 蓝色
        } else if (box.label == "bicycle" || box.label == "motorcycle") {
            color = cv::Scalar(0, 255, 0);  // 绿色
        } else {
            color = cv::Scalar(128, 128, 128);  // 灰色
        }
        
        // 绘制旋转矩形
        cv::Point2f center(center_x, center_y);
        cv::Size2f size(box_width, box_length);
        cv::RotatedRect rect(center, size, -box.yaw * 180.0f / M_PI);  // 转换为度
        
        cv::Point2f vertices[4];
        rect.points(vertices);
        
        // 绘制矩形
        for (int i = 0; i < 4; ++i) {
            cv::line(bev_img, vertices[i], vertices[(i + 1) % 4], color, 2);
        }
        
        // 绘制朝向箭头
        float arrow_length = std::min(box_width, box_length) * 0.3f;
        float arrow_x = center_x + arrow_length * cos(-box.yaw);
        float arrow_y = center_y + arrow_length * sin(-box.yaw);
        cv::arrowedLine(bev_img, cv::Point(center_x, center_y), 
                       cv::Point(static_cast<int>(arrow_x), static_cast<int>(arrow_y)), 
                       color, 2, 8, 0, 0.3);
        
        // 添加标签文本
        std::string label_text = box.label + " " + std::to_string(static_cast<int>(box.confidence * 100)) + "%";
        cv::putText(bev_img, label_text, cv::Point(center_x - 20, center_y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    }
    
    // 绘制真值框（绿色虚线）
    for (const auto& box : gt_boxes) {
        // 过滤超出范围的目标
        if (std::abs(box.x) > range_x || std::abs(box.y) > range_y) {
            continue;
        }
        
        // 转换到图像坐标
        int center_x = static_cast<int>((box.x + range_x) * scale_x);
        int center_y = static_cast<int>((box.y + range_y) * scale_y);
        
        // 计算框的尺寸
        int box_width = static_cast<int>(box.w * scale_x);
        int box_length = static_cast<int>(box.l * scale_y);
        
        // 绘制旋转矩形（虚线）
        cv::Point2f center(center_x, center_y);
        cv::Size2f size(box_width, box_length);
        cv::RotatedRect rect(center, size, -box.yaw * 180.0f / M_PI);
        
        cv::Point2f vertices[4];
        rect.points(vertices);
        
        // 绘制虚线矩形
        for (int i = 0; i < 4; ++i) {
            cv::Point2f start = vertices[i];
            cv::Point2f end = vertices[(i + 1) % 4];
            
            // 绘制虚线
            float length = cv::norm(end - start);
            int num_segments = static_cast<int>(length / 5.0f);
            for (int j = 0; j < num_segments; j += 2) {
                float t1 = static_cast<float>(j) / num_segments;
                float t2 = static_cast<float>(j + 1) / num_segments;
                cv::Point2f p1 = start + t1 * (end - start);
                cv::Point2f p2 = start + t2 * (end - start);
                cv::line(bev_img, p1, p2, cv::Scalar(0, 255, 0), 2);
            }
        }
        
        // 添加GT标签
        std::string label_text = "GT: " + box.label;
        cv::putText(bev_img, label_text, cv::Point(center_x - 20, center_y + 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    }
    
    // 添加图例
    cv::putText(bev_img, "Prediction", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(bev_img, "Ground Truth", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(bev_img, "Range: " + std::to_string(static_cast<int>(range_x)) + "m x " + std::to_string(static_cast<int>(range_y)) + "m", 
               cv::Point(10, img_height - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    // 保存图像
    cv::imwrite(save_path, bev_img);
    LOG(INFO) << "BEV visualization saved to: " << save_path;
}

} // namespace sparse_bev_8_6

// 导出给main.cpp使用的主函数
void main_sparse_bev_8_6() {
    try {
        // 设置路径
        std::string deploy_path = "/share/Code/Sparse4d/C++/Output/";
        sparse_bev_8_6::g_save_dir = deploy_path + "vis/";
        std::string asset_path = "/share/Code/Sparse4d/script/tutorial/asset/";

        // 创建保存目录
        if (!std::filesystem::exists(sparse_bev_8_6::g_save_dir)) {
            std::filesystem::create_directories(sparse_bev_8_6::g_save_dir);
            LOG(INFO) << "创建保存目录: " << sparse_bev_8_6::g_save_dir;
        }

        // 算法接口调用流程 - 使用SparseBEV8.6版本
        ISparseBEVAlg* l_pObj = CreateSparseBEV8_6AlgObj(deploy_path);
        
        // 初始化算法接口对象
        l_pObj->initAlgorithm(deploy_path, sparse_bev_8_6::testSparseBEV8_6Alg, nullptr);

        // 处理多个样本的数据
        int num_samples = 10;  // asset中有3个样本 (0, 1, 2)
        for (int i = 0; i < num_samples; i++)
        {
            LOG(INFO) << "Processing sample " << i << " with SparseBEV8.6";
            
            // 设置当前样本索引
            sparse_bev_8_6::g_current_frame_index = i;
            
            // 设置算法内部模块的当前样本索引
            // 注意：这里需要通过算法接口设置样本索引，具体实现需要根据接口设计
            // 暂时通过全局变量或其他方式传递
            
            // 加载asset数据
            CTimeMatchSrcData multi_modal_data = sparse_bev_8_6::loadAssetData(asset_path, i);
            
            // 检查数据是否加载成功
            if (multi_modal_data.vecVideoSrcData().empty()) {
                LOG(WARNING) << "No video data loaded for sample " << i << ", skipping...";
                continue;
            }
            
            // 更新全局变量用于可视化
            sparse_bev_8_6::g_asset_path = asset_path;
            
            // 运行算法
            l_pObj->runAlgorithm(&multi_modal_data);
            
            LOG(INFO) << "Sample " << i << " processed successfully with SparseBEV8.6";
        }
        
        LOG(INFO) << "All samples processed successfully with SparseBEV8.6";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "错误: " << e.what();
        throw;
    }
} 