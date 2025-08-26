#pragma once

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include "log.h"

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

namespace sparse_bev {

// 全局变量
std::string g_rgb_path;
std::string g_save_dir;
int g_current_frame_index = 0;  // 添加当前帧索引

// 函数声明
CTimeMatchSrcData loadOfflineData(std::string data_path, int index);
void testSparseBEVAlg(const CAlgResult& alg_result, void* p_handle);
std::vector<Point3D> loadPointCloud(const std::string& file_path);
std::vector<BBox3D> loadGroundTruthLabels(const std::string& file_path);
std::vector<BBox3D> convertDetectionsToBBox3D(const std::vector<CObjectResult>& detections);
void visualizeBEV(const std::vector<Point3D>& points, 
                  const std::vector<BBox3D>& gt_boxes,
                  const std::vector<BBox3D>& pred_boxes,
                  const std::string& save_path);

CTimeMatchSrcData loadOfflineData(std::string data_path, int index) 
{   
    CTimeMatchSrcData data;
    
    // nuScenes数据集的多视角相机配置
    std::vector<std::string> camera_types = {
        "CAM_FRONT",      // 前视相机
        "CAM_FRONT_RIGHT", // 前右相机
        "CAM_FRONT_LEFT",  // 前左相机
        "CAM_BACK",       // 后视相机
        "CAM_BACK_LEFT",  // 后左相机
        "CAM_BACK_RIGHT"  // 后右相机
    };
    
    std::vector<CVideoSrcData> video_data;
    
    // 加载每个相机的图像数据
    for (size_t cam_idx = 0; cam_idx < camera_types.size(); ++cam_idx) {
        std::string camera_type = camera_types[cam_idx];
        
        // 构建图像文件路径（假设图像已预处理为模型输入格式）
        std::string img_path = data_path + "images/" + camera_type + "/" + 
                              std::to_string(index) + ".jpg";
        
        // 加载图像
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) {
            LOG(ERROR) << "Failed to load image: " << img_path;
            continue;
        }
        
        // 创建视频源数据
        CVideoSrcData video_src_data;
        video_src_data.ucCameraId(cam_idx);  // 相机ID
        video_src_data.usBmpWidth(img.cols);  // 图像宽度
        video_src_data.usBmpLength(img.rows); // 图像高度
        video_src_data.unBmpBytes(img.total() * img.elemSize()); // 图像字节数
        
        // 转换图像数据为字节数组
        std::vector<uint8_t> img_vec;
        if (img.isContinuous()) {
            img_vec.assign(img.data, img.data + img.total() * img.elemSize());
        } else {
            for (int i = 0; i < img.rows; ++i) {
                img_vec.insert(img_vec.end(), 
                              img.ptr<uint8_t>(i), 
                              img.ptr<uint8_t>(i) + img.cols * img.elemSize());
            }
        }
        video_src_data.vecImageBuf(img_vec);
        
        video_data.push_back(video_src_data);
        LOG(INFO) << "Loaded image for " << camera_type << ": " << img_path 
                  << " (size: " << img.cols << "x" << img.rows << ")";
    }
    
    // 设置视频源数据
    data.vecVideoSrcData(video_data);
    
    // 设置基础时间戳信息（使用继承自CDataBase的lTimeStamp字段）
    data.lTimeStamp(static_cast<unsigned long long>(index * 100));  // 转换为毫秒时间戳
    data.unFrameId(index);  // 设置帧号
    
    // 加载时间戳和变换信息
    std::string temporal_path = data_path + "temporal/temporal_" + std::to_string(index) + ".json";
    std::ifstream temporal_file(temporal_path);
    if (temporal_file.is_open()) {
        try {
            // 读取JSON文件内容
            std::string content((std::istreambuf_iterator<char>(temporal_file)),
                               std::istreambuf_iterator<char>());
            
            // 解析时间戳信息并设置到lTimeStamp
            size_t timestamp_pos = content.find("\"timestamp\":");
            if (timestamp_pos != std::string::npos) {
                size_t timestamp_start = content.find_first_not_of(" \t\r\n", timestamp_pos + 12);
                size_t timestamp_end = content.find_first_of(",}", timestamp_start);
                if (timestamp_end != std::string::npos) {
                    std::string timestamp_str = content.substr(timestamp_start, timestamp_end - timestamp_start);
                    try {
                        double timestamp_seconds = std::stod(timestamp_str);
                        // 转换为毫秒时间戳
                        data.lTimeStamp(static_cast<unsigned long long>(timestamp_seconds * 1000));
                        LOG(INFO) << "Loaded timestamp: " << timestamp_seconds << "s (" << data.lTimeStamp() << "ms)";
                    } catch (const std::exception& e) {
                        LOG(WARNING) << "Failed to parse timestamp: " << timestamp_str << ", using default";
                        data.lTimeStamp(static_cast<unsigned long long>(index * 100));
                    }
                }
            }
            
            // 解析ego pose信息
            CEgoPoseInfo ego_pose_info;
            std::vector<double> ego_translation = {index * 10.0, 0.0, 0.0};
            std::vector<double> ego_rotation = {1.0, 0.0, 0.0, 0.0};
            
            // 尝试从JSON中解析ego pose信息
            size_t ego_trans_pos = content.find("\"ego2global_translation\":");
            if (ego_trans_pos != std::string::npos) {
                size_t trans_start = content.find("[", ego_trans_pos);
                size_t trans_end = content.find("]", trans_start);
                if (trans_start != std::string::npos && trans_end != std::string::npos) {
                    std::string trans_str = content.substr(trans_start + 1, trans_end - trans_start - 1);
                    std::istringstream trans_iss(trans_str);
                    ego_translation.clear();
                    std::string value;
                    while (std::getline(trans_iss, value, ',')) {
                        value.erase(0, value.find_first_not_of(" \t\r\n"));
                        value.erase(value.find_last_not_of(" \t\r\n") + 1);
                        if (!value.empty()) {
                            ego_translation.push_back(std::stod(value));
                        }
                    }
                }
            }
            
            ego_pose_info.ego2global_translation(ego_translation);
            ego_pose_info.ego2global_rotation(ego_rotation);
            ego_pose_info.ego_pose_token("ego_pose_" + std::to_string(index));
            data.ego_pose_info(ego_pose_info);
            
            // 解析lidar2ego信息
            CLidar2EgoInfo lidar2ego_info;
            std::vector<double> lidar2ego_translation = {0.0, 0.0, 1.8};
            std::vector<double> lidar2ego_rotation = {1.0, 0.0, 0.0, 0.0};
            
            // 尝试从JSON中解析lidar2ego信息
            size_t lidar_trans_pos = content.find("\"lidar2ego_translation\":");
            if (lidar_trans_pos != std::string::npos) {
                size_t trans_start = content.find("[", lidar_trans_pos);
                size_t trans_end = content.find("]", trans_start);
                if (trans_start != std::string::npos && trans_end != std::string::npos) {
                    std::string trans_str = content.substr(trans_start + 1, trans_end - trans_start - 1);
                    std::istringstream trans_iss(trans_str);
                    lidar2ego_translation.clear();
                    std::string value;
                    while (std::getline(trans_iss, value, ',')) {
                        value.erase(0, value.find_first_not_of(" \t\r\n"));
                        value.erase(value.find_last_not_of(" \t\r\n") + 1);
                        if (!value.empty()) {
                            lidar2ego_translation.push_back(std::stod(value));
                        }
                    }
                }
            }
            
            lidar2ego_info.lidar2ego_translation(lidar2ego_translation);
            lidar2ego_info.lidar2ego_rotation(lidar2ego_rotation);
            lidar2ego_info.calibrated_sensor_token("calib_" + std::to_string(index));
            data.lidar2ego_info(lidar2ego_info);
            
            // 解析变换矩阵信息
            CTransformInfo transform_info;
            // 构建4x4变换矩阵（16个元素）
            std::vector<double> lidar2ego_matrix(16, 0.0);
            std::vector<double> ego2global_matrix(16, 0.0);
            std::vector<double> lidar2global_matrix(16, 0.0);
            std::vector<double> global2lidar_matrix(16, 0.0);
            
            // 设置单位矩阵
            for (int i = 0; i < 4; ++i) {
                lidar2ego_matrix[i * 4 + i] = 1.0;
                ego2global_matrix[i * 4 + i] = 1.0;
                lidar2global_matrix[i * 4 + i] = 1.0;
                global2lidar_matrix[i * 4 + i] = 1.0;
            }
            
            // 设置平移
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
            
            LOG(INFO) << "Loaded temporal info from: " << temporal_path;
            
        } catch (const std::exception& e) {
            LOG(WARNING) << "Error parsing temporal info: " << e.what();
        }
        temporal_file.close();
    } else {
        LOG(WARNING) << "Failed to open temporal file: " << temporal_path;
        // 使用默认的ego pose和lidar2ego信息
        CEgoPoseInfo ego_pose_info;
        std::vector<double> ego_translation = {index * 10.0, 0.0, 0.0};
        std::vector<double> ego_rotation = {1.0, 0.0, 0.0, 0.0};
        ego_pose_info.ego2global_translation(ego_translation);
        ego_pose_info.ego2global_rotation(ego_rotation);
        ego_pose_info.ego_pose_token("ego_pose_" + std::to_string(index));
        data.ego_pose_info(ego_pose_info);
        
        CLidar2EgoInfo lidar2ego_info;
        std::vector<double> lidar2ego_translation = {0.0, 0.0, 1.8};
        std::vector<double> lidar2ego_rotation = {1.0, 0.0, 0.0, 0.0};
        lidar2ego_info.lidar2ego_translation(lidar2ego_translation);
        lidar2ego_info.lidar2ego_rotation(lidar2ego_rotation);
        lidar2ego_info.calibrated_sensor_token("calib_" + std::to_string(index));
        data.lidar2ego_info(lidar2ego_info);
        
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
    }
    
    // 加载激光雷达点云数据
    std::string lidar_path = data_path + "lidar/lidar_" + std::to_string(index) + ".bin";
    std::ifstream lidar_file(lidar_path, std::ios::binary);
    if (lidar_file.is_open()) {
        // 获取文件大小
        lidar_file.seekg(0, std::ios::end);
        size_t file_size = lidar_file.tellg();
        lidar_file.seekg(0, std::ios::beg);
        
        // 计算点云数量
        size_t num_points = file_size / (5 * sizeof(float));  // 每个点5个float值
        std::vector<float> points_data(num_points * 5);
        
        lidar_file.read(reinterpret_cast<char*>(points_data.data()), file_size);
        lidar_file.close();
        
        CLidarData lidar_data;
        lidar_data.num_points(num_points);
        lidar_data.points(points_data);
        data.lidar_data(lidar_data);
        
        LOG(INFO) << "Loaded lidar data: " << num_points << " points from " << lidar_path;
    } else {
        LOG(WARNING) << "Failed to open lidar file: " << lidar_path;
        // 创建空的点云数据
        CLidarData lidar_data;
        lidar_data.num_points(0);
        data.lidar_data(lidar_data);
    }
    
    // 加载标定参数
    std::string calib_path = data_path + "calib/lidar2img_" + std::to_string(index) + ".bin";
    std::ifstream calib_file(calib_path, std::ios::binary);
    if (calib_file.is_open()) {
        // 获取文件大小
        calib_file.seekg(0, std::ios::end);
        size_t file_size = calib_file.tellg();
        calib_file.seekg(0, std::ios::beg);
        
        // 计算float数量
        size_t num_floats = file_size / sizeof(float);
        std::vector<float> lidar2img_matrices(num_floats);
        
        calib_file.read(reinterpret_cast<char*>(lidar2img_matrices.data()), file_size);
        calib_file.close();
        
        // 应用图像变换矩阵（与SparseBEV.cpp中的computeTransformedLidar2img方法对应）
        // 获取预处理参数（这些参数应该与配置文件中的一致）
        float resize_ratio = 1.0f;  // 缩放比例
        uint32_t crop_height = 0;   // 裁剪高度
        uint32_t crop_width = 0;    // 裁剪宽度
        
        // 创建图像变换矩阵（4x4）
        std::vector<float> transform_matrix(16, 0.0f);
        
        // 设置单位矩阵
        transform_matrix[0] = 1.0f;   // [0,0]
        transform_matrix[5] = 1.0f;   // [1,1]
        transform_matrix[10] = 1.0f;  // [2,2]
        transform_matrix[15] = 1.0f;  // [3,3]
        
        // 1. 缩放变换
        transform_matrix[0] = resize_ratio;  // [0,0] = resize_ratio
        transform_matrix[5] = resize_ratio;  // [1,1] = resize_ratio
        
        // 2. 裁剪变换（平移）
        transform_matrix[3] = -static_cast<float>(crop_width);   // [0,3] = -crop_width
        transform_matrix[7] = -static_cast<float>(crop_height);  // [1,3] = -crop_height
        
        LOG(INFO) << "Image transform matrix:";
        LOG(INFO) << "  [" << transform_matrix[0] << ", " << transform_matrix[1] << ", " << transform_matrix[2] << ", " << transform_matrix[3] << "]";
        LOG(INFO) << "  [" << transform_matrix[4] << ", " << transform_matrix[5] << ", " << transform_matrix[6] << ", " << transform_matrix[7] << "]";
        LOG(INFO) << "  [" << transform_matrix[8] << ", " << transform_matrix[9] << ", " << transform_matrix[10] << ", " << transform_matrix[11] << "]";
        LOG(INFO) << "  [" << transform_matrix[12] << ", " << transform_matrix[13] << ", " << transform_matrix[14] << ", " << transform_matrix[15] << "]";
        
        // 保存原始lidar2img矩阵（用于调试）
        std::vector<float> original_lidar2img = lidar2img_matrices;
        
        // 应用变换到所有相机的lidar2img矩阵
        int num_cams = 6;  // nuScenes有6个相机
        for (int cam_idx = 0; cam_idx < num_cams; ++cam_idx) {
            // 提取当前相机的原始lidar2img矩阵
            std::vector<float> lidar2img_matrix(16);
            for (int i = 0; i < 16; ++i) {
                lidar2img_matrix[i] = original_lidar2img[cam_idx * 16 + i];
            }
            
            // 应用变换：new_lidar2img = transform_matrix @ original_lidar2img
            std::vector<float> transformed_matrix(16, 0.0f);
            
            // 矩阵乘法：4x4 @ 4x4
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        transformed_matrix[i * 4 + j] += transform_matrix[i * 4 + k] * lidar2img_matrix[k * 4 + j];
                    }
                }
            }
            
            // 将变换后的矩阵存储回数组
            for (int i = 0; i < 16; ++i) {
                lidar2img_matrices[cam_idx * 16 + i] = transformed_matrix[i];
            }
            
            LOG(INFO) << "Camera " << cam_idx << " lidar2img matrix updated";
            
            // 打印变换前后的矩阵（调试用）
            if (cam_idx == 0) {  // 只打印第一个相机的矩阵
                LOG(INFO) << "Camera 0 - Original lidar2img matrix:";
                for (int row = 0; row < 4; ++row) {
                    std::stringstream ss;
                    ss << "  Row " << row << ": ";
                    for (int col = 0; col < 4; ++col) {
                        ss << std::fixed << std::setprecision(6) 
                           << original_lidar2img[cam_idx * 16 + row * 4 + col] << " ";
                    }
                    LOG(INFO) << ss.str();
                }
                
                LOG(INFO) << "Camera 0 - Transformed lidar2img matrix:";
                for (int row = 0; row < 4; ++row) {
                    std::stringstream ss;
                    ss << "  Row " << row << ": ";
                    for (int col = 0; col < 4; ++col) {
                        ss << std::fixed << std::setprecision(6) 
                           << lidar2img_matrices[cam_idx * 16 + row * 4 + col] << " ";
                    }
                    LOG(INFO) << ss.str();
                }
            }
        }
        
        LOG(INFO) << "Successfully computed transformed lidar2img matrices for " << num_cams << " cameras";
        
        CCalibrationData calib_data;
        calib_data.lidar2img_matrices(lidar2img_matrices);
        calib_data.num_cameras(6);  // nuScenes有6个相机
        calib_data.matrix_size(16); // 每个矩阵4x4=16个元素
        data.calibration_data(calib_data);
        
        LOG(INFO) << "Loaded and transformed calibration data: " << num_floats << " floats from " << calib_path;
    } else {
        LOG(WARNING) << "Failed to open calibration file: " << calib_path;
        // 创建默认的标定数据
        CCalibrationData calib_data;
        std::vector<float> default_matrices(6 * 4 * 4, 0.0f);
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
    
    // 尝试加载预处理后的图像数据到测试数据中
    CTestSrcData test_data;
    std::string preprocessed_imgs_path = data_path + "preprocessed/preprocessed_" + std::to_string(index) + ".bin";
    std::ifstream preprocessed_file(preprocessed_imgs_path, std::ios::binary);
    if (preprocessed_file.is_open()) {
        preprocessed_file.seekg(0, std::ios::end);
        size_t preprocessed_size = preprocessed_file.tellg();
        preprocessed_file.seekg(0, std::ios::beg);
        
        size_t num_floats = preprocessed_size / sizeof(float);
        std::vector<float> preprocessed_imgs(num_floats);
        preprocessed_file.read(reinterpret_cast<char*>(preprocessed_imgs.data()), preprocessed_size);
        preprocessed_file.close();
        
        test_data.preprocessed_imgs(preprocessed_imgs);
        test_data.img_channels(3);
        test_data.img_height(256);
        test_data.img_width(704);
        test_data.test_data_path(data_path);
        test_data.test_data_type(1); // 1表示离线数据
        LOG(INFO) << "Loaded preprocessed images to test data: " << preprocessed_imgs.size() << " floats from " << preprocessed_imgs_path;
    } else {
        LOG(WARNING) << "Preprocessed images file not found: " << preprocessed_imgs_path << ", using empty test data";
        test_data.preprocessed_imgs(std::vector<float>());
        test_data.img_channels(0);
        test_data.img_height(0);
        test_data.img_width(0);
        test_data.test_data_path(data_path);
        test_data.test_data_type(1);
    }
    data.test_data(test_data);
    
    LOG(INFO) << "Successfully loaded offline data for frame " << index 
              << " with " << video_data.size() << " cameras";
    
    return data;
}

void testSparseBEVAlg(const CAlgResult& alg_result, void* p_handle)
{
    // 获取检测结果
    const auto& detections = alg_result.vecFrameResult().at(0).vecObjectResult();
    
    // 加载原始图像用于可视化（使用前视相机图像）
    // 使用当前帧索引构建正确的路径
    std::string front_camera_path = g_rgb_path + "images/CAM_FRONT/" + 
                                   std::to_string(g_current_frame_index) + ".jpg";
    cv::Mat rgb_img = cv::imread(front_camera_path);
    if (rgb_img.empty()) {
        LOG(ERROR) << "无法加载前视相机图像用于可视化: " << front_camera_path;
        return;
    }

    // 可视化结果
    for (const auto& det : detections) {
        // 获取3D目标信息
        float x = det.x();           // 3D中心点x坐标
        float y = det.y();           // 3D中心点y坐标
        float z = det.z();           // 3D中心点z坐标
        float w = det.w();           // 3D边界框宽度
        float l = det.l();           // 3D边界框长度
        float h = det.h();           // 3D边界框高度
        float yaw = det.yaw();       // 3D边界框偏航角
        float conf = det.confidence(); // 置信度
        std::string cls_name = det.strClass();
        long track_id = det.trackid(); // 跟踪ID

        // 将3D边界框投影到2D图像平面
        // 这里需要lidar2img变换矩阵，暂时使用简化的投影
        // 实际应用中应该使用正确的相机内参和外参
        
        // 简化的投影：假设相机在原点，z轴向前
        if (z > 0) {  // 只显示相机前方的目标
            // 简单的透视投影
            float scale = 1000.0f / z;  // 简单的深度缩放
            float x_2d = rgb_img.cols / 2 + x * scale;
            float y_2d = rgb_img.rows / 2 - y * scale;  // 注意y轴方向
            
            // 计算2D边界框大小
            float w_2d = w * scale;
            float h_2d = h * scale;
            
            // 确保边界框在图像范围内
            float x1 = std::max(0.0f, x_2d - w_2d / 2);
            float y1 = std::max(0.0f, y_2d - h_2d / 2);
            float x2 = std::min(static_cast<float>(rgb_img.cols), x_2d + w_2d / 2);
            float y2 = std::min(static_cast<float>(rgb_img.rows), y_2d + h_2d / 2);
            
            // 根据类别选择颜色
            cv::Scalar color;
            if (cls_name == "car") color = cv::Scalar(0, 255, 0);      // 绿色
            else if (cls_name == "truck") color = cv::Scalar(0, 255, 255); // 黄色
            else if (cls_name == "bus") color = cv::Scalar(255, 0, 0);     // 蓝色
            else if (cls_name == "pedestrian") color = cv::Scalar(0, 0, 255); // 红色
            else if (cls_name == "motorcycle") color = cv::Scalar(255, 0, 255); // 紫色
            else if (cls_name == "bicycle") color = cv::Scalar(255, 255, 0); // 青色
            else color = cv::Scalar(128, 128, 128); // 灰色

            // 绘制2D边界框
            cv::rectangle(rgb_img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

            // 准备标签文本
            std::stringstream ss;
            ss << cls_name << " " << std::fixed << std::setprecision(2) << conf;
            if (track_id >= 0) {
                ss << " ID:" << track_id;
            }
            std::string label = ss.str();

            // 计算文本大小
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);

            // 绘制标签背景
            cv::rectangle(rgb_img, 
                         cv::Point(x1, y1 - text_size.height - 10),
                         cv::Point(x1 + text_size.width, y1),
                         color, -1);

            // 绘制标签文本
            cv::putText(rgb_img, label, 
                       cv::Point(x1, y1 - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);

            // 绘制3D信息
            std::stringstream info_ss;
            info_ss << "3D:(" << std::fixed << std::setprecision(1) 
                    << x << "," << y << "," << z << ")";
            std::string info_text = info_ss.str();
            
            cv::putText(rgb_img, info_text, 
                       cv::Point(x1, y2 + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }
    }

    // 保存相机视角结果
    std::string camera_save_path = g_save_dir + 
        std::filesystem::path(g_rgb_path).stem().string() + "_sparse4d.jpg";
        
    // 检查保存目录是否存在
    if (!std::filesystem::exists(g_save_dir)) {
        std::filesystem::create_directories(g_save_dir);
        LOG(INFO) << "创建保存目录: " << g_save_dir;
    }
    
    // 保存相机视角图像
    bool save_success = cv::imwrite(camera_save_path, rgb_img);
    if (!save_success) {
        LOG(ERROR) << "保存相机视角图像失败: " << camera_save_path;
        return;
    }
    
    LOG(INFO) << "相机视角推理完成，结果已保存到: " << camera_save_path;
    
    // ========== BEV可视化部分 ==========
    try {
        // 加载点云数据
        std::string pointcloud_path = g_rgb_path + "lidar/lidar_" + 
                                     std::to_string(g_current_frame_index) + ".bin";
        std::vector<Point3D> points = loadPointCloud(pointcloud_path);
        
        // 加载真值标签
        std::string labels_path = g_rgb_path + "labels/labels_" + 
                                 std::to_string(g_current_frame_index) + ".json";
        std::vector<BBox3D> gt_boxes = loadGroundTruthLabels(labels_path);
        
        // 转换检测结果为BBox3D格式
        std::vector<BBox3D> pred_boxes = convertDetectionsToBBox3D(detections);
        
        // 生成BEV可视化
        std::string bev_save_path = g_save_dir + 
            std::filesystem::path(g_rgb_path).stem().string() + "_bev.jpg";
        visualizeBEV(points, gt_boxes, pred_boxes, bev_save_path);
        
        LOG(INFO) << "BEV可视化完成，结果已保存到: " << bev_save_path;
        
    } catch (const std::exception& e) {
        LOG(WARNING) << "BEV可视化失败: " << e.what() << "，继续执行相机视角可视化";
    }
    
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
}

std::vector<Point3D> loadPointCloud(const std::string& file_path) {
    std::vector<Point3D> points;
    
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG(WARNING) << "无法打开点云文件: " << file_path;
        return points;
    }
    
    // 读取二进制格式
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 尝试不同的点云格式
    size_t num_points_4 = file_size / (4 * sizeof(float));  // x, y, z, intensity
    size_t num_points_5 = file_size / (5 * sizeof(float));  // x, y, z, intensity, ring
    
    size_t num_points = 0;
    size_t floats_per_point = 0;
    
    if (file_size % (4 * sizeof(float)) == 0) {
        num_points = num_points_4;
        floats_per_point = 4;
        LOG(INFO) << "检测到4个float值格式的点云数据";
    } else if (file_size % (5 * sizeof(float)) == 0) {
        num_points = num_points_5;
        floats_per_point = 5;
        LOG(INFO) << "检测到5个float值格式的点云数据";
    } else {
        LOG(ERROR) << "无法识别的点云数据格式，文件大小: " << file_size << " 字节";
        return points;
    }
    
    points.reserve(num_points);
    std::vector<float> buffer(num_points * floats_per_point);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);
    
    for (size_t i = 0; i < num_points; ++i) {
        Point3D point;
        point.x = buffer[i * floats_per_point + 0];
        point.y = buffer[i * floats_per_point + 1];
        point.z = buffer[i * floats_per_point + 2];
        point.intensity = buffer[i * floats_per_point + 3];
        points.push_back(point);
    }
    
    LOG(INFO) << "加载点云数据: " << points.size() << " 个点";
    
    // 统计点云范围
    if (!points.empty()) {
        float min_x = points[0].x, max_x = points[0].x;
        float min_y = points[0].y, max_y = points[0].y;
        float min_z = points[0].z, max_z = points[0].z;
        
        for (const auto& point : points) {
            min_x = std::min(min_x, point.x);
            max_x = std::max(max_x, point.x);
            min_y = std::min(min_y, point.y);
            max_y = std::max(max_y, point.y);
            min_z = std::min(min_z, point.z);
            max_z = std::max(max_z, point.z);
        }
        
        LOG(INFO) << "点云范围 - X: [" << min_x << ", " << max_x 
                  << "], Y: [" << min_y << ", " << max_y 
                  << "], Z: [" << min_z << ", " << max_z << "]";
    }
    
    return points;
}

std::vector<BBox3D> loadGroundTruthLabels(const std::string& file_path) {
    std::vector<BBox3D> gt_boxes;
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        LOG(WARNING) << "无法打开标签文件: " << file_path;
        return gt_boxes;
    }
    
    // 读取JSON文件内容
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    
    LOG(INFO) << "读取标签文件内容，大小: " << content.size() << " 字符";
    
    // 简单的JSON解析（针对nuScenes格式）
    // 查找所有包含bbox_3d的条目
    size_t pos = 0;
    int parsed_count = 0;
    
    while ((pos = content.find("\"bbox_3d\":", pos)) != std::string::npos) {
        try {
            // 找到category_name
            size_t cat_pos = content.rfind("\"category_name\":", pos);
            if (cat_pos == std::string::npos || cat_pos > pos) {
                pos += 10; // 移动到下一个位置
                continue;
            }
            
            // 提取类别名称
            size_t cat_start = content.find("\"", cat_pos + 16) + 1;
            if (cat_start == std::string::npos) {
                pos += 10;
                continue;
            }
            size_t cat_end = content.find("\"", cat_start);
            if (cat_end == std::string::npos) {
                pos += 10;
                continue;
            }
            std::string category = content.substr(cat_start, cat_end - cat_start);
            
            // 找到bbox_3d数组
            size_t bbox_start = content.find("[", pos);
            if (bbox_start == std::string::npos) {
                pos += 10;
                continue;
            }
            size_t bbox_end = content.find("]", bbox_start);
            if (bbox_end == std::string::npos) {
                pos += 10;
                continue;
            }
            std::string bbox_str = content.substr(bbox_start + 1, bbox_end - bbox_start - 1);
            
            // 解析bbox_3d数组 [x, y, z, w, l, h, yaw]
            std::istringstream bbox_iss(bbox_str);
            std::vector<float> bbox_values;
            std::string value;
            while (std::getline(bbox_iss, value, ',')) {
                // 去除空格
                value.erase(0, value.find_first_not_of(" \t\r\n"));
                value.erase(value.find_last_not_of(" \t\r\n") + 1);
                if (!value.empty()) {
                    bbox_values.push_back(std::stof(value));
                }
            }
            
            if (bbox_values.size() >= 7) {
                BBox3D box;
                box.x = bbox_values[0];  // x
                box.y = bbox_values[1];  // y
                box.z = bbox_values[2];  // z
                box.l = bbox_values[3];  // 长度 (length) - 与Python脚本一致
                box.w = bbox_values[4];  // 宽度 (width)  - 与Python脚本一致
                box.h = bbox_values[5];  // 高度 (height) - 与Python脚本一致
                box.yaw = bbox_values[6]; // 偏航角
                box.label = category;
                box.confidence = 1.0f;   // 真值置信度为1.0
                box.track_id = -1;       // 真值没有跟踪ID
                box.is_gt = true;
                
                gt_boxes.push_back(box);
                parsed_count++;
                
                LOG(INFO) << "解析标签: " << category << " 位置(" << box.x << ", " << box.y << ", " << box.z 
                           << ") 尺寸(" << box.l << ", " << box.w << ", " << box.h << ") 偏航角: " << box.yaw;
            } else {
                LOG(WARNING) << "bbox_3d数组元素不足，期望7个，实际" << bbox_values.size() << "个";
            }
            
            pos = bbox_end;
        } catch (const std::exception& e) {
            LOG(WARNING) << "解析标签时出错: " << e.what();
            pos += 10;
        }
    }
    
    LOG(INFO) << "成功解析 " << parsed_count << " 个真值标签";
    
    // 统计标签范围
    if (!gt_boxes.empty()) {
        float min_x = gt_boxes[0].x, max_x = gt_boxes[0].x;
        float min_y = gt_boxes[0].y, max_y = gt_boxes[0].y;
        
        for (const auto& box : gt_boxes) {
            min_x = std::min(min_x, box.x);
            max_x = std::max(max_x, box.x);
            min_y = std::min(min_y, box.y);
            max_y = std::max(max_y, box.y);
        }
        
        LOG(INFO) << "标签范围 - X: [" << min_x << ", " << max_x 
                  << "], Y: [" << min_y << ", " << max_y << "]";
    }
    
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
    const int img_size = 800;
    
    // 改进的可视范围计算 - 确保点云居中
    float min_x = FLT_MAX, max_x = -FLT_MAX;
    float min_y = FLT_MAX, max_y = -FLT_MAX;
    
    // 从点云数据计算范围
    if (!points.empty()) {
        for (const auto& point : points) {
            min_x = std::min(min_x, point.x);
            max_x = std::max(max_x, point.x);
            min_y = std::min(min_y, point.y);
            max_y = std::max(max_y, point.y);
        }
    }
    
    // 从标签数据计算范围
    for (const auto& box : gt_boxes) {
        min_x = std::min(min_x, box.x - box.l/2);
        max_x = std::max(max_x, box.x + box.l/2);
        min_y = std::min(min_y, box.y - box.w/2);
        max_y = std::max(max_y, box.y + box.w/2);
    }
    
    for (const auto& box : pred_boxes) {
        min_x = std::min(min_x, box.x - box.l/2);
        max_x = std::max(max_x, box.x + box.l/2);
        min_y = std::min(min_y, box.y - box.w/2);
        max_y = std::max(max_y, box.y + box.w/2);
    }
    
    // 如果没有数据，使用默认范围
    if (min_x == FLT_MAX) {
        min_x = -50.0f; max_x = 50.0f;
        min_y = -50.0f; max_y = 50.0f;
    }
    
    // 计算中心点和范围
    float center_x = (min_x + max_x) / 2.0f;
    float center_y = (min_y + max_y) / 2.0f;
    float range_x = max_x - min_x;
    float range_y = max_y - min_y;
    
    // 确保最小可视范围，并添加边距
    float min_range = 20.0f;  // 最小20米范围
    float margin = 5.0f;      // 5米边距
    float range = std::max({range_x, range_y, min_range}) / 2.0f + margin;
    
    // 重新计算可视范围，确保正方形
    float visual_min_x = center_x - range;
    float visual_max_x = center_x + range;
    float visual_min_y = center_y - range;
    float visual_max_y = center_y + range;
    
    LOG(INFO) << "BEV可视范围 - X: [" << visual_min_x << ", " << visual_max_x 
              << "], Y: [" << visual_min_y << ", " << visual_max_y << "]";
    LOG(INFO) << "中心点: (" << center_x << ", " << center_y << "), 范围: " << range;
    
    const float resolution = range * 2.0f / img_size; // 分辨率
    
    // 创建BEV图像
    cv::Mat bev_img = cv::Mat::zeros(img_size, img_size, CV_8UC3);
    
    // 绘制点云（俯视图，只显示x-y平面）
    int valid_points = 0;
    for (const auto& point : points) {
        // 过滤超出范围的点
        if (point.x < visual_min_x || point.x > visual_max_x || 
            point.y < visual_min_y || point.y > visual_max_y) {
            continue;
        }
        
        // 坐标转换：从世界坐标到图像坐标
        int img_x = static_cast<int>((point.x - visual_min_x) / resolution);
        int img_y = static_cast<int>((visual_max_y - point.y) / resolution); // 翻转Y轴
        
        // 确保在图像范围内
        if (img_x >= 0 && img_x < img_size && img_y >= 0 && img_y < img_size) {
            // 根据强度值设置颜色
            int intensity = static_cast<int>(std::min(255.0f, point.intensity * 255.0f));
            cv::Vec3b color(intensity, intensity, intensity);
            bev_img.at<cv::Vec3b>(img_y, img_x) = color;
            valid_points++;
        }
    }
    
    LOG(INFO) << "绘制了 " << valid_points << " 个有效点云点";
    
    // 绘制真值边界框（绿色）
    for (const auto& box : gt_boxes) {
        // 检查边界框是否在可视范围内
        if (box.x < visual_min_x || box.x > visual_max_x || 
            box.y < visual_min_y || box.y > visual_max_y) {
            continue;
        }
        
        // 计算边界框的四个角点
        float cos_yaw = std::cos(box.yaw);
        float sin_yaw = std::sin(box.yaw);
        
        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(-box.l/2, -box.w/2); // 左下
        corners[1] = cv::Point2f( box.l/2, -box.w/2); // 右下
        corners[2] = cv::Point2f( box.l/2,  box.w/2); // 右上
        corners[3] = cv::Point2f(-box.l/2,  box.w/2); // 左上
        
        // 旋转和平移
        for (auto& corner : corners) {
            float x = corner.x * cos_yaw - corner.y * sin_yaw + box.x;
            float y = corner.x * sin_yaw + corner.y * cos_yaw + box.y;
            
            // 转换到图像坐标
            corner.x = (x - visual_min_x) / resolution;
            corner.y = (visual_max_y - y) / resolution;
        }
        
        // 检查坐标是否在图像范围内
        bool valid_corners = true;
        for (const auto& corner : corners) {
            if (corner.x < 0 || corner.x >= img_size || corner.y < 0 || corner.y >= img_size) {
                valid_corners = false;
                break;
            }
        }
        
        if (valid_corners) {
            // 转换为整数坐标
            std::vector<cv::Point> int_corners;
            for (const auto& corner : corners) {
                int_corners.push_back(cv::Point(static_cast<int>(corner.x), static_cast<int>(corner.y)));
            }
            
            // 绘制边界框
            cv::polylines(bev_img, std::vector<std::vector<cv::Point>>{int_corners}, 
                         true, cv::Scalar(0, 255, 0), 2);
            
            // 绘制标签
            std::string label = box.label + "_GT";
            cv::putText(bev_img, label, cv::Point(int_corners[0].x, int_corners[0].y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        }
    }
    
    // 绘制预测边界框（红色）
    for (const auto& box : pred_boxes) {
        // 检查边界框是否在可视范围内
        if (box.x < visual_min_x || box.x > visual_max_x || 
            box.y < visual_min_y || box.y > visual_max_y) {
            continue;
        }
        
        // 计算边界框的四个角点
        float cos_yaw = std::cos(box.yaw);
        float sin_yaw = std::sin(box.yaw);
        
        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(-box.l/2, -box.w/2); // 左下
        corners[1] = cv::Point2f( box.l/2, -box.w/2); // 右下
        corners[2] = cv::Point2f( box.l/2,  box.w/2); // 右上
        corners[3] = cv::Point2f(-box.l/2,  box.w/2); // 左上
        
        // 旋转和平移
        for (auto& corner : corners) {
            float x = corner.x * cos_yaw - corner.y * sin_yaw + box.x;
            float y = corner.x * sin_yaw + corner.y * cos_yaw + box.y;
            
            // 转换到图像坐标
            corner.x = (x - visual_min_x) / resolution;
            corner.y = (visual_max_y - y) / resolution;
        }
        
        // 检查坐标是否在图像范围内
        bool valid_corners = true;
        for (const auto& corner : corners) {
            if (corner.x < 0 || corner.x >= img_size || corner.y < 0 || corner.y >= img_size) {
                valid_corners = false;
                break;
            }
        }
        
        if (valid_corners) {
            // 转换为整数坐标
            std::vector<cv::Point> int_corners;
            for (const auto& corner : corners) {
                int_corners.push_back(cv::Point(static_cast<int>(corner.x), static_cast<int>(corner.y)));
            }
            
            // 绘制边界框
            cv::polylines(bev_img, std::vector<std::vector<cv::Point>>{int_corners}, 
                         true, cv::Scalar(0, 0, 255), 2);
            
            // 绘制标签和置信度
            std::stringstream ss;
            ss << box.label << " " << std::fixed << std::setprecision(2) << box.confidence;
            if (box.track_id >= 0) {
                ss << " ID:" << box.track_id;
            }
            std::string label = ss.str();
            
            cv::putText(bev_img, label, cv::Point(int_corners[0].x, int_corners[0].y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
        }
    }
    
    // 绘制坐标轴（在图像中心）
    int center_img_x = img_size / 2;
    int center_img_y = img_size / 2;
    int axis_length = 50;
    
    cv::arrowedLine(bev_img, cv::Point(center_img_x, center_img_y), 
                   cv::Point(center_img_x + axis_length, center_img_y), cv::Scalar(255, 0, 0), 2); // X轴（红色）
    cv::arrowedLine(bev_img, cv::Point(center_img_x, center_img_y), 
                   cv::Point(center_img_x, center_img_y - axis_length), cv::Scalar(0, 255, 0), 2); // Y轴（绿色）
    
    // 添加图例
    cv::putText(bev_img, "X", cv::Point(center_img_x + axis_length + 5, center_img_y + 5),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    cv::putText(bev_img, "Y", cv::Point(center_img_x - 5, center_img_y - axis_length - 5),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    
    // 添加标题和统计信息
    cv::putText(bev_img, "BEV View (Green: GT, Red: Prediction)", 
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // 添加坐标信息
    std::stringstream coord_info;
    coord_info << "Center: (" << std::fixed << std::setprecision(1) << center_x << ", " << center_y << ")";
    cv::putText(bev_img, coord_info.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    std::stringstream range_info;
    range_info << "Range: " << std::fixed << std::setprecision(1) << range * 2 << "m";
    cv::putText(bev_img, range_info.str(), cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    // 保存图像
    cv::imwrite(save_path, bev_img);
    LOG(INFO) << "BEV可视化已保存到: " << save_path;
}

} // namespace sparse_bev

// 导出给main.cpp使用的主函数
void main_sparse_bev() {
    try {
        // 设置默认路径
        std::string deploy_path = "/share/Code/Sparse4d/C++/Output/";
        sparse_bev::g_save_dir = deploy_path + "vis/";
        std::string data_path = "/share/Code/Sparse4d/C++/Data/sparse/";  // 匹配Python脚本输出路径

        // 算法接口调用流程基本如下：
        ISparseBEVAlg* l_pObj = CreateSparseBEVAlgObj(deploy_path);

        // // 准备算法参数
        // CSelfAlgParam *l_stTestAlgParam = new CSelfAlgParam();
        // l_stTestAlgParam->m_strRootPath = deploy_path;
        
        // 初始化算法接口对象
        l_pObj->initAlgorithm(deploy_path, sparse_bev::testSparseBEVAlg, nullptr);

        // 处理多个帧的数据
        int num_frames = 2;  // 处理10帧数据
        for (int i = 0; i < num_frames; i++)
        {
            LOG(INFO) << "Processing frame " << i;
            
            // 设置当前帧索引
            sparse_bev::g_current_frame_index = i;
            
            // 加载离线数据
            CTimeMatchSrcData multi_modal_data = sparse_bev::loadOfflineData(data_path, i);
            
            // 检查数据是否加载成功
            if (multi_modal_data.vecVideoSrcData().empty()) {
                LOG(WARNING) << "No video data loaded for frame " << i << ", skipping...";
                continue;
            }
            
            // 更新全局变量用于可视化
            sparse_bev::g_rgb_path = data_path;
            
            // 运行算法
            l_pObj->runAlgorithm(&multi_modal_data);
            
            LOG(INFO) << "Frame " << i << " processed successfully";
        }
        
        LOG(INFO) << "All frames processed successfully";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "错误: " << e.what();
        throw;
    }
}