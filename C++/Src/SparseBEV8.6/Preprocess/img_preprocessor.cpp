// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "img_preprocessor.h"
#include "../../Common/Core/FunctionHub.h"
#include "../../Common/TensorRT/TensorRT.h"
#include "../data/SparseBEVInputData.h"
#include "log.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <algorithm>

// 添加DIVUP宏定义，与CUDA内核文件保持一致
#define DIVUP(a, b) ((a % b != 0) ? (a / b + 1) : (a / b))

// 注册模块
REGISTER_MODULE("SparseBEV", ImagePreprocessor, ImagePreprocessor)

/**
 * @brief 析构函数
 */
ImagePreprocessor::~ImagePreprocessor() {
    // 清理资源
}

/**
 * @brief 初始化模块
 * @param p_pAlgParam 算法参数指针
 * @return 初始化是否成功
 */
bool ImagePreprocessor::init(void* p_pAlgParam) {
    // 解析任务配置
    m_taskConfig = *static_cast<const sparsebev::TaskConfig*>(p_pAlgParam);
    // 强制使用全精度，取消半精度支持
    use_half_precision_ = false;
    
    LOG(INFO) << "[INFO] Task config loaded, forced to use full precision (FP32)";
    
    // 计算输出大小
    size_t output_size = m_taskConfig.preprocessor_params().num_cams() * 
                        m_taskConfig.preprocessor_params().model_input_img_c() *
                        m_taskConfig.preprocessor_params().model_input_img_h() * 
                        m_taskConfig.preprocessor_params().model_input_img_w();
    
    // 根据精度预先分配GPU内存
    if (!m_float_output_wrapper.allocate(output_size)) {
        LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for float output!";
        status_ = false;
        return status_;
    }
    LOG(INFO) << "[INFO] Pre-allocated float precision GPU memory, size: " << output_size;
    
    status_ = true;
    return status_;
}

/**
 * @brief 设置输入数据
 * @param input 输入数据指针
 */
void ImagePreprocessor::setInput(void* input) {
    if (input != nullptr) {
        m_inputImage = *static_cast<CTimeMatchSrcData*>(input);
    }
}

/**
 * @brief 获取输出数据
 * @return 输出数据指针
 */
void* ImagePreprocessor::getOutput() 
{
    // 返回SparseBEVInputWrapper而不是CudaWrapper
    return static_cast<void*>(m_output_wrapper.get());
}

/**
 * @brief 执行预处理
 */
void ImagePreprocessor::execute() 
{
    if (!status_) {
        LOG(ERROR) << "[ERROR] Module not initialized!";
        return;
    }
    
    // // 验证输入数据的完整性
    // LOG(INFO) << "[INFO] Validating input data completeness:";
    // LOG(INFO) << "[INFO] - Video data count: " << m_inputImage.vecVideoSrcData().size();
    // LOG(INFO) << "[INFO] - Timestamp: " << m_inputImage.lTimeStamp();
    // LOG(INFO) << "[INFO] - Ego pose translation: " << m_inputImage.ego_pose_info().ego2global_translation().size() << " elements";
    // LOG(INFO) << "[INFO] - Ego pose rotation: " << m_inputImage.ego_pose_info().ego2global_rotation().size() << " elements";
    // LOG(INFO) << "[INFO] - Lidar2ego translation: " << m_inputImage.lidar2ego_info().lidar2ego_translation().size() << " elements";
    // LOG(INFO) << "[INFO] - Lidar2ego rotation: " << m_inputImage.lidar2ego_info().lidar2ego_rotation().size() << " elements";
    // LOG(INFO) << "[INFO] - Lidar points: " << m_inputImage.lidar_data().num_points();
    // LOG(INFO) << "[INFO] - Calibration matrices: " << m_inputImage.calibration_data().lidar2img_matrices().size() << " elements";
    
    // 从m_inputImage获取输入数据
    if (m_inputImage.vecVideoSrcData().empty()) {
        LOG(ERROR) << "[ERROR] No input video data available!";
        return;
    }
    
    // 计算总的图像数据大小
    size_t total_image_size = 0;
    for (const auto& video_data : m_inputImage.vecVideoSrcData()) {
        total_image_size += video_data.vecImageBuf().size();
    }
    
    // 验证数据大小是否与配置匹配
    size_t expected_size = m_taskConfig.preprocessor_params().num_cams() * 
                          m_taskConfig.preprocessor_params().raw_img_c() *
                          m_taskConfig.preprocessor_params().raw_img_h() * 
                          m_taskConfig.preprocessor_params().raw_img_w();
    
    // LOG(INFO) << "[DEBUG] Preprocessor size calculation:";
    // LOG(INFO) << "[DEBUG]   num_cams: " << m_taskConfig.preprocessor_params().num_cams();
    // LOG(INFO) << "[DEBUG]   raw_img_c: " << m_taskConfig.preprocessor_params().raw_img_c();
    // LOG(INFO) << "[DEBUG]   raw_img_h: " << m_taskConfig.preprocessor_params().raw_img_h();
    // LOG(INFO) << "[DEBUG]   raw_img_w: " << m_taskConfig.preprocessor_params().raw_img_w();
    // LOG(INFO) << "[DEBUG]   Expected size: " << expected_size;
    // LOG(INFO) << "[DEBUG]   Actual total size: " << total_image_size;
    
    if (total_image_size != expected_size) {
        LOG(ERROR) << "[ERROR] Input data size mismatch! Expected: " << expected_size 
                   << ", Got: " << total_image_size;
        return;
    }
    
    // 创建GPU内存包装器用于输入数据
    CudaWrapper<std::uint8_t> raw_imgs_wrapper;
    
    // 分配GPU内存用于输入数据
    if (!raw_imgs_wrapper.allocate(expected_size)) {
        LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for raw images!";
        return;
    }
    
    // 将输入数据复制到GPU
    size_t offset = 0;
    for (const auto& video_data : m_inputImage.vecVideoSrcData()) {
        if (!raw_imgs_wrapper.copyFromHost(video_data.vecImageBuf().data(), 
                                         offset, video_data.vecImageBuf().size())) {
            LOG(ERROR) << "[ERROR] Failed to copy video data to GPU!";
            return;
        }
        offset += video_data.vecImageBuf().size();
    }
    
    // 创建CUDA流
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        LOG(ERROR) << "[ERROR] Failed to create CUDA stream!";
        return;
    }
    
    // 根据精度选择调用相应的forward函数，使用预先分配的成员变量
    Status result;
    LOG(INFO) << "[INFO] Using float precision for image preprocessing";
    // 直接使用原始uint8数据进行预处理
    result = forward(raw_imgs_wrapper, stream, m_float_output_wrapper);
    
    // 清理CUDA流
    cudaStreamDestroy(stream);
    
    if (result != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Image preprocessing failed with status: " << static_cast<int>(result);
        return;
    }
    
    // 检查run_status，如果为false则保存预处理数据为bin文件
    if (!m_taskConfig.run_status()) {
        saveOriginalDataToBin();  // 保存原始数据
        savePreprocessedDataToBin();  // 保存预处理后的数据
    }
    
    // 创建完整的输入数据结构
    m_output_wrapper = createInputWrapper();
    
    LOG(INFO) << "[INFO] Image preprocessing completed successfully";
    LOG(INFO) << "[INFO] Created complete input data wrapper for SparseBEV";
}

/**
 * @brief 图像预处理主函数（uint8到float版本）
 * @param input_imgs 输入图像数据，uint8格式，存储在GPU内存中
 * @param stream CUDA流，用于异步GPU操作
 * @param output_imgs 输出缓冲区，存储预处理后的float类型图像
 * @return 处理状态码，成功返回kSuccess，失败返回相应错误码
 */
Status ImagePreprocessor::forward(const CudaWrapper<std::uint8_t>& input_imgs, 
                                  const cudaStream_t& stream,
                                  CudaWrapper<float>& output_imgs) 
{
    // 验证输入图像尺寸：检查输入图像的总像素数是否与配置参数匹配
    if (input_imgs.getSize() != m_taskConfig.preprocessor_params().num_cams() *
                              (m_taskConfig.preprocessor_params().raw_img_c() * 
                               m_taskConfig.preprocessor_params().raw_img_h() *
                               m_taskConfig.preprocessor_params().raw_img_w())) {
        LOG(ERROR) << "[ERROR] Input imgs' size mismatches with params!";
        return Status::kImgPreprocesSizeErr;
    }

    // 验证输出缓冲区尺寸：检查输出图像的总像素数是否与配置参数匹配
    if (output_imgs.getSize() !=
        m_taskConfig.preprocessor_params().num_cams() * 
        m_taskConfig.preprocessor_params().model_input_img_c() *
        m_taskConfig.preprocessor_params().model_input_img_h() * 
        m_taskConfig.preprocessor_params().model_input_img_w()) {
        LOG(ERROR) << "[ERROR] Model input imgs' size mismatches with params!";
        return Status::kImgPreprocesSizeErr;
    }

    // 调用CUDA内核执行图像预处理操作
    // 包括：尺寸调整、裁剪、归一化等
    Status Ret_Code = imgPreprocessLauncher(
        input_imgs.getCudaPtr(),                                    // 输入图像GPU指针
        m_taskConfig.preprocessor_params().num_cams(),           // 相机数量
        m_taskConfig.preprocessor_params().raw_img_c(),          // 原始图像通道数
        m_taskConfig.preprocessor_params().raw_img_h(),          // 原始图像高度
        m_taskConfig.preprocessor_params().raw_img_w(),          // 原始图像宽度
        m_taskConfig.preprocessor_params().model_input_img_h(),  // 模型输入图像高度
        m_taskConfig.preprocessor_params().model_input_img_w(),  // 模型输入图像宽度
        m_taskConfig.preprocessor_params().resize_ratio(),       // 缩放比例
        m_taskConfig.preprocessor_params().crop_height(),        // 裁剪高度
        m_taskConfig.preprocessor_params().crop_width(),         // 裁剪宽度
        stream,                                                   // CUDA流
        output_imgs.getCudaPtr());                               // 输出图像GPU指针

    if (Ret_Code != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Image preprocessing kernel execution failed!";
        return Ret_Code;
    }

    LOG(INFO) << "[INFO] Image preprocessing completed successfully (uint8 to float)";
    return Status::kSuccess;
}



/**
 * @brief 创建完整的输入数据结构
 * @return 包含所有必要信息的SparseBEVInputWrapper
 */
std::shared_ptr<sparsebev::SparseBEVInputWrapper> ImagePreprocessor::createInputWrapper() {
    // 1. 创建图像特征数据 - 直接使用GPU内存，不进行CPU拷贝
    std::shared_ptr<CudaWrapper<float>> processed_images;
    // 对于float精度，直接使用现有的GPU内存，避免不必要的拷贝
    // 创建一个新的CudaWrapper，直接使用现有的GPU内存指针
    processed_images = std::make_shared<CudaWrapper<float>>();
    // 直接分配相同大小的GPU内存，然后进行GPU到GPU的拷贝
    if (processed_images->allocate(m_float_output_wrapper.getSize())) {
        // 进行GPU到GPU的拷贝，避免CPU中转
        cudaMemcpy(processed_images->getCudaPtr(), 
                  m_float_output_wrapper.getCudaPtr(), 
                  m_float_output_wrapper.getSize() * sizeof(float), 
                  cudaMemcpyDeviceToDevice);
        LOG(INFO) << "[INFO] Using float precision data with GPU-to-GPU copy";
    } else {
        LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for float data copy";
        return nullptr;
    }
    
    // 2. 创建图像预处理结果
    sparsebev::ImagePreprocessResult image_data(processed_images);
    
    // 5. 创建输入数据包装器
    auto input_data = std::make_shared<sparsebev::SparseBEVInputData>();
    
    // 设置图像特征
    input_data->image_feature = sparsebev::ImageFeature(processed_images);
    
    // 设置时间间隔
    // 将时间戳转换为float并包装为CudaWrapper
    std::vector<float> time_interval_data = {static_cast<float>(m_inputImage.lTimeStamp())};
    auto time_interval_wrapper = std::make_shared<CudaWrapper<float>>(time_interval_data);
    input_data->time_interval = sparsebev::TimeInterval(time_interval_wrapper);
    
    // 设置图像标定 - 从CTimeMatchSrcData中获取lidar2img数据
    if (m_inputImage.calibration_data().lidar2img_matrices().size() >= m_taskConfig.preprocessor_params().num_cams() * 16) {
        // 创建图像宽高数据
        std::vector<float> image_wh = {
            static_cast<float>(m_taskConfig.preprocessor_params().model_input_img_w()),
            static_cast<float>(m_taskConfig.preprocessor_params().model_input_img_h())
        };
        auto image_wh_wrapper = std::make_shared<CudaWrapper<float>>(image_wh);
        
        // 创建lidar2img数据
        const auto& matrices = m_inputImage.calibration_data().lidar2img_matrices();
        auto lidar2img_wrapper = std::make_shared<CudaWrapper<float>>(matrices);
        
        input_data->image_calibration = sparsebev::ImageCalibration(image_wh_wrapper, lidar2img_wrapper);
        
        LOG(INFO) << "[INFO] Set image calibration with lidar2img data from CTimeMatchSrcData";
        LOG(INFO) << "[INFO] Lidar2img matrices size: " << matrices.size() << " floats";
        LOG(INFO) << "[INFO] Number of cameras: " << m_taskConfig.preprocessor_params().num_cams();
        
        // 验证预处理图像数据（如果存在）
        const auto& test_data = m_inputImage.test_data();
        if (!test_data.preprocessed_imgs().empty()) {
            LOG(INFO) << "[INFO] Found preprocessed image data in CTestSrcData for validation";
            LOG(INFO) << "[INFO] Preprocessed data size: " << test_data.preprocessed_imgs().size() << " floats";
            LOG(INFO) << "[INFO] Expected size: " << (test_data.img_channels() * test_data.img_height() * test_data.img_width() * m_taskConfig.preprocessor_params().num_cams()) << " floats";
            
            // 验证数据大小是否匹配
            size_t expected_size = test_data.img_channels() * test_data.img_height() * test_data.img_width() * m_taskConfig.preprocessor_params().num_cams();
            if (test_data.preprocessed_imgs().size() == expected_size) {
                LOG(INFO) << "[INFO] Preprocessed image data size validation: PASSED";
                
                // 使用FunctionHub中的工具函数验证数据有效性
                if (validate_preprocessed_image_data(
                    test_data.preprocessed_imgs(),
                    m_taskConfig.preprocessor_params().num_cams(),
                    test_data.img_channels(),
                    test_data.img_height(),
                    test_data.img_width())) {
                    LOG(INFO) << "[INFO] Preprocessed image data validation using FunctionHub: PASSED";
                    
                    // 示例：提取第一个相机的R通道数据进行验证
                    std::vector<float> camera0_r_channel = extract_preprocessed_image_data(
                        test_data.preprocessed_imgs(),
                        m_taskConfig.preprocessor_params().num_cams(),
                        test_data.img_channels(),
                        test_data.img_height(),
                        test_data.img_width(),
                        0,  // camera_id = 0 (第一个相机)
                        0   // channel = 0 (R通道)
                    );
                    
                    if (!camera0_r_channel.empty()) {
                        LOG(INFO) << "[INFO] Successfully extracted camera 0 R channel data: " << camera0_r_channel.size() << " floats";
                        
                        // 计算R通道的统计信息
                        float min_val = *std::min_element(camera0_r_channel.begin(), camera0_r_channel.end());
                        float max_val = *std::max_element(camera0_r_channel.begin(), camera0_r_channel.end());
                        float sum = std::accumulate(camera0_r_channel.begin(), camera0_r_channel.end(), 0.0f);
                        float mean = sum / camera0_r_channel.size();
                        
                        LOG(INFO) << "[INFO] Camera 0 R channel statistics:";
                        LOG(INFO) << "[INFO] - Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean;
                    }
                    
                    // 示例：提取所有相机的数据
                    std::vector<float> all_cameras_data = extract_preprocessed_image_data(
                        test_data.preprocessed_imgs(),
                        m_taskConfig.preprocessor_params().num_cams(),
                        test_data.img_channels(),
                        test_data.img_height(),
                        test_data.img_width()
                    );
                    
                    LOG(INFO) << "[INFO] Successfully extracted all cameras data: " << all_cameras_data.size() << " floats";
                } else {
                    LOG(WARNING) << "[WARNING] Preprocessed image data validation using FunctionHub: FAILED";
                }
                
                // 获取当前处理的图像数据用于比较
                std::vector<float> current_processed_data = m_float_output_wrapper.cudaMemcpyD2HResWrap();
                
                // 比较数据
                if (current_processed_data.size() == test_data.preprocessed_imgs().size()) {
                    float max_error = 0.0f;
                    
                    // 计算最大误差
                    for (size_t i = 0; i < current_processed_data.size(); ++i) {
                        float error = std::abs(current_processed_data[i] - test_data.preprocessed_imgs()[i]);
                        if (error > max_error) {
                            max_error = error;
                        }
                    }
                    
                    // 只输出最大误差
                    LOG(INFO) << "[INFO] Max error: " << max_error;
                    
                    // 如果误差在可接受范围内，记录验证成功
                    if (max_error < 0.1f) {
                        LOG(INFO) << "[INFO] Preprocessed image data validation: PASSED";
                    } else {
                        LOG(WARNING) << "[WARNING] Preprocessed image data validation: FAILED - errors too large";
                    }
                } else {
                    LOG(WARNING) << "[WARNING] Preprocessed image data size mismatch for comparison";
                }
            } else {
                LOG(WARNING) << "[WARNING] Preprocessed image data size validation: FAILED";
            }
        } else {
            LOG(INFO) << "[INFO] No preprocessed image data found in CTestSrcData for validation";
        }
    } else {
        LOG(WARNING) << "[WARNING] No valid lidar2img data in CTimeMatchSrcData, using default values";
        // 使用默认单位矩阵
        std::vector<float> image_wh = {
            static_cast<float>(m_taskConfig.preprocessor_params().model_input_img_w()),
            static_cast<float>(m_taskConfig.preprocessor_params().model_input_img_h())
        };
        auto image_wh_wrapper = std::make_shared<CudaWrapper<float>>(image_wh);
        
        std::vector<float> default_matrices(m_taskConfig.preprocessor_params().num_cams() * 16, 0.0f);
        for (uint32_t i = 0; i < m_taskConfig.preprocessor_params().num_cams(); ++i) {
            for (uint32_t j = 0; j < 16; ++j) {
                default_matrices[i * 16 + j] = (j % 5 == 0) ? 1.0f : 0.0f;
            }
        }
        auto lidar2img_wrapper = std::make_shared<CudaWrapper<float>>(default_matrices);
        
        input_data->image_calibration = sparsebev::ImageCalibration(image_wh_wrapper, lidar2img_wrapper);
    }
    
    // 设置坐标变换：将 global2lidar 矩阵传递到推理
    {
        const auto& tinfo = m_inputImage.transform_info();
        const auto& g2l = tinfo.global2lidar_matrix();
        if (g2l.size() == 16) {
            Eigen::Matrix<double, 4, 4> global2lidar_mat;
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    global2lidar_mat(r, c) = static_cast<double>(g2l[r * 4 + c]);
                }
            }

            // 计算逆矩阵：lidar_to_global
            Eigen::Matrix<double, 4, 4> lidar2global_mat = global2lidar_mat.inverse();

            // 写入到输入数据结构
            input_data->coordinate_transform.setTransformMatrices(global2lidar_mat, lidar2global_mat);

            // 调试输出
            LOG(INFO) << "[INFO] Set coordinate transform from CTimeMatchSrcData (global2lidar)";
            LOG(INFO) << "[INFO] global2lidar matrix:";
            for (int r = 0; r < 4; ++r) {
                std::stringstream ss;
                ss << "  Row " << r << ": ";
                for (int c = 0; c < 4; ++c) {
                    ss << std::fixed << std::setprecision(6) << global2lidar_mat(r, c) << " ";
                }
                LOG(INFO) << ss.str();
            }
        } else {
            LOG(WARNING) << "[WARNING] global2lidar_matrix size is not 16, skip setting coordinate transform (size=" << g2l.size() << ")";
        }
    }

    input_data->data_valid = true;
    
    // 6. 创建包装器
    auto wrapper = std::make_shared<sparsebev::SparseBEVInputWrapper>(input_data);
    
    LOG(INFO) << "[INFO] Created SparseBEVInputWrapper with:";
    LOG(INFO) << "[INFO] - Image data valid: " << (image_data.processed_image != nullptr ? "Yes" : "No");
    LOG(INFO) << "[INFO] - Number of cameras: " << m_taskConfig.preprocessor_params().num_cams();
    LOG(INFO) << "[INFO] - Image size: " << m_taskConfig.preprocessor_params().model_input_img_w() << "x" << m_taskConfig.preprocessor_params().model_input_img_h() << "x" << m_taskConfig.preprocessor_params().model_input_img_c();
    LOG(INFO) << "[INFO] - Overall data valid: " << (input_data->data_valid ? "Yes" : "No");
    LOG(INFO) << "[INFO] - Is first frame: " << (input_data->isFirstFrame() ? "Yes" : "No");
    
    return wrapper;
}

/**
 * @brief 保存预处理数据为bin文件
 * 当run_status为false时，将预处理后的图像数据保存到Output/val_bin/目录
 */
void ImagePreprocessor::savePreprocessedDataToBin() {
    try {
        // 创建保存目录
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        // 获取当前时间戳作为文件名的一部分
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        
        // 获取预处理后的数据
        std::vector<float> processed_data = m_float_output_wrapper.cudaMemcpyD2HResWrap();
        
        // 计算数据形状
        int num_cams = m_taskConfig.preprocessor_params().num_cams();
        int channels = m_taskConfig.preprocessor_params().model_input_img_c();
        int height = m_taskConfig.preprocessor_params().model_input_img_h();
        int width = m_taskConfig.preprocessor_params().model_input_img_w();
        
        // 生成文件名 - 使用当前样本索引，与Python脚本格式保持一致
        std::string shape_str = "1*" + std::to_string(num_cams) + "*" + 
                               std::to_string(channels) + "*" + 
                               std::to_string(height) + "*" + 
                               std::to_string(width);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_imgs_" + 
                              shape_str + "_float32.bin";
        
        // 保存数据到bin文件
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(processed_data.data()), 
                         processed_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 预处理数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存预处理数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存原始图像数据为bin文件
 */
void ImagePreprocessor::saveOriginalDataToBin() {
    try {
        // 创建保存目录
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        // 获取原始图像数据 - 从输入数据中获取
        std::vector<float> original_data;
        
        // 检查是否有输入数据
        if (m_inputImage.vecVideoSrcData().empty()) {
            LOG(WARNING) << "[WARNING] 没有输入图像数据，跳过保存原始数据";
            return;
        }
        
        // 从第一个视频源数据中获取原始图像
        const auto& video_data = m_inputImage.vecVideoSrcData()[0];
        if (video_data.vecImageBuf().empty()) {
            LOG(WARNING) << "[WARNING] 视频数据中没有图像，跳过保存原始数据";
            return;
        }
        
        // 获取图像数据
        const auto& img_buffer = video_data.vecImageBuf();
        
        // 将uint8数据转换为float
        original_data.resize(img_buffer.size());
        for (size_t i = 0; i < img_buffer.size(); ++i) {
            original_data[i] = static_cast<float>(img_buffer[i]) / 255.0f;  // 归一化到[0,1]
        }
        
        // 计算数据形状
        int num_cams = m_taskConfig.preprocessor_params().num_cams();
        int channels = m_taskConfig.preprocessor_params().raw_img_c();
        int original_height = m_taskConfig.preprocessor_params().raw_img_h();
        int original_width = m_taskConfig.preprocessor_params().raw_img_w();
        
        // 生成文件名 - 使用当前样本索引，与Python脚本格式保持一致
        std::string shape_str = "1*" + std::to_string(num_cams) + "*" + 
                               std::to_string(channels) + "*" + 
                               std::to_string(original_height) + "*" + 
                               std::to_string(original_width);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_ori_imgs_" + 
                              shape_str + "_float32.bin";
        
        // 保存数据到bin文件
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(original_data.data()), 
                         original_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 原始图像数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存原始图像数据时发生异常: " << e.what();
    }
}

void ImagePreprocessor::setCurrentSampleIndex(int sample_index) {
    current_sample_index_ = sample_index;
}