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
    if (h_pinned_) {
        cudaFreeHost(h_pinned_);
        h_pinned_ = nullptr;
        h_pinned_bytes_ = 0;
    }
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
    use_half_precision_ = m_taskConfig.use_half_precision();
    // datatype_ = m_taskConfig.datatype();
    
    LOG(INFO) << "[INFO] Task config loaded, datatype: " << static_cast<int>(use_half_precision_);
    
    // 计算输出大小
    size_t output_size = m_taskConfig.preprocessor_params().num_cams() * 
                        m_taskConfig.preprocessor_params().model_input_img_c() *
                        m_taskConfig.preprocessor_params().model_input_img_h() * 
                        m_taskConfig.preprocessor_params().model_input_img_w();
    
    // 根据精度预先分配GPU内存
    bool result = false;
    if(!use_half_precision_)
    {
        result = m_float_output_wrapper.allocate(output_size);
    }
    else
    {
        result = m_half_output_wrapper.allocate(output_size);
    }
    // else if(datatype_ == sparsebev::DataType::int8)
    // {
    //     bool result = m_int8_output_wrapper.allocate(output_size, int8_t{});
    // }
    
    if (!result) {
        LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for float output!";
        status_ = false;
        return status_;
    }
    LOG(INFO) << "[INFO] Pre-allocated float precision GPU memory, size: " << output_size;
    
    // 预分配原始图像 GPU 缓冲与 pinned 主机缓冲（一次拷贝）
    raw_size_ = m_taskConfig.preprocessor_params().num_cams() * 
                     m_taskConfig.preprocessor_params().raw_img_c() *
                     m_taskConfig.preprocessor_params().raw_img_h() * 
                     m_taskConfig.preprocessor_params().raw_img_w();

    if (!m_raw_imgs_wrapper.allocate(raw_size_)) {
        LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for raw images!";
        status_ = false;
        return status_;
    }
    if (!h_pinned_ || h_pinned_bytes_ != raw_size_) {
        if (h_pinned_) cudaFreeHost(h_pinned_);
        if (cudaSuccess != cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_), raw_size_, cudaHostAllocDefault)) {
            LOG(ERROR) << "[ERROR] Failed to allocate pinned host buffer!";
            status_ = false;
            return status_;
        }
        h_pinned_bytes_ = raw_size_;
    }
    
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
    LOG(INFO) << "[INFO] - Timestamp: " << m_inputImage.lTimeStamp();

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
    if (total_image_size != raw_size_) {
        LOG(ERROR) << "[ERROR] Input data size mismatch! Expected: " << raw_size_ 
                   << ", Got: " << total_image_size;
        return;
    }
    
    // 检查 pinned 缓冲大小是否匹配（按字节数）
    size_t expected_bytes = raw_size_ * sizeof(std::uint8_t);
    if (h_pinned_bytes_ != expected_bytes) {
        LOG(ERROR) << "[ERROR] Pinned buffer size mismatch! Expected: " << expected_bytes 
                   << " bytes, Got: " << h_pinned_bytes_ << " bytes";
        return;
    }
    
    // 将输入数据复制到 pinned 主机缓冲（一次性 H2D）
    size_t offset = 0;
    for (const auto& video_data : m_inputImage.vecVideoSrcData()) {
        const auto& buf = video_data.vecImageBuf();
        memcpy(h_pinned_ + offset, buf.data(), buf.size());
        offset += buf.size();
    }
    
    // 创建CUDA流（非阻塞）
    cudaStream_t stream;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        LOG(ERROR) << "[ERROR] Failed to create CUDA stream!";
        return;
    }

    // 一次性 H2D 拷贝到 GPU
    cudaError_t h2d = cudaMemcpyAsync(m_raw_imgs_wrapper.getCudaPtr(),
                                      h_pinned_,
                                      h_pinned_bytes_,
                                      cudaMemcpyHostToDevice,
                                      stream);
    if (h2d != cudaSuccess) {
        LOG(ERROR) << "[ERROR] Failed to H2D copy raw images: " << cudaGetErrorString(h2d);
        cudaStreamDestroy(stream);
        return;
    }

    // 等待H2D拷贝完成，确保数据完全传输后再执行kernel
    cudaError_t sync_result = cudaStreamSynchronize(stream);
    if (sync_result != cudaSuccess) {
        LOG(ERROR) << "[ERROR] Failed to synchronize H2D copy: " << cudaGetErrorString(sync_result);
        cudaStreamDestroy(stream);
        return;
    }

    // 根据精度选择调用相应的forward函数，使用预先分配的成员变量
    LOG(INFO) << "[INFO] Using float precision for image preprocessing";
    // 直接使用原始uint8数据进行预处理
    Status result = Status::kSuccess;
    if(!use_half_precision_)
    {
        result = forward(m_raw_imgs_wrapper, stream, m_float_output_wrapper);
    }
    else
    {
        result = forward(m_raw_imgs_wrapper, stream, m_half_output_wrapper);
    }
    // else if(datatype_ == sparsebev::DataType::int8)
    // {
    //     result = forward(m_raw_imgs_wrapper, stream, m_int8_output_wrapper);
    // }
    
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

    // LOG(INFO) << "[INFO] Image preprocessing completed successfully (uint8 to float)";
    return Status::kSuccess;
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
                                  CudaWrapper<half>& output_imgs) 
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

    // LOG(INFO) << "[INFO] Image preprocessing completed successfully (uint8 to float)";
    return Status::kSuccess;
}

/**
 * @brief 创建完整的输入数据结构
 * @return 包含所有必要信息的SparseBEVInputWrapper
 */
std::shared_ptr<sparsebev::SparseBEVInputWrapper> ImagePreprocessor::createInputWrapper() {
    // 1) 基于配置选择输出精度并创建新的包装器，执行 D2D 拷贝，避免使用 attachExternal
    std::shared_ptr<sparsebev::SparseBEVInputData> input_data = std::make_shared<sparsebev::SparseBEVInputData>();

    if(!use_half_precision_)
    {
        auto processed_images = std::make_shared<CudaWrapper<float>>();
        // 分配与成员输出相同大小
        if (!processed_images->allocate(m_float_output_wrapper.getSize())) {
            LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for processed float images";
            return nullptr;
        }
        // D2D 拷贝
        checkCudaErrors(cudaMemcpy(processed_images->getCudaPtr(),
                                   m_float_output_wrapper.getCudaPtr(),
                                   m_float_output_wrapper.getSize() * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
        LOG(INFO) << "[INFO] Using float precision data with D2D alias";
        // 设置图像特征
        input_data->image_feature = sparsebev::ImageFeature(processed_images);
    }
    else
    {
        auto processed_images = std::make_shared<CudaWrapper<half>>();
        if (!processed_images->allocate(m_half_output_wrapper.getSize())) {
            LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for processed half images";
            return nullptr;
        }
        checkCudaErrors(cudaMemcpy(processed_images->getCudaPtr(),
                                   m_half_output_wrapper.getCudaPtr(),
                                   m_half_output_wrapper.getSize() * sizeof(half),
                                   cudaMemcpyDeviceToDevice));
        LOG(INFO) << "[INFO] Using half precision data with D2D alias";
    input_data->image_feature = sparsebev::ImageFeature(processed_images);
    }
    
    // 2) 设置时间戳
    int64_t input_timestamp_ms = m_inputImage.lTimeStamp();
    std::vector<float> timestamp_data = {static_cast<float>(input_timestamp_ms)};
    LOG(INFO) << "[INFO] Passing timestamp to inference module: " << input_timestamp_ms << " ms";
    auto timestamp_wrapper = std::make_shared<CudaWrapper<float>>(timestamp_data);
    input_data->time_interval = sparsebev::TimeInterval(timestamp_wrapper);
    
    // 3) 设置图像标定
    if (m_inputImage.calibration_data().lidar2img_matrices().size() >= m_taskConfig.preprocessor_params().num_cams() * 16) {
        std::vector<float> image_wh = {
            static_cast<float>(m_taskConfig.preprocessor_params().model_input_img_w()),
            static_cast<float>(m_taskConfig.preprocessor_params().model_input_img_h())
        };
        auto image_wh_wrapper = std::make_shared<CudaWrapper<float>>(image_wh);
        
        const auto& matrices = m_inputImage.calibration_data().lidar2img_matrices();
        auto lidar2img_wrapper = std::make_shared<CudaWrapper<float>>(matrices);
        
        input_data->image_calibration = sparsebev::ImageCalibration(image_wh_wrapper, lidar2img_wrapper);
        
        LOG(INFO) << "[INFO] Set image calibration with lidar2img data from CTimeMatchSrcData";
        LOG(INFO) << "[INFO] Lidar2img matrices size: " << matrices.size() << " floats";
        LOG(INFO) << "[INFO] Number of cameras: " << m_taskConfig.preprocessor_params().num_cams();
    }

    // 4) 返回包装器
    auto wrapper = std::make_shared<sparsebev::SparseBEVInputWrapper>(input_data);
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