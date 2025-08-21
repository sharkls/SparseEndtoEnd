// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "img_preprocessor.h"
#include "img_aug_with_bilinearinterpolation_kernel.h"
#include <filesystem>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <fstream>

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
    // 解析任务配置参数
    if (p_pAlgParam != nullptr) {
        m_taskConfig = *static_cast<sparsebev::TaskConfig*>(p_pAlgParam);
        use_half_precision_ = m_taskConfig.use_half_precision();
        
        // 计算输出大小
        size_t output_size = m_taskConfig.preprocessor_params().num_cams() * 
                            m_taskConfig.preprocessor_params().model_input_img_c() *
                            m_taskConfig.preprocessor_params().model_input_img_h() * 
                            m_taskConfig.preprocessor_params().model_input_img_w();
        
        // 根据精度预先分配GPU内存
        if (use_half_precision_) {
            if (!m_half_output_wrapper.allocate(output_size)) {
                LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for half precision output!";
                status_ = false;
                return status_;
            }
            LOG(INFO) << "[INFO] Pre-allocated half precision GPU memory, size: " << output_size;
        } else {
            if (!m_float_output_wrapper.allocate(output_size)) {
                LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for float output!";
                status_ = false;
                return status_;
            }
            LOG(INFO) << "[INFO] Pre-allocated float precision GPU memory, size: " << output_size;
        }
        
        status_ = true;
    } else {
        LOG(ERROR) << "[ERROR] Failed to get algorithm parameters!";
        status_ = false;
    }
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
void* ImagePreprocessor::getOutput() {
    // 优先返回完整的输入数据结构（推荐）
    if (m_output_wrapper != nullptr && m_output_wrapper->isValid()) {
        return static_cast<void*>(m_output_wrapper.get());
    }
    
    // 向后兼容：返回GPU内存包装器
    if (use_half_precision_) {
        return static_cast<void*>(&m_half_output_wrapper);
    } else {
        return static_cast<void*>(&m_float_output_wrapper);
    }
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
    
    // 验证输入数据的完整性
    LOG(INFO) << "[INFO] Validating input data completeness:";
    LOG(INFO) << "[INFO] - Video data count: " << m_inputImage.vecVideoSrcData().size();
    LOG(INFO) << "[INFO] - Timestamp: " << m_inputImage.lTimeStamp();
    LOG(INFO) << "[INFO] - Ego pose translation: " << m_inputImage.ego_pose_info().ego2global_translation().size() << " elements";
    LOG(INFO) << "[INFO] - Ego pose rotation: " << m_inputImage.ego_pose_info().ego2global_rotation().size() << " elements";
    LOG(INFO) << "[INFO] - Lidar2ego translation: " << m_inputImage.lidar2ego_info().lidar2ego_translation().size() << " elements";
    LOG(INFO) << "[INFO] - Lidar2ego rotation: " << m_inputImage.lidar2ego_info().lidar2ego_rotation().size() << " elements";
    LOG(INFO) << "[INFO] - Lidar points: " << m_inputImage.lidar_data().num_points();
    LOG(INFO) << "[INFO] - Calibration matrices: " << m_inputImage.calibration_data().lidar2img_matrices().size() << " elements";
    
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
    
    LOG(INFO) << "[DEBUG] Preprocessor size calculation:";
    LOG(INFO) << "[DEBUG]   num_cams: " << m_taskConfig.preprocessor_params().num_cams();
    LOG(INFO) << "[DEBUG]   raw_img_c: " << m_taskConfig.preprocessor_params().raw_img_c();
    LOG(INFO) << "[DEBUG]   raw_img_h: " << m_taskConfig.preprocessor_params().raw_img_h();
    LOG(INFO) << "[DEBUG]   raw_img_w: " << m_taskConfig.preprocessor_params().raw_img_w();
    LOG(INFO) << "[DEBUG]   Expected size: " << expected_size;
    LOG(INFO) << "[DEBUG]   Actual total size: " << total_image_size;
    
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
    if (use_half_precision_) {
        LOG(INFO) << "[INFO] Using half precision for image preprocessing";
        result = forward(raw_imgs_wrapper, stream, m_half_output_wrapper);
    } else {
        LOG(INFO) << "[INFO] Using float precision for image preprocessing";
        result = forward(raw_imgs_wrapper, stream, m_float_output_wrapper);
    }
    
    // 清理CUDA流
    cudaStreamDestroy(stream);
    
    if (result != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Image preprocessing failed with status: " << static_cast<int>(result);
        return;
    }
    
    // 检查run_status，如果为false则保存预处理数据为bin文件
    if (!m_taskConfig.run_status()) {
        savePreprocessedDataToBin();
    }
    
    // 创建完整的输入数据结构
    m_output_wrapper = createInputWrapper();
    
    LOG(INFO) << "[INFO] Image preprocessing completed successfully";
    LOG(INFO) << "[INFO] Created complete input data wrapper for SparseBEV";
}

/**
 * @brief 图像预处理主函数（float版本）
 * @param raw_imgs 原始图像数据，uint8格式，存储在GPU内存中
 * @param stream CUDA流，用于异步GPU操作
 * @param model_input_imgs 输出缓冲区，存储预处理后的float类型图像
 * @return 处理状态码，成功返回kSuccess，失败返回相应错误码
 */
Status ImagePreprocessor::forward(const CudaWrapper<std::uint8_t>& raw_imgs,
                                        const cudaStream_t& stream,
                                        const CudaWrapper<float>& model_input_imgs) 
{
    // ========== 详细参数打印开始 ==========
    LOG(INFO) << "[DEBUG] ========== Image Preprocessor Parameters ==========";
    LOG(INFO) << "[DEBUG] Input parameters:";
    LOG(INFO) << "[DEBUG]   - num_cams: " << m_taskConfig.preprocessor_params().num_cams();
    LOG(INFO) << "[DEBUG]   - raw_img_c: " << m_taskConfig.preprocessor_params().raw_img_c();
    LOG(INFO) << "[DEBUG]   - raw_img_h: " << m_taskConfig.preprocessor_params().raw_img_h();
    LOG(INFO) << "[DEBUG]   - raw_img_w: " << m_taskConfig.preprocessor_params().raw_img_w();
    LOG(INFO) << "[DEBUG]   - model_input_img_c: " << m_taskConfig.preprocessor_params().model_input_img_c();
    LOG(INFO) << "[DEBUG]   - model_input_img_h: " << m_taskConfig.preprocessor_params().model_input_img_h();
    LOG(INFO) << "[DEBUG]   - model_input_img_w: " << m_taskConfig.preprocessor_params().model_input_img_w();
    LOG(INFO) << "[DEBUG]   - resize_ratio: " << m_taskConfig.preprocessor_params().resize_ratio();
    LOG(INFO) << "[DEBUG]   - crop_height: " << m_taskConfig.preprocessor_params().crop_height();
    LOG(INFO) << "[DEBUG]   - crop_width: " << m_taskConfig.preprocessor_params().crop_width();
    LOG(INFO) << "[DEBUG]   - use_half_precision: " << (use_half_precision_ ? "true" : "false");
    
    LOG(INFO) << "[DEBUG] Data sizes:";
    LOG(INFO) << "[DEBUG]   - raw_imgs.getSize(): " << raw_imgs.getSize();
    LOG(INFO) << "[DEBUG]   - model_input_imgs.getSize(): " << model_input_imgs.getSize();
    
    // 计算期望的输入输出大小
    size_t expected_input_size = m_taskConfig.preprocessor_params().num_cams() * 
                                m_taskConfig.preprocessor_params().raw_img_c() *
                                m_taskConfig.preprocessor_params().raw_img_h() * 
                                m_taskConfig.preprocessor_params().raw_img_w();
    
    size_t expected_output_size = m_taskConfig.preprocessor_params().num_cams() *
                                 m_taskConfig.preprocessor_params().model_input_img_c() * 
                                 m_taskConfig.preprocessor_params().model_input_img_h() *
                                 m_taskConfig.preprocessor_params().model_input_img_w();
    
    LOG(INFO) << "[DEBUG] Expected sizes:";
    LOG(INFO) << "[DEBUG]   - expected_input_size: " << expected_input_size;
    LOG(INFO) << "[DEBUG]   - expected_output_size: " << expected_output_size;
    
    // 检查输入数据的前几个值
    if (raw_imgs.getSize() > 0) {
        std::vector<uint8_t> sample_input = raw_imgs.cudaMemcpyD2HResWrap();
        LOG(INFO) << "[DEBUG] Input data sample (first 20 values):";
        for (int i = 0; i < std::min(20UL, sample_input.size()); ++i) {
            LOG(INFO) << "[DEBUG]   input[" << i << "] = " << static_cast<int>(sample_input[i]);
        }
    }
    LOG(INFO) << "[DEBUG] ========== End Parameters ==========";
    // ========== 详细参数打印结束 ==========

    // 验证输入图像尺寸：检查原始图像的总像素数是否与配置参数匹配
    // 总像素数 = 相机数量 × 通道数 × 高度 × 宽度
    if (raw_imgs.getSize() != m_taskConfig.preprocessor_params().num_cams() * 
                              m_taskConfig.preprocessor_params().raw_img_c() *
                              m_taskConfig.preprocessor_params().raw_img_h() * 
                              m_taskConfig.preprocessor_params().raw_img_w()) 
    {
        LOG(ERROR) << "[ERROR] Input raw imgs' size mismatches with params!";
        LOG(ERROR) << "[ERROR] Expected: " << expected_input_size << ", Got: " << raw_imgs.getSize();
        return Status::kImgPreprocesSizeErr;
    }

    // 验证输出缓冲区尺寸：检查输出图像的总像素数是否与配置参数匹配
    // 总像素数 = 相机数量 × 通道数 × 模型输入高度 × 模型输入宽度
    if (model_input_imgs.getSize() !=
        m_taskConfig.preprocessor_params().num_cams() *
        (m_taskConfig.preprocessor_params().model_input_img_c() * 
         m_taskConfig.preprocessor_params().model_input_img_h() *
         m_taskConfig.preprocessor_params().model_input_img_w())) 
    {
        LOG(ERROR) << "[ERROR] Model input imgs' size mismatches with params!";
        LOG(ERROR) << "[ERROR] Expected: " << expected_output_size << ", Got: " << model_input_imgs.getSize();
        return Status::kImgPreprocesSizeErr;
    }

    LOG(INFO) << "[DEBUG] Calling imgPreprocessLauncher with parameters:";
    LOG(INFO) << "[DEBUG]   raw_imgs.getCudaPtr(): " << raw_imgs.getCudaPtr();
    LOG(INFO) << "[DEBUG]   model_input_imgs.getCudaPtr(): " << model_input_imgs.getCudaPtr();
    LOG(INFO) << "[DEBUG]   stream: " << stream;

    // 调用CUDA内核执行图像预处理操作
    // 包括：尺寸调整、裁剪、格式转换（uint8 -> float）、归一化等
    const Status Ret_Code = imgPreprocessLauncher(
        raw_imgs.getCudaPtr(),                                    // 原始图像GPU指针
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
        model_input_imgs.getCudaPtr());                          // 输出图像GPU指针

    LOG(INFO) << "[DEBUG] imgPreprocessLauncher returned: " << static_cast<int>(Ret_Code);
    
    // 检查输出数据的前几个值
    if (Ret_Code == Status::kSuccess && model_input_imgs.getSize() > 0) {
        std::vector<float> sample_output = model_input_imgs.cudaMemcpyD2HResWrap();
        LOG(INFO) << "[DEBUG] Output data sample (first 20 values):";
        for (int i = 0; i < std::min(20UL, sample_output.size()); ++i) {
            LOG(INFO) << "[DEBUG]   output[" << i << "] = " << sample_output[i];
        }
        
        // 计算输出数据的统计信息
        float min_val = *std::min_element(sample_output.begin(), sample_output.end());
        float max_val = *std::max_element(sample_output.begin(), sample_output.end());
        float sum = std::accumulate(sample_output.begin(), sample_output.end(), 0.0f);
        float mean = sum / sample_output.size();
        
        LOG(INFO) << "[DEBUG] Output data statistics:";
        LOG(INFO) << "[DEBUG]   - Min value: " << min_val;
        LOG(INFO) << "[DEBUG]   - Max value: " << max_val;
        LOG(INFO) << "[DEBUG]   - Mean value: " << mean;
        LOG(INFO) << "[DEBUG]   - Total elements: " << sample_output.size();
    }

    return Ret_Code;
}

/**
 * @brief 图像预处理主函数（half精度版本）
 * @param raw_imgs 原始图像数据，uint8格式，存储在GPU内存中
 * @param stream CUDA流，用于异步GPU操作
 * @param model_input_imgs 输出缓冲区，存储预处理后的half类型图像
 * @return 处理状态码，成功返回kSuccess，失败返回相应错误码
 */
Status ImagePreprocessor::forward(const CudaWrapper<std::uint8_t>& raw_imgs,
                                          const cudaStream_t& stream,
                                          const CudaWrapper<half>& model_input_imgs) 
{
    // 验证输入图像尺寸：检查原始图像的总像素数是否与配置参数匹配
    if (raw_imgs.getSize() != m_taskConfig.preprocessor_params().num_cams() *
                              (m_taskConfig.preprocessor_params().raw_img_c() * 
                               m_taskConfig.preprocessor_params().raw_img_h() *
                               m_taskConfig.preprocessor_params().raw_img_w())) {
        LOG(ERROR) << "[ERROR] Input raw imgs' size mismatches with params!";
        return Status::kImgPreprocesSizeErr;
    }

    // 验证输出缓冲区尺寸：检查输出图像的总像素数是否与配置参数匹配
    if (model_input_imgs.getSize() !=
        m_taskConfig.preprocessor_params().num_cams() * 
        m_taskConfig.preprocessor_params().model_input_img_c() *
        m_taskConfig.preprocessor_params().model_input_img_h() * 
        m_taskConfig.preprocessor_params().model_input_img_w()) {
        LOG(ERROR) << "[ERROR] Model input imgs' size mismatches with params!";
        return Status::kImgPreprocesSizeErr;
    }

    // 调用CUDA内核执行图像预处理操作
    // 包括：尺寸调整、裁剪、格式转换（uint8 -> half）、归一化等
    // half精度版本可以减少内存使用和提高计算速度
    Status Ret_Code = imgPreprocessLauncher(
        raw_imgs.getCudaPtr(),                                    // 原始图像GPU指针
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
        model_input_imgs.getCudaPtr());                          // 输出图像GPU指针

    return Ret_Code;
}

/**
 * @brief 创建完整的输入数据结构
 * @return 包含所有必要信息的SparseBEVInputWrapper
 */
std::shared_ptr<sparsebev::SparseBEVInputWrapper> ImagePreprocessor::createInputWrapper() {
    // 1. 创建图像特征数据 - 直接使用GPU内存，不进行CPU拷贝
    std::shared_ptr<CudaWrapper<float>> processed_images;
    if (use_half_precision_) {
        // 对于half精度，需要转换为float精度，但保持在GPU上
        size_t data_size = m_half_output_wrapper.getSize();
        processed_images = std::make_shared<CudaWrapper<float>>();
        if (processed_images->allocate(data_size)) {
            // 在GPU上进行half到float的转换
            auto half_data = m_half_output_wrapper.cudaMemcpyD2HResWrap();
            std::vector<float> float_data(half_data.begin(), half_data.end());
            processed_images->copyFromHost(float_data.data(), 0, data_size);
            LOG(INFO) << "[INFO] Converted half precision to float precision on GPU";
        } else {
            LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for float conversion";
            return nullptr;
        }
    } else {
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
    }
    
    // 2. 创建图像预处理结果
    sparsebev::ImagePreprocessResult image_data(processed_images);
    
    // 5. 创建输入数据包装器
    auto input_data = std::make_shared<sparsebev::SparseBEVInputData>();
    
    // 设置图像特征
    input_data->image_feature = sparsebev::ImageFeature(processed_images);
    
    // 设置时间间隔（暂时为空）
    input_data->time_interval = sparsebev::TimeInterval();
    
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
        std::vector<float> processed_data;
        if (use_half_precision_) {
            // 将half精度数据转换为float精度
            auto half_data = m_half_output_wrapper.cudaMemcpyD2HResWrap();
            processed_data.assign(half_data.begin(), half_data.end());
        } else {
            // 直接获取float精度数据
            processed_data = m_float_output_wrapper.cudaMemcpyD2HResWrap();
        }
        
        // 计算数据形状
        int num_cams = m_taskConfig.preprocessor_params().num_cams();
        int channels = m_taskConfig.preprocessor_params().model_input_img_c();
        int height = m_taskConfig.preprocessor_params().model_input_img_h();
        int width = m_taskConfig.preprocessor_params().model_input_img_w();
        
        // 生成文件名
        std::string filename = save_dir + "preprocessed_imgs_" + 
                              std::to_string(num_cams) + "*" + 
                              std::to_string(channels) + "*" + 
                              std::to_string(height) + "*" + 
                              std::to_string(width) + "_float32_" + 
                              std::to_string(timestamp) + ".bin";
        
        // 保存数据到bin文件
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(processed_data.data()), 
                         processed_data.size() * sizeof(float));
            outfile.close();
            
            LOG(INFO) << "[INFO] Preprocessed data saved to: " << filename;
            LOG(INFO) << "[INFO] Data shape: [" << num_cams << ", " << channels << ", " << height << ", " << width << "]";
            LOG(INFO) << "[INFO] Total elements: " << processed_data.size();
            LOG(INFO) << "[INFO] File size: " << (processed_data.size() * sizeof(float)) << " bytes";
            
            // 保存数据信息到文本文件
            std::string info_filename = save_dir + "preprocessed_info_" + std::to_string(timestamp) + ".txt";
            std::ofstream info_file(info_filename);
            if (info_file.is_open()) {
                info_file << "Preprocessed Image Data Information" << std::endl;
                info_file << "===================================" << std::endl;
                info_file << "Timestamp: " << timestamp << std::endl;
                info_file << "Data file: " << filename << std::endl;
                info_file << "Number of cameras: " << num_cams << std::endl;
                info_file << "Channels: " << channels << std::endl;
                info_file << "Height: " << height << std::endl;
                info_file << "Width: " << width << std::endl;
                info_file << "Total elements: " << processed_data.size() << std::endl;
                info_file << "Data type: float32" << std::endl;
                info_file << "File size: " << (processed_data.size() * sizeof(float)) << " bytes" << std::endl;
                info_file << "Precision: " << (use_half_precision_ ? "half" : "float") << std::endl;
                info_file << std::endl;
                info_file << "Data statistics:" << std::endl;
                
                // 计算数据统计信息
                float min_val = *std::min_element(processed_data.begin(), processed_data.end());
                float max_val = *std::max_element(processed_data.begin(), processed_data.end());
                float sum = std::accumulate(processed_data.begin(), processed_data.end(), 0.0f);
                float mean = sum / processed_data.size();
                
                float variance = 0.0f;
                for (float val : processed_data) {
                    variance += (val - mean) * (val - mean);
                }
                variance /= processed_data.size();
                float std_dev = std::sqrt(variance);
                
                info_file << "  Min value: " << min_val << std::endl;
                info_file << "  Max value: " << max_val << std::endl;
                info_file << "  Mean value: " << mean << std::endl;
                info_file << "  Standard deviation: " << std_dev << std::endl;
                info_file << "  Sum: " << sum << std::endl;
                
                info_file.close();
                LOG(INFO) << "[INFO] Data information saved to: " << info_filename;
            }
            
        } else {
            LOG(ERROR) << "[ERROR] Failed to open file for writing: " << filename;
        }
        
        // 同时保存lidar2img标定数据
        if (!m_inputImage.calibration_data().lidar2img_matrices().empty()) {
            std::string calib_filename = save_dir + "lidar2img_" + 
                                       std::to_string(num_cams) + "*4*4_float32_" + 
                                       std::to_string(timestamp) + ".bin";
            
            std::ofstream calib_file(calib_filename, std::ios::binary);
            if (calib_file.is_open()) {
                const auto& calib_data = m_inputImage.calibration_data().lidar2img_matrices();
                calib_file.write(reinterpret_cast<const char*>(calib_data.data()), 
                               calib_data.size() * sizeof(float));
                calib_file.close();
                
                LOG(INFO) << "[INFO] Lidar2img calibration data saved to: " << calib_filename;
                LOG(INFO) << "[INFO] Calibration data size: " << calib_data.size() << " floats";
            } else {
                LOG(ERROR) << "[ERROR] Failed to open calibration file for writing: " << calib_filename;
            }
        }
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] Exception while saving preprocessed data: " << e.what();
    }
}