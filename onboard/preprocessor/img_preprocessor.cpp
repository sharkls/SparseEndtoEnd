// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include "img_preprocessor.h"

#include <iostream>
#include <algorithm>  // 添加这个头文件用于std::min_element, std::max_element
#include <numeric>    // 添加这个头文件用于std::accumulate

#include "img_aug_with_bilinearinterpolation_kernel.h"

// 添加DIVUP宏定义，与CUDA内核文件保持一致
#define DIVUP(a, b) ((a % b != 0) ? (a / b + 1) : (a / b))

namespace sparse_end2end {
namespace preprocessor {

/**
 * @brief 构造函数：初始化图像预处理器
 * @param params 端到端参数配置，包含预处理器所需的所有参数
 */
ImagePreprocessor::ImagePreprocessor(const common::E2EParams& params) : params_(params) {}

/**
 * @brief 图像预处理主函数（float版本）
 * @param raw_imgs 原始图像数据，uint8格式，存储在GPU内存中
 * @param stream CUDA流，用于异步GPU操作
 * @param model_input_imgs 输出缓冲区，存储预处理后的float类型图像
 * @return 处理状态码，成功返回kSuccess，失败返回相应错误码
 */
common::Status ImagePreprocessor::forward(const common::CudaWrapper<std::uint8_t>& raw_imgs,
                                          const cudaStream_t& stream,
                                          const common::CudaWrapper<float>& model_input_imgs) 
{
  // ========== 详细参数打印开始 ==========
  std::cout << "[DEBUG] ========== Onboard Image Preprocessor Parameters ==========" << std::endl;
  std::cout << "[DEBUG] Input parameters:" << std::endl;
  std::cout << "[DEBUG]   - num_cams: " << params_.preprocessor_params.num_cams << std::endl;
  std::cout << "[DEBUG]   - raw_img_c: " << params_.preprocessor_params.raw_img_c << std::endl;
  std::cout << "[DEBUG]   - raw_img_h: " << params_.preprocessor_params.raw_img_h << std::endl;
  std::cout << "[DEBUG]   - raw_img_w: " << params_.preprocessor_params.raw_img_w << std::endl;
  std::cout << "[DEBUG]   - model_input_img_c: " << params_.preprocessor_params.model_input_img_c << std::endl;
  std::cout << "[DEBUG]   - model_input_img_h: " << params_.preprocessor_params.model_input_img_h << std::endl;
  std::cout << "[DEBUG]   - model_input_img_w: " << params_.preprocessor_params.model_input_img_w << std::endl;
  std::cout << "[DEBUG]   - resize_ratio: " << params_.preprocessor_params.resize_ratio << std::endl;
  std::cout << "[DEBUG]   - crop_height: " << params_.preprocessor_params.crop_height << std::endl;
  std::cout << "[DEBUG]   - crop_width: " << params_.preprocessor_params.crop_width << std::endl;
  
  std::cout << "[DEBUG] Data sizes:" << std::endl;
  std::cout << "[DEBUG]   - raw_imgs.getSize(): " << raw_imgs.getSize() << std::endl;
  std::cout << "[DEBUG]   - model_input_imgs.getSize(): " << model_input_imgs.getSize() << std::endl;
  
  // 计算期望的输入输出大小
  size_t expected_input_size = params_.preprocessor_params.num_cams * 
                              params_.preprocessor_params.raw_img_c *
                              params_.preprocessor_params.raw_img_h * 
                              params_.preprocessor_params.raw_img_w;
  
  size_t expected_output_size = params_.preprocessor_params.num_cams *
                               params_.preprocessor_params.model_input_img_c * 
                               params_.preprocessor_params.model_input_img_h *
                               params_.preprocessor_params.model_input_img_w;
  
  std::cout << "[DEBUG] Expected sizes:" << std::endl;
  std::cout << "[DEBUG]   - expected_input_size: " << expected_input_size << std::endl;
  std::cout << "[DEBUG]   - expected_output_size: " << expected_output_size << std::endl;
  
  // 检查输入数据的前几个值
  if (raw_imgs.getSize() > 0) {
    std::vector<uint8_t> sample_input = raw_imgs.cudaMemcpyD2HResWrap();
    std::cout << "[DEBUG] Input data sample (first 20 values):" << std::endl;
    for (int i = 0; i < std::min(20UL, sample_input.size()); ++i) {
      std::cout << "[DEBUG]   input[" << i << "] = " << static_cast<int>(sample_input[i]) << std::endl;
    }
  }
  std::cout << "[DEBUG] ========== End Parameters ==========" << std::endl;
  // ========== 详细参数打印结束 ==========

  // 验证输入图像尺寸：检查原始图像的总像素数是否与配置参数匹配
  // 总像素数 = 相机数量 × 通道数 × 高度 × 宽度
  if (raw_imgs.getSize() != params_.preprocessor_params.num_cams * params_.preprocessor_params.raw_img_c *
                                params_.preprocessor_params.raw_img_h * params_.preprocessor_params.raw_img_w) 
  {
    // LOG_ERROR(kLogContext, "Input raw imgs' size mismatches with params.");
    std::cout << "[ERROR] Input raw imgs' size mismatches with params!" << std::endl;
    std::cout << "[ERROR] Expected: " << expected_input_size << ", Got: " << raw_imgs.getSize() << std::endl;

    return common::Status::kImgPreprocesSizeErr;
  }

  // 验证输出缓冲区尺寸：检查输出图像的总像素数是否与配置参数匹配
  // 总像素数 = 相机数量 × 通道数 × 模型输入高度 × 模型输入宽度
  if (model_input_imgs.getSize() !=
      params_.preprocessor_params.num_cams *
          (params_.preprocessor_params.model_input_img_c * params_.preprocessor_params.model_input_img_h *
           params_.preprocessor_params.model_input_img_w)) 
  {
    // LOG_ERROR(kLogContext, "Model input imgs' size mismatches with params.");
    std::cout << "[ERROR] Model input imgs' size mismatches with params!" << std::endl;
    std::cout << "[ERROR] Expected: " << expected_output_size << ", Got: " << model_input_imgs.getSize() << std::endl;

    return common::Status::kImgPreprocesSizeErr;
  }

  std::cout << "[DEBUG] Calling imgPreprocessLauncher with parameters:" << std::endl;
  // std::cout << "[DEBUG]   raw_imgs.getCudaPtr(): " << raw_imgs.getCudaPtr() << std::endl;
  // std::cout << "[DEBUG]   model_input_imgs.getCudaPtr(): " << model_input_imgs.getCudaPtr() << std::endl;
  // std::cout << "[DEBUG]   stream: " << stream << std::endl;

  // 添加CUDA设备检查
  int device_count = 0;
  cudaError_t cuda_error = cudaGetDeviceCount(&device_count);
  if (cuda_error != cudaSuccess) {
    std::cout << "[ERROR] Failed to get CUDA device count: " << cudaGetErrorString(cuda_error) << std::endl;
    return common::Status::kImgPreprocessLaunchErr;
  }
  
  if (device_count == 0) {
    std::cout << "[ERROR] No CUDA devices found!" << std::endl;
    return common::Status::kImgPreprocessLaunchErr;
  }
  
  int current_device = 0;
  cuda_error = cudaGetDevice(&current_device);
  if (cuda_error != cudaSuccess) {
    std::cout << "[ERROR] Failed to get current CUDA device: " << cudaGetErrorString(cuda_error) << std::endl;
    return common::Status::kImgPreprocessLaunchErr;
  }
  
  std::cout << "[DEBUG] CUDA device info:" << std::endl;
  std::cout << "[DEBUG]   - Device count: " << device_count << std::endl;
  std::cout << "[DEBUG]   - Current device: " << current_device << std::endl;
  
  // 检查CUDA内核配置
  const std::uint32_t thread_num = 32U;
  dim3 blocks_dim_in_each_grid(params_.preprocessor_params.num_cams, 
                               DIVUP(params_.preprocessor_params.model_input_img_h, thread_num), 
                               DIVUP(params_.preprocessor_params.model_input_img_w, thread_num));
  dim3 threads_dim_in_each_block(thread_num, thread_num);
  
  std::cout << "[DEBUG] CUDA kernel configuration:" << std::endl;
  std::cout << "[DEBUG]   - Grid dimensions: (" << blocks_dim_in_each_grid.x << ", " 
            << blocks_dim_in_each_grid.y << ", " << blocks_dim_in_each_grid.z << ")" << std::endl;
  std::cout << "[DEBUG]   - Block dimensions: (" << threads_dim_in_each_block.x << ", " 
            << threads_dim_in_each_block.y << ", " << threads_dim_in_each_block.z << ")" << std::endl;
  
  // 检查内存指针是否有效
  if (raw_imgs.getCudaPtr() == nullptr) {
    std::cout << "[ERROR] raw_imgs.getCudaPtr() is nullptr!" << std::endl;
    return common::Status::kImgPreprocessLaunchErr;
  }
  
  if (model_input_imgs.getCudaPtr() == nullptr) {
    std::cout << "[ERROR] model_input_imgs.getCudaPtr() is nullptr!" << std::endl;
    return common::Status::kImgPreprocessLaunchErr;
  }

  std::cout << "[DEBUG] About to call imgPreprocessLauncher..." << std::endl;
  std::cout << "[DEBUG] Function parameters:" << std::endl;
  // std::cout << "[DEBUG]   - raw_imgs_cuda_ptr: " << raw_imgs.getCudaPtr() << std::endl;
  std::cout << "[DEBUG]   - num_cams: " << params_.preprocessor_params.num_cams << std::endl;
  std::cout << "[DEBUG]   - raw_img_c: " << params_.preprocessor_params.raw_img_c << std::endl;
  std::cout << "[DEBUG]   - raw_img_h: " << params_.preprocessor_params.raw_img_h << std::endl;
  std::cout << "[DEBUG]   - raw_img_w: " << params_.preprocessor_params.raw_img_w << std::endl;
  std::cout << "[DEBUG]   - model_input_img_h: " << params_.preprocessor_params.model_input_img_h << std::endl;
  std::cout << "[DEBUG]   - model_input_img_w: " << params_.preprocessor_params.model_input_img_w << std::endl;
  std::cout << "[DEBUG]   - resize_ratio: " << params_.preprocessor_params.resize_ratio << std::endl;
  std::cout << "[DEBUG]   - crop_height: " << params_.preprocessor_params.crop_height << std::endl;
  std::cout << "[DEBUG]   - crop_width: " << params_.preprocessor_params.crop_width << std::endl;
  std::cout << "[DEBUG]   - stream: " << stream << std::endl;
  // std::cout << "[DEBUG]   - model_input_imgs_cuda_ptr: " << model_input_imgs.getCudaPtr() << std::endl;

  // 调用CUDA内核执行图像预处理操作
  // 包括：尺寸调整、裁剪、格式转换（uint8 -> float）、归一化等
  cudaError_t err_before = cudaGetLastError();
  std::cout << "[DEBUG] Before imgPreprocessLauncher, cudaGetLastError: " << cudaGetErrorString(err_before) << std::endl;

  const common::Status Ret_Code = imgPreprocessLauncher(
      raw_imgs.getCudaPtr(),           // 原始图像GPU指针
      params_.preprocessor_params.num_cams,                    // 相机数量
      params_.preprocessor_params.raw_img_c,                   // 原始图像通道数
      params_.preprocessor_params.raw_img_h,                   // 原始图像高度
      params_.preprocessor_params.raw_img_w,                   // 原始图像宽度
      params_.preprocessor_params.model_input_img_h,           // 模型输入图像高度
      params_.preprocessor_params.model_input_img_w,           // 模型输入图像宽度
      params_.preprocessor_params.resize_ratio,                // 缩放比例
      params_.preprocessor_params.crop_height,                 // 裁剪高度
      params_.preprocessor_params.crop_width,                  // 裁剪宽度
      stream,                                                  // CUDA流
      model_input_imgs.getCudaPtr());                         // 输出图像GPU指针

  cudaError_t err_after = cudaDeviceSynchronize();
  std::cout << "[DEBUG] After imgPreprocessLauncher, cudaDeviceSynchronize: " << cudaGetErrorString(err_after) << std::endl;

  std::cout << "[DEBUG] imgPreprocessLauncher returned: " << static_cast<int>(Ret_Code) << std::endl;
  
  // 检查输出数据的前几个值
  if (Ret_Code == common::Status::kSuccess && model_input_imgs.getSize() > 0) {
    std::vector<float> sample_output = model_input_imgs.cudaMemcpyD2HResWrap();
    std::cout << "[DEBUG] Output data sample (first 20 values):" << std::endl;
    for (int i = 0; i < std::min(20UL, sample_output.size()); ++i) {
      std::cout << "[DEBUG]   output[" << i << "] = " << sample_output[i] << std::endl;
    }
    
    // 计算输出数据的统计信息
    float min_val = *std::min_element(sample_output.begin(), sample_output.end());
    float max_val = *std::max_element(sample_output.begin(), sample_output.end());
    float sum = std::accumulate(sample_output.begin(), sample_output.end(), 0.0f);
    float mean = sum / sample_output.size();
    
    std::cout << "[DEBUG] Output data statistics:" << std::endl;
    std::cout << "[DEBUG]   - Min value: " << min_val << std::endl;
    std::cout << "[DEBUG]   - Max value: " << max_val << std::endl;
    std::cout << "[DEBUG]   - Mean value: " << mean << std::endl;
    std::cout << "[DEBUG]   - Total elements: " << sample_output.size() << std::endl;
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
common::Status ImagePreprocessor::forward(const common::CudaWrapper<std::uint8_t>& raw_imgs,
                                          const cudaStream_t& stream,
                                          const common::CudaWrapper<half>& model_input_imgs) 
{
  // ========== 详细参数打印开始 ==========
  std::cout << "[DEBUG] ========== Onboard Image Preprocessor Parameters (HALF) ==========" << std::endl;
  std::cout << "[DEBUG] Input parameters:" << std::endl;
  std::cout << "[DEBUG]   - num_cams: " << params_.preprocessor_params.num_cams << std::endl;
  std::cout << "[DEBUG]   - raw_img_c: " << params_.preprocessor_params.raw_img_c << std::endl;
  std::cout << "[DEBUG]   - raw_img_h: " << params_.preprocessor_params.raw_img_h << std::endl;
  std::cout << "[DEBUG]   - raw_img_w: " << params_.preprocessor_params.raw_img_w << std::endl;
  std::cout << "[DEBUG]   - model_input_img_c: " << params_.preprocessor_params.model_input_img_c << std::endl;
  std::cout << "[DEBUG]   - model_input_img_h: " << params_.preprocessor_params.model_input_img_h << std::endl;
  std::cout << "[DEBUG]   - model_input_img_w: " << params_.preprocessor_params.model_input_img_w << std::endl;
  std::cout << "[DEBUG]   - resize_ratio: " << params_.preprocessor_params.resize_ratio << std::endl;
  std::cout << "[DEBUG]   - crop_height: " << params_.preprocessor_params.crop_height << std::endl;
  std::cout << "[DEBUG]   - crop_width: " << params_.preprocessor_params.crop_width << std::endl;
  std::cout << "[DEBUG]   - precision: HALF" << std::endl;
  
  std::cout << "[DEBUG] Data sizes:" << std::endl;
  std::cout << "[DEBUG]   - raw_imgs.getSize(): " << raw_imgs.getSize() << std::endl;
  std::cout << "[DEBUG]   - model_input_imgs.getSize(): " << model_input_imgs.getSize() << std::endl;
  
  // 计算期望的输入输出大小
  size_t expected_input_size = params_.preprocessor_params.num_cams * 
                              params_.preprocessor_params.raw_img_c *
                              params_.preprocessor_params.raw_img_h * 
                              params_.preprocessor_params.raw_img_w;
  
  size_t expected_output_size = params_.preprocessor_params.num_cams *
                               params_.preprocessor_params.model_input_img_c * 
                               params_.preprocessor_params.model_input_img_h *
                               params_.preprocessor_params.model_input_img_w;
  
  std::cout << "[DEBUG] Expected sizes:" << std::endl;
  std::cout << "[DEBUG]   - expected_input_size: " << expected_input_size << std::endl;
  std::cout << "[DEBUG]   - expected_output_size: " << expected_output_size << std::endl;
  
  // 检查输入数据的前几个值
  if (raw_imgs.getSize() > 0) {
    std::vector<uint8_t> sample_input = raw_imgs.cudaMemcpyD2HResWrap();
    std::cout << "[DEBUG] Input data sample (first 20 values):" << std::endl;
    for (int i = 0; i < std::min(20UL, sample_input.size()); ++i) {
      std::cout << "[DEBUG]   input[" << i << "] = " << static_cast<int>(sample_input[i]) << std::endl;
    }
  }
  std::cout << "[DEBUG] ========== End Parameters ==========" << std::endl;
  // ========== 详细参数打印结束 ==========

  // 验证输入图像尺寸：检查原始图像的总像素数是否与配置参数匹配
  if (raw_imgs.getSize() != params_.preprocessor_params.num_cams *
                                (params_.preprocessor_params.raw_img_c * params_.preprocessor_params.raw_img_h *
                                 params_.preprocessor_params.raw_img_w)) {
    // LOG_ERROR(kLogContext, "Input raw imgs' size mismatches with params.");
    std::cout << "[ERROR] Input raw imgs' size mismatches with params!" << std::endl;
    std::cout << "[ERROR] Expected: " << expected_input_size << ", Got: " << raw_imgs.getSize() << std::endl;

    return common::Status::kImgPreprocesSizeErr;
  }

  // 验证输出缓冲区尺寸：检查输出图像的总像素数是否与配置参数匹配
  if (model_input_imgs.getSize() !=
      params_.preprocessor_params.num_cams * params_.preprocessor_params.model_input_img_c *
          params_.preprocessor_params.model_input_img_h * params_.preprocessor_params.model_input_img_w) {
    // LOG_ERROR(kLogContext, "Model input imgs' size mismatches with params.");
    std::cout << "[ERROR] Model input imgs' size mismatches with params!" << std::endl;
    std::cout << "[ERROR] Expected: " << expected_output_size << ", Got: " << model_input_imgs.getSize() << std::endl;

    return common::Status::kImgPreprocesSizeErr;
  }

  std::cout << "[DEBUG] Calling imgPreprocessLauncher with parameters (HALF):" << std::endl;
  // std::cout << "[DEBUG]   raw_imgs.getCudaPtr(): " << raw_imgs.getCudaPtr() << std::endl;
  // std::cout << "[DEBUG]   model_input_imgs.getCudaPtr(): " << model_input_imgs.getCudaPtr() << std::endl;
  // std::cout << "[DEBUG]   stream: " << stream << std::endl;

  // 添加CUDA设备检查
  int device_count = 0;
  cudaError_t cuda_error = cudaGetDeviceCount(&device_count);
  if (cuda_error != cudaSuccess) {
    std::cout << "[ERROR] Failed to get CUDA device count: " << cudaGetErrorString(cuda_error) << std::endl;
    return common::Status::kImgPreprocessLaunchErr;
  }
  
  if (device_count == 0) {
    std::cout << "[ERROR] No CUDA devices found!" << std::endl;
    return common::Status::kImgPreprocessLaunchErr;
  }
  
  int current_device = 0;
  cuda_error = cudaGetDevice(&current_device);
  if (cuda_error != cudaSuccess) {
    std::cout << "[ERROR] Failed to get current CUDA device: " << cudaGetErrorString(cuda_error) << std::endl;
    return common::Status::kImgPreprocessLaunchErr;
  }
  
  std::cout << "[DEBUG] CUDA device info:" << std::endl;
  std::cout << "[DEBUG]   - Device count: " << device_count << std::endl;
  std::cout << "[DEBUG]   - Current device: " << current_device << std::endl;
  
  // 检查CUDA内核配置
  const std::uint32_t thread_num = 32U;
  dim3 blocks_dim_in_each_grid(params_.preprocessor_params.num_cams, 
                               DIVUP(params_.preprocessor_params.model_input_img_h, thread_num), 
                               DIVUP(params_.preprocessor_params.model_input_img_w, thread_num));
  dim3 threads_dim_in_each_block(thread_num, thread_num);
  
  std::cout << "[DEBUG] CUDA kernel configuration:" << std::endl;
  std::cout << "[DEBUG]   - Grid dimensions: (" << blocks_dim_in_each_grid.x << ", " 
            << blocks_dim_in_each_grid.y << ", " << blocks_dim_in_each_grid.z << ")" << std::endl;
  std::cout << "[DEBUG]   - Block dimensions: (" << threads_dim_in_each_block.x << ", " 
            << threads_dim_in_each_block.y << ", " << threads_dim_in_each_block.z << ")" << std::endl;
  
  // 检查内存指针是否有效
  if (raw_imgs.getCudaPtr() == nullptr) {
    std::cout << "[ERROR] raw_imgs.getCudaPtr() is nullptr!" << std::endl;
    return common::Status::kImgPreprocessLaunchErr;
  }
  
  if (model_input_imgs.getCudaPtr() == nullptr) {
    std::cout << "[ERROR] model_input_imgs.getCudaPtr() is nullptr!" << std::endl;
    return common::Status::kImgPreprocessLaunchErr;
  }

  // 调用CUDA内核执行图像预处理操作
  // 包括：尺寸调整、裁剪、格式转换（uint8 -> half）、归一化等
  // half精度版本可以减少内存使用和提高计算速度
  cudaError_t err_before = cudaGetLastError();
  std::cout << "[DEBUG] Before imgPreprocessLauncher, cudaGetLastError: " << cudaGetErrorString(err_before) << std::endl;

  common::Status Ret_Code = imgPreprocessLauncher(
      raw_imgs.getCudaPtr(),           // 原始图像GPU指针
      params_.preprocessor_params.num_cams,                    // 相机数量
      params_.preprocessor_params.raw_img_c,                   // 原始图像通道数
      params_.preprocessor_params.raw_img_h,                   // 原始图像高度
      params_.preprocessor_params.raw_img_w,                   // 原始图像宽度
      params_.preprocessor_params.model_input_img_h,           // 模型输入图像高度
      params_.preprocessor_params.model_input_img_w,           // 模型输入图像宽度
      params_.preprocessor_params.resize_ratio,                // 缩放比例
      params_.preprocessor_params.crop_height,                 // 裁剪高度
      params_.preprocessor_params.crop_width,                  // 裁剪宽度
      stream,                                                  // CUDA流
      model_input_imgs.getCudaPtr());                         // 输出图像GPU指针

  cudaError_t err_after = cudaDeviceSynchronize();
  std::cout << "[DEBUG] After imgPreprocessLauncher, cudaDeviceSynchronize: " << cudaGetErrorString(err_after) << std::endl;

  std::cout << "[DEBUG] imgPreprocessLauncher returned: " << static_cast<int>(Ret_Code) << std::endl;
  
  // 检查输出数据的前几个值
  if (Ret_Code == common::Status::kSuccess && model_input_imgs.getSize() > 0) {
    std::vector<half> sample_output = model_input_imgs.cudaMemcpyD2HResWrap();
    std::cout << "[DEBUG] Output data sample (first 20 values, HALF):" << std::endl;
    for (int i = 0; i < std::min(20UL, sample_output.size()); ++i) {
      std::cout << "[DEBUG]   output[" << i << "] = " << static_cast<float>(sample_output[i]) << std::endl;
    }
    
    // 计算输出数据的统计信息
    float min_val = static_cast<float>(*std::min_element(sample_output.begin(), sample_output.end()));
    float max_val = static_cast<float>(*std::max_element(sample_output.begin(), sample_output.end()));
    float sum = 0.0f;
    for (const auto& val : sample_output) {
      sum += static_cast<float>(val);
    }
    float mean = sum / sample_output.size();
    
    std::cout << "[DEBUG] Output data statistics (HALF):" << std::endl;
    std::cout << "[DEBUG]   - Min value: " << min_val << std::endl;
    std::cout << "[DEBUG]   - Max value: " << max_val << std::endl;
    std::cout << "[DEBUG]   - Mean value: " << mean << std::endl;
    std::cout << "[DEBUG]   - Total elements: " << sample_output.size() << std::endl;
  }

  return Ret_Code;
}

}  // namespace preprocessor
}  // namespace sparse_end2end