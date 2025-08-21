/*******************************************************
 文件名：img_preprocessor.h
 作者：sharkls
 描述：图像预处理类，负责将图像预处理为模型输入格式
 版本：v1.0
 日期：2025-06-18
 *******************************************************/
#ifndef __IMG_AUG_WITH_BILINEARINTERPOLATION_KERNEL_H__
#define __IMG_AUG_WITH_BILINEARINTERPOLATION_KERNEL_H__

#include <cuda_fp16.h>
#include <cstdint>

#include "../../../Include/Common/GlobalContext.h"
#include "../../../Include/Common/CudaWrapper.h"


/// @brief RGB format image prepocess CUDA launcher: resize(bilinear
/// interpolation) -> crop -> normalization.
/// @return image augmentation with output dtype: fp32.
Status imgPreprocessLauncher(const std::uint8_t* raw_imgs_cuda_ptr,
                                     const std::uint32_t& num_cams,
                                     const std::uint32_t& raw_img_c,
                                     const std::uint32_t& raw_img_h,
                                     const std::uint32_t& raw_img_w,
                                     const std::uint32_t& model_input_img_h,
                                     const std::uint32_t& model_input_img_w,
                                     const float& resize_ratio,
                                     const std::uint32_t& crop_height,
                                     const std::uint32_t& crop_width,
                                     const cudaStream_t& stream,
                                     float* model_input_imgs_cuda_ptr);

/// @brief RGB format image prepocess CUDA launcher: resize(bilinear
/// interpolation) -> crop -> normalization.
/// @return image augmentation with output dtype: fp16.
Status imgPreprocessLauncher(const std::uint8_t* raw_imgs_cuda_ptr,
                                     const std::uint32_t& num_cams,
                                     const std::uint32_t& raw_img_c,
                                     const std::uint32_t& raw_img_h,
                                     const std::uint32_t& raw_img_w,
                                     const std::uint32_t& model_input_img_h,
                                     const std::uint32_t& model_input_img_w,
                                     const float& resize_ratio,
                                     const std::uint32_t& crop_height,
                                     const std::uint32_t& crop_width,
                                     const cudaStream_t& stream,
                                     half* model_input_imgs_cuda_ptr);


#endif  // __IMG_AUG_WITH_BILINEARINTERPOLATION_KERNEL_H__