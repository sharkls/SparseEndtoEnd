// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// FP32版本的bilinear采样
__device__ float thomas_bilinear_sampling(const float*& bottom_data,
                                         const int& height,
                                         const int& width,
                                         const int& num_embeds,
                                         const float& h_im,
                                         const float& w_im,
                                         const int& base_ptr)
{
    const int h_low = floorf(h_im);
    const int w_low = floorf(w_im);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float lh = h_im - h_low;
    const float lw = w_im - w_low;
    const float hh = 1 - lh, hw = 1 - lw;
    // 特征展开形式是 h,w,c
    const int w_stride = num_embeds;                            // 每个像素的通道数
    const int h_stride = width * w_stride;                      // 每行的总通道数
    const int h_low_ptr_offset = h_low * h_stride;              // 行偏移
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;              // 列偏移
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

    float v1 = 0;
    if (h_low >= 0 && w_low >= 0)
    {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
    }
    float v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
    {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
    }
    float v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
    {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
    }
    float v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
    }

    // 距离谁更近，权重越大
    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

// FP16版本的bilinear采样
__device__ __half thomas_bilinear_sampling_half(const __half*& bottom_data,
                                         const int& height,
                                         const int& width,
                                         const int& num_embeds,
                                         const float& h_im,
                                         const float& w_im,
                                         const int& base_ptr)
{
    const int h_low = floorf(h_im);
    const int w_low = floorf(w_im);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float lh = h_im - h_low;
    const float lw = w_im - w_low;
    const float hh = 1 - lh, hw = 1 - lw;
    // 特征展开形式是 h,w,c
    const int w_stride = num_embeds;                            // 每个像素的通道数
    const int h_stride = width * w_stride;                      // 每行的总通道数
    const int h_low_ptr_offset = h_low * h_stride;              // 行偏移
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;              // 列偏移
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

    __half v1 = __float2half(0.0f);
    if (h_low >= 0 && w_low >= 0)
    {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
    }
    __half v2 = __float2half(0.0f);
    if (h_low >= 0 && w_high <= width - 1)
    {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
    }
    __half v3 = __float2half(0.0f);
    if (h_high <= height - 1 && w_low >= 0)
    {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
    }
    __half v4 = __float2half(0.0f);
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
    }

    // 距离谁更近，权重越大
    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    // 使用FP32进行中间计算，然后转换回FP16
    const float val = (w1 * __half2float(v1) + w2 * __half2float(v2) + 
                       w3 * __half2float(v3) + w4 * __half2float(v4));
    return __float2half(val);
}

__global__ void thomas_deformable_aggregation_kernel(
    const int num_kernels,         // batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    float* output,                     // batch_size * num_anchors * num_embeds
    const float* mc_ms_feat,           // batch_size * num_feat * num_embeds
    const int* spatial_shape,      // num_cams * num_scale * 2
    const int* scale_start_index,  /// num_cams * num_scale
    const float* sample_location,      /// batch_size * num_anchors * num_pts * num_cams * 2
    const float* weights,              /// batch_size * num_anchors * num_pts * num_cams * num_scale * num_groups
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups)
{
    // 与PyTorch版本一致的实现，增加必要的边界检查防止越界访问
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    // 权重索引计算 - 在idx被修改前计算（与PyTorch一致）
    const float weight = *(weights + idx / (num_embeds / num_groups));
    
    // 计算各个维度的索引（与PyTorch完全一致的顺序）
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;
    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    // 确认3D关键点映射到图像上的采样点在图像范围内（与PyTorch一致）
    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    // 边界检查：确保value_offset在有效范围内（防止越界读取导致崩溃）
    const int max_read_offset = batch_size * num_feat * num_embeds;
    if (value_offset < 0 || value_offset >= max_read_offset) return;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    // 边界检查：确保输出索引在有效范围内（防止越界写入导致崩溃）
    const int out_idx = anchor_index * num_embeds + channel_index;
    const int max_write_idx = batch_size * num_anchors * num_embeds;
    if (out_idx < 0 || out_idx >= max_write_idx) return;

    // 与PyTorch一致：执行atomicAdd（bilinear_sampling内部处理边界）
    atomicAdd(
        output + out_idx,
        thomas_bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight
    );
}

// FP16版本的kernel - 优化版本：使用float临时缓冲区减少原子操作开销
__global__ void thomas_deformable_aggregation_kernel_half(
    const int num_kernels,         // batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    __half* output,                     // batch_size * num_anchors * num_embeds
    const __half* mc_ms_feat,           // batch_size * num_feat * num_embeds
    const int* spatial_shape,      // num_cams * num_scale * 2
    const int* scale_start_index,  /// num_cams * num_scale
    const __half* sample_location,      /// batch_size * num_anchors * num_pts * num_cams * 2
    const __half* weights,              /// batch_size * num_anchors * num_pts * num_cams * num_scale * num_groups
    float* temp_output,                 // 临时FP32缓冲区 [batch_size * num_anchors * num_embeds]
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups)
{
    // 与PyTorch版本一致的实现（FP16输入版本），增加必要的边界检查
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    // 权重索引计算 - 在idx被修改前计算（与PyTorch一致）
    const __half weight = *(weights + idx / (num_embeds / num_groups));
    
    // 计算各个维度的索引（与PyTorch完全一致的顺序）
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;
    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    // 确认3D关键点映射到图像上的采样点在图像范围内（与PyTorch一致）
    const float loc_w = __half2float(sample_location[loc_offset]);
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = __half2float(sample_location[loc_offset + 1]);
    if (loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    // 边界检查：确保value_offset在有效范围内
    const int max_read_offset = batch_size * num_feat * num_embeds;
    if (value_offset < 0 || value_offset >= max_read_offset) return;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    // 边界检查：确保输出索引在有效范围内
    const int out_idx = anchor_index * num_embeds + channel_index;
    const int max_write_idx = batch_size * num_anchors * num_embeds;
    if (out_idx < 0 || out_idx >= max_write_idx) return;

    // 使用FP16进行采样
    const __half sampled_half = thomas_bilinear_sampling_half(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset);
    
    // 转换为FP32进行乘法和累加（使用FP32的atomicAdd，更快且精度更高）
    const float sampled_val = __half2float(sampled_half);
    const float weight_val = __half2float(weight);
    const float result = sampled_val * weight_val;
    
    atomicAdd(temp_output + out_idx, result);
}

// 混合精度版本的聚合kernel：FP16 value + FP32 keypoints
// 关键优化：关键点使用FP32精度，减少精度损失
__global__ void thomas_deformable_aggregation_kernel_mixed(
    const int num_kernels,
    __half* output,                          // 输出（FP16，但不会被使用）
    const __half* mc_ms_feat,                // FP16特征值
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,            // FP32关键点位置（混合精度）
    const float* weights,                    // FP32注意力权重（混合精度）
    float* temp_output,                      // 临时FP32缓冲区 [batch_size * num_anchors * num_embeds]
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups)
{
    // 与PyTorch版本一致的实现（混合精度版本），增加必要的边界检查
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    // 权重索引计算 - 在idx被修改前计算（与PyTorch一致）
    const float weight = weights[idx / (num_embeds / num_groups)];  // FP32权重
    
    // 计算各个维度的索引（与PyTorch完全一致的顺序）
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;
    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    // 确认3D关键点映射到图像上的采样点在图像范围内（使用FP32精度）
    const float loc_w = sample_location[loc_offset];  // 直接使用FP32，无需转换
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];  // 直接使用FP32，无需转换
    if (loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    // 边界检查：确保value_offset在有效范围内
    const int max_read_offset = batch_size * num_feat * num_embeds;
    if (value_offset < 0 || value_offset >= max_read_offset) return;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    // 边界检查：确保输出索引在有效范围内
    const int out_idx = anchor_index * num_embeds + channel_index;
    const int max_write_idx = batch_size * num_anchors * num_embeds;
    if (out_idx < 0 || out_idx >= max_write_idx) return;

    // 使用FP16进行采样
    const __half sampled_half = thomas_bilinear_sampling_half(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset);
    
    // 转换为FP32进行乘法和累加
    const float sampled_val = __half2float(sampled_half);
    const float result = sampled_val * weight;  // weight已经是FP32，无需转换
    
    atomicAdd(temp_output + out_idx, result);
}

// FP16版本的转换kernel：将FP32临时缓冲区转换为FP16输出
__global__ void convert_float_to_half_kernel(
    const float* input,
    __half* output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = __float2half(input[idx]);
    }
}

int thomas_deform_attn_cuda_forward(cudaStream_t stream,
                                       const float* value,
                                       const int* spatialShapes,
                                       const int* levelStartIndex,
                                       const float* samplingLoc,
                                       const float* attnWeight,
                                       float* output,
                                       int batch_size,                      // batch_size
                                       int num_cams,                        // num_cams
                                       int num_feat,                        // num_feat (spatial_size)
                                       int num_embeds,                      // num_embeds (channels)
                                       int num_scale,                       // num_scale (num_levels)
                                       int num_anchors,                     // num_anchors (num_query)
                                       int num_pts,                         // num_pts (num_point)
                                       int num_groups)
{
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    const int output_size = batch_size * num_anchors * num_embeds;
    cudaError_t err = cudaSuccess;

    // 打印调试信息
    // printf("[DFA-PLUGIN-DEBUG] Parameters: batch_size=%d, num_cams=%d, num_feat=%d, num_embeds=%d, num_scale=%d, num_anchors=%d, num_pts=%d, num_groups=%d\n",
    //        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups);
    // printf("[DFA-PLUGIN-DEBUG] num_kernels=%d, output_size=%d\n", num_kernels, output_size);

    // 初始化输出内存为零
    err = cudaMemsetAsync(output, 0, output_size * sizeof(float), stream);
    if (err != cudaSuccess)
    {
        printf("[DFA-PLUGIN-ERROR] Output memory initialization failed with Error \"%s : %s\".\n",
               cudaGetErrorString(err),
               cudaGetErrorName(err));
        return -1;
    }

    thomas_deformable_aggregation_kernel<<<(int)ceil(((double)num_kernels / 128)), 128, 0, stream>>>(num_kernels,
                                                                                                        output,
                                                                                                        value,
                                                                                                        spatialShapes,
                                                                                                        levelStartIndex,
                                                                                                        samplingLoc,
                                                                                                        attnWeight,
                                                                                                        batch_size,
                                                                                                        num_cams,
                                                                                                        num_feat,
                                                                                                        num_embeds,
                                                                                                        num_scale,
                                                                                                        num_anchors,
                                                                                                        num_pts,
                                                                                                        num_groups);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("[DFA-PLUGIN-ERROR] Kernel Launch Failed with Error \"%s : %s\".\n",
               cudaGetErrorString(err),
               cudaGetErrorName(err));
    }

    return 0;
}

// FP16版本的前向传播函数 - 优化版本：使用FP32临时缓冲区（通过workspace提供）
int thomas_deform_attn_cuda_forward_half(cudaStream_t stream,
                                       const __half* value,
                                       const int* spatialShapes,
                                       const int* levelStartIndex,
                                       const __half* samplingLoc,
                                       const __half* attnWeight,
                                       __half* output,
                                       float* workspace,                    // TensorRT提供的workspace（FP32临时缓冲区）
                                       int batch_size,                      // batch_size
                                       int num_cams,                        // num_cams
                                       int num_feat,                        // num_feat (spatial_size)
                                       int num_embeds,                      // num_embeds (channels)
                                       int num_scale,                       // num_scale (num_levels)
                                       int num_anchors,                     // num_anchors (num_query)
                                       int num_pts,                         // num_pts (num_point)
                                       int num_groups)
{
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    const int output_size = batch_size * num_anchors * num_embeds;
    cudaError_t err = cudaSuccess;

    // 使用TensorRT提供的workspace（FP32临时缓冲区）
    float* temp_output = workspace;

    // 初始化临时缓冲区为零
    err = cudaMemsetAsync(temp_output, 0, output_size * sizeof(float), stream);
    if (err != cudaSuccess)
    {
        printf("[DFA-PLUGIN-ERROR] Temporary buffer initialization failed with Error \"%s : %s\".\n",
               cudaGetErrorString(err),
               cudaGetErrorName(err));
        return -1;
    }

    // 执行聚合kernel（使用FP32临时缓冲区，atomicAdd更快）
    thomas_deformable_aggregation_kernel_half<<<(int)ceil(((double)num_kernels / 128)), 128, 0, stream>>>(
        num_kernels,
        output,  // 虽然传入，但不会被使用
        value,
        spatialShapes,
        levelStartIndex,
        samplingLoc,
        attnWeight,
        temp_output,  // 使用FP32临时缓冲区
        batch_size,
        num_cams,
        num_feat,
        num_embeds,
        num_scale,
        num_anchors,
        num_pts,
        num_groups);
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("[DFA-PLUGIN-ERROR] Kernel Launch Failed with Error \"%s : %s\".\n",
               cudaGetErrorString(err),
               cudaGetErrorName(err));
        return -1;
    }

    // 将FP32临时缓冲区转换为FP16输出（向量化转换，更快）
    const int num_threads_convert = 256;
    const int num_blocks_convert = (output_size + num_threads_convert - 1) / num_threads_convert;
    convert_float_to_half_kernel<<<num_blocks_convert, num_threads_convert, 0, stream>>>(
        temp_output,
        output,
        output_size);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("[DFA-PLUGIN-ERROR] Convert kernel launch failed with Error \"%s : %s\".\n",
               cudaGetErrorString(err),
               cudaGetErrorName(err));
        return -1;
    }

    return 0;
}

// 混合精度版本的前向传播函数：FP16 value + FP32 keypoints
// 关键优化：关键点使用FP32精度，减少精度损失，提高最终anchor精度
int thomas_deform_attn_cuda_forward_mixed(cudaStream_t stream,
                                         const __half* value,          // FP16特征值
                                         const int* spatialShapes,
                                         const int* levelStartIndex,
                                         const float* samplingLoc,      // FP32关键点位置
                                         const float* attnWeight,       // FP32注意力权重
                                         __half* output,               // FP16输出
                                         float* workspace,              // TensorRT提供的workspace（FP32临时缓冲区）
                                         int batch_size,                // batch_size
                                         int num_cams,                  // num_cams
                                         int num_feat,                  // num_feat (spatial_size)
                                         int num_embeds,                // num_embeds (channels)
                                         int num_scale,                 // num_scale (num_levels)
                                         int num_anchors,               // num_anchors (num_query)
                                         int num_pts,                   // num_pts (num_point)
                                         int num_groups)
{
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    const int output_size = batch_size * num_anchors * num_embeds;
    cudaError_t err = cudaSuccess;

    // 使用TensorRT提供的workspace（FP32临时缓冲区）
    float* temp_output = workspace;

    // 初始化临时缓冲区为零
    err = cudaMemsetAsync(temp_output, 0, output_size * sizeof(float), stream);
    if (err != cudaSuccess)
    {
        printf("[DFA-PLUGIN-ERROR] Mixed precision: Temporary buffer initialization failed with Error \"%s : %s\".\n",
               cudaGetErrorString(err),
               cudaGetErrorName(err));
        return -1;
    }

    // 执行混合精度聚合kernel（FP16 value + FP32 keypoints，使用FP32临时缓冲区）
    // 修复参数顺序：正确的调用顺序应该和thomas_deform_attn_cuda_forward一致
    thomas_deformable_aggregation_kernel_mixed<<<(int)ceil(((double)num_kernels / 128)), 128, 0, stream>>>(
        num_kernels,
        output,  // 虽然传入，但不会被使用
        value,   // FP16特征值
        spatialShapes,
        levelStartIndex,
        samplingLoc,  // FP32关键点位置
        attnWeight,   // FP32注意力权重
        temp_output,  // 使用FP32临时缓冲区
        batch_size,
        num_cams,
        num_feat,
        num_embeds,
        num_scale,
        num_anchors,
        num_pts,
        num_groups);
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("[DFA-PLUGIN-ERROR] Mixed precision: Kernel Launch Failed with Error \"%s : %s\".\n",
               cudaGetErrorString(err),
               cudaGetErrorName(err));
        return -1;
    }

    // 将FP32临时缓冲区转换为FP16输出（向量化转换，更快）
    const int num_threads_convert = 256;
    const int num_blocks_convert = (output_size + num_threads_convert - 1) / num_threads_convert;
    convert_float_to_half_kernel<<<num_blocks_convert, num_threads_convert, 0, stream>>>(
        temp_output,
        output,
        output_size);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("[DFA-PLUGIN-ERROR] Mixed precision: Convert kernel launch failed with Error \"%s : %s\".\n",
               cudaGetErrorString(err),
               cudaGetErrorName(err));
        return -1;
    }

    return 0;
}