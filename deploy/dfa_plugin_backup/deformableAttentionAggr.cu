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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels)
        return;
    
    // 保存原始的全局线程索引用于权重计算
    const int original_idx = idx;
    
    // 计算各个维度的索引
    const int channel_index = idx % num_embeds;      // 获取通道索引值
    idx /= num_embeds;
    const int scale_index = idx % num_scale;         // 获取尺度索引值
    idx /= num_scale;
    const int cam_index = idx % num_cams;            // 获取相机索引值
    idx /= num_cams;
    const int pts_index = idx % num_pts;             // 获取映射点索引值
    idx /= num_pts;
    int anchor_index = idx % num_anchors;            // 获取锚点索引值
    idx /= num_anchors;
    const int batch_index = idx % batch_size;        // 获取批次索引值
    
    // 修复：使用与PyTorch版本一致的权重索引计算
    const float weight = *(weights + original_idx / (num_embeds / num_groups));

    anchor_index = batch_index * num_anchors + anchor_index;     // 计算当前线程中锚点索引值
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;  // 计算当前线程中关键映射点索引值

    // 确认3D关键点映射到图像上的采样点在图像范围内
    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1)
        return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1)
        return;

    int cam_scale_index = cam_index * num_scale + scale_index;   // 计算当前线程中相机尺度的索引值
    const int value_offset =
        (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;  // 计算当前线程中特征值的偏移量

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    // 计算采样点的像素坐标
    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    if (h_im > -1 && w_im > -1 && h_im < h && w_im < w)
    {
        atomicAdd(output + anchor_index * num_embeds + channel_index,
                  thomas_bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight);
    }
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels)
        return;
    
    // 保存原始的全局线程索引用于权重计算
    const int original_idx = idx;
    
    // 计算各个维度的索引
    const int channel_index = idx % num_embeds;      // 获取通道索引值
    idx /= num_embeds;
    const int scale_index = idx % num_scale;         // 获取尺度索引值
    idx /= num_scale;
    const int cam_index = idx % num_cams;            // 获取相机索引值
    idx /= num_cams;
    const int pts_index = idx % num_pts;             // 获取映射点索引值
    idx /= num_pts;
    int anchor_index = idx % num_anchors;            // 获取锚点索引值
    idx /= num_anchors;
    const int batch_index = idx % batch_size;        // 获取批次索引值
    
    // 修复：使用与PyTorch版本一致的权重索引计算
    const __half weight = *(weights + original_idx / (num_embeds / num_groups));

    anchor_index = batch_index * num_anchors + anchor_index;     // 计算当前线程中锚点索引值
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;  // 计算当前线程中关键映射点索引值

    // 确认3D关键点映射到图像上的采样点在图像范围内
    const float loc_w = __half2float(sample_location[loc_offset]);
    if (loc_w <= 0 || loc_w >= 1)
        return;
    const float loc_h = __half2float(sample_location[loc_offset + 1]);
    if (loc_h <= 0 || loc_h >= 1)
        return;

    int cam_scale_index = cam_index * num_scale + scale_index;   // 计算当前线程中相机尺度的索引值
    const int value_offset =
        (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;  // 计算当前线程中特征值的偏移量

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    // 计算采样点的像素坐标
    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    if (h_im > -1 && w_im > -1 && h_im < h && w_im < w)
    {
        // 使用FP16进行采样（减少转换开销）
        const __half sampled_half = thomas_bilinear_sampling_half(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset);
        
        // 转换为FP32进行乘法和累加（使用FP32的atomicAdd，更快）
        const float sampled_val = __half2float(sampled_half);
        const float weight_val = __half2float(weight);
        const float result = sampled_val * weight_val;
        
        // 使用FP32的atomicAdd（比half的atomicCAS快得多）
        atomicAdd(temp_output + anchor_index * num_embeds + channel_index, result);
    }
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