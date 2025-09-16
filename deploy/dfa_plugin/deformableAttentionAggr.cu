// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <cstdio>

#include <cuda_runtime.h>
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