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

// 3D 坐标投影辅助函数 (FP32)
__device__ inline void project_point_3d_to_2d(
    const float* key_points,    // [BS*Q, P, 3]
    const float* lidar2img,     // [BS, C, 4, 4]
    const float* image_wh,      // [BS, C, 2]
    int batch_idx,
    int global_anchor_idx,
    int p,
    int c,
    int num_pts,
    int num_cams,
    float& loc_w,
    float& loc_h)
{
    int pts_offset = (global_anchor_idx * num_pts + p) * 3;
    float x = key_points[pts_offset];
    float y = key_points[pts_offset + 1];
    float z = key_points[pts_offset + 2];

    int mat_offset = (batch_idx * num_cams + c) * 16;
    const float* mat = lidar2img + mat_offset;

    // 矩阵乘法 [4,4] * [x, y, z, 1]^T
    float p_x = mat[0] * x + mat[1] * y + mat[2] * z + mat[3];
    float p_y = mat[4] * x + mat[5] * y + mat[6] * z + mat[7];
    float p_z = mat[8] * x + mat[9] * y + mat[10] * z + mat[11];

    if (p_z > 1e-5f) {
        float inv_z = 1.0f / p_z;
        float u = p_x * inv_z;
        float v = p_y * inv_z;

        int wh_offset = (batch_idx * num_cams + c) * 2;
        loc_w = u / image_wh[wh_offset];
        loc_h = v / image_wh[wh_offset + 1];
    } else {
        loc_w = -1.0f;
        loc_h = -1.0f;
    }
}

// 3D 坐标投影辅助函数 (FP16)
__device__ inline void project_point_3d_to_2d_half(
    const __half* key_points,    // [BS*Q, P, 3]
    const __half* lidar2img,     // [BS, C, 4, 4]
    const __half* image_wh,      // [BS, C, 2]
    int batch_idx,
    int global_anchor_idx,
    int p,
    int c,
    int num_pts,
    int num_cams,
    float& loc_w,
    float& loc_h)
{
    int pts_offset = (global_anchor_idx * num_pts + p) * 3;
    float x = __half2float(key_points[pts_offset]);
    float y = __half2float(key_points[pts_offset + 1]);
    float z = __half2float(key_points[pts_offset + 2]);

    int mat_offset = (batch_idx * num_cams + c) * 16;
    const __half* mat = lidar2img + mat_offset;

    float p_x = __half2float(mat[0]) * x + __half2float(mat[1]) * y + __half2float(mat[2]) * z + __half2float(mat[3]);
    float p_y = __half2float(mat[4]) * x + __half2float(mat[5]) * y + __half2float(mat[6]) * z + __half2float(mat[7]);
    float p_z = __half2float(mat[8]) * x + __half2float(mat[9]) * y + __half2float(mat[10]) * z + __half2float(mat[11]);

    if (p_z > 1e-5f) {
        float inv_z = 1.0f / p_z;
        float u = p_x * inv_z;
        float v = p_y * inv_z;

        int wh_offset = (batch_idx * num_cams + c) * 2;
        loc_w = u / __half2float(image_wh[wh_offset]);
        loc_h = v / __half2float(image_wh[wh_offset + 1]);
    } else {
        loc_w = -1.0f;
        loc_h = -1.0f;
    }
}

// INT8 版本的 bilinear 采样
__device__ float thomas_bilinear_sampling_int8(const int8_t*& bottom_data,
                                              float scale,
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
        v1 = static_cast<float>(bottom_data[ptr1]) * scale;
    }
    float v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
    {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = static_cast<float>(bottom_data[ptr2]) * scale;
    }
    float v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
    {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = static_cast<float>(bottom_data[ptr3]) * scale;
    }
    float v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = static_cast<float>(bottom_data[ptr4]) * scale;
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

// 优化后的Gather模式Kernel，消除atomicAdd
__global__ void thomas_deformable_aggregation_kernel_gather(
    const int num_outputs,         // batch * num_anchors * num_embeds
    __half* output,                // Output (FP16)
    const __half* mc_ms_feat,      // Input Features (FP16)
    const int* spatial_shape,
    const int* scale_start_index,
    const __half* key_points,      // 3D 采样点 [BS*Q, P, 3] (FP16)
    const __half* lidar2img,       // 投影矩阵 [BS, C, 4, 4] (FP16)
    const __half* image_wh,        // 图像宽高 [BS, C, 2] (FP16)
    const __half* weights,         // Attention Weights (FP16)
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups)
{
    // 每个线程处理一个 (Batch, Anchor, Channel)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_outputs) return;

    // 解析索引
    int channel_idx = idx % num_embeds;
    int tmp = idx / num_embeds;
    int anchor_idx = tmp % num_anchors;
    int batch_idx = tmp / num_anchors;

    // 权重分组索引
    int group_idx = channel_idx / (num_embeds / num_groups);
    int global_anchor_idx = batch_idx * num_anchors + anchor_idx;

    float res = 0.0f;

    // 循环聚合 (Gather Loop)
    for (int p = 0; p < num_pts; ++p) {
        for (int c = 0; c < num_cams; ++c) {
            // 集成 3D 投影逻辑 (FP16)
            float loc_w, loc_h;
            project_point_3d_to_2d_half(key_points, lidar2img, image_wh, batch_idx, global_anchor_idx, p, c, num_pts, num_cams, loc_w, loc_h);

            // 边界检查优化：尽早剪枝
            if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
                for (int s = 0; s < num_scale; ++s) {
                    // 权重读取
                    // weights layout: [batch, anchors, pts, cams, scales, groups]
                    int weight_offset = ((((global_anchor_idx * num_pts + p) * num_cams + c) * num_scale + s) * num_groups + group_idx);
                    float weight = __half2float(weights[weight_offset]);
                    
                    if (fabsf(weight) < 1e-6f) continue; // 权重极小时跳过采样

                    // 空间尺寸
                    int cam_scale_idx = c * num_scale + s;
                    int h = spatial_shape[cam_scale_idx << 1];
                    int w = spatial_shape[(cam_scale_idx << 1) + 1];

                    // 坐标转换
                    float h_im = loc_h * h - 0.5f;
                    float w_im = loc_w * w - 0.5f;

                    // 特征值偏移
                    int value_offset = (batch_idx * num_feat + scale_start_index[cam_scale_idx]) * num_embeds + channel_idx;

                    // 双线性采样
                    __half sampled_val = thomas_bilinear_sampling_half(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset);
                    
                    res += __half2float(sampled_val) * weight;
                }
            }
        }
    }

    output[idx] = __float2half(res);
}

// 优化后的Gather模式Kernel（FP32版本）
__global__ void thomas_deformable_aggregation_kernel_gather_fp32(
    const int num_outputs,         // batch * num_anchors * num_embeds
    float* output,                 // Output (FP32)
    const float* mc_ms_feat,       // Input Features (FP32)
    const int* spatial_shape,
    const int* scale_start_index,
    const float* key_points,       // 3D 采样点 [BS*Q, P, 3]
    const float* lidar2img,        // 投影矩阵 [BS, C, 4, 4]
    const float* image_wh,         // 图像宽高 [BS, C, 2]
    const float* weights,          // Attention Weights (FP32)
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups)
{
    // 每个线程处理一个 (Batch, Anchor, Channel)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_outputs) return;

    // 解析索引
    int channel_idx = idx % num_embeds;
    int tmp = idx / num_embeds;
    int anchor_idx = tmp % num_anchors;
    int batch_idx = tmp / num_anchors;

    // 权重分组索引
    int group_idx = channel_idx / (num_embeds / num_groups);
    int global_anchor_idx = batch_idx * num_anchors + anchor_idx;

    float res = 0.0f;

    // 循环聚合 (Gather Loop)
    for (int p = 0; p < num_pts; ++p) {
        for (int c = 0; c < num_cams; ++c) {
            // 集成 3D 投影逻辑，直接计算 loc_w 和 loc_h
            float loc_w, loc_h;
            project_point_3d_to_2d(key_points, lidar2img, image_wh, batch_idx, global_anchor_idx, p, c, num_pts, num_cams, loc_w, loc_h);

            // 边界检查优化：尽早剪枝
            if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
                for (int s = 0; s < num_scale; ++s) {
                    // 权重读取
                    int weight_offset = ((((global_anchor_idx * num_pts + p) * num_cams + c) * num_scale + s) * num_groups + group_idx);
                    float weight = weights[weight_offset];
                    
                    if (fabsf(weight) < 1e-6f) continue;

                    // 空间尺寸
                    int cam_scale_idx = c * num_scale + s;
                    int h = spatial_shape[cam_scale_idx << 1];
                    int w = spatial_shape[(cam_scale_idx << 1) + 1];

                    // 坐标转换
                    float h_im = loc_h * h - 0.5f;
                    float w_im = loc_w * w - 0.5f;

                    // 特征值偏移
                    int value_offset = (batch_idx * num_feat + scale_start_index[cam_scale_idx]) * num_embeds + channel_idx;

                    // 双线性采样 (FP32)
                    float sampled_val = thomas_bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset);
                    
                    res += sampled_val * weight;
                }
            }
        }
    }

    output[idx] = res;
}

// 优化后的Gather模式Kernel（INT8 Value + FP32 Loc/Weights）
__global__ void thomas_deformable_aggregation_kernel_gather_int8(
    const int num_outputs,         // batch * num_anchors * num_embeds
    float* output,                 // Output (FP32)
    const int8_t* mc_ms_feat,      // Input Features (INT8)
    float value_scale,             // Dequantization scale
    const int* spatial_shape,
    const int* scale_start_index,
    const float* key_points,       // 3D 采样点 [BS*Q, P, 3]
    const float* lidar2img,        // 投影矩阵 [BS, C, 4, 4]
    const float* image_wh,         // 图像宽高 [BS, C, 2]
    const float* weights,          // Attention Weights (FP32)
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups)
{
    // 每个线程处理一个 (Batch, Anchor, Channel)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_outputs) return;

    // 解析索引
    int channel_idx = idx % num_embeds;
    int tmp = idx / num_embeds;
    int anchor_idx = tmp % num_anchors;
    int batch_idx = tmp / num_anchors;

    // 权重分组索引
    int group_idx = channel_idx / (num_embeds / num_groups);
    int global_anchor_idx = batch_idx * num_anchors + anchor_idx;

    float res = 0.0f;

    // 循环聚合 (Gather Loop)
    for (int p = 0; p < num_pts; ++p) {
        for (int c = 0; c < num_cams; ++c) {
            // 集成 3D 投影逻辑 (FP32)
            float loc_w, loc_h;
            project_point_3d_to_2d(key_points, lidar2img, image_wh, batch_idx, global_anchor_idx, p, c, num_pts, num_cams, loc_w, loc_h);

            // 边界检查优化：尽早剪枝
            if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
                for (int s = 0; s < num_scale; ++s) {
                    // 权重读取
                    int weight_offset = ((((global_anchor_idx * num_pts + p) * num_cams + c) * num_scale + s) * num_groups + group_idx);
                    float weight = weights[weight_offset];
                    
                    if (fabsf(weight) < 1e-6f) continue; // 权重极小时跳过采样

                    // 空间尺寸
                    int cam_scale_idx = c * num_scale + s;
                    int h = spatial_shape[cam_scale_idx << 1];
                    int w = spatial_shape[(cam_scale_idx << 1) + 1];

                    // 坐标转换
                    float h_im = loc_h * h - 0.5f;
                    float w_im = loc_w * w - 0.5f;

                    // 特征值偏移
                    int value_offset = (batch_idx * num_feat + scale_start_index[cam_scale_idx]) * num_embeds + channel_idx;

                    // 双线性采样 (INT8 with dequantization)
                    float sampled_val = thomas_bilinear_sampling_int8(mc_ms_feat, value_scale, h, w, num_embeds, h_im, w_im, value_offset);
                    
                    res += sampled_val * weight;
                }
            }
        }
    }

    output[idx] = res;
}

// 优化后的Gather模式Kernel（混合精度：FP16 Value + FP32 Loc/Weights）
__global__ void thomas_deformable_aggregation_kernel_gather_mixed(
    const int num_outputs,         // batch * num_anchors * num_embeds
    __half* output,                // Output (FP16)
    const __half* mc_ms_feat,      // Input Features (FP16)
    const int* spatial_shape,
    const int* scale_start_index,
    const float* key_points,       // 3D 采样点 [BS*Q, P, 3] (FP32 input)
    const float* lidar2img,        // 投影矩阵 [BS, C, 4, 4] (FP32 input)
    const float* image_wh,         // 图像宽高 [BS, C, 2] (FP32 input)
    const float* weights,          // Attention Weights (FP32)
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups)
{
    // 每个线程处理一个 (Batch, Anchor, Channel)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_outputs) return;

    // 解析索引
    int channel_idx = idx % num_embeds;
    int tmp = idx / num_embeds;
    int anchor_idx = tmp % num_anchors;
    int batch_idx = tmp / num_anchors;

    // 权重分组索引
    int group_idx = channel_idx / (num_embeds / num_groups);
    int global_anchor_idx = batch_idx * num_anchors + anchor_idx;

    float res = 0.0f;

    // 循环聚合 (Gather Loop)
    for (int p = 0; p < num_pts; ++p) {
        for (int c = 0; c < num_cams; ++c) {
            // 集成 3D 投影逻辑 (FP32)
            float loc_w, loc_h;
            project_point_3d_to_2d(key_points, lidar2img, image_wh, batch_idx, global_anchor_idx, p, c, num_pts, num_cams, loc_w, loc_h);

            // 边界检查优化：尽早剪枝
            if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
                for (int s = 0; s < num_scale; ++s) {
                    // 权重读取
                    int weight_offset = ((((global_anchor_idx * num_pts + p) * num_cams + c) * num_scale + s) * num_groups + group_idx);
                    // 使用FP32读取权重
                    float weight = weights[weight_offset];
                    
                    if (fabsf(weight) < 1e-6f) continue; // 权重极小时跳过采样

                    // 空间尺寸
                    int cam_scale_idx = c * num_scale + s;
                    int h = spatial_shape[cam_scale_idx << 1];
                    int w = spatial_shape[(cam_scale_idx << 1) + 1];

                    // 坐标转换
                    float h_im = loc_h * h - 0.5f;
                    float w_im = loc_w * w - 0.5f;

                    // 特征值偏移
                    int value_offset = (batch_idx * num_feat + scale_start_index[cam_scale_idx]) * num_embeds + channel_idx;

                    // 双线性采样 (FP16 Feature)
                    __half sampled_val = thomas_bilinear_sampling_half(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset);
                    
                    res += __half2float(sampled_val) * weight;
                }
            }
        }
    }

    output[idx] = __float2half(res);
}

// // 优化后的Gather模式Kernel（混合精度：int8 Value + FP32 Loc/Weights）
// __global__ void deformable_aggregation_kernel_gather_int8(
//     const int num_outputs,         // batch * num_anchors * num_embeds
//     int8_t* output,                // Output (INT8)
//     const int8_t* mc_ms_feat,      // Input Features (INT8)
//     const int* spatial_shape,
//     const int* scale_start_index,
//     const float* sample_location,  // Sampling Locations (FP32)
//     const float* weights,          // Attention Weights (FP32)
//     int batch_size,
//     int num_cams,
//     int num_feat,
//     int num_embeds,
//     int num_scale,
//     int num_anchors,
//     int num_pts,
//     int num_groups)
// {
//     // 每个线程处理一个 (Batch, Anchor, Channel)
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= num_outputs) return;

//     // 解析索引
//     int channel_idx = idx % num_embeds;
//     int tmp = idx / num_embeds;
//     int anchor_idx = tmp % num_anchors;
//     int batch_idx = tmp / num_anchors;

//     // 权重分组索引
//     int group_idx = channel_idx / (num_embeds / num_groups);
//     int global_anchor_idx = batch_idx * num_anchors + anchor_idx;

//     float res = 0.0f;

//     // 循环聚合 (Gather Loop)
//     for (int p = 0; p < num_pts; ++p) {
//         for (int c = 0; c < num_cams; ++c) {
//             // 计算 Location 偏移
//             int loc_offset = ((global_anchor_idx * num_pts + p) * num_cams + c) << 1;
            
//             // 使用FP32读取坐标
//             float loc_w = sample_location[loc_offset];
//             float loc_h = sample_location[loc_offset + 1];

//             // 边界检查优化：尽早剪枝
//             if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
//                 for (int s = 0; s < num_scale; ++s) {
//                     // 权重读取
//                     int weight_offset = ((((global_anchor_idx * num_pts + p) * num_cams + c) * num_scale + s) * num_groups + group_idx);
//                     // 使用FP32读取权重
//                     float weight = weights[weight_offset];
                    
//                     if (fabsf(weight) < 1e-6f) continue; // 权重极小时跳过采样

//                     // 空间尺寸
//                     int cam_scale_idx = c * num_scale + s;
//                     int h = spatial_shape[cam_scale_idx << 1];
//                     int w = spatial_shape[(cam_scale_idx << 1) + 1];

//                     // 坐标转换
//                     float h_im = loc_h * h - 0.5f;
//                     float w_im = loc_w * w - 0.5f;

//                     // 特征值偏移
//                     int value_offset = (batch_idx * num_feat + scale_start_index[cam_scale_idx]) * num_embeds + channel_idx;

//                     // 双线性采样 (FP16 Feature)
//                     __half sampled_val = thomas_bilinear_sampling_half(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset);
                    
//                     res += __half2float(sampled_val) * weight;
//                 }
//             }
//         }
//     }

//     output[idx] = __float2half(res);
// }

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
    if (idx >= num_kernels) return;

    const float weight = *(weights + idx / (num_embeds / num_groups));
    
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

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    const int max_read_offset = batch_size * num_feat * num_embeds;
    if (value_offset < 0 || value_offset >= max_read_offset) return;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    const int out_idx = anchor_index * num_embeds + channel_index;
    const int max_write_idx = batch_size * num_anchors * num_embeds;
    if (out_idx < 0 || out_idx >= max_write_idx) return;

    atomicAdd(
        output + out_idx,
        thomas_bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight
    );
}

// INT8 Kernel
__global__ void thomas_deformable_aggregation_kernel_int8(
    const int num_kernels,
    float* output,
    const int8_t* mc_ms_feat,
    float value_scale,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
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
    if (idx >= num_kernels) return;

    const float weight = *(weights + idx / (num_embeds / num_groups));
    
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

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    const int max_read_offset = batch_size * num_feat * num_embeds;
    if (value_offset < 0 || value_offset >= max_read_offset) return;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    const int out_idx = anchor_index * num_embeds + channel_index;
    const int max_write_idx = batch_size * num_anchors * num_embeds;
    if (out_idx < 0 || out_idx >= max_write_idx) return;

    atomicAdd(
        output + out_idx,
        thomas_bilinear_sampling_int8(mc_ms_feat, value_scale, h, w, num_embeds, h_im, w_im, value_offset) * weight
    );
}


// FP16版本的kernel
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
    if (idx >= num_kernels) return;

    const __half weight = *(weights + idx / (num_embeds / num_groups));
    
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

    const float loc_w = __half2float(sample_location[loc_offset]);
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = __half2float(sample_location[loc_offset + 1]);
    if (loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    const int max_read_offset = batch_size * num_feat * num_embeds;
    if (value_offset < 0 || value_offset >= max_read_offset) return;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    const int out_idx = anchor_index * num_embeds + channel_index;
    const int max_write_idx = batch_size * num_anchors * num_embeds;
    if (out_idx < 0 || out_idx >= max_write_idx) return;

    const __half sampled_half = thomas_bilinear_sampling_half(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset);
    
    const float sampled_val = __half2float(sampled_half);
    const float weight_val = __half2float(weight);
    const float result = sampled_val * weight_val;
    
    atomicAdd(temp_output + out_idx, result);
}

// 混合精度版本的聚合kernel
__global__ void thomas_deformable_aggregation_kernel_mixed(
    const int num_kernels,
    __half* output,
    const __half* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    float* temp_output,
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
    if (idx >= num_kernels) return;

    const float weight = weights[idx / (num_embeds / num_groups)];
    
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

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;
    
    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    const int max_read_offset = batch_size * num_feat * num_embeds;
    if (value_offset < 0 || value_offset >= max_read_offset) return;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    const int out_idx = anchor_index * num_embeds + channel_index;
    const int max_write_idx = batch_size * num_anchors * num_embeds;
    if (out_idx < 0 || out_idx >= max_write_idx) return;

    const __half sampled_half = thomas_bilinear_sampling_half(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset);
    
    const float sampled_val = __half2float(sampled_half);
    const float result = sampled_val * weight;
    
    atomicAdd(temp_output + out_idx, result);
}

// FP16版本的转换kernel
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
                                       const float* key_points,
                                       const float* lidar2img,
                                       const float* image_wh,
                                       const float* attnWeight,
                                       float* output,
                                       int batch_size,
                                       int num_cams,
                                       int num_feat,
                                       int num_embeds,
                                       int num_scale,
                                       int num_anchors,
                                       int num_pts,
                                       int num_groups)
{
    // Gather模式：线程数为输出元素个数
    const int num_outputs = batch_size * num_anchors * num_embeds;
    cudaError_t err = cudaSuccess;

    const int threadsPerBlock = 256;
    const int blocks = (num_outputs + threadsPerBlock - 1) / threadsPerBlock;

    thomas_deformable_aggregation_kernel_gather_fp32<<<blocks, threadsPerBlock, 0, stream>>>(
        num_outputs,
        output,
        value,
        spatialShapes,
        levelStartIndex,
        key_points,
        lidar2img,
        image_wh,
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
    if (err != cudaSuccess) return -1;

    return 0;
}

int thomas_deform_attn_cuda_forward_half(cudaStream_t stream,
                                       const __half* value,
                                       const int* spatialShapes,
                                       const int* levelStartIndex,
                                       const __half* key_points,
                                       const __half* lidar2img,
                                       const __half* image_wh,
                                       const __half* attnWeight,
                                       __half* output,
                                       float* workspace,
                                       int batch_size,
                                       int num_cams,
                                       int num_feat,
                                       int num_embeds,
                                       int num_scale,
                                       int num_anchors,
                                       int num_pts,
                                       int num_groups)
{
    // 线程总数变为输出尺寸 (Gather模式)
    const int num_outputs = batch_size * num_anchors * num_embeds;
    cudaError_t err = cudaSuccess;

    const int threadsPerBlock = 256;
    const int blocks = (num_outputs + threadsPerBlock - 1) / threadsPerBlock;

    thomas_deformable_aggregation_kernel_gather<<<blocks, threadsPerBlock, 0, stream>>>(
        num_outputs,
        output,
        value,
        spatialShapes,
        levelStartIndex,
        key_points,
        lidar2img,
        image_wh,
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
    if (err != cudaSuccess) return -1;

    return 0;
}

int thomas_deform_attn_cuda_forward_mixed(cudaStream_t stream,
                                         const __half* value,
                                         const int* spatialShapes,
                                         const int* levelStartIndex,
                                         const float* key_points,
                                         const float* lidar2img,
                                         const float* image_wh,
                                         const float* attnWeight,
                                         __half* output,
                                         float* workspace,
                                         int batch_size,
                                         int num_cams,
                                         int num_feat,
                                         int num_embeds,
                                         int num_scale,
                                         int num_anchors,
                                         int num_pts,
                                         int num_groups)
{
    // 线程总数变为输出尺寸 (Gather模式)
    const int num_outputs = batch_size * num_anchors * num_embeds;
    cudaError_t err = cudaSuccess;

    const int threadsPerBlock = 256;
    const int blocks = (num_outputs + threadsPerBlock - 1) / threadsPerBlock;

    thomas_deformable_aggregation_kernel_gather_mixed<<<blocks, threadsPerBlock, 0, stream>>>(
        num_outputs,
        output,
        value,
        spatialShapes,
        levelStartIndex,
        key_points,
        lidar2img,
        image_wh,
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
    if (err != cudaSuccess) return -1;

    return 0;
}

int thomas_deform_attn_cuda_forward_int8(cudaStream_t stream,
                                         const int8_t* value,
                                         float value_scale,
                                         const int* spatialShapes,
                                         const int* levelStartIndex,
                                         const float* key_points,
                                         const float* lidar2img,
                                         const float* image_wh,
                                         const float* attnWeight,
                                         float* output,
                                         int batch_size,
                                         int num_cams,
                                         int num_feat,
                                         int num_embeds,
                                         int num_scale,
                                         int num_anchors,
                                         int num_pts,
                                         int num_groups)
{
    // 使用GATHER模式：线程数为输出元素个数
    const int num_outputs = batch_size * num_anchors * num_embeds;
    cudaError_t err = cudaSuccess;

    const int threadsPerBlock = 256;
    const int blocks = (num_outputs + threadsPerBlock - 1) / threadsPerBlock;

    thomas_deformable_aggregation_kernel_gather_int8<<<blocks, threadsPerBlock, 0, stream>>>(
        num_outputs,
        output,
        value,
        value_scale,
        spatialShapes,
        levelStartIndex,
        key_points,
        lidar2img,
        image_wh,
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
    if (err != cudaSuccess) return -1;

    return 0;
}
