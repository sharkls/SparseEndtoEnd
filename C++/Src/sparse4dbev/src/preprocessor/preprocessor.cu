#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include "../common/cuda_utils_templates.hpp"

#define R_MEAN 0.485F
#define G_MEAN 0.456F
#define B_MEAN 0.406F
#define R_STD 0.229F
#define G_STD 0.224F
#define B_STD 0.225F

#define DIVUP(a, b) ((a % b != 0) ? (a / b + 1) : (a / b))

namespace sparse4d {
namespace bev {

template <typename T>
__global__ void img_preprocess_kernel(
    const uint8_t* __restrict__ src,
    uint32_t raw_h, uint32_t raw_w,
    uint32_t net_h, uint32_t net_w,
    float resize_ratio,
    uint32_t crop_h, uint32_t crop_w,
    T* __restrict__ dst
) {
    // Sparse4DFP16 Grid Layout: 
    // Grid: (num_cams, H_blocks, W_blocks)
    // Block: (16, 16)
    
    const uint32_t cam_id = blockIdx.x;
    // Sparse4DFP16: dst_y = blockIdx.y * blockDim.x + threadIdx.x
    // blockDim.x = 16, blockDim.y = 16
    const uint32_t y = blockIdx.y * blockDim.x + threadIdx.x;
    // Sparse4DFP16: dst_x = blockIdx.z * blockDim.y + threadIdx.y
    const uint32_t x = blockIdx.z * blockDim.y + threadIdx.y;

    if (x >= net_w || y >= net_h) return;

    // Sparse4DFP16 Coordinate Mapping
    const float resize_ratio_x = static_cast<float>(raw_w) / static_cast<float>(floor(raw_w * resize_ratio));
    const float resize_ratio_y = static_cast<float>(raw_h) / static_cast<float>(floor(raw_h * resize_ratio));

    const float src_x = (x + crop_w + 0.5F) * resize_ratio_x - 0.5F;
    const float src_y = (y + crop_h + 0.5F) * resize_ratio_y - 0.5F;

    int32_t low_x = floor(src_x);
    int32_t low_y = floor(src_y);

    int32_t high_x = min(low_x + 1, (int32_t)raw_w - 1);
    int32_t high_y = min(low_y + 1, (int32_t)raw_h - 1);

    low_x = max(0, low_x);
    low_y = max(0, low_y);

    // Boundary check
    if (low_x >= raw_w || low_y >= raw_h || high_x >= raw_w || high_y >= raw_h) {
        uint32_t dst_offset = cam_id * 3 * net_h * net_w + y * net_w + x;
        uint32_t dst_stride = net_h * net_w;
        dst[dst_offset + 0 * dst_stride] = common::float_to_val<T>(0.0f);
        dst[dst_offset + 1 * dst_stride] = common::float_to_val<T>(0.0f);
        dst[dst_offset + 2 * dst_stride] = common::float_to_val<T>(0.0f);
        return;
    }

    const float ly = src_y - low_y;
    const float lx = src_x - low_x;
    const float hy = 1.0F - ly;
    const float hx = 1.0F - lx;

    const float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    // Output index: NCHW (num_cams, 3, H, W)
    uint32_t dst_offset = cam_id * 3 * net_h * net_w + y * net_w + x;
    uint32_t dst_stride = net_h * net_w;

    // Process 3 channels (RGB)
    // Raw Input Indices (CHW Planar)
    // src is [N, C, H, W]
    const uint8_t* cam_base = src + cam_id * (3 * raw_h * raw_w);
    
    const float means[3] = {R_MEAN, G_MEAN, B_MEAN};
    const float stds[3] = {R_STD, G_STD, B_STD};

    for (int c = 0; c < 3; ++c) {
        uint32_t channel_offset = c * raw_h * raw_w;
        
        // Value 1: low_y, low_x
        float p00 = (float)cam_base[channel_offset + low_y * raw_w + low_x];
        // Value 2: low_y, high_x
        float p01 = (float)cam_base[channel_offset + low_y * raw_w + high_x];
        // Value 3: high_y, low_x
        float p10 = (float)cam_base[channel_offset + high_y * raw_w + low_x];
        // Value 4: high_y, high_x
        float p11 = (float)cam_base[channel_offset + high_y * raw_w + high_x];

        float val = p00 * w1 + p01 * w2 + p10 * w3 + p11 * w4;

        // Normalization: (val/255 - mean) / std
        val = val / 255.0F;
        val = (val - means[c]) / stds[c];

        dst[dst_offset + c * dst_stride] = common::float_to_val<T>(val);
    }
}

template <typename T>
void launch_img_preprocess(
    const uint8_t* d_in,
    uint32_t num_cams,
    uint32_t raw_c, uint32_t raw_h, uint32_t raw_w,
    uint32_t net_h, uint32_t net_w,
    float resize_ratio,
    uint32_t crop_h, uint32_t crop_w,
    cudaStream_t stream,
    T* d_out
) {
    // Sparse4DFP16 Block/Grid Config
    const uint32_t thread_num = 16;
    dim3 block(thread_num, thread_num);
    dim3 grid(
        num_cams, 
        DIVUP(net_h, thread_num), 
        DIVUP(net_w, thread_num)
    );

    img_preprocess_kernel<T><<<grid, block, 0, stream>>>(
        d_in, raw_h, raw_w, net_h, net_w, 
        resize_ratio, crop_h, crop_w, d_out
    );
}

// Explicit Instantiation
template void launch_img_preprocess<float>(
    const uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t, 
    uint32_t, uint32_t, float, uint32_t, uint32_t, 
    cudaStream_t, float*
);

template void launch_img_preprocess<half>(
    const uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t, 
    uint32_t, uint32_t, float, uint32_t, uint32_t, 
    cudaStream_t, half*
);

} // namespace bev
} // namespace sparse4d
