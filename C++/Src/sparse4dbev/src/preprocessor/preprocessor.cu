#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "../common/cuda_utils_templates.hpp"

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
    // 3D Grid: x=width, y=height, z=camera_id
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t cam_id = blockIdx.z;

    if (x >= net_w || y >= net_h) return;

    // Output index: [cam, c, h, w] (NCHW) or [cam, h, w, c] (NHWC)?
    // Sparse4D typically expects NCHW (num_cams, 3, H, W)
    // Destination offset
    uint32_t dst_offset = cam_id * 3 * net_h * net_w + 
                          0 * net_h * net_w + // C=0 (R)
                          y * net_w + x;
    uint32_t dst_stride = net_h * net_w; // Stride between channels

    // Coordinate mapping (Inverse mapping)
    // dst(y, x) -> src(src_y, src_x)
    // crop first, then resize
    // effective_src_y = (y / resize_ratio) + crop_h
    // effective_src_x = (x / resize_ratio) + crop_w
    
    float src_y_f = (float)y / resize_ratio + (float)crop_h;
    float src_x_f = (float)x / resize_ratio + (float)crop_w;

    // Bilinear Interpolation
    int x0 = (int)src_y_f;
    int y0 = (int)src_x_f; // Note: raw image storage usually HWC or CHW. 
                           // Assuming Raw is HWC (OpenCV style) or CHW?
                           // Most camera drivers give packed RGB/BGR (HWC).
                           // Let's assume standard packed RGB HWC for raw input.
                           
    // Raw Offset helper: [cam, y, x, c]
    // Raw stride
    uint32_t raw_cam_stride = raw_h * raw_w * 3;
    uint32_t raw_line_stride = raw_w * 3;

    // Check bounds
    if (x0 < 0 || x0 >= raw_h - 1 || y0 < 0 || y0 >= raw_w - 1) {
        // Zero padding for out of bounds
        dst[dst_offset + 0 * dst_stride] = common::float_to_val<T>(0.0f);
        dst[dst_offset + 1 * dst_stride] = common::float_to_val<T>(0.0f);
        dst[dst_offset + 2 * dst_stride] = common::float_to_val<T>(0.0f);
        return;
    }

    // Interpolation weights
    float dy = src_y_f - x0;
    float dx = src_x_f - y0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w10 = dx * (1.0f - dy);
    float w01 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    // Mean and Std for normalization (typical ImageNet)
    // mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    const float mean[3] = {123.675f, 116.28f, 103.53f};
    const float std[3] = {58.395f, 57.12f, 57.375f};

    // Process 3 channels
    for (int c = 0; c < 3; ++c) {
        // Raw Input Indices (HWC)
        // cam_offset + y*row_stride + x*3 + c
        const uint8_t* cam_base = src + cam_id * raw_cam_stride;
        
        uint8_t p00 = cam_base[x0 * raw_line_stride + y0 * 3 + c];
        uint8_t p10 = cam_base[x0 * raw_line_stride + (y0 + 1) * 3 + c];
        uint8_t p01 = cam_base[(x0 + 1) * raw_line_stride + y0 * 3 + c];
        uint8_t p11 = cam_base[(x0 + 1) * raw_line_stride + (y0 + 1) * 3 + c];

        float val = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
        
        // Normalize
        val = (val - mean[c]) / std[c];

        // Store to dst (NCHW planar)
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
    dim3 block(32, 32);
    dim3 grid(
        (net_w + block.x - 1) / block.x,
        (net_h + block.y - 1) / block.y,
        num_cams
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

