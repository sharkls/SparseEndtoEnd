#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../common/cuda_utils_templates.hpp"
#include "../common/common_types.hpp"

namespace sparse4d {
namespace bev {

template <typename T>
__global__ void update_bank_kernel(
    const T* src_feat, const T* src_anchor, const T* src_conf, const int32_t* src_ids,
    T* dst_feat, T* dst_anchor, T* dst_conf, int32_t* dst_ids,
    int num_queries, int feat_dim, int anchor_dim,
    int topk,
    const int* topk_indices,
    float conf_decay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= topk) return;

    int src_idx = topk_indices[idx];
    
    // Copy Feature
    for (int i = 0; i < feat_dim; ++i) {
        dst_feat[idx * feat_dim + i] = src_feat[src_idx * feat_dim + i];
    }
    
    // Copy Anchor
    for (int i = 0; i < anchor_dim; ++i) {
        dst_anchor[idx * anchor_dim + i] = src_anchor[src_idx * anchor_dim + i];
    }
    
    // Update Confidence with decay
    float score = common::val_to_float(src_conf[src_idx]);
    // Apply decay? 
    // Sparse4D v1/v2 logic: usually just keep score or decay it.
    // Let's assume decay is done outside or simply keep it.
    // If conf_decay < 1.0, apply it.
    // score *= conf_decay; // Optional
    dst_conf[idx] = common::float_to_val<T>(score); 
    
    // Copy Track ID
    dst_ids[idx] = (src_ids) ? src_ids[src_idx] : -1;
}

template <typename T>
__global__ void anchor_projection_kernel(
    T* anchors,
    int num_anchors,
    int anchor_dim,
    float dt,
    const float* transform_matrix // 4x4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    int off = idx * anchor_dim;
    
    // Read current pos
    float x = common::val_to_float(anchors[off + 0]);
    float y = common::val_to_float(anchors[off + 1]);
    float z = common::val_to_float(anchors[off + 2]);
    
    // Apply Transform: P_t = T * P_{t-1}
    float x_new = transform_matrix[0]*x + transform_matrix[1]*y + transform_matrix[2]*z + transform_matrix[3];
    float y_new = transform_matrix[4]*x + transform_matrix[5]*y + transform_matrix[6]*z + transform_matrix[7];
    float z_new = transform_matrix[8]*x + transform_matrix[9]*y + transform_matrix[10]*z + transform_matrix[11];
    
    anchors[off + 0] = common::float_to_val<T>(x_new);
    anchors[off + 1] = common::float_to_val<T>(y_new);
    anchors[off + 2] = common::float_to_val<T>(z_new);
    
    // Velocity projection: x = x + vx * dt
    float vx = common::val_to_float(anchors[off + 8]);
    float vy = common::val_to_float(anchors[off + 9]);
    float vz = common::val_to_float(anchors[off + 10]);
    
    float x_final = x_new + vx * dt;
    float y_final = y_new + vy * dt;
    float z_final = z_new + vz * dt;
    
    anchors[off + 0] = common::float_to_val<T>(x_final);
    anchors[off + 1] = common::float_to_val<T>(y_final);
    anchors[off + 2] = common::float_to_val<T>(z_final);
}

// Launcher functions
template <typename T>
void launch_anchor_projection(
    T* anchors,
    int num_anchors,
    int anchor_dim,
    float dt,
    const float* transform_matrix,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (num_anchors + block - 1) / block;
    anchor_projection_kernel<T><<<grid, block, 0, stream>>>(anchors, num_anchors, anchor_dim, dt, transform_matrix);
}

template <typename T>
void launch_update_bank(
    const T* src_feat, const T* src_anchor, const T* src_conf, const int32_t* src_ids,
    T* dst_feat, T* dst_anchor, T* dst_conf, int32_t* dst_ids,
    int num_queries, int feat_dim, int anchor_dim,
    int topk,
    const int* topk_indices,
    float conf_decay,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (topk + block - 1) / block;
    update_bank_kernel<T><<<grid, block, 0, stream>>>(
        src_feat, src_anchor, src_conf, src_ids,
        dst_feat, dst_anchor, dst_conf, dst_ids,
        num_queries, feat_dim, anchor_dim,
        topk, topk_indices, conf_decay
    );
}

// Explicit Instantiations
template void launch_anchor_projection<float>(float*, int, int, float, const float*, cudaStream_t);
template void launch_anchor_projection<half>(half*, int, int, float, const float*, cudaStream_t);

template void launch_update_bank<float>(const float*, const float*, const float*, const int32_t*, float*, float*, float*, int32_t*, int, int, int, int, const int*, float, cudaStream_t);
template void launch_update_bank<half>(const half*, const half*, const half*, const int32_t*, half*, half*, half*, int32_t*, int, int, int, int, const int*, float, cudaStream_t);

} // namespace bev
} // namespace sparse4d
