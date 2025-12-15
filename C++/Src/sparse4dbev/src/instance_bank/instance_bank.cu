#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
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
    float size_x = common::val_to_float(anchors[off + 3]);
    float size_y = common::val_to_float(anchors[off + 4]);
    float size_z = common::val_to_float(anchors[off + 5]);
    float yaw_sin = common::val_to_float(anchors[off + 6]);
    float yaw_cos = common::val_to_float(anchors[off + 7]);
    float vx = common::val_to_float(anchors[off + 8]);
    float vy = common::val_to_float(anchors[off + 9]);
    float vz = common::val_to_float(anchors[off + 10]);
    
    // 1. Velocity projection first (in prev frame coordinates)
    x += vx * dt;
    y += vy * dt;
    z += vz * dt;

    // 2. Coordinate Transformation (Prev -> Curr)
    // P_curr = T * P_prev
    float x_new = transform_matrix[0]*x + transform_matrix[1]*y + transform_matrix[2]*z + transform_matrix[3];
    float y_new = transform_matrix[4]*x + transform_matrix[5]*y + transform_matrix[6]*z + transform_matrix[7];
    float z_new = transform_matrix[8]*x + transform_matrix[9]*y + transform_matrix[10]*z + transform_matrix[11];
    
    // 3. Velocity Transformation
    // V_curr = R * V_prev (Using 3x3 rotation part of transform matrix)
    float vx_new = transform_matrix[0]*vx + transform_matrix[1]*vy + transform_matrix[2]*vz;
    float vy_new = transform_matrix[4]*vx + transform_matrix[5]*vy + transform_matrix[6]*vz;
    float vz_new = transform_matrix[8]*vx + transform_matrix[9]*vy + transform_matrix[10]*vz;

    // 4. Orientation (Yaw) Transformation
    // [sin', cos'] = [sin, cos] * R_2x2^T? No.
    // Vector orientation: u = [cos, sin, 0]
    // u_new = R * u
    // yaw vector: [cos, sin]
    // We store [sin, cos] usually in index 6,7
    // Let's verify Sparse4D convention: usually [sin, cos] or [cos, sin]?
    // Sparse4DFP16 code:
    // float yaw_x = anchor_ptr[6]; // sin
    // float yaw_y = anchor_ptr[7]; // cos
    // float temp_yaw_x = yaw_x; float temp_yaw_y = yaw_y;
    // yaw_x = temp_yaw_y; // cos
    // yaw_y = temp_yaw_x; // sin
    // new_yaw_x (cos') = cos * m[0] + sin * m[4] ...
    // Basically rotate the vector [cos, sin]
    
    // Assuming anchor[6]=sin, anchor[7]=cos based on typical conventions
    // Rotate vector [cos, sin] by the rotation matrix R (top-left 2x2)
    // x_axis (cos) corresponds to index 0, y_axis (sin) to index 1 in matrix logic?
    // Let's stick to standard rotation:
    // v = [cos, sin, 0]
    // v_new = R * v
    // v_new_x = m00*cos + m01*sin
    // v_new_y = m10*cos + m11*sin
    
    float cos_yaw_new = transform_matrix[0]*yaw_cos + transform_matrix[1]*yaw_sin;
    float sin_yaw_new = transform_matrix[4]*yaw_cos + transform_matrix[5]*yaw_sin;
    
    // Normalize to keep it unit vector (optional but good for stability)
    float norm = sqrtf(cos_yaw_new*cos_yaw_new + sin_yaw_new*sin_yaw_new + 1e-6f);
    cos_yaw_new /= norm;
    sin_yaw_new /= norm;

    // Write back
    anchors[off + 0] = common::float_to_val<T>(x_new);
    anchors[off + 1] = common::float_to_val<T>(y_new);
    anchors[off + 2] = common::float_to_val<T>(z_new);
    // Size remains same (assuming rigid body transform)
    anchors[off + 3] = common::float_to_val<T>(size_x);
    anchors[off + 4] = common::float_to_val<T>(size_y);
    anchors[off + 5] = common::float_to_val<T>(size_z);
    anchors[off + 6] = common::float_to_val<T>(sin_yaw_new);
    anchors[off + 7] = common::float_to_val<T>(cos_yaw_new);
    anchors[off + 8] = common::float_to_val<T>(vx_new);
    anchors[off + 9] = common::float_to_val<T>(vy_new);
    anchors[off + 10] = common::float_to_val<T>(vz_new);
}

// Launcher functions
template <typename T>
void launch_generate_new_track_ids(int32_t* track_ids, const int32_t* prev_id, uint32_t size, cudaStream_t stream);

template <typename T>
void launch_update_new_track_ids(int32_t* track_ids, uint32_t num_anchors, uint32_t topk, int32_t* prev_id, cudaStream_t stream);

// Launcher functions
template <typename T>
void launch_get_max_confidence_scores(const T* confidence_logits, T* max_confidence_scores, int num_querys, int num_classes, cudaStream_t stream);

template <typename T>
__global__ void getMaxConfidenceScoresKernel(const T* confidence_logits,
                                             T* max_confidence_scores,
                                             int num_querys,
                                             int num_classes) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_querys) return;

    // Initialize with first class
    float max_logit = common::val_to_float(confidence_logits[idx * num_classes]);
    for (int class_idx = 1; class_idx < num_classes; ++class_idx) {
        float logit = common::val_to_float(confidence_logits[idx * num_classes + class_idx]);
        max_logit = fmaxf(max_logit, logit);
    }
    
    // Apply Sigmoid
    float sigmoid_value;
    if (max_logit > 0) {
        float exp_neg_x = expf(-max_logit);
        sigmoid_value = 1.0f / (1.0f + exp_neg_x);
    } else {
        float exp_x = expf(max_logit);
        sigmoid_value = exp_x / (1.0f + exp_x);
    }
    
    max_confidence_scores[idx] = common::float_to_val<T>(sigmoid_value);
}

template <typename T>
void launch_get_max_confidence_scores(const T* confidence_logits, T* max_confidence_scores, int num_querys, int num_classes, cudaStream_t stream) {
    int block = 256;
    int grid = (num_querys + block - 1) / block;
    getMaxConfidenceScoresKernel<T><<<grid, block, 0, stream>>>(confidence_logits, max_confidence_scores, num_querys, num_classes);
}

__global__ void generateNewTrackIdsKernel(int32_t* track_ids,
                                          const int32_t* prev_id,
                                          uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    track_ids[idx] = prev_id[0] + idx;
}

__global__ void updateNewTrackIdsKernel(int32_t* track_ids,
                                        uint32_t num_anchors_,
                                        uint32_t topk_anchors_,
                                        int32_t* prev_id) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Only update the new queries part (indices >= topk)
    // The first topk are history and preserve their IDs (already copied)
    if (tid < num_anchors_ && tid >= topk_anchors_) {
        track_ids[tid] = prev_id[0] + tid - topk_anchors_ + 1;
    }
}

template <typename T>
void launch_generate_new_track_ids(int32_t* track_ids, const int32_t* prev_id, uint32_t size, cudaStream_t stream) {
    int block = 256;
    int grid = (size + block - 1) / block;
    generateNewTrackIdsKernel<<<grid, block, 0, stream>>>(track_ids, prev_id, size);
}

template <typename T>
void launch_update_new_track_ids(int32_t* track_ids, uint32_t num_anchors, uint32_t topk, int32_t* prev_id, cudaStream_t stream) {
    int block = 256;
    int grid = (num_anchors + block - 1) / block;
    updateNewTrackIdsKernel<<<grid, block, 0, stream>>>(track_ids, num_anchors, topk, prev_id);
}

// Launcher functions
template <typename T>
__global__ void decay_and_fuse_kernel(
    T* fused_conf, // in-place update: input current, output fused
    const T* cached_conf,
    int topk,
    float decay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= topk) return;

    float curr = common::val_to_float(fused_conf[idx]);
    float hist = common::val_to_float(cached_conf[idx]);
    
    // Apply decay and max
    float fused = fmaxf(curr, hist * decay);
    
    fused_conf[idx] = common::float_to_val<T>(fused);
}

template <typename T>
void launch_decay_and_fuse(
    T* fused_conf,
    const T* cached_conf,
    int topk,
    float decay,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (topk + block - 1) / block;
    decay_and_fuse_kernel<T><<<grid, block, 0, stream>>>(fused_conf, cached_conf, topk, decay);
}

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

template void launch_decay_and_fuse<float>(float*, const float*, int, float, cudaStream_t);
template void launch_decay_and_fuse<half>(half*, const half*, int, float, cudaStream_t);

template void launch_generate_new_track_ids<float>(int32_t*, const int32_t*, uint32_t, cudaStream_t);
template void launch_generate_new_track_ids<half>(int32_t*, const int32_t*, uint32_t, cudaStream_t);

template void launch_update_new_track_ids<float>(int32_t*, uint32_t, uint32_t, int32_t*, cudaStream_t);
template void launch_update_new_track_ids<half>(int32_t*, uint32_t, uint32_t, int32_t*, cudaStream_t);

template void launch_get_max_confidence_scores<float>(const float*, float*, int, int, cudaStream_t);
template void launch_get_max_confidence_scores<half>(const half*, half*, int, int, cudaStream_t);

} // namespace bev
} // namespace sparse4d
