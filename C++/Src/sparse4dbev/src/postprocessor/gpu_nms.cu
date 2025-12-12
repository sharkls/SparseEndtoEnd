#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../common/cuda_utils_templates.hpp"
#include "../common/common_types.hpp"

namespace sparse4d {
namespace bev {

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <typename T>
__global__ void decode_and_filter_kernel(
    const T* anchor,
    const T* cls_score,
    const T* quality_score,
    const int32_t* track_ids,
    int num_objects,
    int num_classes,
    int anchor_dim,
    float conf_thresh,
    BoundingBox3D* out_boxes,
    int* valid_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;

    // 1. Class Score (Sigmoid & Max)
    float max_score = -1e10f;
    int best_label = 0;
    
    for (int c = 0; c < num_classes; ++c) {
        float s = common::val_to_float(cls_score[idx * num_classes + c]);
        s = sigmoid_f(s);
        if (s > max_score) {
            max_score = s;
            best_label = c;
        }
    }

    // 2. Quality Score (Centerness)
    // Quality is [num_objects, 2], index 0 is centerness
    float centerness = common::val_to_float(quality_score[idx * 2 + 0]);
    centerness = sigmoid_f(centerness);

    // 3. Final Confidence
    float confidence = max_score * centerness;

    if (confidence <= conf_thresh) return;

    // 4. Decode Box
    // anchor: [x, y, z, log(w), log(l), log(h), sin, cos, vx, vy, vz]
    int off = idx * anchor_dim;
    float x = common::val_to_float(anchor[off + 0]);
    float y = common::val_to_float(anchor[off + 1]);
    float z = common::val_to_float(anchor[off + 2]);
    float w = expf(common::val_to_float(anchor[off + 3]));
    float l = expf(common::val_to_float(anchor[off + 4]));
    float h = expf(common::val_to_float(anchor[off + 5]));
    float sin_y = common::val_to_float(anchor[off + 6]);
    float cos_y = common::val_to_float(anchor[off + 7]);
    float yaw = atan2f(sin_y, cos_y);

    int tid = (track_ids) ? track_ids[idx] : -1;

    BoundingBox3D box;
    box.x = x; box.y = y; box.z = z;
    box.w = w; box.l = l; box.h = h;
    box.yaw = yaw;
    box.confidence = confidence;
    box.label = best_label;
    box.index = idx;
    box.track_id = tid;

    int out_idx = atomicAdd(valid_count, 1);
    out_boxes[out_idx] = box;
}

__device__ float iou_bev_2d(const BoundingBox3D& b1, const BoundingBox3D& b2) {
    // Simple center distance check first for speed
    float dist = sqrtf((b1.x - b2.x)*(b1.x - b2.x) + (b1.y - b2.y)*(b1.y - b2.y));
    float max_dim = fmaxf(fmaxf(b1.l, b1.w), fmaxf(b2.l, b2.w));
    if (dist > max_dim) return 0.0f; // No overlap possible

    float inter_l = fmaxf(0.0f, fminf(b1.x + b1.l/2, b2.x + b2.l/2) - fmaxf(b1.x - b1.l/2, b2.x - b2.l/2));
    float inter_w = fmaxf(0.0f, fminf(b1.y + b1.w/2, b2.y + b2.w/2) - fmaxf(b1.y - b1.w/2, b2.y - b2.w/2));
    float inter_area = inter_l * inter_w;
    float union_area = b1.l * b1.w + b2.l * b2.w - inter_area;
    
    return (union_area > 1e-6f) ? (inter_area / union_area) : 0.0f;
}

__global__ void nms_kernel(
    const BoundingBox3D* sorted_boxes,
    int* suppressed,
    int num_boxes,
    float iou_thresh
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    if (suppressed[idx]) return;

    const BoundingBox3D& cur_box = sorted_boxes[idx];
    
    for (int i = 0; i < idx; ++i) {
        if (suppressed[i]) continue;
        
        const BoundingBox3D& higher_box = sorted_boxes[i];
        if (cur_box.label == higher_box.label) {
            float iou = iou_bev_2d(cur_box, higher_box);
            if (iou > iou_thresh) {
                suppressed[idx] = 1;
                return;
            }
        }
    }
}

__global__ void collect_nms_result(
    const BoundingBox3D* sorted_boxes,
    const int* suppressed,
    int num_boxes,
    BoundingBox3D* out_boxes,
    int* out_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    if (!suppressed[idx]) {
        int out_idx = atomicAdd(out_count, 1);
        out_boxes[out_idx] = sorted_boxes[idx];
    }
}

template <typename T>
void launch_decode_filter(
    const T* pred_anchor,
    const T* pred_class_score,
    const T* pred_quality_score,
    const int32_t* pred_track_ids,
    int num_objects,
    int num_classes,
    int anchor_dim,
    float confidence_thresh,
    cudaStream_t stream,
    BoundingBox3D* d_all_boxes,
    int* d_valid_count
) {
    cudaMemsetAsync(d_valid_count, 0, sizeof(int), stream);
    int block_size = 256;
    int grid_size = (num_objects + block_size - 1) / block_size;
    
    decode_and_filter_kernel<T><<<grid_size, block_size, 0, stream>>>(
        pred_anchor, pred_class_score, pred_quality_score, pred_track_ids,
        num_objects, num_classes, anchor_dim, confidence_thresh,
        d_all_boxes, d_valid_count
    );
}

void launch_nms_collect(
    const BoundingBox3D* d_temp_boxes,
    int* d_suppressed,
    int nms_input_num,
    float nms_thresh,
    cudaStream_t stream,
    BoundingBox3D* d_output_boxes,
    int* d_output_count
) {
    cudaMemsetAsync(d_output_count, 0, sizeof(int), stream);
    cudaMemsetAsync(d_suppressed, 0, nms_input_num * sizeof(int), stream);

    int block_size = 256;
    int grid_size = (nms_input_num + block_size - 1) / block_size;

    nms_kernel<<<grid_size, block_size, 0, stream>>>(d_temp_boxes, d_suppressed, nms_input_num, nms_thresh);
    collect_nms_result<<<grid_size, block_size, 0, stream>>>(d_temp_boxes, d_suppressed, nms_input_num, d_output_boxes, d_output_count);
}

// Explicit Instantiations
template void launch_decode_filter<float>(const float*, const float*, const float*, const int32_t*, int, int, int, float, cudaStream_t, BoundingBox3D*, int*);
template void launch_decode_filter<half>(const half*, const half*, const half*, const int32_t*, int, int, int, float, cudaStream_t, BoundingBox3D*, int*);

} // namespace bev
} // namespace sparse4d
