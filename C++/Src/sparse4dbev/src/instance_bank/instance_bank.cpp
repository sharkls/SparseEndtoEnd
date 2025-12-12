#include "instance_bank.hpp"
#include "log.h"
#include <algorithm>

namespace sparse4d {
namespace bev {

// Declare CUDA launcher functions
template <typename T>
void launch_anchor_projection(
    T* anchors,
    int num_anchors,
    int anchor_dim,
    float dt,
    const float* transform_matrix,
    cudaStream_t stream
);

template <typename T>
void launch_update_bank(
    const T* src_feat, const T* src_anchor, const T* src_conf, const int32_t* src_ids,
    T* dst_feat, T* dst_anchor, T* dst_conf, int32_t* dst_ids,
    int num_queries, int feat_dim, int anchor_dim,
    int topk,
    const int* topk_indices,
    float conf_decay,
    cudaStream_t stream
);

template <typename T>
InstanceBank<T>::InstanceBank() = default;

template <typename T>
InstanceBank<T>::~InstanceBank() = default;

template <typename T>
bool InstanceBank<T>::init(const TaskConfig& config) {
    config_ = config;
    const auto& params = config.instance_bank_params();
    
    int topk = params.topk_querys();
    int dims = params.query_dims();
    int feat_dims = config.model_cfg_params().embedfeat_dims(); // 256
    
    // Allocate Buffers
    temp_features_.allocate(topk * feat_dims);
    temp_anchors_.allocate(topk * dims);
    track_ids_.allocate(topk);
    mask_.allocate(1); // Single scalar mask? Or per-query? Usually global mask for "is_first_frame".
    cached_confidence_.allocate(topk);
    device_time_interval_.allocate(1);
    
    // Load K-Means Anchors (Fallback)
    // In real implement, load from file specified in config
    // For now, zero init or simple default
    // Ambiguity fix: cast 0 to float, then to T, or construct T
    temp_anchors_.cudaMemSetWrap(T(0.0f));
    temp_features_.cudaMemSetWrap(T(0.0f));
    
    reset();
    return true;
}

template <typename T>
void InstanceBank<T>::reset() {
    is_first_frame_ = true;
    mask_.cudaMemSetWrap(0); // 0 means no history
    track_ids_.cudaMemSetWrap(-1); // -1
}

template <typename T>
bool InstanceBank<T>::project_anchors(const double current_timestamp, 
                                      const Eigen::Matrix4d& lidar_to_global, 
                                      cudaStream_t stream) {
    if (is_first_frame_) {
        dt_ = 0.0f;
        // Update state
        last_timestamp_ = current_timestamp;
        last_lidar_to_global_ = lidar_to_global;
        
        int mask_val = 0;
        cudaMemcpyAsync(mask_.getCudaPtr(), &mask_val, sizeof(int), cudaMemcpyHostToDevice, stream);
        return true;
    }
    
    dt_ = (float)(current_timestamp - last_timestamp_);
    // Cap dt
    if (dt_ > config_.instance_bank_params().max_time_interval()) {
        dt_ = config_.instance_bank_params().max_time_interval();
    }
    
    // Compute Transform Matrix
    // P_t = Inv(L2G_t) * L2G_{t-1} * P_{t-1}
    Eigen::Matrix4d global_to_lidar_cur = lidar_to_global.inverse();
    Eigen::Matrix4d transform = global_to_lidar_cur * last_lidar_to_global_;
    
    // Upload Matrix
    float h_mat[16];
    for(int r=0; r<4; ++r)
        for(int c=0; c<4; ++c)
            h_mat[r*4+c] = (float)transform(r, c);
            
    float* d_mat;
    cudaMallocAsync(&d_mat, 16 * sizeof(float), stream);
    cudaMemcpyAsync(d_mat, h_mat, 16 * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // Launch Kernel
    int topk = config_.instance_bank_params().topk_querys();
    int dims = config_.instance_bank_params().query_dims();
    
    launch_anchor_projection<T>(
        temp_anchors_.getCudaPtr(),
        topk, dims, dt_, d_mat, stream
    );
    
    cudaFreeAsync(d_mat, stream);
    
    // Update Mask = 1
    int mask_val = 1;
    cudaMemcpyAsync(mask_.getCudaPtr(), &mask_val, sizeof(int), cudaMemcpyHostToDevice, stream);
    
    // Update Time Interval
    std::vector<T> dt_vec(1);
    if constexpr (std::is_same<T, float>::value) {
        dt_vec[0] = (float)dt_;
    } else {
        dt_vec[0] = __float2half((float)dt_);
    }
    device_time_interval_.cudaMemUpdateWrapAsync(dt_vec, stream);

    // Update state
    last_timestamp_ = current_timestamp;
    last_lidar_to_global_ = lidar_to_global;
    
    return true;
}

// Helper for host side sorting of indices
struct ConfIndex {
    float conf;
    int index;
    bool operator>(const ConfIndex& other) const {
        return conf > other.conf;
    }
};

template <typename T>
bool InstanceBank<T>::update(const CudaWrapper<T>& pred_features,
                             const CudaWrapper<T>& pred_anchors,
                             const CudaWrapper<T>& pred_confidence,
                             const CudaWrapper<int32_t>& pred_track_ids,
                             cudaStream_t stream) {
    int num_queries = pred_confidence.getSize();
    int topk = config_.instance_bank_params().topk_querys();
    int feat_dim = config_.model_cfg_params().embedfeat_dims();
    int anchor_dim = config_.instance_bank_params().query_dims();

    // 1. Copy confidence to Host
    // T might be half, we need to convert to float for sorting
    std::vector<T> host_conf_T(num_queries);
    cudaMemcpyAsync(host_conf_T.data(), pred_confidence.getCudaPtr(), num_queries * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // Sync to get data

    std::vector<ConfIndex> sorted_indices(num_queries);
    for (int i = 0; i < num_queries; ++i) {
        if constexpr (std::is_same<T, float>::value) {
            sorted_indices[i].conf = host_conf_T[i];
        } else {
            sorted_indices[i].conf = __half2float(host_conf_T[i]);
        }
        sorted_indices[i].index = i;
    }
    
    // Sort descending
    // We only need TopK
    std::partial_sort(sorted_indices.begin(), sorted_indices.begin() + topk, sorted_indices.end(), 
                      [](const ConfIndex& a, const ConfIndex& b) { return a.conf > b.conf; });
    
    std::vector<int> topk_indices(topk);
    for(int i=0; i<topk; ++i) topk_indices[i] = sorted_indices[i].index;
    
    // Copy indices to Device
    int* d_indices;
    cudaMallocAsync(&d_indices, topk * sizeof(int), stream);
    cudaMemcpyAsync(d_indices, topk_indices.data(), topk * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 2. Launch Update Kernel
    float conf_decay = config_.instance_bank_params().confidence_decay();
    
    launch_update_bank<T>(
        pred_features.getCudaPtr(),
        pred_anchors.getCudaPtr(),
        pred_confidence.getCudaPtr(),
        pred_track_ids.getCudaPtr(),
        temp_features_.getCudaPtr(),
        temp_anchors_.getCudaPtr(),
        cached_confidence_.getCudaPtr(),
        track_ids_.getCudaPtr(),
        num_queries, feat_dim, anchor_dim,
        topk, d_indices, conf_decay, stream
    );

    cudaFreeAsync(d_indices, stream);

    is_first_frame_ = false;
    return true;
}

// Instantiate
template class InstanceBank<float>;
template class InstanceBank<half>;

} // namespace bev
} // namespace sparse4d

