#include "instance_bank.hpp"
#include "log.h"
#include <algorithm>
#include <fstream>

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
    int num_queries = params.num_querys(); // 900
    
    // Allocate Buffers
    // NOTE: These buffers must be FULL SIZE (num_queries) because they are used as Engine Inputs
    temp_features_.allocate(num_queries * feat_dims);
    temp_anchors_.allocate(num_queries * dims);
    track_ids_.allocate(num_queries);
    mask_.allocate(1); // Single scalar mask? Or per-query? Usually global mask for "is_first_frame".
    cached_confidence_.allocate(topk);
    device_time_interval_.allocate(1);
    
    // Init buffers for New Queries
    init_features_.allocate(num_queries * feat_dims);
    init_anchors_.allocate(num_queries * dims);
    
    // Load Init Data from File
    // We assume Sparse4DImpl has already loaded these into its own buffers?
    // No, Sparse4DImpl has init_anchor_ and init_instance_feature_ but they are separate.
    // We should load them here too, or pass them in.
    // For simplicity, let's reload or assume caller will set them?
    // Actually, let's load them here.
    
    std::string anchor_path = config.instance_bank_params().instance_bank_anchor_path();
    if (!anchor_path.empty()) {
        std::ifstream ifs(anchor_path, std::ios::binary | std::ios::ate);
        if (ifs) {
            std::streamsize size = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            size_t num = size / sizeof(float);
            std::vector<float> anchors_f(num);
            ifs.read(reinterpret_cast<char*>(anchors_f.data()), size);
            
            std::vector<T> anchors_t(num);
            for(size_t i=0; i<num; ++i) anchors_t[i] = (T)anchors_f[i];
            
            init_anchors_.cudaMemUpdateWrap(anchors_t);
        }
    }
    
    std::string feat_path = config.instance_bank_params().instance_bank_feature_path();
    if (!feat_path.empty()) {
        // Load features if available
    } else {
        // Zero init
        init_features_.cudaMemSetWrap(T(0.0f));
    }
    
    // Alloc Prev ID
    device_prev_id_.allocate(1);
    
    reset();
    return true;
}

template <typename T>
void InstanceBank<T>::reset() {
    is_first_frame_ = true;
    mask_.cudaMemSetWrap(0); // 0 means no history
    track_ids_.cudaMemSetWrap(-1); // -1
    
    host_prev_id_ = 0;
    std::vector<int32_t> prev_id_vec = {0};
    device_prev_id_.cudaMemUpdateWrap(prev_id_vec);
    
    // Reset temp buffers to init state
    // Important: Copy init to temp so that the first frame (or reset state) has valid inputs
    // (Though Head1 uses init buffers directly, Head2 uses temp. Reset prepares for next usage.)
    int num_queries = config_.instance_bank_params().num_querys();
    int feat_dims = config_.model_cfg_params().embedfeat_dims();
    int anchor_dims = config_.instance_bank_params().query_dims();
    
    cudaMemcpy(temp_features_.getCudaPtr(), init_features_.getCudaPtr(), num_queries * feat_dims * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(temp_anchors_.getCudaPtr(), init_anchors_.getCudaPtr(), num_queries * anchor_dims * sizeof(T), cudaMemcpyDeviceToDevice);
}

template <typename T>
void launch_generate_new_track_ids(int32_t* track_ids, const int32_t* prev_id, uint32_t size, cudaStream_t stream);

template <typename T>
void launch_update_new_track_ids(int32_t* track_ids, uint32_t num_anchors, uint32_t topk, int32_t* prev_id, cudaStream_t stream);

template <typename T>
void InstanceBank<T>::compute_track_ids(CudaWrapper<int32_t>& pred_track_ids, cudaStream_t stream) {
    int num_queries = config_.instance_bank_params().num_querys();
    int topk = config_.instance_bank_params().topk_querys();
    
    if (is_first_frame_) {
        // Generate [0, 1, ..., N-1]
        launch_generate_new_track_ids<T>(
            pred_track_ids.getCudaPtr(),
            device_prev_id_.getCudaPtr(),
            num_queries,
            stream
        );
        
        // Update prev_id
        host_prev_id_ += (num_queries - 1); // Or num_queries? Usually max ID. Last ID is start + N - 1.
        // Sparse4DFP16 logic: prev_id_vec = {num_anchors_ - 1};
        // It sets prev_id to the MAX ID assigned.
        host_prev_id_ = num_queries - 1;
        
    } else {
        // Update New Queries [TopK...N-1]
        // The first TopK are already set by the Engine (passed through) or need to be preserved?
        // In Sparse4D, Engine output `pred_track_id` for history queries corresponds to the input track_id.
        // But we rely on the input `track_id` buffer having been set up correctly in `update`.
        // Wait, `InstanceBank::update` updates `track_ids_` (the member).
        // Then `project_anchors` (get) prepares `track_ids_` for the engine.
        // The engine outputs `pred_track_id`.
        // Does the engine propagate IDs? Most TRT engines for Sparse4D just pass through `track_id` input to output for the history part.
        // Or output is just index?
        // Assuming engine propagates: `pred_track_id` has valid history IDs.
        // We just need to update the new ones.
        
        launch_update_new_track_ids<T>(
            pred_track_ids.getCudaPtr(),
            num_queries,
            topk,
            device_prev_id_.getCudaPtr(),
            stream
        );
        
        // Update prev_id
        // new queries count = num - topk
        host_prev_id_ += (num_queries - topk);
    }
    
    // Sync prev_id to GPU for next frame
    std::vector<int32_t> prev_id_vec = {host_prev_id_};
    device_prev_id_.cudaMemUpdateWrapAsync(prev_id_vec, stream);
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
    int num_queries = config_.instance_bank_params().num_querys();
    int feat_dims = config_.model_cfg_params().embedfeat_dims();
    
    // 1. Reset Temp Buffers with Init (for New Queries part)
    // We copy the full Init buffer to Temp buffer first.
    // This ensures [TopK..N] has valid new query embeddings/anchors.
    // Optimization: Only copy the [TopK..N] part? Or full?
    // Copying full is safer and simpler.
    cudaMemcpyAsync(temp_anchors_.getCudaPtr(), init_anchors_.getCudaPtr(), num_queries * dims * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(temp_features_.getCudaPtr(), init_features_.getCudaPtr(), num_queries * feat_dims * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    
    // 2. Project Cached Anchors (History)
    // Note: We project `temp_anchors_` (which now has Init values).
    // BUT we should project the *History* anchors which are stored in `temp_anchors_[0..TopK]` from previous `update`?
    // Wait, `update` gathered TopK into `temp_[0..TopK]`.
    // So `temp_` ALREADY contains the history at the start (from previous frame's update).
    // So we should NOT overwrite `temp_` with `init_` completely!
    // We should only overwrite the `[TopK..N]` part with `init_`.
    
    // Correct Logic:
    // `temp_anchors_` currently holds: [History(TopK) from prev update, Garbage(TopK..N)].
    // We want: [ProjectedHistory(TopK), NewAnchors(TopK..N)].
    
    // Step A: Copy New Anchors to [TopK..N]
    size_t offset_anchors = topk * dims * sizeof(T);
    size_t size_anchors_new = (num_queries - topk) * dims * sizeof(T);
    cudaMemcpyAsync((uint8_t*)temp_anchors_.getCudaPtr() + offset_anchors, 
                    (uint8_t*)init_anchors_.getCudaPtr() + offset_anchors, // Assuming init has new anchors at the end
                    size_anchors_new, cudaMemcpyDeviceToDevice, stream);

    size_t offset_feats = topk * feat_dims * sizeof(T);
    size_t size_feats_new = (num_queries - topk) * feat_dims * sizeof(T);
    cudaMemcpyAsync((uint8_t*)temp_features_.getCudaPtr() + offset_feats, 
                    (uint8_t*)init_features_.getCudaPtr() + offset_feats,
                    size_feats_new, cudaMemcpyDeviceToDevice, stream);

    // Step B: Project History Anchors [0..TopK]
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

// Launcher functions
template <typename T>
void launch_decay_and_fuse(
    T* fused_conf,
    const T* cached_conf,
    int topk,
    float decay,
    cudaStream_t stream
);

template <typename T>
void launch_get_max_confidence_scores(const T* confidence_logits, T* max_confidence_scores, int num_querys, int num_classes, cudaStream_t stream);

template <typename T>
bool InstanceBank<T>::update(const CudaWrapper<T>& pred_features,
                             const CudaWrapper<T>& pred_anchors,
                             const CudaWrapper<T>& pred_class_scores, // Corrected name: input is [900, 10]
                             const CudaWrapper<int32_t>& pred_track_ids,
                             cudaStream_t stream) {
    int num_queries = config_.instance_bank_params().num_querys();
    int topk = config_.instance_bank_params().topk_querys();
    int feat_dim = config_.model_cfg_params().embedfeat_dims();
    int anchor_dim = config_.instance_bank_params().query_dims();
    float conf_decay = config_.instance_bank_params().confidence_decay();
    int num_classes = config_.model_cfg_params().num_classes();

    // 0. Get Max Confidence Scores (Reduce [900, 10] -> [900])
    // Use cached_confidence_ as temp buffer if size matches? 
    // cached_confidence_ is topk (600). We need 900.
    // So we need a temporary buffer of size 900.
    // Let's assume we can reuse or allocate.
    // Using a CudaWrapper on stack is safe (RAII).
    CudaWrapper<T> current_confidence(num_queries);
    
    launch_get_max_confidence_scores(
        pred_class_scores.getCudaPtr(),
        current_confidence.getCudaPtr(),
        num_queries,
        num_classes,
        stream
    );

    // 1. Temporal Fusion (Decay & Fuse)
    // Allocate temp fused confidence buffer
    CudaWrapper<T> fused_confidence(num_queries);
    cudaMemcpyAsync(fused_confidence.getCudaPtr(), current_confidence.getCudaPtr(), num_queries * sizeof(T), cudaMemcpyDeviceToDevice, stream);

    if (!is_first_frame_) {
        // Apply decay and max fusion for the first topk queries (history)
        launch_decay_and_fuse<T>(
            fused_confidence.getCudaPtr(), // In/Out
            cached_confidence_.getCudaPtr(), // History
            topk,
            conf_decay,
            stream
        );
    }

    // 2. Sort and Update
    // Copy fused confidence to Host for sorting
    std::vector<T> host_conf_T(num_queries);
    cudaMemcpyAsync(host_conf_T.data(), fused_confidence.getCudaPtr(), num_queries * sizeof(T), cudaMemcpyDeviceToHost, stream);
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

    // 3. Launch Update Kernel (Gather TopK to cache)
    launch_update_bank<T>(
        pred_features.getCudaPtr(),
        pred_anchors.getCudaPtr(),
        fused_confidence.getCudaPtr(), // Use fused confidence
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

