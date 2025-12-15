#ifndef __SPARSE4D_BEV_INSTANCE_BANK_HPP__
#define __SPARSE4D_BEV_INSTANCE_BANK_HPP__

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <Eigen/Dense>
#include "Sparse4D_conf.pb.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"

namespace sparse4d {
namespace bev {

template <typename T>
class InstanceBank {
public:
    InstanceBank();
    ~InstanceBank();

    bool init(const TaskConfig& config);
    
    // Project cached anchors to current time t
    bool project_anchors(const double current_timestamp, 
                         const Eigen::Matrix4d& lidar_to_global, 
                         cudaStream_t stream);

    // Update bank with new detections
    // topk_indices: [1, k] indices of top confidence queries
    bool update(const CudaWrapper<T>& pred_features,
                const CudaWrapper<T>& pred_anchors,
                const CudaWrapper<T>& pred_confidence,
                const CudaWrapper<int32_t>& pred_track_ids,
                cudaStream_t stream);

    // Getters for Head Inputs
    CudaWrapper<T>& get_temp_features() { return temp_features_; }
    CudaWrapper<T>& get_temp_anchors() { return temp_anchors_; }
    CudaWrapper<int32_t>& get_track_ids() { return track_ids_; }
    CudaWrapper<int32_t>& get_mask() { return mask_; }
    
    // Provide time interval (dt) to network
    CudaWrapper<T>& get_time_interval() { return device_time_interval_; }
    
    bool is_first_frame() const { return is_first_frame_; }

    void reset();

private:
    TaskConfig config_;
    
    // State
    bool is_first_frame_ = true;
    double last_timestamp_ = 0.0;
    Eigen::Matrix4d last_lidar_to_global_;
    float dt_ = 0.0f;

    // Buffers (Cached Tensors for Next Frame)
    // Size: [1, topk_queries, dims]
    CudaWrapper<T> temp_features_;
    CudaWrapper<T> temp_anchors_;
    CudaWrapper<int32_t> track_ids_; 
    
    // Mask indicates which queries are valid history (0 or 1)
    CudaWrapper<int32_t> mask_;
    
    // Helper buffer for confidence decay update if needed
    CudaWrapper<T> cached_confidence_;

    // Time interval tensor (shape [1])
    CudaWrapper<T> device_time_interval_;
    
    // Default Anchors (K-Means) and Features
    CudaWrapper<T> init_features_;
    CudaWrapper<T> init_anchors_;
    
    // Helper for Track ID generation
    CudaWrapper<int32_t> device_prev_id_; // Stores the max ID from previous frame
    int32_t host_prev_id_ = 0;
    
public:
    // Compute/Generate Track IDs
    void compute_track_ids(CudaWrapper<int32_t>& pred_track_ids, cudaStream_t stream);
};

} // namespace bev
} // namespace sparse4d

#endif // __SPARSE4D_BEV_INSTANCE_BANK_HPP__

