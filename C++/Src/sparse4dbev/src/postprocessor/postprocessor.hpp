#ifndef __SPARSE4D_BEV_POSTPROCESSOR_HPP__
#define __SPARSE4D_BEV_POSTPROCESSOR_HPP__

#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "Sparse4D_conf.pb.h"
#include "CAlgResult.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"
#include "../common/common_types.hpp"

namespace sparse4d {
namespace bev {

template <typename T>
class Postprocessor {
public:
    virtual ~Postprocessor() = default;
    
    virtual bool init(const TaskConfig& config) = 0;
    
    // Inputs are raw tensors from Head output (Device Memory)
    virtual bool forward(const CudaWrapper<T>& pred_anchor,
                         const CudaWrapper<T>& pred_class_score,
                         const CudaWrapper<T>& pred_quality_score,
                         const CudaWrapper<int32_t>& pred_track_ids,
                         const cudaStream_t& stream,
                         CAlgResult& result) = 0;
};

template <typename T>
class PostprocessorImpl : public Postprocessor<T> {
public:
    PostprocessorImpl();
    ~PostprocessorImpl() override;
    
    bool init(const TaskConfig& config) override;
    
    bool forward(const CudaWrapper<T>& pred_anchor,
                 const CudaWrapper<T>& pred_class_score,
                 const CudaWrapper<T>& pred_quality_score,
                 const CudaWrapper<int32_t>& pred_track_ids,
                 const cudaStream_t& stream,
                 CAlgResult& result) override;

private:
    TaskConfig config_;
    
    // NMS Params
    bool use_gpu_nms_ = true;
    float nms_threshold_ = 0.5f;
    float confidence_threshold_ = 0.1f;
    int max_output_boxes_ = 300;
    int num_classes_ = 10;
    int anchor_dim_ = 11;

    // GPU Internal Buffers
    BoundingBox3D* d_boxes_all_ = nullptr;     // All decoded boxes before filter
    int* d_valid_count_ = nullptr;             // Count of valid boxes after threshold
    BoundingBox3D* d_boxes_sorted_ = nullptr;  // Top-K boxes for NMS
    int* d_suppressed_ = nullptr;              // NMS suppression mask
    BoundingBox3D* d_output_boxes_ = nullptr;  // Final output boxes
    int* d_output_count_ = nullptr;            // Final output count
    
    size_t max_objects_capacity_ = 900; // Typically 900 queries
};

} // namespace bev
} // namespace sparse4d

#endif // __SPARSE4D_BEV_POSTPROCESSOR_HPP__

