#ifndef __SPARSE4D_IMPL_HPP__
#define __SPARSE4D_IMPL_HPP__

#include "Sparse4D_conf.pb.h"
#include "CTimeMatchSrcData.h"
#include "CAlgResult.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <memory>

// Include component headers
#include "preprocessor/preprocessor.hpp"
#include "engine/engine_wrapper.hpp"
#include "instance_bank/instance_bank.hpp"
#include "postprocessor/postprocessor.hpp"
#include "../../../Include/Common/Utils/CudaWrapper.h"

namespace sparse4d {
namespace bev {

// Abstract base class for PIMPL
class ISparse4DImpl {
public:
    virtual ~ISparse4DImpl() = default;
    
    virtual bool init(const TaskConfig& config, const std::string& exe_path) = 0;
    
    virtual void forward(const CTimeMatchSrcData* src_data, CAlgResult& result) = 0;
    
    virtual bool update_params(void* param) = 0;
};

// Templated implementation class
template <typename T>
class Sparse4DImpl : public ISparse4DImpl {
public:
    Sparse4DImpl();
    ~Sparse4DImpl() override;
    
    bool init(const TaskConfig& config, const std::string& exe_path) override;
    void forward(const CTimeMatchSrcData* src_data, CAlgResult& result) override;
    bool update_params(void* param) override;

private:
    bool init_memory();
    bool init_aux_data();

private:
    TaskConfig config_;
    cudaStream_t stream_ = nullptr;
    
    // Components
    std::unique_ptr<Preprocessor<T>> preprocessor_;
    std::unique_ptr<InstanceBank<T>> instance_bank_;
    std::unique_ptr<Postprocessor<T>> postprocessor_;
    
    std::unique_ptr<EngineWrapper> backbone_;
    std::unique_ptr<EngineWrapper> head1_;
    std::unique_ptr<EngineWrapper> head2_;

    // Tensor Buffers
    // Use CudaWrapper for automatic memory management
    
    // 1. Preprocessor Output / Backbone Input
    CudaWrapper<T> input_imgs_; 
    
    // 2. Backbone Output
    CudaWrapper<T> feature_maps_; 
    
    // 3. Head Outputs (Shared for Head1 and Head2 outputs to save memory if possible)
    CudaWrapper<T> pred_instance_feature_;
    CudaWrapper<T> pred_anchor_;
    CudaWrapper<T> pred_class_score_;
    CudaWrapper<T> pred_quality_score_;
    CudaWrapper<int32_t> pred_track_id_;

    // 4. Auxiliary Inputs (Loaded from file or consts)
    CudaWrapper<int32_t> spatial_shapes_;
    CudaWrapper<int32_t> level_start_index_;
    CudaWrapper<T> lidar2img_; // Projection matrix
    CudaWrapper<T> image_wh_;
    
    // Initial State (for Head1 or First Frame)
    CudaWrapper<T> init_anchor_;           // K-Means anchors
    CudaWrapper<T> init_instance_feature_; // Zero initialized

    // Bindings vectors for Engines
    std::vector<void*> backbone_bindings_;
    std::vector<void*> head1_bindings_;
    std::vector<void*> head2_bindings_;
};

} // namespace bev
} // namespace sparse4d

#endif // __SPARSE4D_IMPL_HPP__
