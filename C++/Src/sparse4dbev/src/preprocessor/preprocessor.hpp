#ifndef __SPARSE4D_BEV_PREPROCESSOR_HPP__
#define __SPARSE4D_BEV_PREPROCESSOR_HPP__

#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "Sparse4D_conf.pb.h"
#include "CTimeMatchSrcData.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"

namespace sparse4d {
namespace bev {

template <typename T>
class Preprocessor {
public:
    virtual ~Preprocessor() = default;
    
    virtual bool init(const TaskConfig& config) = 0;
    
    // Core function: uint8 (Host) -> T (Device)
    virtual bool forward(const CTimeMatchSrcData* src_data, 
                         const cudaStream_t& stream,
                         CudaWrapper<T>& output_buffer) = 0;
};

template <typename T>
class PreprocessorImpl : public Preprocessor<T> {
public:
    PreprocessorImpl();
    ~PreprocessorImpl() override;
    
    bool init(const TaskConfig& config) override;
    bool forward(const CTimeMatchSrcData* src_data, 
                 const cudaStream_t& stream,
                 CudaWrapper<T>& output_buffer) override;

private:
    TaskConfig config_;
    size_t expected_size_raw_img_ = 0;
    size_t per_cam_size_ = 0;
    uint32_t num_cams_ = 0;
    
    // Pinned memory for host buffering
    uint8_t* h_pinned_ = nullptr;
    size_t h_pinned_bytes_ = 0;
    
    // Device memory for raw input
    uint8_t* d_raw_ = nullptr;
    size_t d_raw_bytes_ = 0;
};

} // namespace bev
} // namespace sparse4d

#endif // __SPARSE4D_BEV_PREPROCESSOR_HPP__