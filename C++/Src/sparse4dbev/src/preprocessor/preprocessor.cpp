#include "preprocessor.hpp"
#include "log.h"
#include <cuda_runtime.h>
#include <cstring>

namespace sparse4d {
namespace bev {

// CUDA Kernel declaration (to be implemented in .cu)
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
);

template <typename T>
PreprocessorImpl<T>::PreprocessorImpl() = default;

template <typename T>
PreprocessorImpl<T>::~PreprocessorImpl() {
    if (d_raw_) cudaFree(d_raw_);
    if (h_pinned_) cudaFreeHost(h_pinned_);
}

template <typename T>
bool PreprocessorImpl<T>::init(const TaskConfig& config) {
    config_ = config;
    const auto& params = config.preprocessor_params();
    
    num_cams_ = params.num_cams();
    per_cam_size_ = params.raw_img_c() * params.raw_img_h() * params.raw_img_w();
    expected_size_raw_img_ = num_cams_ * per_cam_size_;
    
    return true;
}

template <typename T>
bool PreprocessorImpl<T>::forward(const CTimeMatchSrcData* src_data, 
                                  const cudaStream_t& stream,
                                  CudaWrapper<T>& output_buffer) {
    if (!src_data) return false;
    
    const auto& videos = src_data->vecVideoSrcData();
    if (videos.size() != num_cams_) {
        LOG(ERROR) << "Camera count mismatch";
        return false;
    }

    // 1. Prepare Pinned Memory
    if (!h_pinned_ || h_pinned_bytes_ < expected_size_raw_img_) {
        if (h_pinned_) cudaFreeHost(h_pinned_);
        cudaError_t err = cudaHostAlloc((void**)&h_pinned_, expected_size_raw_img_, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            LOG(ERROR) << "CUDA Host Alloc failed: " << cudaGetErrorString(err);
            return false;
        }
        h_pinned_bytes_ = expected_size_raw_img_;
    }

    // 2. Copy to Pinned Memory
    size_t offset = 0;
    for (const auto& v : videos) {
        const auto& buf = v.vecImageBuf();
        if (buf.size() != per_cam_size_) {
            LOG(ERROR) << "Image buffer size mismatch";
            return false;
        }
        memcpy(h_pinned_ + offset, buf.data(), buf.size());
        offset += buf.size();
    }

    // 3. Prepare Device Memory
    if (!d_raw_ || d_raw_bytes_ < expected_size_raw_img_) {
        if (d_raw_) cudaFree(d_raw_);
        cudaError_t err = cudaMalloc((void**)&d_raw_, expected_size_raw_img_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "CUDA Malloc failed: " << cudaGetErrorString(err);
            return false;
        }
        d_raw_bytes_ = expected_size_raw_img_;
    }

    // 4. H2D Copy
    cudaMemcpyAsync(d_raw_, h_pinned_, expected_size_raw_img_, cudaMemcpyHostToDevice, stream);

    // 5. Launch Kernel
    const auto& params = config_.preprocessor_params();
    launch_img_preprocess<T>(
        d_raw_,
        num_cams_,
        params.raw_img_c(), params.raw_img_h(), params.raw_img_w(),
        params.model_input_img_h(), params.model_input_img_w(),
        params.resize_ratio(),
        params.crop_height(), params.crop_width(),
        stream,
        output_buffer.getCudaPtr()
    );

    return true;
}

// Instantiate
template class PreprocessorImpl<float>;
template class PreprocessorImpl<half>;

} // namespace bev
} // namespace sparse4d

