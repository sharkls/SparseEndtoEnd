#include "postprocessor.hpp"
#include "log.h"
#include <algorithm>

namespace sparse4d {
namespace bev {

// CUDA Kernel declaration
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
);

void launch_nms_collect(
    const BoundingBox3D* d_temp_boxes,
    int* d_suppressed,
    int nms_input_num,
    float nms_thresh,
    cudaStream_t stream,
    BoundingBox3D* d_output_boxes,
    int* d_output_count
);

template <typename T>
PostprocessorImpl<T>::PostprocessorImpl() = default;

template <typename T>
PostprocessorImpl<T>::~PostprocessorImpl() {
    if (d_boxes_all_) cudaFree(d_boxes_all_);
    if (d_valid_count_) cudaFree(d_valid_count_);
    if (d_boxes_sorted_) cudaFree(d_boxes_sorted_);
    if (d_suppressed_) cudaFree(d_suppressed_);
    if (d_output_boxes_) cudaFree(d_output_boxes_);
    if (d_output_count_) cudaFree(d_output_count_);
}

template <typename T>
bool PostprocessorImpl<T>::init(const TaskConfig& config) {
    config_ = config;
    const auto& params = config.postprocessor_params();
    const auto& model_params = config.model_cfg_params();
    
    use_gpu_nms_ = params.use_gpu_nms();
    nms_threshold_ = params.gpu_nms_threshold();
    confidence_threshold_ = params.post_process_threshold();
    max_output_boxes_ = params.max_output_boxes();
    num_classes_ = model_params.num_classes();
    
    // Pre-allocate GPU memory (Assuming max 2000 queries for safety, typically 900)
    max_objects_capacity_ = 2000; 
    
    cudaMalloc(&d_boxes_all_, max_objects_capacity_ * sizeof(BoundingBox3D));
    cudaMalloc(&d_valid_count_, sizeof(int));
    cudaMalloc(&d_boxes_sorted_, max_output_boxes_ * sizeof(BoundingBox3D)); // Temp buffer for NMS
    cudaMalloc(&d_suppressed_, max_output_boxes_ * sizeof(int));
    cudaMalloc(&d_output_boxes_, max_output_boxes_ * sizeof(BoundingBox3D));
    cudaMalloc(&d_output_count_, sizeof(int));
    
    return true;
}

template <typename T>
bool PostprocessorImpl<T>::forward(const CudaWrapper<T>& pred_anchor,
                                   const CudaWrapper<T>& pred_class_score,
                                   const CudaWrapper<T>& pred_quality_score,
                                   const CudaWrapper<int32_t>& pred_track_ids,
                                   const cudaStream_t& stream,
                                   CAlgResult& result) {
    
    int num_objects = pred_anchor.getSize() / anchor_dim_;
    
    if (num_objects > max_objects_capacity_) {
        LOG(ERROR) << "Too many objects for pre-allocated buffer: " << num_objects;
        return false;
    }

    // 1. Decode and Filter (GPU)
    launch_decode_filter<T>(
        pred_anchor.getCudaPtr(),
        pred_class_score.getCudaPtr(),
        pred_quality_score.getCudaPtr(),
        pred_track_ids.getCudaPtr(),
        num_objects,
        num_classes_,
        anchor_dim_,
        confidence_threshold_,
        stream,
        d_boxes_all_,
        d_valid_count_
    );

    // 2. Sort (Host)
    int valid_num = 0;
    cudaMemcpyAsync(&valid_num, d_valid_count_, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int count = 0;
    if (valid_num > 0) {
        std::vector<BoundingBox3D> host_boxes(valid_num);
        cudaMemcpyAsync(host_boxes.data(), d_boxes_all_, valid_num * sizeof(BoundingBox3D), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Sort desc
        std::sort(host_boxes.begin(), host_boxes.end(), [](const BoundingBox3D& a, const BoundingBox3D& b) {
            return a.confidence > b.confidence;
        });

        // Copy back top-K for NMS
        int nms_input_num = (valid_num > max_output_boxes_ * 2) ? max_output_boxes_ * 2 : valid_num;
        cudaMemcpyAsync(d_boxes_sorted_, host_boxes.data(), nms_input_num * sizeof(BoundingBox3D), cudaMemcpyHostToDevice, stream);

        // 3. NMS (GPU)
        launch_nms_collect(
            d_boxes_sorted_,
            d_suppressed_,
            nms_input_num,
            nms_threshold_,
            stream,
            d_output_boxes_,
            d_output_count_
        );

        cudaMemcpyAsync(&count, d_output_count_, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        if (count > max_output_boxes_) count = max_output_boxes_;
    }

    if (count < 0) return false; // Should not happen

    // Retrieve results to Host
    std::vector<BoundingBox3D> host_boxes_final(count);
    if (count > 0) {
        cudaMemcpyAsync(host_boxes_final.data(), d_output_boxes_, count * sizeof(BoundingBox3D), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // Fill CAlgResult
    CFrameResult frame;
    std::vector<CObjectResult> detections;
    detections.reserve(count);
    
    for (const auto& box : host_boxes_final) {
        CObjectResult obj;
        obj.x(box.x);
        obj.y(box.y);
        obj.z(box.z);
        obj.l(box.l);
        obj.w(box.w);
        obj.h(box.h);
        obj.yaw(box.yaw);
        obj.confidence(box.confidence);
        obj.label(static_cast<uint8_t>(box.label));
        obj.trackid(box.track_id);
        detections.push_back(std::move(obj));
    }
    
    frame.vecObjectResult(std::move(detections));
    std::vector<CFrameResult> frames;
    frames.push_back(std::move(frame));
    result.vecFrameResult(std::move(frames));

    return true;
}

// Instantiate
template class PostprocessorImpl<float>;
template class PostprocessorImpl<half>;

} // namespace bev
} // namespace sparse4d

