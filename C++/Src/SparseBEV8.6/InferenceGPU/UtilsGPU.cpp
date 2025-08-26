#include "UtilsGPU.h"
#include "UtilsGPU_kernels.cuh"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// CUDA kernel函数声明
extern "C" {
    // Top-K相关kernel
    __global__ void getTopkInstanceKernel(const float* confidence,
                                         const float* instance_feature,
                                         const float* anchor,
                                         float* temp_topk_confidence,
                                         float* temp_topk_instance_feature,
                                         float* temp_topk_anchors,
                                         int32_t* temp_topk_index,
                                         uint32_t num_querys,
                                         uint32_t query_dims,
                                         uint32_t embedfeat_dims,
                                         uint32_t num_topk_querys);

    __global__ void getTopKScoresKernel(const float* confidence,
                                        float* topk_confidence,
                                        uint32_t* topk_indices,
                                        uint32_t anchor_nums,
                                        uint32_t k);

    __global__ void getTopKTrackIDKernel(const float* confidence,
                                         const int* track_ids,
                                         int* topk_track_ids,
                                         uint32_t anchor_nums,
                                         uint32_t k);

    // 置信度处理相关kernel
    __global__ void applyConfidenceDecayKernel(float* output_confidence,
                                              const float* input_confidence,
                                              float decay_factor,
                                              uint32_t size);

    __global__ void fuseConfidenceKernel(float* new_confidence,
                                         const float* cached_confidence,
                                         uint32_t size);

    // Anchor投影相关kernel
    __global__ void anchorProjectionKernel(float* temp_anchor,
                                           const float* transform_matrix,
                                           float time_interval,
                                           uint32_t topk_querys,
                                           uint32_t query_dims);

    // 置信度计算相关kernel
    __global__ void getMaxConfidenceScoresKernel(const float* confidence_logits,
                                                 float* max_confidence_scores,
                                                 uint32_t num_querys,
                                                 uint32_t class_nums);

    __global__ void applySigmoidKernel(float* logits, uint32_t size);

    // 跟踪ID生成相关kernel
    __global__ void generateNewTrackIdsKernel(int32_t* track_ids,
                                              int32_t* prev_id,
                                              uint32_t num_querys);

    // 后处理相关kernel
    __global__ void topKKernel(const float* topk_cls_scores_origin,
                               const uint32_t* topk_index,
                               const float* fusioned_scores,
                               const uint8_t* cls_ids,
                               const float* box_preds,
                               const int* track_ids,
                               float* topk_cls_scores,
                               float* topk_fusioned_scores,
                               uint8_t* topk_cls_ids,
                               float* topk_box_preds,
                               int* topk_track_ids,
                               uint32_t k,
                               float threshold,
                               uint32_t kmeans_anchor_dims,
                               uint32_t* actual_topk_out);
}

Status UtilsGPU::getTopkInstanceOnGPU(const CudaWrapper<float>& confidence,
                                      const CudaWrapper<float>& instance_feature,
                                      const CudaWrapper<float>& anchor,
                                      const uint32_t& num_querys,
                                      const uint32_t& query_dims,
                                      const uint32_t& embedfeat_dims,
                                      const uint32_t& num_topk_querys,
                                      CudaWrapper<float>& temp_topk_confidence,
                                      CudaWrapper<float>& temp_topk_instance_feature,
                                      CudaWrapper<float>& temp_topk_anchors,
                                      CudaWrapper<int32_t>& temp_topk_index,
                                      cudaStream_t stream) {
    
    // 计算CUDA kernel的grid和block大小
    dim3 blockSize(256);
    dim3 gridSize((num_topk_querys + blockSize.x - 1) / blockSize.x);
    
    // 调用CUDA kernel
    getTopkInstanceKernel<<<gridSize, blockSize, 0, stream>>>(
        confidence.getCudaPtr(),
        instance_feature.getCudaPtr(),
        anchor.getCudaPtr(),
        temp_topk_confidence.getCudaPtr(),
        temp_topk_instance_feature.getCudaPtr(),
        temp_topk_anchors.getCudaPtr(),
        temp_topk_index.getCudaPtr(),
        num_querys,
        query_dims,
        embedfeat_dims,
        num_topk_querys
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA kernel error in getTopkInstanceKernel: " << cudaGetErrorString(err);
        return Status::kError;
    }
    
    return Status::kSuccess;
}

Status UtilsGPU::getTopKScoresOnGPU(const CudaWrapper<float>& confidence,
                                    const uint32_t& anchor_nums,
                                    const uint32_t& k,
                                    CudaWrapper<float>& topk_confidence,
                                    CudaWrapper<uint32_t>& topk_indices,
                                    cudaStream_t stream) {
    
    // 计算CUDA kernel的grid和block大小
    dim3 blockSize(256);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x);
    
    // 调用CUDA kernel
    getTopKScoresKernel<<<gridSize, blockSize, 0, stream>>>(
        confidence.getCudaPtr(),
        topk_confidence.getCudaPtr(),
        topk_indices.getCudaPtr(),
        anchor_nums,
        k
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA kernel error in getTopKScoresKernel: " << cudaGetErrorString(err);
        return Status::kError;
    }
    
    return Status::kSuccess;
}

Status UtilsGPU::getTopKTrackIDOnGPU(const CudaWrapper<float>& confidence,
                                     const uint32_t& anchor_nums,
                                     const uint32_t& k,
                                     const CudaWrapper<int>& track_ids,
                                     CudaWrapper<int>& topk_track_ids,
                                     cudaStream_t stream) {
    
    // 计算CUDA kernel的grid和block大小
    dim3 blockSize(256);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x);
    
    // 调用CUDA kernel
    getTopKTrackIDKernel<<<gridSize, blockSize, 0, stream>>>(
        confidence.getCudaPtr(),
        track_ids.getCudaPtr(),
        topk_track_ids.getCudaPtr(),
        anchor_nums,
        k
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA kernel error in getTopKTrackIDKernel: " << cudaGetErrorString(err);
        return Status::kError;
    }
    
    return Status::kSuccess;
}

void UtilsGPU::applyConfidenceDecayOnGPU(const CudaWrapper<float>& input_confidence,
                                         CudaWrapper<float>& output_confidence,
                                         float decay_factor,
                                         cudaStream_t stream) {
    
    // 计算CUDA kernel的grid和block大小
    uint32_t size = input_confidence.getSize();
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    
    // 调用CUDA kernel
    applyConfidenceDecayKernel<<<gridSize, blockSize, 0, stream>>>(
        output_confidence.getCudaPtr(),
        input_confidence.getCudaPtr(),
        decay_factor,
        size
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA kernel error in applyConfidenceDecayKernel: " << cudaGetErrorString(err);
    }
}

void UtilsGPU::fuseConfidenceOnGPU(CudaWrapper<float>& new_confidence,
                                   const CudaWrapper<float>& cached_confidence,
                                   cudaStream_t stream) {
    
    // 计算CUDA kernel的grid和block大小
    uint32_t size = new_confidence.getSize();
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    
    // 调用CUDA kernel
    fuseConfidenceKernel<<<gridSize, blockSize, 0, stream>>>(
        new_confidence.getCudaPtr(),
        cached_confidence.getCudaPtr(),
        size
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA kernel error in fuseConfidenceKernel: " << cudaGetErrorString(err);
    }
}

void UtilsGPU::anchorProjectionOnGPU(CudaWrapper<float>& temp_anchor,
                                     const Eigen::Matrix<float, 4, 4>& temp2cur_mat,
                                     float time_interval,
                                     uint32_t topk_querys,
                                     uint32_t query_dims,
                                     cudaStream_t stream) {
    
    // 将变换矩阵转换为GPU内存
    std::vector<float> transform_vec(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            transform_vec[i * 4 + j] = temp2cur_mat(i, j);
        }
    }
    
    // 创建临时的GPU内存用于变换矩阵
    CudaWrapper<float> gpu_transform_matrix;
    gpu_transform_matrix.allocate(16);
    gpu_transform_matrix.cudaMemUpdateWrap(transform_vec);
    
    // 计算CUDA kernel的grid和block大小
    dim3 blockSize(256);
    dim3 gridSize((topk_querys + blockSize.x - 1) / blockSize.x);
    
    // 调用CUDA kernel
    anchorProjectionKernel<<<gridSize, blockSize, 0, stream>>>(
        temp_anchor.getCudaPtr(),
        gpu_transform_matrix.getCudaPtr(),
        time_interval,
        topk_querys,
        query_dims
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA kernel error in anchorProjectionKernel: " << cudaGetErrorString(err);
    }
}

void UtilsGPU::getMaxConfidenceScoresOnGPU(const CudaWrapper<float>& confidence_logits,
                                           CudaWrapper<float>& max_confidence_scores,
                                           const uint32_t& num_querys,
                                           cudaStream_t stream) {
    
    // 计算CUDA kernel的grid和block大小
    dim3 blockSize(256);
    dim3 gridSize((num_querys + blockSize.x - 1) / blockSize.x);
    
    // 调用CUDA kernel
    getMaxConfidenceScoresKernel<<<gridSize, blockSize, 0, stream>>>(
        confidence_logits.getCudaPtr(),
        max_confidence_scores.getCudaPtr(),
        num_querys,
        static_cast<uint32_t>(10)  // 假设有10个类别
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA kernel error in getMaxConfidenceScoresKernel: " << cudaGetErrorString(err);
    }
}

void UtilsGPU::applySigmoidOnGPU(CudaWrapper<float>& logits, cudaStream_t stream) {
    
    // 计算CUDA kernel的grid和block大小
    uint32_t size = logits.getSize();
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    
    // 调用CUDA kernel
    applySigmoidKernel<<<gridSize, blockSize, 0, stream>>>(
        logits.getCudaPtr(),
        size
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA kernel error in applySigmoidKernel: " << cudaGetErrorString(err);
    }
}

void UtilsGPU::generateNewTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                        const CudaWrapper<int32_t>& prev_id,
                                        cudaStream_t stream) {
    
    // 计算CUDA kernel的grid和block大小
    uint32_t size = track_ids.getSize();
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    
    // 调用CUDA kernel
    generateNewTrackIdsKernel<<<gridSize, blockSize, 0, stream>>>(
        track_ids.getCudaPtr(),
        const_cast<int32_t*>(prev_id.getCudaPtr()),
        size
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA kernel error in generateNewTrackIdsKernel: " << cudaGetErrorString(err);
    }
}

Status UtilsGPU::topKOnGPU(const CudaWrapper<float>& topk_cls_scores_origin,
                            const CudaWrapper<uint32_t>& topk_index,
                            const CudaWrapper<float>& fusioned_scores,
                            const CudaWrapper<uint8_t>& cls_ids,
                            const CudaWrapper<float>& box_preds,
                            const CudaWrapper<int>& track_ids,
                            const float& threshold,
                            const uint32_t& kmeans_anchor_dims,
                            const uint32_t& k,
                            CudaWrapper<float>& topk_cls_scores,
                            CudaWrapper<float>& topk_fusioned_scores,
                            CudaWrapper<uint8_t>& topk_cls_ids,
                            CudaWrapper<float>& topk_box_preds,
                            CudaWrapper<int>& topk_track_ids,
                            uint32_t& actual_topk_out,
                            cudaStream_t stream) {
    
    // 计算CUDA kernel的grid和block大小
    dim3 blockSize(256);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x);
    
    // 创建临时GPU内存用于actual_topk_out
    CudaWrapper<uint32_t> gpu_actual_topk_out;
    gpu_actual_topk_out.allocate(1);
    gpu_actual_topk_out.cudaMemSetWrap(0);
    
    // 调用CUDA kernel
    topKKernel<<<gridSize, blockSize, 0, stream>>>(
        topk_cls_scores_origin.getCudaPtr(),
        topk_index.getCudaPtr(),
        fusioned_scores.getCudaPtr(),
        cls_ids.getCudaPtr(),
        box_preds.getCudaPtr(),
        track_ids.getCudaPtr(),
        topk_cls_scores.getCudaPtr(),
        topk_fusioned_scores.getCudaPtr(),
        topk_cls_ids.getCudaPtr(),
        topk_box_preds.getCudaPtr(),
        topk_track_ids.getCudaPtr(),
        k,
        threshold,
        kmeans_anchor_dims,
        gpu_actual_topk_out.getCudaPtr()
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA kernel error in topKKernel: " << cudaGetErrorString(err);
        return Status::kError;
    }
    
    // 将结果从GPU拷贝到CPU
    auto actual_topk_cpu = gpu_actual_topk_out.cudaMemcpyD2HResWrap();
    actual_topk_out = actual_topk_cpu[0];
    
    return Status::kSuccess;
}

void UtilsGPU::sortOnGPU(CudaWrapper<float>& values, CudaWrapper<uint32_t>& indices, 
                          uint32_t size, cudaStream_t stream) {
    
    // 使用Thrust库进行GPU排序
    thrust::device_ptr<float> d_values_ptr(values.getCudaPtr());
    thrust::device_ptr<uint32_t> d_indices_ptr(indices.getCudaPtr());
    
    // 创建索引数组
    thrust::sequence(d_indices_ptr, d_indices_ptr + size);
    
    // 根据值进行排序，同时保持索引对应关系
    thrust::sort_by_key(thrust::cuda::par.on(stream), d_values_ptr, d_values_ptr + size, d_indices_ptr, thrust::greater<float>());
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA error in sortOnGPU: " << cudaGetErrorString(err);
    }
}