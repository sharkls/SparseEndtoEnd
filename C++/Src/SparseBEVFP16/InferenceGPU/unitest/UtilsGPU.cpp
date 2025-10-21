#include "UtilsGPU.h"
#include "UtilsGPU_kernels.cuh"
#include "log.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

Status UtilsGPU::getTopkInstanceOnGPU(const CudaWrapper<float>& confidence,
                                      const CudaWrapper<float>& instance_feature,
                                      const CudaWrapper<float>& anchor,
                                      const uint32_t& num_querys,
                                      const uint32_t& query_dims,
                                      const uint32_t& embedfeat_dims,
                                      const uint32_t& num_topk_querys,
                                      CudaWrapper<float>& output_confidence,
                                      CudaWrapper<float>& output_instance_feature,
                                      CudaWrapper<float>& output_anchor,
                                      CudaWrapper<int32_t>& output_track_ids,
                                      cudaStream_t stream) {
    if (confidence.getSize() != num_querys ||
        instance_feature.getSize() != num_querys * embedfeat_dims ||
        anchor.getSize() != num_querys * query_dims) {
        LOG(ERROR) << "[ERROR] Input size mismatch in getTopkInstanceOnGPU";
        return kInvalidInput;
    }
    
    uint32_t block_size = 256;
    uint32_t grid_size = (num_querys + block_size - 1) / block_size;
    
    getTopkInstanceKernel<<<grid_size, block_size, 0, stream>>>(
        confidence.getCudaPtr(), instance_feature.getCudaPtr(), anchor.getCudaPtr(),
        output_confidence.getCudaPtr(), output_instance_feature.getCudaPtr(),
        output_anchor.getCudaPtr(), output_track_ids.getCudaPtr(),
        num_querys, query_dims, embedfeat_dims, num_topk_querys
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        LOG(ERROR) << "[ERROR] CUDA kernel execution failed in getTopkInstanceOnGPU";
        return kInferenceErr;
    }
    
    return kSuccess;
}

Status UtilsGPU::getTopKScoresOnGPU(const CudaWrapper<float>& confidence,
                                    const uint32_t& num_querys,
                                    const uint32_t& k,
                                    CudaWrapper<float>& topk_confidence,
                                    CudaWrapper<uint32_t>& topk_indices,
                                    cudaStream_t stream) {
    if (confidence.getSize() != num_querys) {
        LOG(ERROR) << "[ERROR] Input size mismatch in getTopKScoresOnGPU";
        return kInvalidInput;
    }
    
    int block_size = 256;
    int grid_size = (k + block_size - 1) / block_size;
    
    getTopKScoresKernel<<<grid_size, block_size, 0, stream>>>(
        confidence.getCudaPtr(), topk_confidence.getCudaPtr(), topk_indices.getCudaPtr(),
        num_querys, k
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        LOG(ERROR) << "[ERROR] CUDA kernel execution failed in getTopKScoresOnGPU";
        return kInferenceErr;
    }
    
    return kSuccess;
}

Status UtilsGPU::getTrackIdOnGPU(const CudaWrapper<float>& confidence,
                                 const uint32_t& num_querys,
                                 const uint32_t& num_topk_querys,
                                 CudaWrapper<int32_t>& output_track_ids,
                                 cudaStream_t stream) {
    if (confidence.getSize() != num_querys) {
        LOG(ERROR) << "[ERROR] Input size mismatch in getTrackIdOnGPU";
        return kInvalidInput;
    }
    
    int block_size = 256;
    int grid_size = (num_querys + block_size - 1) / block_size;
    
    getTrackIdKernel<<<grid_size, block_size, 0, stream>>>(
        confidence.getCudaPtr(), output_track_ids.getCudaPtr(),
        num_querys, num_topk_querys
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        LOG(ERROR) << "[ERROR] CUDA kernel execution failed in getTrackIdOnGPU";
        return kInferenceErr;
    }
    
    return kSuccess;
}

Status UtilsGPU::cacheFeatureOnGPU(const CudaWrapper<float>& instance_feature,
                                   const CudaWrapper<float>& anchor,
                                   const CudaWrapper<float>& confidence,
                                   const CudaWrapper<int32_t>& track_ids,
                                   const uint32_t& num_querys,
                                   const uint32_t& query_dims,
                                   const uint32_t& embedfeat_dims,
                                   CudaWrapper<float>& cached_feature,
                                   CudaWrapper<float>& cached_anchor,
                                   CudaWrapper<float>& cached_confidence,
                                   CudaWrapper<int32_t>& cached_track_ids,
                                   cudaStream_t stream) {
    if (instance_feature.getSize() != num_querys * embedfeat_dims ||
        anchor.getSize() != num_querys * query_dims ||
        confidence.getSize() != num_querys ||
        track_ids.getSize() != num_querys) {
        LOG(ERROR) << "[ERROR] Input size mismatch in cacheFeatureOnGPU";
        return kInvalidInput;
    }
    
    int block_size = 256;
    int grid_size = (num_querys + block_size - 1) / block_size;
    
    cacheFeatureKernel<<<grid_size, block_size, 0, stream>>>(
        instance_feature.getCudaPtr(), anchor.getCudaPtr(), confidence.getCudaPtr(), track_ids.getCudaPtr(),
        cached_feature.getCudaPtr(), cached_anchor.getCudaPtr(), cached_confidence.getCudaPtr(), cached_track_ids.getCudaPtr(),
        num_querys, query_dims, embedfeat_dims
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        LOG(ERROR) << "[ERROR] CUDA kernel execution failed in cacheFeatureOnGPU";
        return kInferenceErr;
    }
    
    return kSuccess;
}

void UtilsGPU::applyConfidenceDecayOnGPU(const CudaWrapper<float>& input_confidence,
                                         CudaWrapper<float>& output_confidence,
                                         float decay_factor,
                                         cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (input_confidence.getSize() + block_size - 1) / block_size;
    
    applyConfidenceDecayKernel<<<grid_size, block_size, 0, stream>>>(
        output_confidence.getCudaPtr(), input_confidence.getCudaPtr(), decay_factor, input_confidence.getSize()
    );
    
    checkCudaError(cudaGetLastError());
}

void UtilsGPU::fuseConfidenceOnGPU(CudaWrapper<float>& new_confidence,
                                   const CudaWrapper<float>& cached_confidence,
                                   cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (new_confidence.getSize() + block_size - 1) / block_size;
    
    fuseConfidenceKernel<<<grid_size, block_size, 0, stream>>>(
        new_confidence.getCudaPtr(), cached_confidence.getCudaPtr(), new_confidence.getSize()
    );
    
    checkCudaError(cudaGetLastError());
}

void UtilsGPU::anchorProjectionOnGPU(CudaWrapper<float>& temp_anchor,
                                     const Eigen::Matrix<float, 4, 4>& temp2cur_mat,
                                     float time_interval,
                                     uint32_t topk_querys,
                                     uint32_t query_dims,
                                     cudaStream_t stream) {
    // 拷贝矩阵到Device临时缓冲
    float h_tf[16];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            h_tf[r * 4 + c] = temp2cur_mat(r, c);
        }
    }
    float* d_tf = nullptr;
    cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&d_tf), sizeof(float) * 16, stream);
    if (!checkCudaError(err)) { return; }
    err = cudaMemcpyAsync(d_tf, h_tf, sizeof(float) * 16, cudaMemcpyHostToDevice, stream);
    if (!checkCudaError(err)) { cudaFreeAsync(d_tf, stream); return; }

    int block_size = 256;
    int grid_size = (topk_querys + block_size - 1) / block_size;
    anchorProjectionKernel<<<grid_size, block_size, 0, stream>>>(
        temp_anchor.getCudaPtr(), d_tf, time_interval, topk_querys, query_dims
    );
    checkCudaError(cudaGetLastError());
    cudaFreeAsync(d_tf, stream);
}

void UtilsGPU::getMaxConfidenceScoresOnGPU(const CudaWrapper<float>& confidence_logits,
                                           CudaWrapper<float>& max_confidence_scores,
                                           const uint32_t& num_querys,
                                           cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_querys + block_size - 1) / block_size;
    
    getMaxConfidenceScoresKernel<<<grid_size, block_size, 0, stream>>>(
        confidence_logits.getCudaPtr(), max_confidence_scores.getCudaPtr(), num_querys, 1  // 假设只有1个类别
    );
    
    checkCudaError(cudaGetLastError());
}

void UtilsGPU::applySigmoidOnGPU(CudaWrapper<float>& logits, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (logits.getSize() + block_size - 1) / block_size;
    
    applySigmoidKernel<<<grid_size, block_size, 0, stream>>>(
        logits.getCudaPtr(), logits.getSize()
    );
    
    checkCudaError(cudaGetLastError());
}

void UtilsGPU::generateNewTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                        const CudaWrapper<int32_t>& prev_id,
                                        cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (track_ids.getSize() + block_size - 1) / block_size;
    
    generateNewTrackIdsKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), prev_id.getCudaPtr(), track_ids.getSize()
    );
    
    checkCudaError(cudaGetLastError());
}

void UtilsGPU::sortOnGPU(CudaWrapper<float>& values, CudaWrapper<uint32_t>& indices, 
                          uint32_t size, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    radixSortKernel<<<grid_size, block_size, 0, stream>>>(
        values.getCudaPtr(), indices.getCudaPtr(), size
    );
    
    checkCudaError(cudaGetLastError());
}

bool UtilsGPU::checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        LOG(ERROR) << "[CUDA ERROR] " << cudaGetErrorString(error);
        return false;
    }
    return true;
}

Status UtilsGPU::countNegativeValuesOnGPU(const CudaWrapper<int32_t>& track_ids,
                                         uint32_t& negative_count,
                                         uint32_t array_size,
                                         cudaStream_t stream) {
    negative_count = 0;
    CudaWrapper<uint32_t> gpu_counter(1);
    gpu_counter.cudaMemSetWrap();

    uint32_t block_size = 256;
    uint32_t grid_size = (array_size + block_size - 1) / block_size;

    countNegativeValuesKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), gpu_counter.getCudaPtr(), array_size
    );
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }

    std::vector<uint32_t> h(1);
    gpu_counter.cudaMemcpyD2H(h);
    negative_count = h[0];
    return kSuccess;
}

Status UtilsGPU::updateNegativeTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                             const std::vector<int32_t>& new_track_ids,
                                             uint32_t array_size,
                                             cudaStream_t stream) {
    CudaWrapper<int32_t> gpu_new_track_ids(new_track_ids);

    uint32_t block_size = 256;
    uint32_t grid_size = (array_size + block_size - 1) / block_size;

    updateNegativeTrackIdsKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), gpu_new_track_ids.getCudaPtr(), array_size, static_cast<uint32_t>(new_track_ids.size())
    );
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status UtilsGPU::updateLastNTrackIdsOnGPU(CudaWrapper<int32_t>& track_ids,
                                         const std::vector<int32_t>& new_track_ids,
                                         uint32_t total_size,
                                         uint32_t update_count,
                                         cudaStream_t stream) {
    CudaWrapper<int32_t> gpu_new_track_ids(new_track_ids);

    uint32_t block_size = 256;
    uint32_t grid_size = (update_count + block_size - 1) / block_size;

    updateLastNTrackIdsKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), gpu_new_track_ids.getCudaPtr(), total_size, update_count
    );
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status UtilsGPU::selectTrackIdsFromIndicesOnGPU(CudaWrapper<int32_t>& track_ids,
                                                const CudaWrapper<int32_t>& cached_indices,
                                                uint32_t target_length,
                                                cudaStream_t stream) {
    // 修正：source_length应该是cached_indices的长度，不是track_ids的长度
    uint32_t source_length = cached_indices.getSize();

    uint32_t block_size = 256;
    uint32_t grid_size = (target_length + block_size - 1) / block_size;

    // 调用正确的kernel名称
    selectTrackIdsFromIndicesKernel<<<grid_size, block_size, 0, stream>>>(
        track_ids.getCudaPtr(), cached_indices.getCudaPtr(), target_length, source_length
    );
    
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status UtilsGPU::applyConfidenceDecayOnGPU(const CudaWrapper<float>& input_confidence,
                                           CudaWrapper<float>& output_confidence,
                                           float decay_factor,
                                           uint32_t array_size,
                                           cudaStream_t stream) {
    uint32_t block_size = 256;
    uint32_t grid_size = (array_size + block_size - 1) / block_size;

    applyConfidenceDecayKernel<<<grid_size, block_size, 0, stream>>>(
        input_confidence.getCudaPtr(),
        output_confidence.getCudaPtr(),
        decay_factor,
        array_size
    );
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}

Status UtilsGPU::fuseConfidenceOnGPU(const CudaWrapper<float>& current_confidence,
                                     const CudaWrapper<float>& cached_confidence,
                                     uint32_t fusion_length,
                                     cudaStream_t stream) {
    uint32_t block_size = 256;
    uint32_t grid_size = (fusion_length + block_size - 1) / block_size;

    CudaWrapper<float> temp_current = current_confidence;  // 创建临时副本
    
    fuseConfidenceKernel<<<grid_size, block_size, 0, stream>>>(
        temp_current.getCudaPtr(), cached_confidence.getCudaPtr(), fusion_length
    );
    if (!checkCudaError(cudaGetLastError())) {
        return kInferenceErr;
    }
    return kSuccess;
}