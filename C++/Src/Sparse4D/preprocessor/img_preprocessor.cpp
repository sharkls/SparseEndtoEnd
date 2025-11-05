#include "img_preprocessor.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace sparse4d{
namespace preprocessor{

class PreprocessorImplement : public Preprocessor{
public:
    ~PreprocessorImplement() override {
        if (d_raw_ != nullptr) {
            cudaFree(d_raw_);
            d_raw_ = nullptr;
            d_raw_bytes_ = 0;
        }
        if (h_pinned_ != nullptr) {
            cudaFreeHost(h_pinned_);
            h_pinned_ = nullptr;
            h_pinned_bytes_ = 0;
        }
    }

    Status init(const TaskConfig &param) override;

    Status forward(const CTimeMatchSrcData *raw_data, 
                   const cudaStream_t& stream,
                   common::PipelineContext& pipeline_context) override;

private:
    TaskConfig m_taskConfig;            // 任务配置参数

    // 运行参数
    size_t expected_size_raw_img_;
    size_t per_cam_size_;
    std::uint32_t num_cams_;

    // 持久化缓冲（复用）
    std::uint8_t* h_pinned_ = nullptr;
    size_t h_pinned_bytes_ = 0;
    std::uint8_t* d_raw_ = nullptr;
    size_t d_raw_bytes_ = 0;
};


Status PreprocessorImplement::init(const TaskConfig &param)
{
    m_taskConfig = param;

    // 计算期望尺寸
    expected_size_raw_img_ = static_cast<size_t>(m_taskConfig.preprocessor_params().num_cams()) * 
                             m_taskConfig.preprocessor_params().raw_img_c() *
                             m_taskConfig.preprocessor_params().raw_img_h() *
                             m_taskConfig.preprocessor_params().raw_img_w();

    num_cams_ = m_taskConfig.preprocessor_params().num_cams();
    per_cam_size_ = static_cast<size_t>(
        m_taskConfig.preprocessor_params().raw_img_c() *
        m_taskConfig.preprocessor_params().raw_img_h() *
        m_taskConfig.preprocessor_params().raw_img_w());

    return Status::kSuccess;
}

// uint8 -> float
Status PreprocessorImplement::forward(const CTimeMatchSrcData *raw_data,
                                      const cudaStream_t& stream,
                                      common::PipelineContext& pipeline_context)
{
    if (raw_data == nullptr) {
        LOG(ERROR) << "[ERROR] raw_data is null";
        return Status::kInvalidInput;
    }

    // 统计输入总字节数并校验
    size_t total_size = 0;
    const auto &videos = raw_data->vecVideoSrcData();
    
    // 校验相机数量
    if (videos.size() != num_cams_) {
        LOG(ERROR) << "[ERROR] Camera count mismatch! Expected: " << num_cams_ << ", Got: " << videos.size();
        return Status::kImgPreprocesSizeErr;
    }

    for (const auto &v : videos) {
        total_size += v.vecImageBuf().size();
        if (v.vecImageBuf().size() != per_cam_size_) {
            LOG(ERROR) << "[ERROR] One camera buffer size mismatch! Expected per cam: " << per_cam_size_
                       << ", Got: " << v.vecImageBuf().size();
            return Status::kImgPreprocesSizeErr;
        }
    }
    if (total_size != expected_size_raw_img_) {
        LOG(ERROR) << "[ERROR] Input data size mismatch! Expected: " << expected_size_raw_img_ << ", Got: " << total_size;
        return Status::kImgPreprocesSizeErr;
    }

    // 分配/复用 pinned 主机缓冲
    cudaError_t err;
    if (h_pinned_ == nullptr || h_pinned_bytes_ != expected_size_raw_img_) {
        if (h_pinned_ != nullptr) cudaFreeHost(h_pinned_);
        err = cudaHostAlloc(reinterpret_cast<void **>(&h_pinned_), expected_size_raw_img_, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to allocate pinned host buffer: " << cudaGetErrorString(err);
            return Status::kImgPreprocesCudaErr;
        }
        h_pinned_bytes_ = expected_size_raw_img_;
    }
    size_t offset = 0;
    for (const auto &v : videos) {
        const auto &buf = v.vecImageBuf();
        if (!buf.empty()) {
            memcpy(h_pinned_ + offset, buf.data(), buf.size());
            offset += buf.size();
        }
    }

    // 分配/复用 设备端缓冲
    if (d_raw_ == nullptr || d_raw_bytes_ != expected_size_raw_img_) {
        if (d_raw_ != nullptr) cudaFree(d_raw_);
        err = cudaMalloc(reinterpret_cast<void **>(&d_raw_), expected_size_raw_img_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to allocate device buffer: " << cudaGetErrorString(err);
            return Status::kImgPreprocesCudaErr;
        }
        d_raw_bytes_ = expected_size_raw_img_;
    }

    // 同一stream上执行异步拷贝
    err = cudaMemcpyAsync(d_raw_, h_pinned_, expected_size_raw_img_, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] Failed to H2D copy raw images: " << cudaGetErrorString(err);
        return Status::kImgPreprocesCudaErr;
    }

    // 验证输出缓冲区尺寸
    if (pipeline_context.input_images.getSize() != static_cast<std::uint64_t>(
            m_taskConfig.preprocessor_params().num_cams() * 
            m_taskConfig.preprocessor_params().model_input_img_c() *
            m_taskConfig.preprocessor_params().model_input_img_h() * 
            m_taskConfig.preprocessor_params().model_input_img_w())) {
        LOG(ERROR) << "[ERROR] Model input imgs' size mismatches with params!";
        cudaFree(d_raw_);
        cudaFreeHost(h_pinned_);
        return Status::kImgPreprocesSizeErr;
    }

    // 调用CUDA内核执行图像预处理操作（uint8 -> float）
    Status ret_code = imgPreprocessLauncher(
        d_raw_,                                                  // 输入图像GPU指针
        m_taskConfig.preprocessor_params().num_cams(),           // 相机数量
        m_taskConfig.preprocessor_params().raw_img_c(),          // 原始图像通道数
        m_taskConfig.preprocessor_params().raw_img_h(),          // 原始图像高度
        m_taskConfig.preprocessor_params().raw_img_w(),          // 原始图像宽度
        m_taskConfig.preprocessor_params().model_input_img_h(),  // 模型输入图像高度
        m_taskConfig.preprocessor_params().model_input_img_w(),  // 模型输入图像宽度
        m_taskConfig.preprocessor_params().resize_ratio(),       // 缩放比例
        m_taskConfig.preprocessor_params().crop_height(),        // 裁剪高度
        m_taskConfig.preprocessor_params().crop_width(),         // 裁剪宽度
        stream,                                                  // CUDA流
        pipeline_context.input_images.getCudaPtr());             // 输出图像GPU指针

    // 注意：此处不强制同步，也不释放复用缓冲，允许流水线并行

    if (ret_code != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Image preprocessing kernel execution failed!";
        return ret_code;
    }

    return Status::kSuccess;
}

// // uint8 -> half
// Status PreprocessorImplement::forward(const CTimeMatchSrcData *raw_data, 
//                                       const cudaStream_t& stream,
//                                       CudaWrapper<half>& output_imgs)
// {
//     // 验证输入图像尺寸
//     if (input_imgs.getSize() != static_cast<std::uint64_t>(
//             m_taskConfig.preprocessor_params().num_cams() *
//             (m_taskConfig.preprocessor_params().raw_img_c() * 
//              m_taskConfig.preprocessor_params().raw_img_h() *
//              m_taskConfig.preprocessor_params().raw_img_w()))) {
//         LOG(ERROR) << "[ERROR] Input imgs' size mismatches with params!";
//         return Status::kImgPreprocesSizeErr;
//     }

//     // 验证输出缓冲区尺寸
//     if (output_imgs.getSize() != static_cast<std::uint64_t>(
//             m_taskConfig.preprocessor_params().num_cams() * 
//             m_taskConfig.preprocessor_params().model_input_img_c() *
//             m_taskConfig.preprocessor_params().model_input_img_h() * 
//             m_taskConfig.preprocessor_params().model_input_img_w())) {
//         LOG(ERROR) << "[ERROR] Model input imgs' size mismatches with params!";
//         return Status::kImgPreprocesSizeErr;
//     }

//     Status ret_code = imgPreprocessLauncher(
//         input_imgs.getCudaPtr(),
//         m_taskConfig.preprocessor_params().num_cams(),
//         m_taskConfig.preprocessor_params().raw_img_c(),
//         m_taskConfig.preprocessor_params().raw_img_h(),
//         m_taskConfig.preprocessor_params().raw_img_w(),
//         m_taskConfig.preprocessor_params().model_input_img_h(),
//         m_taskConfig.preprocessor_params().model_input_img_w(),
//         m_taskConfig.preprocessor_params().resize_ratio(),
//         m_taskConfig.preprocessor_params().crop_height(),
//         m_taskConfig.preprocessor_params().crop_width(),
//         stream,
//         output_imgs.getCudaPtr());

//     if (ret_code != Status::kSuccess) {
//         LOG(ERROR) << "[ERROR] Image preprocessing kernel execution failed! (half)";
//         return ret_code;
//     }

//     return Status::kSuccess;
// }

// // uint8 -> int8 (未实现：如需量化预处理，这里应包含scale/zero-point等)
// Status PreprocessorImplement::forward(const CTimeMatchSrcData *raw_data,
//                                       const cudaStream_t& /*stream*/,
//                                       CudaWrapper<int8_t>& /*output_imgs*/)
// {
//     LOG(ERROR) << "[ERROR] int8 preprocessor path is not implemented";
//     return Status::kImgPreprocesFormatErr;
// }

std::shared_ptr<Preprocessor> create_preprocessor(const TaskConfig &param) {
    auto instance = std::make_shared<PreprocessorImplement>();
    Status st = instance->init(param);
    if (st != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Preprocessor init failed";
        return nullptr;
    }
    return instance;
}
}//namespace preprocessor
}//namespace sparse4d