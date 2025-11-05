#include "second_head.hpp"
#include <cuda_runtime.h>

namespace sparse4d{
namespace second_head{

/**
 * @brief 第二帧头部实现
 * 
 */
class SecondHeadImplement : public SecondHead
{
    public:
        virtual ~SecondHeadImplement() = default;

        Status init(const TaskConfig &param) override;

        Status forward(common::PipelineContext& pipeline_context,
                       const cudaStream_t& stream,
                       common::HeadOutput& head_output) override;

    private:
        TaskConfig m_taskConfig;
        std::shared_ptr<TensorRT> m_second_head_engine;
};

Status SecondHeadImplement::init(const TaskConfig &param)
{
    LOG(INFO) << "[INFO] Sparse4D::SecondHeadImplement::init start";

    m_taskConfig = param;
    m_second_head_engine = std::make_shared<TensorRT>(
        m_taskConfig.head2nd_engine().engine_path(),
        m_taskConfig.head2nd_engine().plugin_path(),
        std::vector<std::string>(m_taskConfig.head2nd_engine().input_names().begin(), 
                                m_taskConfig.head2nd_engine().input_names().end()),
        std::vector<std::string>(m_taskConfig.head2nd_engine().output_names().begin(), 
                                m_taskConfig.head2nd_engine().output_names().end()));
    LOG(INFO) << "[INFO] second head engine loaded";

    if (!m_second_head_engine) {
        LOG(ERROR) << "[ERROR] Failed to load second head engine";
        return Status::kSecondHeadEngineLoadErr;
    }

    LOG(INFO) << "[INFO] Sparse4D::SecondHeadImplement::init end";
    return Status::kSuccess;
}

Status SecondHeadImplement::forward(common::PipelineContext& pipeline_context,
                                    const cudaStream_t& stream,
                                    common::HeadOutput& head_output)
{
    // LOG(INFO) << "[INFO] Sparse4D::SecondHeadImplement::forward start";

    if (!m_second_head_engine) {
        LOG(ERROR) << "[ERROR] Second head engine is null!";
        return Status::kInferenceErr;
    }

    // track_ids 由 InstanceBank::get() 已写入 pipeline_context.track_ids
    std::vector<void*> input_buffers = {const_cast<void*>(static_cast<const void*>(pipeline_context.features.getCudaPtr())),
                                        static_cast<void*>(pipeline_context.spatial_shapes.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.level_start_index.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.instance_feature.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.anchor.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.time_interval.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.temp_instance_feature.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.temp_anchor.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.mask.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.track_ids.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.image_wh.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.lidar2img.getCudaPtr())};

    std::vector<void*> output_buffers = {static_cast<void*>(head_output.tmp_outs0.getCudaPtr()),
                                            static_cast<void*>(head_output.pred_track_ids.getCudaPtr()),
                                            static_cast<void*>(head_output.tmp_outs1.getCudaPtr()),
                                            static_cast<void*>(head_output.tmp_outs2.getCudaPtr()),
                                            static_cast<void*>(head_output.tmp_outs3.getCudaPtr()),
                                            static_cast<void*>(head_output.tmp_outs4.getCudaPtr()),
                                            static_cast<void*>(head_output.tmp_outs5.getCudaPtr()),
                                            static_cast<void*>(head_output.pred_instance_feature.getCudaPtr()),
                                            static_cast<void*>(head_output.pred_anchor.getCudaPtr()),
                                            static_cast<void*>(head_output.pred_class_score.getCudaPtr()),
                                            static_cast<void*>(head_output.pred_quality_score.getCudaPtr())};

    bool success = m_second_head_engine->infer(input_buffers.data(), output_buffers.data(), stream);

    if (success) {
        // LOG(INFO) << "[INFO] Second head completed successfully";
        return Status::kSuccess;
    } else {
        LOG(ERROR) << "[ERROR] Second head failed";
        return Status::kInferenceErr;
    }
}

std::shared_ptr<SecondHead> create_second_head(const TaskConfig &param) {
    auto instance = std::make_shared<SecondHeadImplement>();
    Status status = instance->init(param);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Failed to create second head";
        return nullptr;
    }
    return instance;
}
}//namespace second_head
}//namespace sparse4d