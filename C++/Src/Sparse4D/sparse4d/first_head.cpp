#include "first_head.hpp"
#include "../../Common/TensorRT/TensorRT.h"
#include "log.h"

namespace sparse4d{
namespace first_head{

class FirstHeadImplement : public FirstHead{
    public:
        ~FirstHeadImplement() override = default;

        Status init(const TaskConfig &param) override;

        Status forward(common::PipelineContext& pipeline_context,
                        const cudaStream_t& stream,
                        common::HeadOutput& head_output) override;
    private:
        TaskConfig m_taskConfig;              // 任务配置参数
        std::shared_ptr<TensorRT> m_first_head_engine;    // 第一帧头部引擎
};

Status FirstHeadImplement::init(const TaskConfig &param)
{
    LOG(INFO) << "[INFO] Sparse4D::FirstHeadImplement::init start";

    m_taskConfig = param;
    // 第一帧头部引擎
    m_first_head_engine = std::make_shared<TensorRT>(
        m_taskConfig.head1st_engine().engine_path(),
        m_taskConfig.head1st_engine().plugin_path(),
        std::vector<std::string>(m_taskConfig.head1st_engine().input_names().begin(), 
                                m_taskConfig.head1st_engine().input_names().end()),
        std::vector<std::string>(m_taskConfig.head1st_engine().output_names().begin(), 
                                m_taskConfig.head1st_engine().output_names().end()));
    LOG(INFO) << "[INFO] first head engine loaded";
    
    if (!m_first_head_engine) {
        LOG(ERROR) << "[ERROR] Failed to load first head engine";
        return Status::kFirstHeadEngineLoadErr;
    }
    
    LOG(INFO) << "[INFO] Sparse4D::FirstHeadImplement::init end";
    return Status::kSuccess;
}

Status FirstHeadImplement::forward(common::PipelineContext& pipeline_context,
                                    const cudaStream_t& stream,
                                    common::HeadOutput& head_output)
{   
    if (m_first_head_engine == nullptr) {
        LOG(ERROR) << "[ERROR] First head engine is null!";
        return Status::kInferenceErr;
    }
    
    // 准备输入输出缓冲区
    std::vector<void*> input_buffers = {const_cast<void*>(static_cast<const void*>(pipeline_context.features.getCudaPtr())),
                                        static_cast<void*>(pipeline_context.spatial_shapes.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.level_start_index.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.instance_feature.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.anchor.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.time_interval.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.image_wh.getCudaPtr()),
                                        static_cast<void*>(pipeline_context.lidar2img.getCudaPtr())};
    std::vector<void*> output_buffers = {static_cast<void*>(head_output.tmp_outs0.getCudaPtr()),
                                         static_cast<void*>(head_output.tmp_outs1.getCudaPtr()),
                                         static_cast<void*>(head_output.tmp_outs2.getCudaPtr()),
                                         static_cast<void*>(head_output.tmp_outs3.getCudaPtr()),
                                         static_cast<void*>(head_output.tmp_outs4.getCudaPtr()),
                                         static_cast<void*>(head_output.tmp_outs5.getCudaPtr()),
                                         static_cast<void*>(head_output.pred_instance_feature.getCudaPtr()),
                                         static_cast<void*>(head_output.pred_anchor.getCudaPtr()),
                                         static_cast<void*>(head_output.pred_class_score.getCudaPtr()),
                                         static_cast<void*>(head_output.pred_quality_score.getCudaPtr())};
    
    // 执行推理
    bool success = m_first_head_engine->infer(input_buffers.data(), output_buffers.data(), stream);
    
    if (success) {
        LOG(INFO) << "[INFO] First head completed successfully";
        return Status::kSuccess;
    } else {
        LOG(ERROR) << "[ERROR] First head failed";
        return Status::kInferenceErr;
    }
}

std::shared_ptr<FirstHead> create_first_head(const TaskConfig &param) {
    auto instance = std::make_shared<FirstHeadImplement>();
    Status status = instance->init(param);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Failed to create first head";
        return nullptr;
    }
    return instance;
}
}//namespace first_head
}//namespace sparse4d
