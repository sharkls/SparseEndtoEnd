#include "backbone.hpp"
#include "../../Common/TensorRT/TensorRT.h"

namespace sparse4d{
namespace backbone{

class BackboneImplement : public Backbone{
public:
    ~BackboneImplement() override = default;

    Status init(const TaskConfig &param) override;

    Status forward(common::PipelineContext& pipeline_context, const cudaStream_t& stream) override;

private:
    TaskConfig m_taskConfig;              // 任务配置参数
    std::shared_ptr<TensorRT> m_backbone_engine;    // 骨干网络引擎
};

Status BackboneImplement::init(const TaskConfig &param)
{
    LOG(INFO) << "[INFO] Sparse4D::BackboneImplement::init start";

    m_taskConfig = param;
    m_backbone_engine = std::make_shared<TensorRT>(
        m_taskConfig.backbone_engine().engine_path(),
        "",
        std::vector<std::string>(m_taskConfig.backbone_engine().input_names().begin(), 
                                m_taskConfig.backbone_engine().input_names().end()),
        std::vector<std::string>(m_taskConfig.backbone_engine().output_names().begin(), 
                                m_taskConfig.backbone_engine().output_names().end()));
    
    if (!m_backbone_engine) {
        LOG(ERROR) << "[ERROR] Failed to load backbone engine";
        return Status::kBackboneEngineLoadErr;
    }
    
    LOG(INFO) << "[INFO] SparseBEV::init end";
    return Status::kSuccess;
}

Status BackboneImplement::forward(common::PipelineContext& pipeline_context, const cudaStream_t& stream)
{   
    if (m_backbone_engine == nullptr) {
        LOG(ERROR) << "[ERROR] Backbone engine is null!";
        return Status::kInferenceErr;
    }
    
    // 准备输入输出缓冲区
    std::vector<void*> input_buffers = {const_cast<void*>(static_cast<const void*>(pipeline_context.input_images.getCudaPtr()))};
    std::vector<void*> output_buffers = {static_cast<void*>(pipeline_context.features.getCudaPtr())};
    
    // 执行推理
    bool success = m_backbone_engine->infer(input_buffers.data(), output_buffers.data(), stream);
    
    if (success) {
        // LOG(INFO) << "[INFO] Backbone forward completed successfully";
        return Status::kSuccess;
    } else {
        LOG(ERROR) << "[ERROR] Backbone forward failed";
        return Status::kInferenceErr;
    }
}

std::shared_ptr<Backbone> create_backbone(const TaskConfig &param) {
    auto instance = std::make_shared<BackboneImplement>();
    Status status = instance->init(param);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Failed to create backbone";
        return nullptr;
    }
    return instance;
}

}//namespace backbone
}//namespace sparse4d
