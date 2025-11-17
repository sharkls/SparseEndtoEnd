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
    
    // 验证输入输出缓冲区指针有效性
    void* input_ptr = const_cast<void*>(static_cast<const void*>(pipeline_context.input_images.getCudaPtr()));
    void* output_ptr = static_cast<void*>(pipeline_context.features.getCudaPtr());
    
    if (input_ptr == nullptr) {
        LOG(ERROR) << "[ERROR] Backbone input_images buffer is null!";
        return Status::kInferenceErr;
    }
    
    if (output_ptr == nullptr) {
        LOG(ERROR) << "[ERROR] Backbone features buffer is null!";
        return Status::kInferenceErr;
    }
    
    // 验证缓冲区大小
    size_t input_size = pipeline_context.input_images.getSize();
    size_t output_size = pipeline_context.features.getSize();
    
    if (input_size == 0) {
        LOG(ERROR) << "[ERROR] Backbone input_images size is 0!";
        return Status::kInferenceErr;
    }
    
    if (output_size == 0) {
        LOG(ERROR) << "[ERROR] Backbone features size is 0!";
        return Status::kInferenceErr;
    }
    
    // 验证 Engine 期望的形状和实际缓冲区大小是否匹配
    nvinfer1::ICudaEngine* engine = m_backbone_engine->getEngine();
    if (engine != nullptr) {
        int numBindings = engine->getNbBindings();
        if (numBindings > 0) {
            // TensorRT 8.x - 检查输入 binding
            for (int i = 0; i < numBindings; ++i) {
                if (engine->bindingIsInput(i)) {
                    const char* binding_name = engine->getBindingName(i);
                    nvinfer1::Dims dims = engine->getBindingDimensions(i);
                    nvinfer1::DataType dtype = engine->getBindingDataType(i);
                    
                    // 计算期望的元素数量
                    int32_t expected_elements = 1;
                    for (int j = 0; j < dims.nbDims; ++j) {
                        expected_elements *= dims.d[j];
                    }
                    
                    // 计算期望的字节数（FP16 = 2 bytes）
                    size_t expected_bytes = expected_elements * (dtype == nvinfer1::DataType::kHALF ? 2 : 4);
                    size_t actual_bytes = input_size * sizeof(half);
                    
                    if (expected_bytes != actual_bytes) {
                        std::string shape_str = "[";
                        for (int j = 0; j < dims.nbDims; ++j) {
                            shape_str += std::to_string(dims.d[j]);
                            if (j < dims.nbDims - 1) shape_str += ", ";
                        }
                        shape_str += "]";
                        LOG(ERROR) << "[ERROR] Backbone input size mismatch! Binding: " << binding_name
                                   << ", Expected: " << expected_elements << " elements (" << expected_bytes << " bytes)"
                                   << ", Actual: " << input_size << " elements (" << actual_bytes << " bytes)";
                        LOG(ERROR) << "[ERROR] Expected shape: " << shape_str
                                   << ", Type: " << (dtype == nvinfer1::DataType::kHALF ? "HALF" : "FLOAT");
                        return Status::kInferenceErr;
                    }
                } else {
                    // 检查输出 binding
                    const char* binding_name = engine->getBindingName(i);
                    nvinfer1::Dims dims = engine->getBindingDimensions(i);
                    nvinfer1::DataType dtype = engine->getBindingDataType(i);
                    
                    // 计算期望的元素数量
                    int32_t expected_elements = 1;
                    for (int j = 0; j < dims.nbDims; ++j) {
                        expected_elements *= dims.d[j];
                    }
                    
                    // 计算期望的字节数（FP16 = 2 bytes）
                    size_t expected_bytes = expected_elements * (dtype == nvinfer1::DataType::kHALF ? 2 : 4);
                    size_t actual_bytes = output_size * sizeof(half);
                    
                    if (expected_bytes != actual_bytes) {
                        std::string shape_str = "[";
                        for (int j = 0; j < dims.nbDims; ++j) {
                            shape_str += std::to_string(dims.d[j]);
                            if (j < dims.nbDims - 1) shape_str += ", ";
                        }
                        shape_str += "]";
                        LOG(ERROR) << "[ERROR] Backbone output size mismatch! Binding: " << binding_name
                                   << ", Expected: " << expected_elements << " elements (" << expected_bytes << " bytes)"
                                   << ", Actual: " << output_size << " elements (" << actual_bytes << " bytes)";
                        LOG(ERROR) << "[ERROR] Expected shape: " << shape_str
                                   << ", Type: " << (dtype == nvinfer1::DataType::kHALF ? "HALF" : "FLOAT");
                        return Status::kInferenceErr;
                    }
                }
            }
        }
    }
    
    // 准备输入输出缓冲区
    std::vector<void*> input_buffers = {input_ptr};
    std::vector<void*> output_buffers = {output_ptr};
    
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
