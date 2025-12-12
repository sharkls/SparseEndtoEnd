#ifndef __SPARSE4D_BEV_ENGINE_HPP__
#define __SPARSE4D_BEV_ENGINE_HPP__

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include "../../../Include/Common/Utils/CudaWrapper.h"
#include "Sparse4D_conf.pb.h"

namespace sparse4d {
namespace bev {

// Simple TRT Logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// Generic TensorRT Engine Wrapper
// It manages input/output bindings but delegates buffer management to the caller/CudaWrapper
class EngineWrapper {
public:
    EngineWrapper();
    ~EngineWrapper();

    bool init(const std::string& engine_path, 
              const std::vector<std::string>& plugin_paths = {});
    
    // Check if engine data type matches template T
    template <typename T>
    bool check_precision();

    // Enqueue inference
    // bindings: vector of device pointers, strictly ordered by engine binding indices
    bool forward(const std::vector<void*>& bindings, cudaStream_t stream);

    // Get binding info
    int get_binding_index(const std::string& name) const;
    nvinfer1::Dims get_binding_shape(int index) const;
    nvinfer1::DataType get_binding_dtype(int index) const;
    int get_num_bindings() const;
    std::string get_binding_name(int index) const;

private:
    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    Logger logger_;
};

} // namespace bev
} // namespace sparse4d

#endif // __SPARSE4D_BEV_ENGINE_HPP__

