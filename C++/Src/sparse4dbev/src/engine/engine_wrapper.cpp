#include "engine_wrapper.hpp"
#include "log.h"
#include <fstream>
#include <dlfcn.h>
#include <iostream>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>

namespace sparse4d {
namespace bev {

// Global logger for Plugin Registry to ensure lifetime safety
static Logger g_plugin_logger;

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        LOG(INFO) << "[TRT] " << msg;
    }
}

EngineWrapper::EngineWrapper() = default;

EngineWrapper::~EngineWrapper() {
    // Smart pointers handle destruction
}

bool EngineWrapper::init(const std::string& engine_path, const std::vector<std::string>& plugin_paths) {
    LOG(INFO) << "Loading engine: " << engine_path;
    
    // 1. Load Plugins
    // Use RTLD_LAZY to match TensorRT.cpp
    for (const auto& path : plugin_paths) {
        // Always dlopen, rely on OS ref counting
        LOG(INFO) << "Loading plugin: " << path;
        void* handle = dlopen(path.c_str(), RTLD_LAZY);
        if (!handle) {
            LOG(ERROR) << "Failed to load plugin: " << path << ", error: " << dlerror();
            return false;
        }
    }
    
    // Call initLibNvInferPlugins only once
    static bool plugins_initialized = false;
    if (!plugins_initialized) {
        initLibNvInferPlugins(&g_plugin_logger, "");
        plugins_initialized = true;
        LOG(INFO) << "LibNvInferPlugins initialized.";
    }
    // LOG(WARNING) << "Skipping plugin loading to debug segfault.";

    // 2. Load Engine File
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        LOG(ERROR) << "Failed to open engine file: " << engine_path;
        return false;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    LOG(INFO) << "Engine file size: " << size;
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // 3. Deserialize
    LOG(INFO) << "Deserializing engine...";
    
    // Wrap in try-catch to handle potential TRT exceptions and avoid ABI unwind issues
    try {
        // Use custom deleter for ABI safety and correct destruction
        struct RuntimeDeleter {
            void operator()(nvinfer1::IRuntime* ptr) { if (ptr) ptr->destroy(); }
        };
        struct EngineDeleter {
            void operator()(nvinfer1::ICudaEngine* ptr) { if (ptr) ptr->destroy(); }
        };
        struct ContextDeleter {
            void operator()(nvinfer1::IExecutionContext* ptr) { if (ptr) ptr->destroy(); }
        };

        // Note: EngineWrapper header needs to be updated to use std::shared_ptr or unique_ptr
        // If header uses shared_ptr, we can constructor shared_ptr with custom deleter.
        
        runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(g_plugin_logger), 
            RuntimeDeleter()
        );
        
        if (!runtime_) {
            LOG(ERROR) << "Failed to create runtime.";
            return false;
        }

        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(engine_data.data(), size),
            EngineDeleter()
        );

        if (!engine_) {
            LOG(ERROR) << "Failed to deserialize engine. Possible plugin mismatch or corrupted file.";
            return false;
        }
        
        context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext(),
            ContextDeleter()
        );
        
        if (!context_) {
            LOG(ERROR) << "Failed to create execution context.";
            return false;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception during TensorRT init: " << e.what();
        return false;
    } catch (...) {
        LOG(ERROR) << "Unknown exception during TensorRT init.";
        return false;
    }

    return true;
}

bool EngineWrapper::forward(const std::vector<void*>& bindings, cudaStream_t stream) {
    if (!context_) return false;
    
    // For TRT 8.5+ use enqueueV3, but enqueueV2 is safer for older versions
    // Assuming standard implicit batch or dynamic batch setup
    
    // Debug checks
    if (bindings.empty()) {
        LOG(ERROR) << "EngineWrapper::forward - Bindings vector is empty!";
        return false;
    }
    
    int num_bindings = engine_->getNbBindings();
    if (bindings.size() != static_cast<size_t>(num_bindings)) {
        LOG(ERROR) << "EngineWrapper::forward - Bindings vector size (" << bindings.size() 
                   << ") does not match engine bindings count (" << num_bindings << ")";
        return false;
    }
    
    for (int i = 0; i < num_bindings; ++i) {
        if (bindings[i] == nullptr) {
            LOG(ERROR) << "EngineWrapper::forward - Binding " << i << " (" << engine_->getBindingName(i) << ") is NULL!";
            return false;
        }
        // Optional: Check pointer alignment
        // if ((reinterpret_cast<uintptr_t>(bindings[i]) % 16) != 0) {
        //     LOG(WARNING) << "EngineWrapper::forward - Binding " << i << " is not 16-byte aligned!";
        // }
    }

    bool status = context_->enqueueV2(bindings.data(), stream, nullptr);
    if (!status) {
        LOG(ERROR) << "EngineWrapper::forward - enqueueV2 failed!";
    }
    return status;
}

int EngineWrapper::get_binding_index(const std::string& name) const {
    return engine_ ? engine_->getBindingIndex(name.c_str()) : -1;
}

nvinfer1::Dims EngineWrapper::get_binding_shape(int index) const {
    return engine_->getBindingDimensions(index);
}

nvinfer1::DataType EngineWrapper::get_binding_dtype(int index) const {
    return engine_->getBindingDataType(index);
}

int EngineWrapper::get_num_bindings() const {
    return engine_->getNbBindings();
}

std::string EngineWrapper::get_binding_name(int index) const {
    return engine_->getBindingName(index);
}

// Template specializations for precision check
template <>
bool EngineWrapper::check_precision<float>() {
    // Heuristic: Check the first input/output tensor. If it's Float, pass.
    // If it's Half, fail (mismatch).
    // Note: Some layers might still be float even in FP16 mode, but usually I/O follows mode.
    // Better check: iterate all bindings.
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (engine_->getBindingDataType(i) == nvinfer1::DataType::kHALF) {
            LOG(WARNING) << "Engine binding " << i << " is HALF, but template T is float. Mismatch likely.";
            return false;
        }
    }
    return true;
}

template <>
bool EngineWrapper::check_precision<half>() {
    // If template is half, we expect bindings to be HALF (or maybe INT32 for IDs/indices).
    // If we see FLOAT binding where we expect feature map, it might be an issue or intentional mixed precision.
    // Strict check:
    bool has_half = false;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        auto dtype = engine_->getBindingDataType(i);
        if (dtype == nvinfer1::DataType::kHALF) has_half = true;
    }
    if (!has_half) {
        LOG(WARNING) << "Template T is half, but no HALF bindings found in engine. Running FP32 engine with FP16 storage?";
        // It's possible, but inefficient.
    }
    return true;
}

} // namespace bev
} // namespace sparse4d

