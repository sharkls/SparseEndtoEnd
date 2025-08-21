/*******************************************************
 文件名：TensorRT.h
 作者：sharkls
 描述：TensorRT推理类，负责加载和执行TensorRT模型
 版本：v1.0
 日期：2025-06-18
 *******************************************************/
#ifndef __TENSORRT_H__
#define __TENSORRT_H__

#include <map>
#include <vector>
#include <dlfcn.h>
#include <iostream>
#include <memory>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "./Logging.h"

class TensorRT {
 public:
  TensorRT(const std::string& engine_path = "",
           const std::string& plugin_path = "",
           const std::vector<std::string>& input_names = {},
           const std::vector<std::string>& output_names = {});
  TensorRT() = delete;
  ~TensorRT();

  bool setInputDimensions(const std::vector<std::vector<std::int32_t>>& input_dims);
  bool infer(void* const* buffers, const cudaStream_t& stream);
  bool infer(void* const* input_buffers, void* const* output_buffers, const cudaStream_t& stream);
  std::map<std::string, std::tuple<std::int32_t, std::int32_t>> getInputIndex();
  std::map<std::string, std::tuple<std::int32_t, std::int32_t>> getOutputIndex();
  void getEngineInfo() const;
  
  // 添加getter方法
  nvinfer1::ICudaEngine* getEngine() const { return engine_.get(); }

 private:
  void init();
  void auto_detect_tensors();
  const std::string engine_path_;
  const std::string plugin_path_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

#endif  // __TENSORRT_H__