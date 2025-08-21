#include "TensorRT.h"
#include <fstream>
#include <memory>

// 添加文件读取函数
namespace {
template <typename T>
std::vector<T> readfile_wrapper(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cout << "[ERROR] Read file failed: " << filename << std::endl;
    return std::vector<T>{};
  }

  file.seekg(0, std::ifstream::end);
  auto fsize = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ifstream::beg);

  std::vector<T> buffer(static_cast<size_t>(fsize) / sizeof(T));
  file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fsize));
  file.close();

  return buffer;
}
}

TensorRT::TensorRT(const std::string& engine_path,
                   const std::string& plugin_path,
                   const std::vector<std::string>& input_names,
                   const std::vector<std::string>& output_names)
    : engine_path_(engine_path), plugin_path_(plugin_path), input_names_(input_names), output_names_(output_names) {
  init();
}

TensorRT::~TensorRT() {
  // 使用智能指针管理资源，不需要手动destroy
}

void TensorRT::init() {
  std::vector<char> engine_data = readfile_wrapper<char>(engine_path_);

  if (!plugin_path_.empty()) {
    void* pluginLibraryHandle = dlopen(plugin_path_.c_str(), RTLD_LAZY);
    if (!pluginLibraryHandle) {
      std::cout << "[ERROR] Failed to load TensorRT plugin: " << plugin_path_ << std::endl;
    }
  }

  initLibNvInferPlugins(&gLogger, "");
  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));                                             // 创建推理运行时
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));   // 反序列化CUDA引擎
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());                                                 // 创建执行上下文

  // 自动检测引擎中的所有输入和输出张量
  auto_detect_tensors();

  getEngineInfo();

  if (!runtime_ || !engine_ || !context_) {
    std::cout << "[ERROR] TensorRT engine initialized failed!" << std::endl;
  }
}

void TensorRT::auto_detect_tensors() {
  // 检测TensorRT版本
  int numBindings = engine_->getNbBindings();
  
  // 清空现有的张量名称列表
  input_names_.clear();
  output_names_.clear();
  
  if (numBindings > 0) {
    // TensorRT 8.x版本 - 使用binding索引
    for (int i = 0; i < numBindings; ++i) {
      const char* tensor_name = engine_->getBindingName(i);
      bool isInput = engine_->bindingIsInput(i);
      
      if (isInput) {
        input_names_.push_back(std::string(tensor_name));
      } else {
        output_names_.push_back(std::string(tensor_name));
      }
    }
  } else {
    // TensorRT 10.x版本 - 使用IOTensor
    int numIOTensors = engine_->getNbIOTensors();
    for (int i = 0; i < numIOTensors; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      bool isInput = engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT;
      
      if (isInput) {
        input_names_.push_back(std::string(tensor_name));
      } else {
        output_names_.push_back(std::string(tensor_name));
      }
    }
  }
  
  std::cout << "[INFO] Auto-detected " << input_names_.size() << " input tensors and " 
            << output_names_.size() << " output tensors from engine" << std::endl;
}

bool TensorRT::infer(void* const* buffers, const cudaStream_t& stream) {
  // 检测TensorRT版本并选择合适的enqueue方法
  int numBindings = engine_->getNbBindings();
  
  // 检查是否为TensorRT 8.x版本（使用binding索引）
  if (numBindings > 0) {
    // TensorRT 8.x版本 - 使用enqueueV2
    std::cout << "[DEBUG] Using TensorRT 8.x enqueueV2 method" << std::endl;
    return context_->enqueueV2(buffers, stream, nullptr);
  } else {
    // TensorRT 10.x版本 - 使用enqueueV3
    std::cout << "[DEBUG] Using TensorRT 10.x enqueueV3 method" << std::endl;
    int numIOTensors = engine_->getNbIOTensors();
    for (int i = 0; i < numIOTensors; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      if (!context_->setTensorAddress(tensor_name, buffers[i])) {
        std::cout << "[ERROR] Failed to set tensor address: " << tensor_name << std::endl;
        return false;
      }
    }
    return context_->enqueueV3(stream);
  }
}

bool TensorRT::infer(void* const* input_buffers, void* const* output_buffers, const cudaStream_t& stream) {
  // 检测TensorRT版本并选择合适的enqueue方法
  int numBindings = engine_->getNbBindings();
  
  if (numBindings > 0) {
    // TensorRT 8.x版本 - 使用binding索引和enqueueV2
    std::cout << "[DEBUG] Using TensorRT 8.x enqueueV2 method" << std::endl;
    
    // 获取输入输出binding索引
    std::vector<int> input_indices;
    std::vector<int> output_indices;
    
    for (int i = 0; i < numBindings; ++i) {
      if (engine_->bindingIsInput(i)) {
        input_indices.push_back(i);
      } else {
        output_indices.push_back(i);
      }
    }
    
    // 验证输入输出数量
    if (input_indices.size() != input_names_.size() || output_indices.size() != output_names_.size()) {
      std::cout << "[ERROR] Tensor count mismatch! Got " 
                << input_indices.size() << " inputs, " << output_indices.size() << " outputs" << std::endl;
      return false;
    }
    
    // 准备所有binding的缓冲区
    std::vector<void*> all_buffers(numBindings);
    
    // 设置输入缓冲区
    for (size_t i = 0; i < input_indices.size(); ++i) {
      all_buffers[input_indices[i]] = input_buffers[i];
      //std::cout << "[DEBUG] Setting input binding " << input_indices[i] 
                //<< " at address: " << input_buffers[i] << std::endl;
    }
    
    // 设置输出缓冲区d
    for (size_t i = 0; i < output_indices.size(); ++i) {
      all_buffers[output_indices[i]] = output_buffers[i];
      //std::cout << "[DEBUG] Setting output binding " << output_indices[i] 
                //<< " at address: " << output_buffers[i] << std::endl;
    }
    
    // 调用enqueueV2
    bool result = context_->enqueueV2(all_buffers.data(), stream, nullptr);
    if (!result) {
      std::cout << "[ERROR] enqueueV2 failed!" << std::endl;
    } else {
      std::cout << "[DEBUG] enqueueV2 succeeded!" << std::endl;
    }
    return result;
    
  } else {
    // TensorRT 10.x版本 - 使用enqueueV3
    std::cout << "[DEBUG] Using TensorRT 10.x enqueueV3 method" << std::endl;
    
    // 获取所有binding的索引和顺序
    std::vector<int> input_indices;
    std::vector<int> output_indices;
    std::vector<std::string> engine_input_names;
    std::vector<std::string> engine_output_names;
    
    int numIOTensors = engine_->getNbIOTensors();
    
    // 按binding index顺序收集输入输出tensor
    for (int i = 0; i < numIOTensors; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      bool isInput = engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT;
      
      if (isInput) {
        input_indices.push_back(i);
        engine_input_names.push_back(std::string(tensor_name));
      } else {
        output_indices.push_back(i);
        engine_output_names.push_back(std::string(tensor_name));
      }
    }
    
    // 验证输入输出数量
    if (input_indices.size() != engine_input_names.size() || output_indices.size() != engine_output_names.size()) {
      std::cout << "[ERROR] Tensor count mismatch! Got " 
                << input_indices.size() << " inputs, " << output_indices.size() << " outputs" << std::endl;
      return false;
    }
    
    // 设置输入tensor地址 - 按照引擎中的顺序
    for (size_t i = 0; i < engine_input_names.size(); ++i) {
      const char* tensor_name = engine_->getIOTensorName(input_indices[i]);
      // std::cout << "[DEBUG] Setting input tensor address for: " << tensor_name 
      //           << " (index " << input_indices[i] << ") at address: " << input_buffers[i] << std::endl;
      
      if (!context_->setTensorAddress(tensor_name, input_buffers[i])) {
        std::cout << "[ERROR] Failed to set input tensor address: " << tensor_name << std::endl;
        return false;
      }
    }
    
    // 设置输出tensor地址 - 按照引擎中的顺序
    for (size_t i = 0; i < engine_output_names.size(); ++i) {
      const char* tensor_name = engine_->getIOTensorName(output_indices[i]);
      // std::cout << "[DEBUG] Setting output tensor address for: " << tensor_name 
      //           << " (index " << output_indices[i] << ") at address: " << output_buffers[i] << std::endl;
      
      if (!context_->setTensorAddress(tensor_name, output_buffers[i])) {
        std::cout << "[ERROR] Failed to set output tensor address: " << tensor_name << std::endl;
        return false;
      }
    }
    
    // 使用enqueueV3方法，适用于TensorRT 10.x
    std::cout << "[DEBUG] Calling enqueueV3..." << std::endl;
    bool result = context_->enqueueV3(stream);
    if (!result) {
      std::cout << "[ERROR] enqueueV3 failed!" << std::endl;
    } else {
      std::cout << "[DEBUG] enqueueV3 succeeded!" << std::endl;
    }
    return result;
  }
}

bool TensorRT::setInputDimensions(const std::vector<std::vector<std::int32_t>>& input_dims) {
  if (input_dims.size() != input_names_.size()) {
    std::cout << "[ERROR] Mismatched number of input dimensions!" << std::endl;
    return false;
  }

  // 检测TensorRT版本
  int numBindings = engine_->getNbBindings();
  
  if (numBindings > 0) {
    // TensorRT 8.x版本 - 使用binding索引
    for (size_t i = 0; i < input_names_.size(); ++i) {
      const std::string& input_name = input_names_[i];
      // 通过名称找到索引
      std::int32_t binding_index = -1;
      for (int j = 0; j < numBindings; ++j) {
        if (std::string(engine_->getBindingName(j)) == input_name) {
          binding_index = j;
          break;
        }
      }
      
      if (binding_index == -1) {
        std::cout << "[ERROR] Input tensor not found: " << input_name << std::endl;
        return false;
      }
      
      nvinfer1::Dims dims = engine_->getBindingDimensions(binding_index);
      if (static_cast<size_t>(dims.nbDims) != input_dims[i].size()) {
        std::cout << "Mismatched number of dimensions for input tensor: " << input_name << " "
                  << static_cast<size_t>(dims.nbDims) << " v.s. " << input_dims[i].size() << std::endl;
        return false;
      }

      for (size_t j = 0; j < static_cast<size_t>(dims.nbDims); ++j) {
        dims.d[j] = input_dims[i][j];
      }
    }
  } else {
    // TensorRT 10.x版本 - 使用IOTensor
    for (size_t i = 0; i < input_names_.size(); ++i) {
      const std::string& input_name = input_names_[i];
      // 通过名称找到索引
      std::int32_t binding_index = -1;
      for (int j = 0; j < engine_->getNbIOTensors(); ++j) {
        const char* tensor_name = engine_->getIOTensorName(j);
        if (std::string(tensor_name) == input_name) {
          binding_index = j;
          break;
        }
      }
      
      if (binding_index == -1) {
        std::cout << "[ERROR] Input tensor not found: " << input_name << std::endl;
        return false;
      }
      
      nvinfer1::Dims dims = engine_->getTensorShape(input_name.c_str());
      if (static_cast<size_t>(dims.nbDims) != input_dims[i].size()) {
        std::cout << "Mismatched number of dimensions for input tensor: " << input_name << " "
                  << static_cast<size_t>(dims.nbDims) << " v.s. " << input_dims[i].size() << std::endl;
        return false;
      }

      for (size_t j = 0; j < static_cast<size_t>(dims.nbDims); ++j) {
        dims.d[j] = input_dims[i][j];
      }
    }
  }
  return true;
}

std::map<std::string, std::tuple<std::int32_t, std::int32_t>> TensorRT::getInputIndex() {
  std::map<std::string, std::tuple<std::int32_t, std::int32_t>> inputs_index_map;
  
  // 检测TensorRT版本
  int numBindings = engine_->getNbBindings();
  
  if (numBindings > 0) {
    // TensorRT 8.x版本 - 使用binding索引
    for (size_t i = 0; i < input_names_.size(); ++i) {
      const std::string input_name = input_names_[i];
      // 通过名称找到索引
      std::int32_t binding_index = -1;
      for (int j = 0; j < numBindings; ++j) {
        if (std::string(engine_->getBindingName(j)) == input_name) {
          binding_index = j;
          break;
        }
      }
      
      if (binding_index == -1) {
        std::cout << "[ERROR] Input tensor not found: " << input_name << std::endl;
        continue;
      }
      
      const nvinfer1::Dims dims = engine_->getBindingDimensions(binding_index);
      std::int32_t tensor_length = 1;
      for (int j = 0; j < dims.nbDims; ++j) {
        tensor_length *= dims.d[j];
      }
      inputs_index_map[input_name] = std::make_tuple(tensor_length, binding_index);
    }
  } else {
    // TensorRT 10.x版本 - 使用IOTensor
    for (size_t i = 0; i < input_names_.size(); ++i) {
      const std::string input_name = input_names_[i];
      // 通过名称找到索引
      std::int32_t binding_index = -1;
      for (int j = 0; j < engine_->getNbIOTensors(); ++j) {
        const char* tensor_name = engine_->getIOTensorName(j);
        if (std::string(tensor_name) == input_name) {
          binding_index = j;
          break;
        }
      }
      
      if (binding_index == -1) {
        std::cout << "[ERROR] Input tensor not found: " << input_name << std::endl;
        continue;
      }
      
      const nvinfer1::Dims dims = engine_->getTensorShape(input_name.c_str());
      std::int32_t tensor_length = 1;
      for (int j = 0; j < dims.nbDims; ++j) {
        tensor_length *= dims.d[j];
      }
      inputs_index_map[input_name] = std::make_tuple(tensor_length, binding_index);
    }
  }
  return inputs_index_map;
}

std::map<std::string, std::tuple<std::int32_t, std::int32_t>> TensorRT::getOutputIndex() {
  std::map<std::string, std::tuple<std::int32_t, std::int32_t>> outputs_index_map;
  
  // 检测TensorRT版本
  int numBindings = engine_->getNbBindings();
  
  if (numBindings > 0) {
    // TensorRT 8.x版本 - 使用binding索引
    for (size_t i = 0; i < output_names_.size(); ++i) {
      const std::string output_name = output_names_[i];
      // 通过名称找到索引
      std::int32_t binding_index = -1;
      for (int j = 0; j < numBindings; ++j) {
        if (std::string(engine_->getBindingName(j)) == output_name) {
          binding_index = j;
          break;
        }
      }
      
      if (binding_index == -1) {
        std::cout << "[ERROR] Output tensor not found: " << output_name << std::endl;
        continue;
      }
      
      const nvinfer1::Dims dims = engine_->getBindingDimensions(binding_index);
      std::int32_t tensor_length = 1;
      for (int j = 0; j < dims.nbDims; ++j) {
        tensor_length *= dims.d[j];
      }
      outputs_index_map[output_name] = std::make_tuple(tensor_length, binding_index);
    }
  } else {
    // TensorRT 10.x版本 - 使用IOTensor
    for (size_t i = 0; i < output_names_.size(); ++i) {
      const std::string output_name = output_names_[i];
      // 通过名称找到索引
      std::int32_t binding_index = -1;
      for (int j = 0; j < engine_->getNbIOTensors(); ++j) {
        const char* tensor_name = engine_->getIOTensorName(j);
        if (std::string(tensor_name) == output_name) {
          binding_index = j;
          break;
        }
      }
      
      if (binding_index == -1) {
        std::cout << "[ERROR] Output tensor not found: " << output_name << std::endl;
        continue;
      }
      
      const nvinfer1::Dims dims = engine_->getTensorShape(output_name.c_str());
      std::int32_t tensor_length = 1;
      for (int j = 0; j < dims.nbDims; ++j) {
        tensor_length *= dims.d[j];
      }
      outputs_index_map[output_name] = std::make_tuple(tensor_length, binding_index);
    }
  }
  return outputs_index_map;
}

void TensorRT::getEngineInfo() const {
  // 检测TensorRT版本
  int numBindings = engine_->getNbBindings();
  
  if (numBindings > 0) {
    // TensorRT 8.x版本 - 使用binding索引
    std::cout << "[INFO] Get TensorRT 8.x Engine Name/Dim/Type:" << std::endl;
    
    for (int i = 0; i < numBindings; ++i) {
      const char* tensor_name = engine_->getBindingName(i);
      bool isInput = engine_->bindingIsInput(i);
      std::string type = isInput ? "Input" : "Output";
      std::cout << type << " binding " << i << ": " << std::endl;
      std::cout << "  Name: " << tensor_name << std::endl;
      
      nvinfer1::Dims dims = engine_->getBindingDimensions(i);
      std::cout << "  Dim: ";
      for (int j = 0; j < dims.nbDims; ++j) {
        std::cout << dims.d[j] << (j < dims.nbDims - 1 ? "x" : "");
      }
      std::cout << std::endl;

      nvinfer1::DataType dtype = engine_->getBindingDataType(i);
      std::cout << " Type: ";
      switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
          std::cout << "FLOAT";
          break;
        case nvinfer1::DataType::kHALF:
          std::cout << "HALF";
          break;
        case nvinfer1::DataType::kINT8:
          std::cout << "INT8";
          break;
        case nvinfer1::DataType::kINT32:
          std::cout << "INT32";
          break;
        default:
          std::cout << "UNKNOWN";
          break;
      }
      std::cout << std::endl;
    }
  } else {
    // TensorRT 10.x版本 - 使用IOTensor
    int numIOTensors = engine_->getNbIOTensors();
    std::cout << "[INFO] Get TensorRT 10.x Engine Name/Dim/Type:" << std::endl;
    
    for (int i = 0; i < numIOTensors; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      bool isInput = engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT;
      std::string type = isInput ? "Input" : "Output";
      std::cout << type << " binding " << i << ": " << std::endl;
      std::cout << "  Name: " << tensor_name << std::endl;
      
      nvinfer1::Dims dims = engine_->getTensorShape(tensor_name);
      std::cout << "  Dim: ";
      for (int j = 0; j < dims.nbDims; ++j) {
        std::cout << dims.d[j] << (j < dims.nbDims - 1 ? "x" : "");
      }
      std::cout << std::endl;

      nvinfer1::DataType dtype = engine_->getTensorDataType(tensor_name);
      std::cout << " Type: ";
      switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
          std::cout << "FLOAT";
          break;
        case nvinfer1::DataType::kHALF:
          std::cout << "HALF";
          break;
        case nvinfer1::DataType::kINT8:
          std::cout << "INT8";
          break;
        case nvinfer1::DataType::kINT32:
          std::cout << "INT32";
          break;
        default:
          std::cout << "UNKNOWN";
          break;
      }
      std::cout << std::endl;
    }
  }
}