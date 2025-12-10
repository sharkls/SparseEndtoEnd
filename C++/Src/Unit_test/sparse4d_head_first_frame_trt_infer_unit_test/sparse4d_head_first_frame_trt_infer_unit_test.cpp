// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <map>
#include <set>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include "TensorRT.h"

// Macro for checking CUDA errors
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T result, char const* const func, const char* const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result),
            _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}
static const char* _cudaGetErrorEnum(cudaError_t error) { return cudaGetErrorName(error); }

// Helper function to read binary files
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

// Simple CUDA Memory Wrapper
template <typename T>
class CudaWrapper {
 public:
  CudaWrapper(size_t size) : size_(size) {
    checkCudaErrors(cudaMalloc(&ptr_, size * sizeof(T)));
  }

  CudaWrapper(const std::vector<T>& host_data) : size_(host_data.size()) {
    checkCudaErrors(cudaMalloc(&ptr_, size_ * sizeof(T)));
    checkCudaErrors(cudaMemcpy(ptr_, host_data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  ~CudaWrapper() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  void* getCudaPtr() const { return ptr_; }

  std::vector<T> cudaMemcpyD2HResWrap() const {
    std::vector<T> host_data(size_);
    checkCudaErrors(cudaMemcpy(host_data.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    return host_data;
  }

 private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};

// Error calculation helper
float GetErrorPercentage(const std::vector<float>& a, const std::vector<float>& b, float threshold) {
  float max_error = 0.0F;
  if (a.size() != b.size()) {
    std::cout << "Size mismatch: " << a.size() << " vs " << b.size() << std::endl;
    return 1.0f;
  }

  std::vector<float> cache_errors;
  for (size_t i = 0; i < a.size(); ++i) {
    const float error = std::abs(a[i] - b[i]);
    cache_errors.push_back(error);
    if (max_error < error) {
      max_error = error;
    }
  }

  std::sort(cache_errors.begin(), cache_errors.end(), [](float a, float b) { return a > b; });

  std::vector<float> cache_roi_erros;
  for (auto x : cache_errors) {
    if (x > threshold) {
      cache_roi_erros.push_back(x);
    }
  }

  float p = 0.0f;
  if (!cache_errors.empty()) {
      p = float(cache_roi_erros.size()) / float(cache_errors.size());
  }
  
  std::cout << "Error >" << threshold << " percentage = " << p << std::endl;
  std::cout << "MaxError = " << max_error << std::endl;

  return p;
}

TEST(Sparse4dHeadFisrstFrameTrtInferUnitTest, TrtInferConsistencyVerification) {
  // Paths
  std::string engine_path = "/share/Code/Sparse4dE2E/deploy/engine/sparse4dhead1st.engine";
  std::vector<std::string> plugin_paths = {
      "/share/Code/Sparse4dE2E/deploy/dfa_plugin/lib/deformableAttentionAggr.so",
      "/share/Code/Sparse4dE2E/deploy/ln_plugin/lib/customLayerNorm.so",
      "/share/Code/Sparse4dE2E/deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so"};

  // Input/Output names (for reference, but we will map by name dynamically)
  std::vector<std::string> input_names = {"feature", "spatial_shapes", "level_start_index", "instance_feature", "anchor", "time_interval", "image_wh", "lidar2img"};
  // Outputs from config
  std::vector<std::string> output_names = {"pred_instance_feature", "pred_anchor", "pred_class_score", "pred_quality_score"};

  // CUDA Resources
  cudaEvent_t start, stop;
  cudaStream_t stream = nullptr;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  // Create TensorRT Engine
  std::cout << "Loading Engine: " << engine_path << std::endl;
  auto trt_engine = std::make_shared<TensorRT>(engine_path, plugin_paths, input_names, output_names);
  
  // Print Engine Info to check data types
  trt_engine->getEngineInfo();

  // Test Data Paths
  std::string data_root = "/share/Code/Sparse4dE2E/script/tutorial/asset/";
  
  // Read Inputs
  auto feature = readfile_wrapper<float>(data_root + "sample_0_feature_1*89760*256_float32.bin");
  auto spatial_shapes = readfile_wrapper<int32_t>(data_root + "sample_0_spatial_shapes_6*4*2_int32.bin");
  auto level_start_index = readfile_wrapper<int32_t>(data_root + "sample_0_level_start_index_6*4_int32.bin");
  auto instance_feature = readfile_wrapper<float>(data_root + "sample_0_instance_feature_1*900*256_float32.bin");
  auto anchor = readfile_wrapper<float>(data_root + "sample_0_anchor_1*900*11_float32.bin");
  auto time_interval = readfile_wrapper<float>(data_root + "sample_0_time_interval_1_float32.bin");
  auto image_wh = readfile_wrapper<float>(data_root + "sample_0_image_wh_1*6*2_float32.bin");
  auto lidar2img = readfile_wrapper<float>(data_root + "sample_0_lidar2img_1*6*4*4_float32.bin");

  // Read Expected Outputs
  auto expected_pred_instance_feature = readfile_wrapper<float>(data_root + "sample_0_pred_instance_feature_1*900*256_float32.bin");
  auto expected_pred_anchor = readfile_wrapper<float>(data_root + "sample_0_pred_anchor_1*900*11_float32.bin");
  auto expected_pred_class_score = readfile_wrapper<float>(data_root + "sample_0_pred_class_score_1*900*10_float32.bin");
  auto expected_pred_quality_score = readfile_wrapper<float>(data_root + "sample_0_pred_quality_score_1*900*2_float32.bin");

  // Prepare GPU memory for Inputs
  CudaWrapper<float> feature_gpu(feature);
  CudaWrapper<int32_t> spatial_shapes_gpu(spatial_shapes);
  CudaWrapper<int32_t> level_start_index_gpu(level_start_index);
  CudaWrapper<float> instance_feature_gpu(instance_feature);
  CudaWrapper<float> anchor_gpu(anchor);
  CudaWrapper<float> time_interval_gpu(time_interval);
  CudaWrapper<float> image_wh_gpu(image_wh);
  CudaWrapper<float> lidar2img_gpu(lidar2img);

  // Prepare GPU memory for Outputs (Sizes based on expected outputs)
  CudaWrapper<float> pred_instance_feature_gpu(expected_pred_instance_feature.size());
  CudaWrapper<float> pred_anchor_gpu(expected_pred_anchor.size());
  CudaWrapper<float> pred_class_score_gpu(expected_pred_class_score.size());
  CudaWrapper<float> pred_quality_score_gpu(expected_pred_quality_score.size());
  
  // Optional Outputs (if they exist in engine)
  CudaWrapper<float> tmp_outs0(1 * 900 * 256); // Placeholder size
  CudaWrapper<float> tmp_outs1(1 * 900 * 256);
  CudaWrapper<float> tmp_outs2(1 * 900 * 256);
  CudaWrapper<float> tmp_outs3(1 * 900 * 256);
  CudaWrapper<float> tmp_outs4(1 * 900 * 256);
  CudaWrapper<float> tmp_outs5(1 * 900 * 256);

  // Map buffers to engine bindings
  auto input_map = trt_engine->getInputIndex();
  auto output_map = trt_engine->getOutputIndex();
  
  int num_bindings = trt_engine->getEngine()->getNbBindings();
  std::vector<void*> buffers(num_bindings, nullptr);

  // Fill Inputs
  auto set_buffer = [&](const std::string& name, void* ptr) {
    if (input_map.count(name)) {
      int idx = std::get<1>(input_map[name]);
      buffers[idx] = ptr;
    } else if (output_map.count(name)) {
      int idx = std::get<1>(output_map[name]);
      buffers[idx] = ptr;
    } else {
        std::cout << "[WARN] Tensor " << name << " not found in engine bindings." << std::endl;
    }
  };

  set_buffer("feature", feature_gpu.getCudaPtr());
  set_buffer("spatial_shapes", spatial_shapes_gpu.getCudaPtr());
  set_buffer("level_start_index", level_start_index_gpu.getCudaPtr());
  set_buffer("instance_feature", instance_feature_gpu.getCudaPtr());
  set_buffer("anchor", anchor_gpu.getCudaPtr());
  set_buffer("time_interval", time_interval_gpu.getCudaPtr());
  set_buffer("image_wh", image_wh_gpu.getCudaPtr());
  set_buffer("lidar2img", lidar2img_gpu.getCudaPtr());

  set_buffer("pred_instance_feature", pred_instance_feature_gpu.getCudaPtr());
  set_buffer("pred_anchor", pred_anchor_gpu.getCudaPtr());
  set_buffer("pred_class_score", pred_class_score_gpu.getCudaPtr());
  set_buffer("pred_quality_score", pred_quality_score_gpu.getCudaPtr());

  // Try setting temp outputs if they exist
  set_buffer("tmp_outs0", tmp_outs0.getCudaPtr());
  set_buffer("tmp_outs1", tmp_outs1.getCudaPtr());
  set_buffer("tmp_outs2", tmp_outs2.getCudaPtr());
  set_buffer("tmp_outs3", tmp_outs3.getCudaPtr());
  set_buffer("tmp_outs4", tmp_outs4.getCudaPtr());
  set_buffer("tmp_outs5", tmp_outs5.getCudaPtr());

  // Verify all bindings are set
  for(int i=0; i<num_bindings; ++i) {
      if(buffers[i] == nullptr) {
          std::cout << "[ERROR] Binding " << i << " (" << trt_engine->getEngine()->getBindingName(i) << ") is NULL!" << std::endl;
      }
  }

  // Warmup
  std::cout << "Warming up..." << std::endl;
  for (int i = 0; i < 5; ++i) {
    trt_engine->infer(buffers.data(), stream);
  }
  cudaStreamSynchronize(stream);

  // Inference
  std::cout << "Running Inference..." << std::endl;
  float time_cost = 0.0f;
  checkCudaErrors(cudaEventRecord(start, stream));
  trt_engine->infer(buffers.data(), stream);
  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&time_cost, start, stop));
  std::cout << "[TensorRT Test] Sparse4d Head First Frame Inference Time Costs = " << time_cost << " [ms]" << std::endl;

  // Verification
  auto pred_instance_feature = pred_instance_feature_gpu.cudaMemcpyD2HResWrap();
  auto pred_anchor = pred_anchor_gpu.cudaMemcpyD2HResWrap();
  auto pred_class_score = pred_class_score_gpu.cudaMemcpyD2HResWrap();
  auto pred_quality_score = pred_quality_score_gpu.cudaMemcpyD2HResWrap();

  std::cout << "\nVerifying Results:" << std::endl;
  
  std::cout << "Checking Instance Feature:" << std::endl;
  float p0 = GetErrorPercentage(pred_instance_feature, expected_pred_instance_feature, 0.1);
  EXPECT_LE(p0, 0.02F);

  std::cout << "Checking Anchor:" << std::endl;
  float p1 = GetErrorPercentage(pred_anchor, expected_pred_anchor, 0.1);
  EXPECT_LE(p1, 0.02F);

  std::cout << "Checking Class Score:" << std::endl;
  float p2 = GetErrorPercentage(pred_class_score, expected_pred_class_score, 0.1);
  EXPECT_LE(p2, 0.01F);

  std::cout << "Checking Quality Score:" << std::endl;
  float p3 = GetErrorPercentage(pred_quality_score, expected_pred_quality_score, 0.1);
  EXPECT_LE(p3, 0.01F);

  // Clean up
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

