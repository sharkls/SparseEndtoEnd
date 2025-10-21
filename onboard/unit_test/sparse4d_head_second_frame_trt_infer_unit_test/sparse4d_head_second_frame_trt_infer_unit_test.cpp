// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

/*
unning main() from /home/thomasvowu/PublishRepos/SparseEnd2End/onboard/third_party/googletest/googletest/src/gtest_main.cc
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from Sparse4dHeadFisrstFrameTrtInferUnitTest
[ RUN      ] Sparse4dHeadFisrstFrameTrtInferUnitTest.TrtInferConsistencyVerification
/home/thomasvowu/PublishRepos/SparseEnd2End/onboard/assets/trt_engine/sparse4dhead1st_polygraphy.engine
[Sparse4dTrtLog][I] Loaded engine size: 84 MiB
[Sparse4dTrtLog][W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[Sparse4dTrtLog][I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +1242, GPU +318, now: CPU 1433, GPU 4523 (MiB)
[Sparse4dTrtLog][I] [MemUsageChange] Init cuDNN: CPU +2, GPU +10, now: CPU 1435, GPU 4533 (MiB)
[Sparse4dTrtLog][W] TensorRT was linked against cuDNN 8.9.0 but loaded cuDNN 8.6.0
[Sparse4dTrtLog][I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +75, now: CPU 0, GPU 75 (MiB)
[Sparse4dTrtLog][I] [MS] Running engine with multi stream info
[Sparse4dTrtLog][I] [MS] Number of aux streams is 1
[Sparse4dTrtLog][I] [MS] Number of total worker streams is 2
[Sparse4dTrtLog][I] [MS] The main stream provided by execute/enqueue calls is the first worker stream
[Sparse4dTrtLog][I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 1435, GPU 4525 (MiB)
[Sparse4dTrtLog][I] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 1435, GPU 4533 (MiB)
[Sparse4dTrtLog][W] TensorRT was linked against cuDNN 8.9.0 but loaded cuDNN 8.6.0
[Sparse4dTrtLog][I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +101, now: CPU 0, GPU 176 (MiB)
[TensorRT Test] Sparse4d Head First Frame Inference (FP32) Time Costs = 28.7016 [ms]
Error >0.1 percentage is: 0
MaxError = 0.000582129
Pred_instance_feature: max=3.25359 min=-4.15375
Expd_instance_feature: max=3.25358 min=-4.15376

Error >0.1 percentage is: 0
MaxError = 0.000371933
Pred_anchor : max=55.3649 min=-54.0855
Expd_anchor: max=55.3649 min=-54.0855

Error >0.1 percentage is: 0
MaxError = 0.000299931
Pred_class_score: max=2.15206 min=-9.23959
Pred_class_score: max=2.15207 min=-9.23956

Error >0.1 percentage is: 0
MaxError = 0.000105441
Pred_quality_score: max=2.10492 min=-2.64644
Expd_quality_score: max=2.10492 min=-2.64644

[       OK ] Sparse4dHeadFisrstFrameTrtInferUnitTest.TrtInferConsistencyVerification (2539 ms)
[----------] 1 test from Sparse4dHeadFisrstFrameTrtInferUnitTest (2539 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (2539 ms total)
[  PASSED  ] 1 test.
*/

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>
#include <cmath>
#include <algorithm>

#include "../../common/cuda_wrapper.cu.h"
#include "../../common/utils.h"
#include "../../preprocessor/parameters_parser.h"
#include "../../tensorrt/tensorrt.h"

namespace sparse_end2end {
namespace engine {

// 计算最大绝对误差
float GetMaxAbsError(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<float>::max();
  }
  
  float max_error = 0.0F;
  for (size_t i = 0; i < a.size(); ++i) {
    const float error = std::abs(a[i] - b[i]);
    if (max_error < error) {
      max_error = error;
    }
  }
  return max_error;
}

// 计算余弦距离
float GetCosineDistance(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return 1.0F; // 最大余弦距离
  }
  
  float dot_product = 0.0F;
  float norm_a = 0.0F;
  float norm_b = 0.0F;
  
  for (size_t i = 0; i < a.size(); ++i) {
    dot_product += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  
  norm_a = std::sqrt(norm_a);
  norm_b = std::sqrt(norm_b);
  
  if (norm_a < 1e-8F || norm_b < 1e-8F) {
    return 1.0F;
  }
  
  return 1.0F - dot_product / (norm_a * norm_b);
}

// 读取单个样本的数据
std::tuple<std::vector<float>, std::vector<int32_t>, std::vector<int32_t>, 
           std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>, std::vector<float>, std::vector<int32_t>, 
           std::vector<int32_t>, std::vector<float>, std::vector<float>,
           std::vector<int32_t>, std::vector<float>, std::vector<float>, 
           std::vector<float>, std::vector<float>>
ReadSampleData(int sample_id) {
  std::string prefix = "/share/Code/Sparse4dE2E/script/tutorial/asset/";
  
  // 输入数据
  auto feature = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_feature_1*89760*256_float32.bin");
  auto spatial_shapes = common::readfile_wrapper<int32_t>(prefix + "sample_" + std::to_string(sample_id) + "_spatial_shapes_6*4*2_int32.bin");
  auto level_start_index = common::readfile_wrapper<int32_t>(prefix + "sample_" + std::to_string(sample_id) + "_level_start_index_6*4_int32.bin");
  auto instance_feature = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_instance_feature_1*900*256_float32.bin");
  auto anchor = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_anchor_1*900*11_float32.bin");
  auto time_interval = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_time_interval_1_float32.bin");
  auto temp_instance_feature = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_temp_instance_feature_1*600*256_float32.bin");
  auto temp_anchor = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_temp_anchor_1*600*11_float32.bin");
  auto mask = common::readfile_wrapper<int32_t>(prefix + "sample_" + std::to_string(sample_id) + "_mask_1_int32.bin");
  auto track_id = common::readfile_wrapper<int32_t>(prefix + "sample_" + std::to_string(sample_id) + "_track_id_1*900_int32.bin");
  auto image_wh = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_image_wh_1*6*2_float32.bin");
  auto lidar2img = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_lidar2img_1*6*4*4_float32.bin");
  
  // 期望输出数据
  auto expected_pred_track_id = common::readfile_wrapper<int32_t>(prefix + "sample_" + std::to_string(sample_id) + "_pred_track_id_1*900_int32.bin");
  auto expected_pred_instance_feature = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_pred_instance_feature_1*900*256_float32.bin");
  auto expected_pred_anchor = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_pred_anchor_1*900*11_float32.bin");
  auto expected_pred_class_score = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_pred_class_score_1*900*10_float32.bin");
  auto expected_pred_quality_score = common::readfile_wrapper<float>(prefix + "sample_" + std::to_string(sample_id) + "_pred_quality_score_1*900*2_float32.bin");
  
  return std::make_tuple(feature, spatial_shapes, level_start_index, instance_feature, anchor, time_interval,
                        temp_instance_feature, temp_anchor, mask, track_id, image_wh, lidar2img,
                        expected_pred_track_id, expected_pred_instance_feature, expected_pred_anchor,
                        expected_pred_class_score, expected_pred_quality_score);
}

// 推理一致性验证函数
void InferenceConsistencyValidation(const std::vector<float>& predicted, const std::vector<float>& expected, 
                                   const std::string& name) {
  float max_abs_error = GetMaxAbsError(predicted, expected);
  float cosine_dist = GetCosineDistance(predicted, expected);
  
  std::cout << "[max(abs()) error] " << name << " = " << max_abs_error << std::endl;
  std::cout << "[cosine_distance ] " << name << " = " << cosine_dist << std::endl;
}

TEST(Sparse4dHeadSecondFrameTrtInferUnitTest, TrtInferConsistencyVerification) {
  // 1.获取当前路径
  std::filesystem::path current_dir = std::filesystem::current_path();
  const common::E2EParams params = preprocessor::parseParams(current_dir / "../../assets/model_cfg.yaml");

  // 2.获取 engine 路径和输入输出名称
  std::string sparse4d_head2nd_engine_path = params.sparse4d_head2nd_engine.engine_path;
  std::string multiview_multiscale_deformable_attention_aggregation_path =
      params.model_cfg.multiview_multiscale_deformable_attention_aggregation_path;
  std::vector<std::string> sparse4d_head2nd_engine_input_names = params.sparse4d_head2nd_engine.input_names;
  std::vector<std::string> sparse4d_head2nd_engine_output_names = params.sparse4d_head2nd_engine.output_names;

  // 3.创建 cuda event 和 stream
  cudaEvent_t start, stop;
  cudaStream_t stream = nullptr;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  // 4.创建 TensorRT 引擎
  std::cout << sparse4d_head2nd_engine_path << std::endl;
  std::shared_ptr<TensorRT> trt_engine = std::make_shared<TensorRT>(
      sparse4d_head2nd_engine_path, multiview_multiscale_deformable_attention_aggregation_path,
      sparse4d_head2nd_engine_input_names, sparse4d_head2nd_engine_output_names);

  // 5.测试多个样本
  for (int sample_id = 1; sample_id <= 2; ++sample_id) {
    std::cout << "Testing sample " << sample_id << std::endl;
    
    // 读取样本数据
    auto [feature, spatial_shapes, level_start_index, instance_feature, anchor, time_interval,
          temp_instance_feature, temp_anchor, mask, track_id, image_wh, lidar2img,
          expected_pred_track_id, expected_pred_instance_feature, expected_pred_anchor,
          expected_pred_class_score, expected_pred_quality_score] = ReadSampleData(sample_id);

    // 6.检查测试样本大小 
    EXPECT_EQ(feature.size(), 1 * 89760 * 256);
    EXPECT_EQ(spatial_shapes.size(), 6 * 4 * 2);
    EXPECT_EQ(level_start_index.size(), 6 * 4);
    EXPECT_EQ(instance_feature.size(), 1 * 900 * 256);
    EXPECT_EQ(anchor.size(), 1 * 900 * 11);
    EXPECT_EQ(time_interval.size(), 1);
    EXPECT_EQ(temp_instance_feature.size(), 1 * 600 * 256);
    EXPECT_EQ(temp_anchor.size(), 1 * 600 * 11);
    EXPECT_EQ(mask.size(), 1);
    EXPECT_EQ(track_id.size(), 1 * 900);
    EXPECT_EQ(image_wh.size(), 1 * 6 * 2);
    EXPECT_EQ(lidar2img.size(), 1 * 6 * 4 * 4);
    EXPECT_EQ(expected_pred_track_id.size(), 1 * 900);
    EXPECT_EQ(expected_pred_instance_feature.size(), 1 * 900 * 256);
    EXPECT_EQ(expected_pred_anchor.size(), 1 * 900 * 11);
    EXPECT_EQ(expected_pred_class_score.size(), 1 * 900 * 10);
    EXPECT_EQ(expected_pred_quality_score.size(), 1 * 900 * 2);

    // 7.创建 warmup 数据
    const common::CudaWrapper<float> warmup_feature_gpu(feature);
    const common::CudaWrapper<int32_t> warmup_spatial_shapes_gpu(spatial_shapes);
    const common::CudaWrapper<int32_t> warmup_level_start_index_gpu(level_start_index);
    const common::CudaWrapper<float> warmup_instance_feature_gpu(instance_feature);
    const common::CudaWrapper<float> warmup_anchor_gpu(anchor);
    const common::CudaWrapper<float> warmup_time_interval_gpu(time_interval);
    const common::CudaWrapper<float> warmup_temp_instance_feature_gpu(temp_instance_feature);
    const common::CudaWrapper<float> warmup_temp_anchor_gpu(temp_anchor);
    const common::CudaWrapper<int32_t> warmup_mask_gpu(mask);
    const common::CudaWrapper<int32_t> warmup_track_id_gpu(track_id);
    const common::CudaWrapper<float> warmup_image_wh_gpu(image_wh);
    const common::CudaWrapper<float> warmup_lidar2img_gpu(lidar2img);

    // 8.创建 warmup 输出
    common::CudaWrapper<float> warmup_tmp_outs0(1 * 900 * 256);
    common::CudaWrapper<float> warmup_tmp_outs1(1 * 900 * 256);
    common::CudaWrapper<float> warmup_tmp_outs2(1 * 900 * 256);
    common::CudaWrapper<float> warmup_tmp_outs3(1 * 900 * 256);
    common::CudaWrapper<float> warmup_tmp_outs4(1 * 900 * 256);
    common::CudaWrapper<float> warmup_tmp_outs5(1 * 900 * 256);
    common::CudaWrapper<int32_t> warmup_pred_track_id_gpu(1 * 900);
    common::CudaWrapper<float> warmup_pred_instance_feature_gpu(1 * 900 * 256);
    common::CudaWrapper<float> warmup_pred_anchor_gpu(1 * 900 * 11);
    common::CudaWrapper<float> warmup_pred_class_score_gpu(1 * 900 * 10);
    common::CudaWrapper<float> warmup_pred_quality_score_gpu(1 * 900 * 2);

    // 9.创建 warmup 输入
    std::vector<void*> warmup_buffers;
    warmup_buffers.push_back(warmup_feature_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_spatial_shapes_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_level_start_index_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_instance_feature_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_anchor_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_time_interval_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_temp_instance_feature_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_temp_anchor_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_mask_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_track_id_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_image_wh_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_lidar2img_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_tmp_outs0.getCudaPtr());
    warmup_buffers.push_back(warmup_tmp_outs1.getCudaPtr());
    warmup_buffers.push_back(warmup_tmp_outs2.getCudaPtr());
    warmup_buffers.push_back(warmup_tmp_outs3.getCudaPtr());
    warmup_buffers.push_back(warmup_tmp_outs4.getCudaPtr());
    warmup_buffers.push_back(warmup_tmp_outs5.getCudaPtr());
    warmup_buffers.push_back(warmup_pred_track_id_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_pred_instance_feature_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_pred_anchor_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_pred_class_score_gpu.getCudaPtr());
    warmup_buffers.push_back(warmup_pred_quality_score_gpu.getCudaPtr());

    // Warmup
    for (int i = 0; i < 5; ++i) {
      if (trt_engine->infer(warmup_buffers.data(), stream) != true) {
        std::cout << "[ERROR] TensorRT engine inference failed during warmup" << std::endl;
      }
      cudaStreamSynchronize(stream);
    }

    // 10.创建实际推理数据
    const common::CudaWrapper<float> feature_gpu(feature);
    const common::CudaWrapper<int32_t> spatial_shapes_gpu(spatial_shapes);
    const common::CudaWrapper<int32_t> level_start_index_gpu(level_start_index);
    const common::CudaWrapper<float> instance_feature_gpu(instance_feature);
    const common::CudaWrapper<float> anchor_gpu(anchor);
    const common::CudaWrapper<float> time_interval_gpu(time_interval);
    const common::CudaWrapper<float> temp_instance_feature_gpu(temp_instance_feature);
    const common::CudaWrapper<float> temp_anchor_gpu(temp_anchor);
    const common::CudaWrapper<int32_t> mask_gpu(mask);
    const common::CudaWrapper<int32_t> track_id_gpu(track_id);
    const common::CudaWrapper<float> image_wh_gpu(image_wh);
    const common::CudaWrapper<float> lidar2img_gpu(lidar2img);
    
    common::CudaWrapper<float> tmp_outs0(1 * 900 * 256);
    common::CudaWrapper<float> tmp_outs1(1 * 900 * 256);
    common::CudaWrapper<float> tmp_outs2(1 * 900 * 256);
    common::CudaWrapper<float> tmp_outs3(1 * 900 * 256);
    common::CudaWrapper<float> tmp_outs4(1 * 900 * 256);
    common::CudaWrapper<float> tmp_outs5(1 * 900 * 256);
    common::CudaWrapper<int32_t> pred_track_id_gpu(1 * 900);
    common::CudaWrapper<float> pred_instance_feature_gpu(1 * 900 * 256);
    common::CudaWrapper<float> pred_anchor_gpu(1 * 900 * 11);
    common::CudaWrapper<float> pred_class_score_gpu(1 * 900 * 10);
    common::CudaWrapper<float> pred_quality_score_gpu(1 * 900 * 2);

    // 11.创建推理缓冲区
    std::vector<void*> buffers;
    buffers.push_back(feature_gpu.getCudaPtr());
    buffers.push_back(spatial_shapes_gpu.getCudaPtr());
    buffers.push_back(level_start_index_gpu.getCudaPtr());
    buffers.push_back(instance_feature_gpu.getCudaPtr());
    buffers.push_back(anchor_gpu.getCudaPtr());
    buffers.push_back(time_interval_gpu.getCudaPtr());
    buffers.push_back(temp_instance_feature_gpu.getCudaPtr());
    buffers.push_back(temp_anchor_gpu.getCudaPtr());
    buffers.push_back(mask_gpu.getCudaPtr());
    buffers.push_back(track_id_gpu.getCudaPtr());
    buffers.push_back(image_wh_gpu.getCudaPtr());
    buffers.push_back(lidar2img_gpu.getCudaPtr());
    buffers.push_back(tmp_outs0.getCudaPtr());
    buffers.push_back(tmp_outs1.getCudaPtr());
    buffers.push_back(tmp_outs2.getCudaPtr());
    buffers.push_back(tmp_outs3.getCudaPtr());
    buffers.push_back(tmp_outs4.getCudaPtr());
    buffers.push_back(tmp_outs5.getCudaPtr());
    buffers.push_back(pred_track_id_gpu.getCudaPtr());
    buffers.push_back(pred_instance_feature_gpu.getCudaPtr());
    buffers.push_back(pred_anchor_gpu.getCudaPtr());
    buffers.push_back(pred_class_score_gpu.getCudaPtr());
    buffers.push_back(pred_quality_score_gpu.getCudaPtr());

    // 12.开始推理
    float time_cost = 0.0f;
    checkCudaErrors(cudaEventRecord(start, stream));
    if (!trt_engine->infer(buffers.data(), stream)) {
      std::cout << "[ERROR] TensorRT engine inference failed " << std::endl;
    }
    checkCudaErrors(cudaEventRecord(stop, stream));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_cost, start, stop));
    std::cout << "[TensorRT Test] Sparse4d Head Second Frame Inference (FP32) Time Costs = " << time_cost << " [ms]"
              << std::endl;
    cudaStreamSynchronize(stream);

    // 13.获取推理结果
    auto pred_track_id = pred_track_id_gpu.cudaMemcpyD2HResWrap();
    auto pred_instance_feature = pred_instance_feature_gpu.cudaMemcpyD2HResWrap();
    auto pred_anchor = pred_anchor_gpu.cudaMemcpyD2HResWrap();
    auto pred_class_score = pred_class_score_gpu.cudaMemcpyD2HResWrap();
    auto pred_quality_score = pred_quality_score_gpu.cudaMemcpyD2HResWrap();

    // 14.进行推理一致性验证
    std::cout << "Sample " << (sample_id - 1) << " inference consistency validatation:" << std::endl;
    
    // 将int32_t转换为float进行计算
    std::vector<float> pred_track_id_float(pred_track_id.begin(), pred_track_id.end());
    std::vector<float> expected_pred_track_id_float(expected_pred_track_id.begin(), expected_pred_track_id.end());
    
    InferenceConsistencyValidation(pred_track_id_float, expected_pred_track_id_float, "pred_track_id");
    InferenceConsistencyValidation(pred_instance_feature, expected_pred_instance_feature, "pred_instance_feature");
    InferenceConsistencyValidation(pred_anchor, expected_pred_anchor, "pred_anchor");
    InferenceConsistencyValidation(pred_class_score, expected_pred_class_score, "pred_class_score");
    InferenceConsistencyValidation(pred_quality_score, expected_pred_quality_score, "pred_quality_score");
    
    std::cout << std::endl;
  }

  // 15.销毁cuda event 和 stream
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));
}

}  // namespace engine
}  // namespace sparse_end2end