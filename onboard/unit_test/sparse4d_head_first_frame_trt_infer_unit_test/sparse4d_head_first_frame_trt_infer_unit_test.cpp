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
Expd_class_score: max=2.15207 min=-9.23956

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
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <map>
#include <set>
#include <cmath>

#include "../../common/cuda_wrapper.cu.h"
#include "../../common/utils.h"
#include "../../preprocessor/parameters_parser.h"
#include "../../tensorrt/tensorrt.h"

namespace sparse_end2end {
namespace engine {

float GetErrorPercentage(const std::vector<float>& a, const std::vector<float>& b, float threshold) {
  float max_error = 0.0F;
  if (a.size() != b.size()) {
    max_error = std::numeric_limits<float>::max();
  }

  std::vector<float> cache_errors;
  for (size_t i = 0; i < a.size(); ++i) {
    const float error = std::abs(a[i] - b[i]);
    cache_errors.push_back(error);
    if (max_error < error) {
      max_error = error;
    }
  }

  std::sort(cache_errors.begin(), cache_errors.end(), [](int a, int b) { return a > b; });

  std::vector<float> cache_roi_erros;
  for (auto x : cache_errors) {
    if (x > threshold) {
      cache_roi_erros.push_back(x);
    }
  }

  float p = float(cache_roi_erros.size()) / float(cache_errors.size());
  std::cout << "Error >" << threshold << " percentage = " << p << std::endl;
  std::cout << "MaxError = " << max_error << std::endl;

  return p;
}

TEST(Sparse4dHeadFisrstFrameTrtInferUnitTest, TrtInferConsistencyVerification) {
  // 1.获取当前路径
  std::filesystem::path current_dir = std::filesystem::current_path();
  const common::E2EParams params = preprocessor::parseParams(current_dir / "../../assets/model_cfg.yaml");

  // 2.获取 engine 路径和输入输出名称
  std::string sparse4d_head1st_engine_path = params.sparse4d_head1st_engine.engine_path;
  std::string multiview_multiscale_deformable_attention_aggregation_path =
      params.model_cfg.multiview_multiscale_deformable_attention_aggregation_path;
  std::vector<std::string> sparse4d_head1st_engine_input_names = params.sparse4d_head1st_engine.input_names;
  std::vector<std::string> sparse4d_head1st_engine_output_names = params.sparse4d_head1st_engine.output_names;

  // 3.获取特征提取 shape
  std::vector<std::uint32_t> sparse4d_extract_feat_shape_lc = params.model_cfg.sparse4d_extract_feat_shape_lc;

  // 4.创建 cuda event 和 stream
  cudaEvent_t start, stop;
  cudaStream_t stream = nullptr;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  // 5.创建 TensorRT 引擎
  std::cout << sparse4d_head1st_engine_path << std::endl;
  std::shared_ptr<TensorRT> trt_engine = std::make_shared<TensorRT>(
      sparse4d_head1st_engine_path, multiview_multiscale_deformable_attention_aggregation_path,
      sparse4d_head1st_engine_input_names, sparse4d_head1st_engine_output_names);

  // 6.获取测试样本
  std::tuple<std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string,
             std::string, std::string, std::string, std::string>
      test_sample{
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_feature_1*89760*256_float32.bin",
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_spatial_shapes_6*4*2_int32.bin",
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_level_start_index_6*4_int32.bin",
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_instance_feature_1*900*256_float32.bin",
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_anchor_1*900*11_float32.bin",
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_time_interval_1_float32.bin",
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_image_wh_1*6*2_float32.bin",
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_lidar2img_1*6*4*4_float32.bin",
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_pred_instance_feature_1*900*256_float32.bin",
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_pred_anchor_1*900*11_float32.bin",
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_pred_class_score_1*900*10_float32.bin",
          "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_pred_quality_score_1*900*2_float32.bin"};

  // 7.读取测试样本
  const auto feature = common::readfile_wrapper<float>(std::get<0>(test_sample));
  const auto spatial_shapes = common::readfile_wrapper<int32_t>(std::get<1>(test_sample));
  const auto level_start_index = common::readfile_wrapper<int32_t>(std::get<2>(test_sample));
  const auto instance_feature = common::readfile_wrapper<float>(std::get<3>(test_sample));
  const auto anchor = common::readfile_wrapper<float>(std::get<4>(test_sample));
  const auto time_interval = common::readfile_wrapper<float>(std::get<5>(test_sample));
  const auto image_wh = common::readfile_wrapper<float>(std::get<6>(test_sample));
  const auto lidar2img = common::readfile_wrapper<float>(std::get<7>(test_sample));
  const auto expected_pred_instance_feature = common::readfile_wrapper<float>(std::get<8>(test_sample));
  const auto expected_pred_anchor = common::readfile_wrapper<float>(std::get<9>(test_sample));
  const auto expected_pred_class_score = common::readfile_wrapper<float>(std::get<10>(test_sample));
  const auto expected_pred_quality_score = common::readfile_wrapper<float>(std::get<11>(test_sample));

  // 8.检查测试样本大小 
  EXPECT_EQ(feature.size(), 1 * 89760 * 256);
  EXPECT_EQ(spatial_shapes.size(), 6 * 4 * 2);
  EXPECT_EQ(level_start_index.size(), 6 * 4);
  EXPECT_EQ(instance_feature.size(), 1 * 900 * 256);
  EXPECT_EQ(anchor.size(), 1 * 900 * 11);
  EXPECT_EQ(time_interval.size(), 1);
  EXPECT_EQ(image_wh.size(), 1 * 6 * 2);
  EXPECT_EQ(lidar2img.size(), 1 * 6 * 4 * 4);
  EXPECT_EQ(expected_pred_instance_feature.size(), 1 * 900 * 256);
  EXPECT_EQ(expected_pred_anchor.size(), 1 * 900 * 11);
  EXPECT_EQ(expected_pred_class_score.size(), 1 * 900 * 10);
  EXPECT_EQ(expected_pred_quality_score.size(), 1 * 900 * 2);

  // 9.创建 warmup 数据
  const common::CudaWrapper<float> warmup_feature_gpu(feature);
  const common::CudaWrapper<int32_t> warmup_spatial_shapes_gpu(spatial_shapes);
  const common::CudaWrapper<int32_t> warmup_level_start_index_gpu(level_start_index);
  const common::CudaWrapper<float> warmup_instance_feature_gpu(instance_feature);
  const common::CudaWrapper<float> warmup_anchor_gpu(anchor);
  const common::CudaWrapper<float> warmup_time_interval_gpu(time_interval);
  const common::CudaWrapper<float> warmup_image_wh_gpu(image_wh);
  const common::CudaWrapper<float> warmup_lidar2img_gpu(lidar2img);

  // 10.创建 warmup 输出
  common::CudaWrapper<float> warmup_tmp_outs0(1 * 900 * 256);
  common::CudaWrapper<float> warmup_tmp_outs1(1 * 900 * 256);
  common::CudaWrapper<float> warmup_tmp_outs2(1 * 900 * 256);
  common::CudaWrapper<float> warmup_tmp_outs3(1 * 900 * 256);
  common::CudaWrapper<float> warmup_tmp_outs4(1 * 900 * 256);
  common::CudaWrapper<float> warmup_tmp_outs5(1 * 900 * 256);
  common::CudaWrapper<float> warmup_pred_instance_feature_gpu(1 * 900 * 256);
  common::CudaWrapper<float> warmup_pred_anchor_gpu(1 * 900 * 11);
  common::CudaWrapper<float> warmup_pred_class_score_gpu(1 * 900 * 10);
  common::CudaWrapper<float> warmup_pred_quality_score_gpu(1 * 900 * 2);

  // 11.创建 warmup 输入
  std::vector<void*> warmup_buffers;
  warmup_buffers.push_back(warmup_feature_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_spatial_shapes_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_level_start_index_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_instance_feature_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_anchor_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_time_interval_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_image_wh_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_lidar2img_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs0.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs1.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs2.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs3.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs4.getCudaPtr());
  warmup_buffers.push_back(warmup_tmp_outs5.getCudaPtr());
  warmup_buffers.push_back(warmup_pred_instance_feature_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_pred_anchor_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_pred_class_score_gpu.getCudaPtr());
  warmup_buffers.push_back(warmup_pred_quality_score_gpu.getCudaPtr());

  // trt_engine->getEngineInfo();

  // // Warmup
  for (int i = 0; i < 5; ++i) {
    if (trt_engine->infer(warmup_buffers.data(), stream) != true) {
      std::cout << "[ERROR] TensorRT engine inference failed during warmup" << std::endl;
    }
    cudaStreamSynchronize(stream);
  }

  // 12.创建输入
  const common::CudaWrapper<float> feature_gpu(feature);
  const common::CudaWrapper<int32_t> spatial_shapes_gpu(spatial_shapes);
  const common::CudaWrapper<int32_t> level_start_index_gpu(level_start_index);
  const common::CudaWrapper<float> instance_feature_gpu(instance_feature);
  const common::CudaWrapper<float> anchor_gpu(anchor);
  const common::CudaWrapper<float> time_interval_gpu(time_interval);
  const common::CudaWrapper<float> image_wh_gpu(image_wh);
  const common::CudaWrapper<float> lidar2img_gpu(lidar2img);
  common::CudaWrapper<float> tmp_outs0(1 * 900 * 256);
  common::CudaWrapper<float> tmp_outs1(1 * 900 * 256);
  common::CudaWrapper<float> tmp_outs2(1 * 900 * 256);
  common::CudaWrapper<float> tmp_outs3(1 * 900 * 256);
  common::CudaWrapper<float> tmp_outs4(1 * 900 * 256);
  common::CudaWrapper<float> tmp_outs5(1 * 900 * 256);
  common::CudaWrapper<float> pred_instance_feature_gpu(1 * 900 * 256);
  common::CudaWrapper<float> pred_anchor_gpu(1 * 900 * 11);
  common::CudaWrapper<float> pred_class_score_gpu(1 * 900 * 10);
  common::CudaWrapper<float> pred_quality_score_gpu(1 * 900 * 2);

  // 13.创建输入
  std::vector<void*> buffers;
  buffers.push_back(feature_gpu.getCudaPtr());
  buffers.push_back(spatial_shapes_gpu.getCudaPtr());
  buffers.push_back(level_start_index_gpu.getCudaPtr());
  buffers.push_back(instance_feature_gpu.getCudaPtr());
  buffers.push_back(anchor_gpu.getCudaPtr());
  buffers.push_back(time_interval_gpu.getCudaPtr());
  buffers.push_back(image_wh_gpu.getCudaPtr());
  buffers.push_back(lidar2img_gpu.getCudaPtr());
  buffers.push_back(tmp_outs0.getCudaPtr());
  buffers.push_back(tmp_outs1.getCudaPtr());
  buffers.push_back(tmp_outs2.getCudaPtr());
  buffers.push_back(tmp_outs3.getCudaPtr());
  buffers.push_back(tmp_outs4.getCudaPtr());
  buffers.push_back(tmp_outs5.getCudaPtr());
  buffers.push_back(pred_instance_feature_gpu.getCudaPtr());
  buffers.push_back(pred_anchor_gpu.getCudaPtr());
  buffers.push_back(pred_class_score_gpu.getCudaPtr());
  buffers.push_back(pred_quality_score_gpu.getCudaPtr());

  // 14.开始推理
  float time_cost = 0.0f;
  checkCudaErrors(cudaEventRecord(start, stream));
  if (!trt_engine->infer(buffers.data(), stream)) {
    std::cout << "[ERROR] TensorRT engine inference failed " << std::endl;
  }
  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&time_cost, start, stop));
  std::cout << "[TensorRT Test] Sparse4d Head First Frame Inference (FP32) Time Costs = " << time_cost << " [ms]"
            << std::endl;
  cudaStreamSynchronize(stream);

  // 15.获取推理结果
  auto pred_instance_feature = pred_instance_feature_gpu.cudaMemcpyD2HResWrap();
  auto pred_anchor = pred_anchor_gpu.cudaMemcpyD2HResWrap();
  auto pred_class_score = pred_class_score_gpu.cudaMemcpyD2HResWrap();
  auto pred_quality_score = pred_quality_score_gpu.cudaMemcpyD2HResWrap();

  // 16.计算instance_feature误差
  const float p0 = GetErrorPercentage(pred_instance_feature, expected_pred_instance_feature, 0.1);
  EXPECT_LE(p0, 0.02F);
  std::cout << "Pred_instance_feature: max="
            << *std::max_element(pred_instance_feature.begin(), pred_instance_feature.end())
            << " min=" << *std::min_element(pred_instance_feature.begin(), pred_instance_feature.end()) << std::endl;
  std::cout << "Expd_instance_feature: max="
            << *std::max_element(expected_pred_instance_feature.begin(), expected_pred_instance_feature.end())
            << " min="
            << *std::min_element(expected_pred_instance_feature.begin(), expected_pred_instance_feature.end())
            << std::endl
            << std::endl;

  // 17.计算anchor误差
  const float p1 = GetErrorPercentage(pred_anchor, expected_pred_anchor, 0.1);
  EXPECT_LE(p1, 0.02F);
  std::cout << "Pred_anchor : max=" << *std::max_element(pred_anchor.begin(), pred_anchor.end())
            << " min=" << *std::min_element(pred_anchor.begin(), pred_anchor.end()) << std::endl;
  std::cout << "Expd_anchor: max=" << *std::max_element(expected_pred_anchor.begin(), expected_pred_anchor.end())
            << " min=" << *std::min_element(expected_pred_anchor.begin(), expected_pred_anchor.end()) << std::endl
            << std::endl;

  // 18.计算class_score误差
  const float p2 = GetErrorPercentage(pred_class_score, expected_pred_class_score, 0.1);
  EXPECT_LE(p2, 0.01F);
  std::cout << "Pred_class_score: max=" << *std::max_element(pred_class_score.begin(), pred_class_score.end())
            << " min=" << *std::min_element(pred_class_score.begin(), pred_class_score.end()) << std::endl;
  std::cout << "Expd_class_score: max="
            << *std::max_element(expected_pred_class_score.begin(), expected_pred_class_score.end())
            << " min=" << *std::min_element(expected_pred_class_score.begin(), expected_pred_class_score.end())
            << std::endl
            << std::endl;

  // 19.计算quality_score误差
  const float p3 = GetErrorPercentage(pred_quality_score, expected_pred_quality_score, 0.1);
  EXPECT_LE(p3, 0.01F);
  std::cout << "Pred_quality_score: max=" << *std::max_element(pred_quality_score.begin(), pred_quality_score.end())
            << " min=" << *std::min_element(pred_quality_score.begin(), pred_quality_score.end()) << std::endl;
  std::cout << "Expd_quality_score: max="
            << *std::max_element(expected_pred_quality_score.begin(), expected_pred_quality_score.end())
            << " min=" << *std::min_element(expected_pred_quality_score.begin(), expected_pred_quality_score.end())
            << std::endl
            << std::endl;

  // 20.置信度过滤、类别过滤和二维NMS去重
  const float confidence_threshold = 0.2f;
  std::cout << "=== 置信度过滤和类别过滤结果 (阈值: " << confidence_threshold << ") ===" << std::endl;
  
  // 定义目标结构体
  struct DetectedObject {
    int index;
    float confidence;
    int class_id;
    std::string class_name;
    float x, y, z, l, w, h, yaw;
    
    DetectedObject() : index(0), confidence(0.0f), class_id(0), x(0.0f), y(0.0f), z(0.0f), 
                      l(0.0f), w(0.0f), h(0.0f), yaw(0.0f) {}
  };
  
  // 类别名称映射
  std::string class_names[] = {"car", "truck", "bus", "trailer", "construction_vehicle", 
                              "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier"};
  
  // 需要过滤的类别ID（障碍物和交通锥桶）
  std::vector<int> filtered_classes = {8, 9};  // traffic_cone, barrier
  
  // 统计通过过滤的目标数量
  int total_objects = 900;
  int confidence_filtered_objects = 0;
  int class_filtered_objects = 0;
  std::vector<DetectedObject> valid_objects;
  
  // 从推理结果中提取目标信息
  for (int i = 0; i < total_objects; ++i) {
    // 获取当前目标的类别分数 (1×900×10 格式)
    float max_confidence = 0.0f;
    int best_class = 0;
    
    for (int j = 0; j < 10; ++j) {  // 10个类别
      int score_idx = i * 10 + j;
      if (score_idx < pred_class_score.size()) {
        // 应用sigmoid激活函数
        float sigmoid_score = 1.0f / (1.0f + std::exp(-pred_class_score[score_idx]));
        if (sigmoid_score > max_confidence) {
          max_confidence = sigmoid_score;
          best_class = j;
        }
      }
    }
    
    // 从quality_score中获取质量分数 (1×900×2 格式)
    float quality_score = 0.0f;
    int quality_idx = i * 2;
    if (quality_idx + 1 < pred_quality_score.size()) {
      float quality_score1 = 1.0f / (1.0f + std::exp(-pred_quality_score[quality_idx]));
      float quality_score2 = 1.0f / (1.0f + std::exp(-pred_quality_score[quality_idx + 1]));
      quality_score = std::max(quality_score1, quality_score2);
    }
    
    // 综合置信度：使用类别分数和质量分数的组合
    float final_confidence = max_confidence * quality_score;
    
    // 如果综合置信度太低，使用质量分数
    if (final_confidence < 0.1f) {
      final_confidence = quality_score;
    }
    
    // 应用置信度阈值过滤
    if (final_confidence >= confidence_threshold) {
      confidence_filtered_objects++;
      
      // 检查是否为需要过滤的类别
      bool should_filter = false;
      for (int filtered_class : filtered_classes) {
        if (best_class == filtered_class) {
          should_filter = true;
          class_filtered_objects++;
          break;
        }
      }
      
      // 如果不属于过滤类别，则保留
      if (!should_filter) {
        DetectedObject obj;
        obj.index = i;
        obj.confidence = final_confidence;
        obj.class_id = best_class;
        obj.class_name = (best_class >= 0 && best_class < 10) ? class_names[best_class] : "unknown";
        
        // 获取目标的位置和尺寸信息 (从pred_anchor中提取)
        int anchor_idx = i * 11;  // 每个目标11个锚点值
        if (anchor_idx + 6 < pred_anchor.size()) {
          obj.x = pred_anchor[anchor_idx + 0];  // 中心点x
          obj.y = pred_anchor[anchor_idx + 1];  // 中心点y
          obj.z = pred_anchor[anchor_idx + 2];  // 中心点z
          obj.w = pred_anchor[anchor_idx + 3];  // 宽度
          obj.l = pred_anchor[anchor_idx + 4];  // 长度
          obj.h = pred_anchor[anchor_idx + 5];  // 高度
          obj.yaw = pred_anchor[anchor_idx + 6]; // 偏航角
        }
        
        valid_objects.push_back(obj);
      }
    }
  }
  
  // 按置信度排序
  std::sort(valid_objects.begin(), valid_objects.end(),
            [](const DetectedObject& a, const DetectedObject& b) {
              return a.confidence > b.confidence;
            });
  
  // 打印过滤结果统计
  std::cout << "总目标数量: " << total_objects << std::endl;
  std::cout << "通过置信度过滤的目标数量: " << confidence_filtered_objects << std::endl;
  std::cout << "被类别过滤的目标数量: " << class_filtered_objects << std::endl;
  std::cout << "置信度过滤后保留的目标数量: " << valid_objects.size() << std::endl;
  
  // 二维NMS去重函数
  auto calculate2DIoU = [](const DetectedObject& box1, const DetectedObject& box2) -> float {
    // 简化的2D IoU计算（基于中心点距离和边界框尺寸）
    float dx = box1.x - box2.x;
    float dy = box1.y - box2.y;
    float distance = std::sqrt(dx * dx + dy * dy);
    
    // 计算边界框对角线长度的一半作为重叠阈值
    float threshold1 = std::sqrt(box1.l * box1.l + box1.w * box1.w) / 2.0f;
    float threshold2 = std::sqrt(box2.l * box2.l + box2.w * box2.w) / 2.0f;
    float overlap_threshold = (threshold1 + threshold2) * 0.5f;
    
    // 如果距离太远，IoU为0
    if (distance > overlap_threshold) {
      return 0.0f;
    }
    
    // 简化的IoU计算
    float overlap_ratio = 1.0f - (distance / overlap_threshold);
    return std::max(0.0f, overlap_ratio);
  };
  
  // 执行二维NMS
  std::vector<DetectedObject> nms_objects;
  std::vector<bool> suppressed(valid_objects.size(), false);
  const float nms_threshold = 0.01f;  // NMS阈值
  
  for (size_t i = 0; i < valid_objects.size(); ++i) {
    if (suppressed[i]) continue;
    
    // 添加当前检测框到结果中
    nms_objects.push_back(valid_objects[i]);
    
    // 抑制与当前检测框IoU大于阈值的检测框
    for (size_t j = i + 1; j < valid_objects.size(); ++j) {
      if (suppressed[j]) continue;
      
      // 只对相同类别的检测框进行NMS
      if (valid_objects[i].class_id == valid_objects[j].class_id) {
        float iou = calculate2DIoU(valid_objects[i], valid_objects[j]);
        // std::cout << "NMS iou: " << iou << std::endl;
        if (iou > nms_threshold) {
          suppressed[j] = true;
        }
      }
    }
  }
  
  std::cout << "NMS去重后保留的目标数量: " << nms_objects.size() << std::endl;
  
  // 打印NMS过滤后的目标详细信息
  std::cout << "\n=== NMS过滤后的目标详细信息 ===" << std::endl;
  std::cout << std::setw(8) << "索引" << std::setw(12) << "置信度" << std::setw(12) << "类别" 
            << std::setw(12) << "X坐标" << std::setw(12) << "Y坐标" << std::setw(12) << "Z坐标"
            << std::setw(10) << "长度" << std::setw(10) << "宽度" << std::setw(10) << "高度"
            << std::setw(12) << "偏航角" << std::endl;
  std::cout << std::string(110, '-') << std::endl;
  
  for (const auto& obj : nms_objects) {
    std::cout << std::setw(8) << obj.index 
              << std::setw(12) << std::fixed << std::setprecision(4) << obj.confidence
              << std::setw(12) << obj.class_name
              << std::setw(12) << std::fixed << std::setprecision(2) << obj.x
              << std::setw(12) << std::fixed << std::setprecision(2) << obj.y
              << std::setw(12) << std::fixed << std::setprecision(2) << obj.z
              << std::setw(10) << std::fixed << std::setprecision(2) << obj.l
              << std::setw(10) << std::fixed << std::setprecision(2) << obj.w
              << std::setw(10) << std::fixed << std::setprecision(2) << obj.h
              << std::setw(12) << std::fixed << std::setprecision(2) << obj.yaw
              << std::endl;
  }
  
  std::cout << "\n=== NMS过滤完成，共保留 " << nms_objects.size() << " 个目标 ===" << std::endl;

  // 21.对预期结果文件进行相同的后处理
  std::cout << "\n=== 对预期结果文件进行后处理 ===" << std::endl;
  
  // 对预期结果进行相同的置信度过滤、类别过滤和NMS去重
  std::vector<DetectedObject> expected_valid_objects;
  
  for (int i = 0; i < total_objects; ++i) {
    // 获取当前目标的类别分数 (1×900×10 格式)
    float max_confidence = 0.0f;
    int best_class = 0;
    
    for (int j = 0; j < 10; ++j) {  // 10个类别
      int score_idx = i * 10 + j;
      if (score_idx < expected_pred_class_score.size()) {
        // 应用sigmoid激活函数
        float sigmoid_score = 1.0f / (1.0f + std::exp(-expected_pred_class_score[score_idx]));
        if (sigmoid_score > max_confidence) {
          max_confidence = sigmoid_score;
          best_class = j;
        }
      }
    }
    
    // 从quality_score中获取质量分数 (1×900×2 格式)
    float quality_score = 0.0f;
    int quality_idx = i * 2;
    if (quality_idx + 1 < expected_pred_quality_score.size()) {
      float quality_score1 = 1.0f / (1.0f + std::exp(-expected_pred_quality_score[quality_idx]));
      float quality_score2 = 1.0f / (1.0f + std::exp(-expected_pred_quality_score[quality_idx + 1]));
      quality_score = std::max(quality_score1, quality_score2);
    }
    
    // 综合置信度：使用类别分数和质量分数的组合
    float final_confidence = max_confidence * quality_score;
    
    // 如果综合置信度太低，使用质量分数
    if (final_confidence < 0.1f) {
      final_confidence = quality_score;
    }
    
    // 应用置信度阈值过滤
    if (final_confidence >= confidence_threshold) {
      // 检查是否为需要过滤的类别
      bool should_filter = false;
      for (int filtered_class : filtered_classes) {
        if (best_class == filtered_class) {
          should_filter = true;
          break;
        }
      }
      
      // 如果不属于过滤类别，则保留
      if (!should_filter) {
        DetectedObject obj;
        obj.index = i;
        obj.confidence = final_confidence;
        obj.class_id = best_class;
        obj.class_name = (best_class >= 0 && best_class < 10) ? class_names[best_class] : "unknown";
        
        // 获取目标的位置和尺寸信息 (从expected_pred_anchor中提取)
        int anchor_idx = i * 11;  // 每个目标11个锚点值
        if (anchor_idx + 6 < expected_pred_anchor.size()) {
          obj.x = expected_pred_anchor[anchor_idx + 0];  // 中心点x
          obj.y = expected_pred_anchor[anchor_idx + 1];  // 中心点y
          obj.z = expected_pred_anchor[anchor_idx + 2];  // 中心点z
          obj.w = expected_pred_anchor[anchor_idx + 3];  // 宽度
          obj.l = expected_pred_anchor[anchor_idx + 4];  // 长度
          obj.h = expected_pred_anchor[anchor_idx + 5];  // 高度
          obj.yaw = expected_pred_anchor[anchor_idx + 6]; // 偏航角
        }
        
        expected_valid_objects.push_back(obj);
      }
    }
  }
  
  // 按置信度排序
  std::sort(expected_valid_objects.begin(), expected_valid_objects.end(),
            [](const DetectedObject& a, const DetectedObject& b) {
              return a.confidence > b.confidence;
            });
  
  // 对预期结果执行二维NMS
  std::vector<DetectedObject> expected_nms_objects;
  std::vector<bool> expected_suppressed(expected_valid_objects.size(), false);
  
  for (size_t i = 0; i < expected_valid_objects.size(); ++i) {
    if (expected_suppressed[i]) continue;
    
    // 添加当前检测框到结果中
    expected_nms_objects.push_back(expected_valid_objects[i]);
    
    // 抑制与当前检测框IoU大于阈值的检测框
    for (size_t j = i + 1; j < expected_valid_objects.size(); ++j) {
      if (expected_suppressed[j]) continue;
      
      // 只对相同类别的检测框进行NMS
      if (expected_valid_objects[i].class_id == expected_valid_objects[j].class_id) {
        float iou = calculate2DIoU(expected_valid_objects[i], expected_valid_objects[j]);
        if (iou > nms_threshold) {
          expected_suppressed[j] = true;
        }
      }
    }
  }
  
  std::cout << "预期结果过滤后保留的目标数量: " << expected_nms_objects.size() << std::endl;
  
  // 打印预期结果的目标详细信息
  std::cout << "\n=== 预期结果NMS过滤后的目标详细信息 ===" << std::endl;
  std::cout << std::setw(8) << "索引" << std::setw(12) << "置信度" << std::setw(12) << "类别" 
            << std::setw(12) << "X坐标" << std::setw(12) << "Y坐标" << std::setw(12) << "Z坐标"
            << std::setw(10) << "长度" << std::setw(10) << "宽度" << std::setw(10) << "高度"
            << std::setw(12) << "偏航角" << std::endl;
  std::cout << std::string(110, '-') << std::endl;
  
  for (const auto& obj : expected_nms_objects) {
    std::cout << std::setw(8) << obj.index 
              << std::setw(12) << std::fixed << std::setprecision(4) << obj.confidence
              << std::setw(12) << obj.class_name
              << std::setw(12) << std::fixed << std::setprecision(2) << obj.x
              << std::setw(12) << std::fixed << std::setprecision(2) << obj.y
              << std::setw(12) << std::fixed << std::setprecision(2) << obj.z
              << std::setw(10) << std::fixed << std::setprecision(2) << obj.l
              << std::setw(10) << std::fixed << std::setprecision(2) << obj.w
              << std::setw(10) << std::fixed << std::setprecision(2) << obj.h
              << std::setw(12) << std::fixed << std::setprecision(2) << obj.yaw
              << std::endl;
  }
  
  // 比较两个结果
  std::cout << "\n=== 结果比较 ===" << std::endl;
  std::cout << "预测结果目标数量: " << nms_objects.size() << std::endl;
  std::cout << "预期结果目标数量: " << expected_nms_objects.size() << std::endl;
  std::cout << "目标数量差异: " << std::abs(static_cast<int>(nms_objects.size()) - static_cast<int>(expected_nms_objects.size())) << std::endl;
  
  // 按类别比较
  std::map<std::string, int> pred_class_counts, exp_class_counts;
  for (const auto& obj : nms_objects) {
    pred_class_counts[obj.class_name]++;
  }
  for (const auto& obj : expected_nms_objects) {
    exp_class_counts[obj.class_name]++;
  }
  
  std::cout << "\n按类别比较:" << std::endl;
  std::cout << std::setw(15) << "类别" << std::setw(10) << "预测" << std::setw(10) << "预期" << std::setw(10) << "差异" << std::endl;
  std::cout << std::string(45, '-') << std::endl;
  
  // 合并所有类别
  std::set<std::string> all_classes;
  for (const auto& pair : pred_class_counts) {
    all_classes.insert(pair.first);
  }
  for (const auto& pair : exp_class_counts) {
    all_classes.insert(pair.first);
  }
  
  for (const auto& class_name : all_classes) {
    int pred_count = pred_class_counts[class_name];
    int exp_count = exp_class_counts[class_name];
    int diff = pred_count - exp_count;
    
    std::cout << std::setw(15) << class_name 
              << std::setw(10) << pred_count
              << std::setw(10) << exp_count
              << std::setw(10) << diff << std::endl;
  }
  
  std::cout << "\n=== 预期结果后处理完成 ===" << std::endl;

  // 20.销毁cuda event 和 stream
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));
}

}  // namespace engine
}  // namespace sparse_end2end