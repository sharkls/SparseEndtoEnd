// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <vector>

#include "common/common.h"
#include "common/cuda_wrapper.cu.h"
#include "common/utils.h"
#include "preprocessor/img_preprocessor.h"
#include "preprocessor/parameters_parser.h"
#include "tensorrt/tensorrt.h"

namespace sparse_end2end {
namespace engine {

/**
 * @brief 计算两个向量之间的误差百分比
 * @param a 第一个向量
 * @param b 第二个向量
 * @param threshold 误差阈值
 * @return 误差百分比
 */
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
  std::cout << "Error >" << threshold << " percentage is: " << p << std::endl;
  std::cout << "MaxError = " << max_error << std::endl;

  return p;
}

}  // namespace engine
}  // namespace sparse_end2end

int main(int argc, char** argv) {
  // 1. 获取当前路径并解析配置文件
  std::filesystem::path current_dir = std::filesystem::current_path();
  const sparse_end2end::common::E2EParams params = sparse_end2end::preprocessor::parseParams(current_dir / "assets/model_cfg.yaml");

  // 2. 获取各个引擎的路径和输入输出名称
  std::string sparse4d_extract_feat_engine_path = params.sparse4d_extract_feat_engine.engine_path;
  std::string sparse4d_head1st_engine_path = params.sparse4d_head1st_engine.engine_path;
  std::string sparse4d_head2nd_engine_path = params.sparse4d_head2nd_engine.engine_path;
  std::string multiview_multiscale_deformable_attention_aggregation_path =
      params.model_cfg.multiview_multiscale_deformable_attention_aggregation_path;

  std::vector<std::string> sparse4d_extract_feat_engine_input_names = params.sparse4d_extract_feat_engine.input_names;
  std::vector<std::string> sparse4d_extract_feat_engine_output_names = params.sparse4d_extract_feat_engine.output_names;
  std::vector<std::string> sparse4d_head1st_engine_input_names = params.sparse4d_head1st_engine.input_names;
  std::vector<std::string> sparse4d_head1st_engine_output_names = params.sparse4d_head1st_engine.output_names;
  std::vector<std::string> sparse4d_head2nd_engine_input_names = params.sparse4d_head2nd_engine.input_names;
  std::vector<std::string> sparse4d_head2nd_engine_output_names = params.sparse4d_head2nd_engine.output_names;

  // 3. 创建CUDA事件和流
  cudaEvent_t start, stop;
  cudaStream_t stream = nullptr;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  // 4. 创建TensorRT引擎
  std::cout << "Loading TensorRT engines..." << std::endl;
  
  // 特征提取引擎
  std::shared_ptr<sparse_end2end::engine::TensorRT> extract_feat_engine = std::make_shared<sparse_end2end::engine::TensorRT>(
      sparse4d_extract_feat_engine_path, "", sparse4d_extract_feat_engine_input_names,
      sparse4d_extract_feat_engine_output_names);

  // 第一帧head引擎
  std::shared_ptr<sparse_end2end::engine::TensorRT> head1st_engine = std::make_shared<sparse_end2end::engine::TensorRT>(
      sparse4d_head1st_engine_path, multiview_multiscale_deformable_attention_aggregation_path,
      sparse4d_head1st_engine_input_names, sparse4d_head1st_engine_output_names);

  // 第二帧head引擎
  std::shared_ptr<sparse_end2end::engine::TensorRT> head2nd_engine = std::make_shared<sparse_end2end::engine::TensorRT>(
      sparse4d_head2nd_engine_path, multiview_multiscale_deformable_attention_aggregation_path,
      sparse4d_head2nd_engine_input_names, sparse4d_head2nd_engine_output_names);

  // 5. 加载测试数据（这里使用示例数据路径，实际使用时需要根据实际情况调整）
  std::string sample_data_path = "/share/Code/SparseEnd2End/script/tutorial/asset/";
  
  // 加载第一帧数据
  const auto imgs = sparse_end2end::common::readfile_wrapper<float>(sample_data_path + "sample_0_imgs_1*6*3*256*704_float32.bin");
  const auto expected_feature = sparse_end2end::common::readfile_wrapper<float>(sample_data_path + "sample_0_feature_1*89760*256_float32.bin");
  
  // 加载第一帧head数据
  const auto spatial_shapes = sparse_end2end::common::readfile_wrapper<int32_t>(sample_data_path + "sample_0_spatial_shapes_6*4*2_int32.bin");
  const auto level_start_index = sparse_end2end::common::readfile_wrapper<int32_t>(sample_data_path + "sample_0_level_start_index_6*4_int32.bin");
  const auto instance_feature = sparse_end2end::common::readfile_wrapper<float>(sample_data_path + "sample_0_instance_feature_1*900*256_float32.bin");
  const auto anchor = sparse_end2end::common::readfile_wrapper<float>(sample_data_path + "sample_0_anchor_1*900*11_float32.bin");
  const auto time_interval = sparse_end2end::common::readfile_wrapper<float>(sample_data_path + "sample_0_time_interval_1_float32.bin");
  const auto image_wh = sparse_end2end::common::readfile_wrapper<float>(sample_data_path + "sample_0_image_wh_1*6*2_float32.bin");
  const auto lidar2img = sparse_end2end::common::readfile_wrapper<float>(sample_data_path + "sample_0_lidar2img_1*6*4*4_float32.bin");
  
  // 加载期望输出
  const auto expected_pred_instance_feature = sparse_end2end::common::readfile_wrapper<float>(sample_data_path + "sample_0_pred_instance_feature_1*900*256_float32.bin");
  const auto expected_pred_anchor = sparse_end2end::common::readfile_wrapper<float>(sample_data_path + "sample_0_pred_anchor_1*900*11_float32.bin");
  const auto expected_pred_class_score = sparse_end2end::common::readfile_wrapper<float>(sample_data_path + "sample_0_pred_class_score_1*900*10_float32.bin");
  const auto expected_pred_quality_score = sparse_end2end::common::readfile_wrapper<float>(sample_data_path + "sample_0_pred_quality_score_1*900*2_float32.bin");

  // 6. 验证数据大小
  if (imgs.size() != 1 * 6 * 3 * 256 * 704) {
    std::cout << "[ERROR] imgs size mismatch!" << std::endl;
    return -1;
  }
  if (expected_feature.size() != 1 * 89760 * 256) {
    std::cout << "[ERROR] expected_feature size mismatch!" << std::endl;
    return -1;
  }
  if (spatial_shapes.size() != 6 * 4 * 2) {
    std::cout << "[ERROR] spatial_shapes size mismatch!" << std::endl;
    return -1;
  }
  if (level_start_index.size() != 6 * 4) {
    std::cout << "[ERROR] level_start_index size mismatch!" << std::endl;
    return -1;
  }
  if (instance_feature.size() != 1 * 900 * 256) {
    std::cout << "[ERROR] instance_feature size mismatch!" << std::endl;
    return -1;
  }
  if (anchor.size() != 1 * 900 * 11) {
    std::cout << "[ERROR] anchor size mismatch!" << std::endl;
    return -1;
  }
  if (time_interval.size() != 1) {
    std::cout << "[ERROR] time_interval size mismatch!" << std::endl;
    return -1;
  }
  if (image_wh.size() != 1 * 6 * 2) {
    std::cout << "[ERROR] image_wh size mismatch!" << std::endl;
    return -1;
  }
  if (lidar2img.size() != 1 * 6 * 4 * 4) {
    std::cout << "[ERROR] lidar2img size mismatch!" << std::endl;
    return -1;
  }

  // 7. 创建GPU内存
  const sparse_end2end::common::CudaWrapper<float> imgs_gpu(imgs);
  sparse_end2end::common::CudaWrapper<float> feature_gpu(1 * 89760 * 256);
  
  const sparse_end2end::common::CudaWrapper<int32_t> spatial_shapes_gpu(spatial_shapes);
  const sparse_end2end::common::CudaWrapper<int32_t> level_start_index_gpu(level_start_index);
  const sparse_end2end::common::CudaWrapper<float> instance_feature_gpu(instance_feature);
  const sparse_end2end::common::CudaWrapper<float> anchor_gpu(anchor);
  const sparse_end2end::common::CudaWrapper<float> time_interval_gpu(time_interval);
  const sparse_end2end::common::CudaWrapper<float> image_wh_gpu(image_wh);
  const sparse_end2end::common::CudaWrapper<float> lidar2img_gpu(lidar2img);

  // 创建输出缓冲区
  sparse_end2end::common::CudaWrapper<float> tmp_outs0(1 * 900 * 256);
  sparse_end2end::common::CudaWrapper<float> tmp_outs1(1 * 900 * 256);
  sparse_end2end::common::CudaWrapper<float> tmp_outs2(1 * 900 * 256);
  sparse_end2end::common::CudaWrapper<float> tmp_outs3(1 * 900 * 256);
  sparse_end2end::common::CudaWrapper<float> tmp_outs4(1 * 900 * 256);
  sparse_end2end::common::CudaWrapper<float> tmp_outs5(1 * 900 * 256);
  sparse_end2end::common::CudaWrapper<float> pred_instance_feature_gpu(1 * 900 * 256);
  sparse_end2end::common::CudaWrapper<float> pred_anchor_gpu(1 * 900 * 11);
  sparse_end2end::common::CudaWrapper<float> pred_class_score_gpu(1 * 900 * 10);
  sparse_end2end::common::CudaWrapper<float> pred_quality_score_gpu(1 * 900 * 2);

  // 8. 执行特征提取推理
  std::cout << "Running feature extraction inference..." << std::endl;
  
  std::vector<void*> extract_feat_buffers;
  extract_feat_buffers.push_back(imgs_gpu.getCudaPtr());
  extract_feat_buffers.push_back(feature_gpu.getCudaPtr());

  checkCudaErrors(cudaEventRecord(start, stream));
  if (!extract_feat_engine->infer(extract_feat_buffers.data(), stream)) {
    std::cout << "[ERROR] Feature extraction inference failed" << std::endl;
    return -1;
  }
  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));
  float extract_time = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&extract_time, start, stop));
  std::cout << "[Feature Extraction] Time Cost = " << extract_time << " [ms]" << std::endl;
  cudaStreamSynchronize(stream);

  // 9. 执行第一帧head推理
  std::cout << "Running first frame head inference..." << std::endl;
  
  std::vector<void*> head1st_buffers;
  head1st_buffers.push_back(feature_gpu.getCudaPtr());
  head1st_buffers.push_back(spatial_shapes_gpu.getCudaPtr());
  head1st_buffers.push_back(level_start_index_gpu.getCudaPtr());
  head1st_buffers.push_back(instance_feature_gpu.getCudaPtr());
  head1st_buffers.push_back(anchor_gpu.getCudaPtr());
  head1st_buffers.push_back(time_interval_gpu.getCudaPtr());
  head1st_buffers.push_back(image_wh_gpu.getCudaPtr());
  head1st_buffers.push_back(lidar2img_gpu.getCudaPtr());
  head1st_buffers.push_back(tmp_outs0.getCudaPtr());
  head1st_buffers.push_back(tmp_outs1.getCudaPtr());
  head1st_buffers.push_back(tmp_outs2.getCudaPtr());
  head1st_buffers.push_back(tmp_outs3.getCudaPtr());
  head1st_buffers.push_back(tmp_outs4.getCudaPtr());
  head1st_buffers.push_back(tmp_outs5.getCudaPtr());
  head1st_buffers.push_back(pred_instance_feature_gpu.getCudaPtr());
  head1st_buffers.push_back(pred_anchor_gpu.getCudaPtr());
  head1st_buffers.push_back(pred_class_score_gpu.getCudaPtr());
  head1st_buffers.push_back(pred_quality_score_gpu.getCudaPtr());

  checkCudaErrors(cudaEventRecord(start, stream));
  if (!head1st_engine->infer(head1st_buffers.data(), stream)) {
    std::cout << "[ERROR] First frame head inference failed" << std::endl;
    return -1;
  }
  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));
  float head1st_time = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&head1st_time, start, stop));
  std::cout << "[First Frame Head] Time Cost = " << head1st_time << " [ms]" << std::endl;
  cudaStreamSynchronize(stream);

  // 10. 获取推理结果并验证
  auto pred_instance_feature = pred_instance_feature_gpu.cudaMemcpyD2HResWrap();
  auto pred_anchor = pred_anchor_gpu.cudaMemcpyD2HResWrap();
  auto pred_class_score = pred_class_score_gpu.cudaMemcpyD2HResWrap();
  auto pred_quality_score = pred_quality_score_gpu.cudaMemcpyD2HResWrap();

  // 11. 计算误差并输出结果
  std::cout << "\n=== Inference Results ===" << std::endl;
  
  const float p0 = sparse_end2end::engine::GetErrorPercentage(pred_instance_feature, expected_pred_instance_feature, 0.1);
  std::cout << "Instance feature error percentage: " << p0 << std::endl;
  std::cout << "Pred_instance_feature: max=" << *std::max_element(pred_instance_feature.begin(), pred_instance_feature.end())
            << " min=" << *std::min_element(pred_instance_feature.begin(), pred_instance_feature.end()) << std::endl;

  const float p1 = sparse_end2end::engine::GetErrorPercentage(pred_anchor, expected_pred_anchor, 0.1);
  std::cout << "Anchor error percentage: " << p1 << std::endl;
  std::cout << "Pred_anchor: max=" << *std::max_element(pred_anchor.begin(), pred_anchor.end())
            << " min=" << *std::min_element(pred_anchor.begin(), pred_anchor.end()) << std::endl;

  const float p2 = sparse_end2end::engine::GetErrorPercentage(pred_class_score, expected_pred_class_score, 0.1);
  std::cout << "Class score error percentage: " << p2 << std::endl;
  std::cout << "Pred_class_score: max=" << *std::max_element(pred_class_score.begin(), pred_class_score.end())
            << " min=" << *std::min_element(pred_class_score.begin(), pred_class_score.end()) << std::endl;

  const float p3 = sparse_end2end::engine::GetErrorPercentage(pred_quality_score, expected_pred_quality_score, 0.1);
  std::cout << "Quality score error percentage: " << p3 << std::endl;
  std::cout << "Pred_quality_score: max=" << *std::max_element(pred_quality_score.begin(), pred_quality_score.end())
            << " min=" << *std::min_element(pred_quality_score.begin(), pred_quality_score.end()) << std::endl;

  std::cout << "\n=== Performance Summary ===" << std::endl;
  std::cout << "Feature Extraction: " << extract_time << " ms" << std::endl;
  std::cout << "First Frame Head: " << head1st_time << " ms" << std::endl;
  std::cout << "Total Time: " << extract_time + head1st_time << " ms" << std::endl;

  // 12. 清理资源
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));

  std::cout << "\nInference completed successfully!" << std::endl;
  
  sparse_end2end::common::Status status = sparse_end2end::common::Status::kSuccess;
  return 0;
}
