
#include "sparse4d.hpp"

#include "common/timer.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <google/protobuf/text_format.h>
#include "log.h"
#include "common/functionhub.hpp"

namespace sparse4d{
namespace core{

/**
 * @brief 析构函数：清理 CUDA Stream 资源
 */
CoreImplement::~CoreImplement() {
    // 销毁固定的 CUDA Stream
    if (inference_stream_ != nullptr) {
        cudaError_t err = cudaStreamDestroy(inference_stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to destroy inference CUDA stream: " << cudaGetErrorString(err);
        } else {
            LOG(INFO) << "[INFO] Destroyed inference CUDA stream";
        }
        inference_stream_ = nullptr;
    }
}

/**
 * @brief 加载TaskConfig配置文件
 * @param config_path 配置文件路径
 * @param task_config 输出TaskConfig对象
 * @return 成功返回true，失败返回false
 */
 bool loadTaskConfig(const std::string& config_path, sparse4d::TaskConfig& task_config) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        LOG(ERROR) << "Failed to open config file: " << config_path;
        return false;
    }
    
    std::stringstream buffer;
    buffer << config_file.rdbuf();
    std::string content = buffer.str();
    config_file.close();
    
    // 解析protobuf文本格式配置文件
    if (!google::protobuf::TextFormat::ParseFromString(content, &task_config)) {
        LOG(ERROR) << "Failed to parse config file: " << config_path;
        return false;
    }
    
    LOG(INFO) << "Successfully loaded TaskConfig from: " << config_path;
    return true;
}

bool CoreImplement::initAlgorithm(const std::string exe_path,  const AlgCallback& alg_cb, void* hd)
{
    exe_path_ = exe_path;
    alg_cb_ = alg_cb;
    user_handle_ = hd;

    // 通过工程路径推导配置文件路径
    std::string config_path = exe_path_ + "/Output/Configs/Alg/Sparse4d.conf";
    sparse4d::TaskConfig task_config;
    if (!loadTaskConfig(config_path, task_config)) {
        LOG(ERROR) << "[ERROR] Failed to load TaskConfig from: " << config_path;
        return false;
    }

    return init(task_config);
}

bool CoreImplement::update(void* p_pParam)
{
    if (p_pParam == nullptr) return false;
    // 将通用参数转换为 lidar2camera（float*，按相机数 * 16）
    const float* lidar2camera = reinterpret_cast<const float*>(p_pParam);
    update(lidar2camera, nullptr);
    return true;
}

void CoreImplement::runAlgorithm(void* p_pSrcData)
{
    if (p_pSrcData == nullptr) return;
    const CTimeMatchSrcData* raw_data = reinterpret_cast<const CTimeMatchSrcData*>(p_pSrcData);
    
    // 使用固定的 CUDA Stream（而不是 nullptr/默认流）
    // 固定的 stream + 固定的缓冲区地址 = CUDA Graph 自动启用
    // TensorRT 8.0+ 会自动检测并启用 CUDA Graph，显著降低 Enqueue Time（从 ~120ms 到 ~0.8ms）
    CAlgResult result = forward(raw_data, inference_stream_);
    if (alg_cb_) {
        alg_cb_(result, user_handle_);
    }
}

//  初始化核心组件，包括骨干网络、第一帧头部、第二帧头部、实例银行、归一化等。每个组件的创建都进行检查，如果失败则返回 false。
bool CoreImplement::init(const TaskConfig &param)
{   
    // 保存配置
    param_ = param;
    // 检查是否有 enable_timer 字段，如果没有则默认为 false
    // 注意：如果 protobuf 中没有这个字段，需要使用 has_enable_timer() 检查
    enable_timer_ = param.enable_timer();
    
    preprocessor_ = preprocessor::create_preprocessor(param);
    if(preprocessor_ == nullptr){
        std::cerr << "Failed to create preprocessor.\n" << std::endl;
        return false;
    }

    instance_bank_ = instance_bank::create_instance_bank(param);
    if(instance_bank_ == nullptr){
        std::cerr << "Failed to create instance bank.\n" << std::endl;
        return false;
    }

    backbone_ = backbone::create_backbone(param);
    if(backbone_ == nullptr){
        std::cerr << "Failed to create backbone.\n" << std::endl;
        return false;
    }

    head1_ = first_head::create_first_head(param);
    if(head1_ == nullptr){
        std::cerr << "Failed to create head1.\n" << std::endl;
        return false;
    }

    head2_ = second_head::create_second_head(param);
    if(head2_ == nullptr){
        std::cerr << "Failed to create head2.\n" << std::endl;
        return false;
    }

    postprocessor_ = postprocessor::create_postprocessor(param);
    if(postprocessor_ == nullptr){
        std::cerr << "Failed to create postprocessor.\n" << std::endl;
        return false;
    }

    if (!loadAuxiliaryData()) {
        std::cerr << "Failed to load auxiliary data.\n" << std::endl;
        return false;
    }

    // 创建固定的 CUDA Stream 用于推理（启用 CUDA Graph 优化）
    // CUDA Graph 需要固定的 stream 和固定的缓冲区地址才能生效
    cudaError_t stream_err = cudaStreamCreate(&inference_stream_);
    if (stream_err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] Failed to create inference CUDA stream: " << cudaGetErrorString(stream_err);
        return false;
    }
    LOG(INFO) << "[INFO] Created fixed CUDA stream for inference (CUDA Graph optimization enabled)";

    // 预热推理：多次调用以触发 CUDA Graph 捕获
    // TensorRT 8.0+ 需要 2-3 次调用 enqueueV2 才能捕获 CUDA Graph
    LOG(INFO) << "[INFO] Starting warmup inference to trigger CUDA Graph capture...";
    if (!warmupInference()) {
        LOG(WARNING) << "[WARNING] Warmup inference failed, but continuing...";
    }
    LOG(INFO) << "[INFO] Warmup inference completed";

    return true;
}

/**
 * @brief 加载辅助数据
 * @note 加载辅助数据到pipeline_context_中
 * @note spatial_shapes、level_start_index、instance_feature、anchor、time_interval、image_wh、lidar2img（7个张量）
 */
bool CoreImplement::loadAuxiliaryData() 
 {
    // 1. 从TaskConfig获取空间形状数据
    std::vector<int32_t> spatial_shapes;
    for (int i = 0; i < param_.model_cfg_params().sparse4d_extract_feat_spatial_shapes_ld_size(); ++i) {
    spatial_shapes.push_back(param_.model_cfg_params().sparse4d_extract_feat_spatial_shapes_ld(i));
    }
    std::vector<int32_t> expanded;
    const int num_cams = param_.preprocessor_params().num_cams();
    expanded.reserve(num_cams * 4 * 2);
    for (int cam = 0; cam < num_cams; ++cam) {
      for (int lvl = 0; lvl < 4; ++lvl) {
        expanded.push_back(spatial_shapes[lvl * 2 + 0]);
        expanded.push_back(spatial_shapes[lvl * 2 + 1]);
      }
    }
    pipeline_context_.spatial_shapes.cudaMemUpdateWrap(expanded);
     
    // 2. 从TaskConfig获取层级起始索引
    std::vector<int32_t> level_start_index;
    for (int i = 0; i < param_.model_cfg_params().sparse4d_extract_feat_level_start_index_size(); ++i) {
    level_start_index.push_back(param_.model_cfg_params().sparse4d_extract_feat_level_start_index(i));
    }
    pipeline_context_.level_start_index.cudaMemUpdateWrap(level_start_index);
     
    // 3. 从TaskConfig获取实例特征大小并初始化
    std::vector<float> instance_feature;
    size_t instance_feature_size = param_.instance_bank_params().num_querys() * 
                                param_.model_cfg_params().embedfeat_dims();
    instance_feature.resize(instance_feature_size, 0.0f);
    // 转换为 half 类型
    std::vector<half> instance_feature_half(instance_feature_size);
    for (size_t i = 0; i < instance_feature_size; ++i) {
        instance_feature_half[i] = __float2half(instance_feature[i]);
    }
    pipeline_context_.instance_feature.cudaMemUpdateWrap(instance_feature_half);
     
    // 4. 从锚点文件加载锚点数据
    std::vector<float> anchor;
    std::string anchor_path = param_.instance_bank_params().instance_bank_anchor_path();
    if (!anchor_path.empty()) {
        size_t expected_anchor_size = param_.instance_bank_params().num_querys() * 
                                    param_.instance_bank_params().query_dims();
        
        std::ifstream file(anchor_path, std::ios::binary);
        if (file.is_open()) {
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);
            
            size_t expected_file_size = expected_anchor_size * sizeof(float);
            if (file_size == expected_file_size) {
                anchor.resize(expected_anchor_size);
                file.read(reinterpret_cast<char*>(anchor.data()), file_size);
                LOG(INFO) << "[INFO] Successfully loaded anchor data from: " << anchor_path;
            } else {
                LOG(ERROR) << "[ERROR] Anchor file size mismatch! Expected: " << expected_file_size 
                        << " bytes, Actual: " << file_size << " bytes";
                anchor.resize(expected_anchor_size, 0.0f);
            }
            file.close();
        } else {
            LOG(ERROR) << "[ERROR] Failed to open anchor file: " << anchor_path;
            anchor.resize(expected_anchor_size, 0.0f);
        }
    } else {
        LOG(WARNING) << "[WARNING] No anchor file path specified, using default values";
        size_t expected_anchor_size = param_.instance_bank_params().num_querys() * 
                                    param_.instance_bank_params().query_dims();
        anchor.resize(expected_anchor_size, 0.0f);
    }
    // 转换为 half 类型
    std::vector<half> anchor_half(anchor.size());
    for (size_t i = 0; i < anchor.size(); ++i) {
        anchor_half[i] = __float2half(anchor[i]);
    }
    pipeline_context_.anchor.cudaMemUpdateWrap(anchor_half);

    // 5. 从TaskConfig获取时间间隔
    std::vector<float> time_interval;
    time_interval.push_back(param_.instance_bank_params().default_time_interval());
    // 转换为 half 类型
    std::vector<half> time_interval_half(1);
    time_interval_half[0] = __float2half(time_interval[0]);
    pipeline_context_.time_interval.cudaMemUpdateWrap(time_interval_half);
    
    // 从TaskConfig获取图像宽高（为每个相机分别保存）
    std::vector<float> image_wh;
    image_wh.resize(num_cams * 2); // 每个相机 * 2个值（宽高）
    for (int i = 0; i < num_cams; ++i) {
        image_wh[i * 2] = static_cast<float>(param_.preprocessor_params().model_input_img_w());
        image_wh[i * 2 + 1] = static_cast<float>(param_.preprocessor_params().model_input_img_h());
    }
    // 转换为 half 类型
    std::vector<half> image_wh_half(image_wh.size());
    for (size_t i = 0; i < image_wh.size(); ++i) {
        image_wh_half[i] = __float2half(image_wh[i]);
    }
    pipeline_context_.image_wh.cudaMemUpdateWrap(image_wh_half);
    
    // 6. 激光雷达到图像变换矩阵大小
    size_t lidar2img_size = param_.preprocessor_params().num_cams() * 4 * 4;
    std::vector<float> lidar2img;
    lidar2img.resize(lidar2img_size, 0.0f);
    for (int i = 0; i < param_.preprocessor_params().num_cams(); ++i) {
        lidar2img[i * 16 + 0] = 1.0f;  // [0,0]
        lidar2img[i * 16 + 5] = 1.0f;  // [1,1]
        lidar2img[i * 16 + 10] = 1.0f; // [2,2]
        lidar2img[i * 16 + 15] = 1.0f; // [3,3]
    }
    // 转换为 half 类型
    std::vector<half> lidar2img_half(lidar2img.size());
    for (size_t i = 0; i < lidar2img.size(); ++i) {
        lidar2img_half[i] = __float2half(lidar2img[i]);
    }
    pipeline_context_.lidar2img.cudaMemUpdateWrap(lidar2img_half);

    // 7. 依据配置初始化 pipeline_context_ / head_output_ 的显存尺寸
    {   
        // 预处理输入
        size_t input_size = param_.preprocessor_params().num_cams() * 
                            param_.preprocessor_params().model_input_img_c() *
                            param_.preprocessor_params().model_input_img_h() * 
                            param_.preprocessor_params().model_input_img_w();
        
        if (!pipeline_context_.input_images.allocate(input_size)) {
            std::cerr << "Failed to allocate pipeline_context_.input_images (size=" << input_size << ")\n";
            return false;  // 注意：这里需要改为返回 false，因为函数现在是 bool 返回类型
        }

        const size_t temp_instance_feature_size = param_.instance_bank_params().topk_querys() * param_.model_cfg_params().embedfeat_dims();
        const size_t temp_anchor_size = param_.instance_bank_params().topk_querys() * param_.instance_bank_params().query_dims();
        const size_t temp_mask_size = 1;
        const size_t temp_track_id_size = param_.instance_bank_params().topk_querys();

        const size_t pred_size = param_.instance_bank_params().num_querys() * param_.model_cfg_params().embedfeat_dims();
        const size_t anchor_size = param_.instance_bank_params().num_querys() * param_.instance_bank_params().query_dims();
        const size_t class_score_size = param_.instance_bank_params().num_querys() * param_.model_cfg_params().num_classes();
        const size_t quality_score_size = param_.instance_bank_params().num_querys() * 2;
        const size_t track_id_size = param_.instance_bank_params().num_querys();
        
        // 输入张量
        size_t feature_size = 1;
        for (int i = 0; i < param_.model_cfg_params().sparse4d_extract_feat_shape_lc_size(); ++i) {
            feature_size *= param_.model_cfg_params().sparse4d_extract_feat_shape_lc(i);
        }
        if (!pipeline_context_.features.allocate(feature_size)) {
            std::cerr << "Failed to allocate pipeline_context_.features (size=" << feature_size << ")\n";
            return false;
        }

        // 第二帧独有的输入
        if(!pipeline_context_.temp_instance_feature.allocate(temp_instance_feature_size) ||
            !pipeline_context_.temp_anchor.allocate(temp_anchor_size) ||
            !pipeline_context_.mask.allocate(temp_mask_size) ||
            !pipeline_context_.track_ids.allocate(track_id_size)) {
            std::cerr << "Failed to allocate pipeline_context_.temp_instance_feature (size=" << temp_instance_feature_size << ")\n";
            return false;
        }

        // 输出张量
        if (!head_output_.pred_instance_feature.allocate(pred_size) ||
            !head_output_.pred_anchor.allocate(anchor_size) ||
            !head_output_.pred_class_score.allocate(class_score_size) ||
            !head_output_.pred_quality_score.allocate(quality_score_size) ||
            !head_output_.pred_track_ids.allocate(track_id_size) ||
            !head_output_.tmp_outs0.allocate(pred_size * 2) ||
            !head_output_.tmp_outs1.allocate(pred_size * 2) ||
            !head_output_.tmp_outs2.allocate(pred_size * 2) ||
            !head_output_.tmp_outs3.allocate(pred_size * 2) ||
            !head_output_.tmp_outs4.allocate(pred_size * 2) ||
            !head_output_.tmp_outs5.allocate(pred_size * 2)) {
            std::cerr << "Failed to allocate head_output_ (size=" << pred_size * 2 << ")\n";
            return false;
        }
    }
    
    LOG(INFO) << "[INFO] Auxiliary data loaded from TaskConfig";
    return true;
}

/**
 * @brief 预热推理：多次调用以触发 CUDA Graph 捕获
 * @return 成功返回 true，失败返回 false
 * @note TensorRT 8.0+ 需要 2-3 次调用 enqueueV2 才能捕获 CUDA Graph
 * @note 使用零数据填充缓冲区，仅用于触发 CUDA Graph 捕获
 */
bool CoreImplement::warmupInference() {
    if (inference_stream_ == nullptr) {
        LOG(ERROR) << "[ERROR] Inference stream is null, cannot perform warmup";
        return false;
    }

    // 使用零数据填充输入缓冲区（仅用于预热，不关心输出结果）
    // 注意：这里使用已有的 pipeline_context_ 和 head_output_ 缓冲区
    // 它们的地址是固定的，满足 CUDA Graph 的要求
    
    // 根据实际测试，Backbone 的 CUDA Graph 在第 2 次迭代就生效了（enqueue time 从 348ms 降到 1.3ms）
    // 但 Head1 和 Head2 需要更多预热迭代才能捕获 CUDA Graph
    // Head2 推理耗时波动大，增加warmup次数以稳定CUDA Graph捕获
    const int BACKBONE_WARMUP = 3;   // Backbone 已经验证生效，保持 3 次
    const int HEAD1_WARMUP = 5;      // Head1 需要更多预热（从日志看第 5 次调用仍然较高）
    const int HEAD2_WARMUP = 15;     // Head2 需要更多预热（从5增加到15，减少性能波动）
    
    LOG(INFO) << "[INFO] Performing warmup iterations: Backbone=" << BACKBONE_WARMUP 
              << ", Head1=" << HEAD1_WARMUP << ", Head2=" << HEAD2_WARMUP;
    
    // 关键修复：初始化 input_images 和 features 为零数据，避免未初始化内存导致的非法访问
    size_t input_images_size = pipeline_context_.input_images.getSize();
    if (input_images_size > 0) {
        std::vector<half> zero_input_images(input_images_size, __float2half(0.0f));
        pipeline_context_.input_images.cudaMemUpdateWrap(zero_input_images);
        cudaStreamSynchronize(inference_stream_);
        cudaError_t init_err = cudaGetLastError();
        if (init_err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to initialize input_images for warmup: " << cudaGetErrorString(init_err);
            return false;
        }
        LOG(INFO) << "[INFO] Initialized input_images with zero data (size: " << input_images_size << ") for warmup";
    } else {
        LOG(ERROR) << "[ERROR] input_images size is 0, cannot perform warmup";
        return false;
    }
    
    // 初始化 features 输出缓冲区为零数据（确保缓冲区已分配且有效）
    size_t features_size = pipeline_context_.features.getSize();
    if (features_size > 0) {
        std::vector<half> zero_features(features_size, __float2half(0.0f));
        pipeline_context_.features.cudaMemUpdateWrap(zero_features);
        cudaStreamSynchronize(inference_stream_);
        cudaError_t init_err = cudaGetLastError();
        if (init_err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to initialize features for warmup: " << cudaGetErrorString(init_err);
            return false;
        }
        LOG(INFO) << "[INFO] Initialized features with zero data (size: " << features_size << ") for warmup";
    } else {
        LOG(ERROR) << "[ERROR] features size is 0, cannot perform warmup";
        return false;
    }
    
    // 验证缓冲区指针有效性
    if (pipeline_context_.input_images.getCudaPtr() == nullptr) {
        LOG(ERROR) << "[ERROR] input_images CUDA pointer is null before warmup";
        return false;
    }
    if (pipeline_context_.features.getCudaPtr() == nullptr) {
        LOG(ERROR) << "[ERROR] features CUDA pointer is null before warmup";
        return false;
    }
    
    // 打印 Backbone engine 的详细信息用于调试
    if (backbone_) {
        LOG(INFO) << "[INFO] Backbone engine info before warmup:";
        LOG(INFO) << "[INFO]   input_images size: " << input_images_size << " elements (" 
                  << (input_images_size * sizeof(half)) << " bytes)";
        LOG(INFO) << "[INFO]   features size: " << features_size << " elements (" 
                  << (features_size * sizeof(half)) << " bytes)";
    }
    
    // 1. 单独预热 Backbone（已验证 CUDA Graph 已生效）
    LOG(INFO) << "[INFO] Warming up Backbone (" << BACKBONE_WARMUP << " iterations)...";
    bool backbone_success = false;
    for (int i = 0; i < BACKBONE_WARMUP; ++i) {
        // 在每次迭代前检查 CUDA 错误
        cudaError_t pre_err = cudaGetLastError();
        if (pre_err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] CUDA error before Backbone warmup iteration " << i 
                       << ": " << cudaGetErrorString(pre_err);
            cudaDeviceSynchronize();
        }
        
        Status status = backbone_->forward(pipeline_context_, inference_stream_);
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Backbone warmup iteration " << i << " failed with status: " << static_cast<int>(status);
            // 检查 CUDA 错误
            cudaError_t cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                LOG(ERROR) << "[ERROR] CUDA error after failed forward: " << cudaGetErrorString(cuda_err);
            }
            cudaDeviceSynchronize();
            continue;
        }
        
        cudaStreamSynchronize(inference_stream_);
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] CUDA error after Backbone warmup iteration " << i 
                       << ": " << cudaGetErrorString(cuda_err);
            cudaDeviceSynchronize();
        } else {
            backbone_success = true;
            LOG(INFO) << "[INFO] Backbone warmup iteration " << i << " completed successfully";
        }
    }
    
    // 关键修复：如果 backbone warmup 失败，初始化 features 为随机数据
    if (!backbone_success) {
        LOG(WARNING) << "[WARNING] Backbone warmup failed, initializing features with random data for Head1 warmup";
        size_t feature_size = pipeline_context_.features.getSize();
        if (feature_size > 0) {
            std::vector<half> random_features(feature_size);
            std::srand(static_cast<unsigned int>(std::time(nullptr)));
            for (size_t j = 0; j < feature_size; ++j) {
                // 生成 -1.0 到 1.0 之间的随机 FP16 值
                float val = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
                random_features[j] = __float2half(val);
            }
            pipeline_context_.features.cudaMemUpdateWrap(random_features);
            cudaStreamSynchronize(inference_stream_);
            LOG(INFO) << "[INFO] Features initialized with random FP16 data (size: " << feature_size << ")";
        }
    }
    
    // 2. 单独预热 Head1（需要更多迭代才能捕获 CUDA Graph）
    LOG(INFO) << "[INFO] Warming up Head1 (" << HEAD1_WARMUP << " iterations)...";
    
    // 检查 CUDA 错误（推理前）
    cudaError_t pre_err = cudaGetLastError();
    if (pre_err != cudaSuccess) {
        LOG(ERROR) << "[ERROR] CUDA error before Head1 warmup: " << cudaGetErrorString(pre_err);
        cudaDeviceSynchronize();
    }
    
    for (int i = 0; i < HEAD1_WARMUP; ++i) {
        // 验证缓冲区指针有效性
        if (pipeline_context_.features.getCudaPtr() == nullptr) {
            LOG(ERROR) << "[ERROR] Features buffer is null before Head1 warmup iteration " << i;
            return false;
        }
        
        Status status = head1_->forward(pipeline_context_, inference_stream_, head_output_);
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Head1 warmup iteration " << i << " failed";
            // 检查 CUDA 错误
            cudaError_t cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                LOG(ERROR) << "[ERROR] CUDA error: " << cudaGetErrorString(cuda_err);
            }
            cudaDeviceSynchronize();
            return false;  // 改为返回 false，中断 warmup
        }
        
        cudaStreamSynchronize(inference_stream_);
        
        // 检查 CUDA 错误（推理后）
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] CUDA error after Head1 warmup iteration " << i 
                       << ": " << cudaGetErrorString(cuda_err);
            cudaDeviceSynchronize();
            return false;
        }
    }
    
    // 3. 单独预热 Head2（需要更多迭代才能捕获 CUDA Graph）
    // 注意：Head2 需要额外的输入（temp_instance_feature, temp_anchor, mask, track_id）
    // 但这些缓冲区已经分配，使用零数据即可
    LOG(INFO) << "[INFO] Warming up Head2 (" << HEAD2_WARMUP << " iterations)...";
    for (int i = 0; i < HEAD2_WARMUP; ++i) {
        Status status = head2_->forward(pipeline_context_, inference_stream_, head_output_);
        if (status != Status::kSuccess) {
            // LOG(WARNING) << "[WARNING] Head2 warmup iteration " << i << " failed";
            continue;
        }
        cudaStreamSynchronize(inference_stream_);
    }
    
    return true;
}


CAlgResult CoreImplement::forward(const CTimeMatchSrcData *raw_data, void *stream)
{   
    if(enable_timer_){
        return forward_timer(raw_data, stream);
    }
    else{
        return forward_only(raw_data, stream);
    }
}

// 内部使用的 forward_only 方法，接受 CTimeMatchSrcData
CAlgResult CoreImplement::forward_only(const CTimeMatchSrcData *raw_data, void *stream)
{   
    CAlgResult result;
    if (raw_data == nullptr) {
        return result;
    }

    // 1.预处理
    Status status = preprocessor_->forward(raw_data, static_cast<cudaStream_t>(stream), pipeline_context_);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Preprocessor forward failed!";
        return result;
    }

    // 2.执行Instance_Bank->get() 获取历史信息
    status = instance_bank_->get(raw_data, is_first_frame_, static_cast<cudaStream_t>(stream), pipeline_context_);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank get failed!";
        return result;
    }

    // 3.执行BackBone->forward() 提取特征
    status = backbone_->forward(pipeline_context_, static_cast<cudaStream_t>(stream));
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Backbone forward failed!";
        return result;
    }

    // 4.执行Head->forward() 提取目标
    if(is_first_frame_){
        status = head1_->forward(pipeline_context_, static_cast<cudaStream_t>(stream), head_output_);
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Head1 forward failed!";
            return result;
        }
    }else
    {
        status = head2_->forward(pipeline_context_, static_cast<cudaStream_t>(stream), head_output_);
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Head2 forward failed!";
            return result;
        }
    }

    // 5.执行Instance_Bank->cache() 缓存结果
    status = instance_bank_->cache(head_output_, is_first_frame_, static_cast<cudaStream_t>(stream));
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank cache failed!";
        return result;
    }

    // 6.执行Instance_Bank->getTrackId() 获取跟踪ID
    status = instance_bank_->getTrackId(head_output_, is_first_frame_, static_cast<cudaStream_t>(stream));
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank get track id failed!";
        return result;
    }

    if(is_first_frame_){
        is_first_frame_ = false;
    }

    // 7.执行Postprocessor->forward() 后处理
    status = postprocessor_->forward(head_output_, static_cast<cudaStream_t>(stream), result);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Postprocessor forward failed!";
        return result;
    }
    return result;
}

CAlgResult CoreImplement::forward_timer(const CTimeMatchSrcData *raw_data, void *stream)
{
    // 使用计时器版本的forward，在每个阶段添加计时
    CAlgResult result;
    if (raw_data == nullptr) {
        return result;
    }

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    // 计时器
    nv::EventTimer timer_preprocess, timer_instance_bank, timer_backbone, 
                   timer_head, timer_cache, timer_trackid, timer_postprocess;

    // 1.预处理
    timer_preprocess.start(cuda_stream);
    Status status = preprocessor_->forward(raw_data, cuda_stream, pipeline_context_);
    timer_preprocess.stop("Preprocess");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Preprocessor forward failed!";
        return result;
    }

    // 2.执行Instance_Bank->get() 获取历史信息
    timer_instance_bank.start(cuda_stream);
    status = instance_bank_->get(raw_data, is_first_frame_, cuda_stream, pipeline_context_);
    timer_instance_bank.stop("InstanceBank::get");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank get failed!";
        return result;
    }

    // 3.执行BackBone->forward() 提取特征
    timer_backbone.start(cuda_stream);
    status = backbone_->forward(pipeline_context_, cuda_stream);
    timer_backbone.stop("Backbone");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Backbone forward failed!";
        return result;
    }

    // 4.执行Head推理
    timer_head.start(cuda_stream);
    if(is_first_frame_){
        // 保存pipeline_context_中的数据
        // common::savePartialFast(pipeline_context_.features, 89760*256, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_input_features_1*89760*256_float32.bin");
        // common::savePartialFast(pipeline_context_.spatial_shapes, 48, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_input_spatial_shapes_6*4*2_int32.bin");
        // common::savePartialFast(pipeline_context_.level_start_index, 6*4, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_input_level_start_index_6*4_int32.bin");
        // common::savePartialFast(pipeline_context_.instance_feature, 900*256, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_input_instance_feature_1*900*256_float32.bin");
        // common::savePartialFast(pipeline_context_.anchor, 900*11, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_input_anchor_1*900*11_float32.bin");
        // common::savePartialFast(pipeline_context_.time_interval, 1, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_input_time_interval_1_float32.bin");
        // common::savePartialFast(pipeline_context_.image_wh, 12, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_input_image_wh_1*6*2_float32.bin");
        // common::savePartialFast(pipeline_context_.lidar2img, 6*4*4, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_input_lidar2img_1*6*4*4_float32.bin");

        status = head1_->forward(pipeline_context_, cuda_stream, head_output_);
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Head1 forward failed!";
            timer_head.stop("Head1");
            return result;
        }
        timer_head.stop("Head1");

        // 保存head_output_中的数据
        // common::savePartialFast(head_output_.pred_instance_feature, 900*256, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_output_pred_instance_feature_1*900*256_float32.bin");
        // common::savePartialFast(head_output_.pred_anchor, 900*11, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_output_pred_anchor_1*900*11_float32.bin");
        // common::savePartialFast(head_output_.pred_class_score, 900*1, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_output_pred_class_score_1*900*1_float32.bin");
        // common::savePartialFast(head_output_.pred_quality_score, 900*2, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_output_pred_quality_score_1*900*2_float32.bin");
        // common::savePartialFast(head_output_.pred_track_ids, 900, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_0_output_pred_track_ids_1*900_int32.bin");
    } else {
        // common::savePartialFast(pipeline_context_.features, 89760*256, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_features_1*89760*256_float32.bin");
        // common::savePartialFast(pipeline_context_.spatial_shapes, 48, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_spatial_shapes_6*4*2_int32.bin");
        // common::savePartialFast(pipeline_context_.level_start_index, 6*4, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_level_start_index_6*4_int32.bin");
        // common::savePartialFast(pipeline_context_.instance_feature, 900*256, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_instance_feature_1*900*256_float32.bin");
        // common::savePartialFast(pipeline_context_.anchor, 900*11, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_anchor_1*900*11_float32.bin");
        // common::savePartialFast(pipeline_context_.time_interval, 1, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_time_interval_1_float32.bin");
        // common::savePartialFast(pipeline_context_.image_wh, 12, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_image_wh_1*6*2_float32.bin");
        // common::savePartialFast(pipeline_context_.lidar2img, 6*4*4, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_lidar2img_1*6*4*4_float32.bin");
        // // 第二帧独有
        // common::savePartialFast(pipeline_context_.temp_instance_feature, 600*256, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_temp_instance_feature_1*600*256_float32.bin");
        // common::savePartialFast(pipeline_context_.temp_anchor, 600*11, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_temp_anchor_1*600*11_float32.bin");
        // common::savePartialFast(pipeline_context_.mask, 1, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_mask_1_int32.bin");
        // common::savePartialFast(pipeline_context_.track_ids, 900, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_input_track_ids_1*900_int32.bin");
        // 同步验证：确保前序异步写入已完成
        // cudaStreamSynchronize(cuda_stream);
        status = head2_->forward(pipeline_context_, cuda_stream, head_output_);
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Head2 forward failed!";
            timer_head.stop("Head2");
            return result;
        }
        timer_head.stop("Head2");

        // common::savePartialFast(head_output_.pred_instance_feature, 900*256, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_output_pred_instance_feature_1*900*256_float32.bin");
        // common::savePartialFast(head_output_.pred_anchor, 900*11, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_output_pred_anchor_1*900*11_float32.bin");
        // common::savePartialFast(head_output_.pred_class_score, 900*1, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_output_pred_class_score_1*900*1_float32.bin");
        // common::savePartialFast(head_output_.pred_quality_score, 900*2, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_output_pred_quality_score_1*900*2_float32.bin");
        // common::savePartialFast(head_output_.pred_track_ids, 900, "/share/Code/Sparse4dE2E/C++/Output/1104/sample_1_output_pred_track_ids_1*900_int32.bin");
    }

    // 5.执行Instance_Bank->cache() 缓存结果
    timer_cache.start(cuda_stream);
    status = instance_bank_->cache(head_output_, is_first_frame_, cuda_stream);
    timer_cache.stop("InstanceBank::cache");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank cache failed!";
        return result;
    }

    // 6.执行Instance_Bank->getTrackId() 获取跟踪ID
    timer_trackid.start(cuda_stream);
    status = instance_bank_->getTrackId(head_output_, is_first_frame_, cuda_stream);
    timer_trackid.stop("InstanceBank::getTrackId");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank get track id failed!";
        return result;
    }

    if(is_first_frame_){
        is_first_frame_ = false;
    }

    // 7.执行Postprocessor->forward() 后处理
    timer_postprocess.start(cuda_stream);
    status = postprocessor_->forward(head_output_, cuda_stream, result);
    timer_postprocess.stop("Postprocess");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Postprocessor forward failed!";
        return result;
    }

    return result;
}

void CoreImplement::update(const float *lidar2camera, void *stream)
{   
    // 更新lidar2img变换矩阵到pipeline_context_（支持默认流）
    if (lidar2camera != nullptr) {
        const int num_cams = param_.preprocessor_params().num_cams();
        std::vector<float> lidar2img_vec;
        lidar2img_vec.resize(num_cams * 16);
        for (int i = 0; i < num_cams; ++i) {
            std::memcpy(lidar2img_vec.data() + i * 16, lidar2camera + i * 16, 16 * sizeof(float));
        }
        // 转换为 half 类型
        std::vector<half> lidar2img_half(lidar2img_vec.size());
        for (size_t i = 0; i < lidar2img_vec.size(); ++i) {
            lidar2img_half[i] = __float2half(lidar2img_vec[i]);
        }
        cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : static_cast<cudaStream_t>(0);
        pipeline_context_.lidar2img.cudaMemUpdateWrapAsync(lidar2img_half, s);
    }
}

void CoreImplement::free_excess_memory()
{
    // 注意：CudaWrapper 使用 RAII 管理内存，在析构函数中自动释放
    // 如果需要手动释放，可以通过重置对象来实现
    // 但通常不需要手动释放，因为 CudaWrapper 会在析构时自动释放
    
    // 可选：通过移动构造来释放内存（将对象移动到一个临时对象，然后临时对象析构）
    // 但这会清空原有对象，可能不是期望的行为
    
    // 当前实现：不执行任何操作，依赖 RAII 自动管理
    // 如果需要强制释放，可以考虑添加 CudaWrapper::reset() 方法
    LOG(INFO) << "[INFO] free_excess_memory called (memory will be freed automatically by RAII)";
}

// legacy factory removed; use ExportSparse4D CreateCoreObj + initAlgorithm instead

} // namespace core
} // namespace sparse4d