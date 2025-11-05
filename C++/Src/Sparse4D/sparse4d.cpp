
#include "sparse4d.hpp"

#include "common/timer.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <google/protobuf/text_format.h>
#include "log.h"
#include "common/functionhub.hpp"

namespace sparse4d{
namespace core{

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
    CAlgResult result = forward(raw_data, nullptr);
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
    pipeline_context_.instance_feature.cudaMemUpdateWrap(instance_feature);
     
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
    pipeline_context_.anchor.cudaMemUpdateWrap(anchor);

    // 5. 从TaskConfig获取时间间隔
    std::vector<float> time_interval;
    time_interval.push_back(param_.instance_bank_params().default_time_interval());
    pipeline_context_.time_interval.cudaMemUpdateWrap(time_interval);
    
    // 从TaskConfig获取图像宽高（为每个相机分别保存）
    std::vector<float> image_wh;
    image_wh.resize(num_cams * 2); // 每个相机 * 2个值（宽高）
    for (int i = 0; i < num_cams; ++i) {
        image_wh[i * 2] = static_cast<float>(param_.preprocessor_params().model_input_img_w());
        image_wh[i * 2 + 1] = static_cast<float>(param_.preprocessor_params().model_input_img_h());
    }
    pipeline_context_.image_wh.cudaMemUpdateWrap(image_wh);
    
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
    pipeline_context_.lidar2img.cudaMemUpdateWrap(lidar2img);

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
        cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : static_cast<cudaStream_t>(0);
        pipeline_context_.lidar2img.cudaMemUpdateWrapAsync(lidar2img_vec, s);
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