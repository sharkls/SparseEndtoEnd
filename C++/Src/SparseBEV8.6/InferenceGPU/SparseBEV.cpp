#include "SparseBEV.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cuda_fp16.h>

// 注册模块
REGISTER_MODULE("SparseBEV", SparseBEV, SparseBEV)

/**
 * @brief 析构函数
 */
SparseBEV::~SparseBEV() {
    // 清理资源
}

/**
 * @brief 初始化模块
 * @param p_pAlgParam 算法参数指针
 * @return 初始化是否成功
 */
bool SparseBEV::init(void* p_pAlgParam) 
{
    LOG(INFO) << "[INFO] SparseBEV::init start";
    
    if (!p_pAlgParam) {
        LOG(ERROR) << "[ERROR] Algorithm parameter is null";
        return false;
    }

    // 解析任务配置
    m_taskConfig = *static_cast<const sparsebev::TaskConfig*>(p_pAlgParam);
    // 强制使用全精度，取消半精度支持
    use_half_precision_ = false;
    
    LOG(INFO) << "[INFO] Task config loaded, forced to use full precision (FP32)";
    
    // 创建TensorRT引擎
    LOG(INFO) << "[INFO] Creating TensorRT engines...";
    
    // 特征提取引擎
    m_extract_feat_engine = std::make_shared<TensorRT>(
        m_taskConfig.extract_feat_engine().engine_path(),
        "",  // 无插件路径
        std::vector<std::string>(m_taskConfig.extract_feat_engine().input_names().begin(), 
                                m_taskConfig.extract_feat_engine().input_names().end()),
        std::vector<std::string>(m_taskConfig.extract_feat_engine().output_names().begin(), 
                                m_taskConfig.extract_feat_engine().output_names().end()));
    
    if (!m_extract_feat_engine) {
        LOG(ERROR) << "[ERROR] Failed to load extract feature engine";
        return false;
    }

    // 第一帧头部引擎
    m_head1st_engine = std::make_shared<TensorRT>(
        m_taskConfig.head1st_engine().engine_path(),
        m_taskConfig.head1st_engine().plugin_path(),
        std::vector<std::string>(m_taskConfig.head1st_engine().input_names().begin(), 
                                m_taskConfig.head1st_engine().input_names().end()),
        std::vector<std::string>(m_taskConfig.head1st_engine().output_names().begin(), 
                                m_taskConfig.head1st_engine().output_names().end()));
    LOG(INFO) << "[INFO] head1st_engine loaded";
    // 第二帧头部引擎
    m_head2nd_engine = std::make_shared<TensorRT>(
        m_taskConfig.head2nd_engine().engine_path(),
        m_taskConfig.head2nd_engine().plugin_path(),
        std::vector<std::string>(m_taskConfig.head2nd_engine().input_names().begin(), 
                                m_taskConfig.head2nd_engine().input_names().end()),
        std::vector<std::string>(m_taskConfig.head2nd_engine().output_names().begin(), 
                                m_taskConfig.head2nd_engine().output_names().end()));
    LOG(INFO) << "[INFO] head2nd_engine loaded";
    // 计算各种数据大小
    size_t input_size = m_taskConfig.preprocessor_params().num_cams() * 
                       m_taskConfig.preprocessor_params().model_input_img_c() *
                       m_taskConfig.preprocessor_params().model_input_img_h() * 
                       m_taskConfig.preprocessor_params().model_input_img_w();
    
    // 从TaskConfig获取特征大小
    size_t feature_size = 1; // batch size
    for (int i = 0; i < m_taskConfig.model_cfg_params().sparse4d_extract_feat_shape_lc_size(); ++i) {
        feature_size *= m_taskConfig.model_cfg_params().sparse4d_extract_feat_shape_lc(i);
    }
    
    // 从TaskConfig获取预测数据大小
    size_t pred_size = 1 * m_taskConfig.instance_bank_params().num_querys() * 
                      m_taskConfig.model_cfg_params().embedfeat_dims();
    size_t anchor_size = 1 * m_taskConfig.instance_bank_params().num_querys() * 
                        m_taskConfig.instance_bank_params().query_dims();
    size_t class_score_size = 1 * m_taskConfig.instance_bank_params().num_querys() * m_taskConfig.model_cfg_params().num_classes();
    size_t quality_score_size = 1 * m_taskConfig.instance_bank_params().num_querys() * 2;
    size_t track_id_size = 1 * m_taskConfig.instance_bank_params().num_querys();

    // 根据精度分配GPU内存 - 强制使用float精度
    if (!m_float_input_wrapper.allocate(input_size) ||
        !m_float_features_wrapper.allocate(feature_size) ||
        !m_float_pred_instance_feature_wrapper.allocate(pred_size) ||
        !m_float_pred_anchor_wrapper.allocate(anchor_size) ||
        !m_float_pred_class_score_wrapper.allocate(class_score_size) ||
        !m_float_pred_quality_score_wrapper.allocate(quality_score_size) ||
        !m_int32_pred_track_id_wrapper.allocate(track_id_size) ||
        !m_float_tmp_outs0_wrapper.allocate(pred_size * 2) ||
        !m_float_tmp_outs1_wrapper.allocate(pred_size * 2) ||
        !m_float_tmp_outs2_wrapper.allocate(pred_size * 2) ||
        !m_float_tmp_outs3_wrapper.allocate(pred_size * 2) ||
        !m_float_tmp_outs4_wrapper.allocate(pred_size * 2) ||
        !m_float_tmp_outs5_wrapper.allocate(pred_size * 2)) {
        LOG(ERROR) << "[ERROR] Failed to allocate float precision GPU memory!";
        status_ = false;
        return status_;
    }
    LOG(INFO) << "[INFO] Pre-allocated float precision GPU memory";

    // 加载辅助数据
    loadAuxiliaryData();

    // 分配辅助数据的GPU内存 - 预分配以避免运行时分配开销
    LOG(INFO) << "[INFO] Pre-allocating auxiliary data GPU memory...";
    if (!m_gpu_spatial_shapes_wrapper.allocate(m_spatial_shapes.size()) ||
        !m_gpu_level_start_index_wrapper.allocate(m_level_start_index.size()) ||
        !m_gpu_instance_feature_wrapper.allocate(m_instance_feature.size()) ||
        !m_gpu_anchor_wrapper.allocate(m_anchor.size()) ||
        !m_gpu_time_interval_wrapper.allocate(m_time_interval.size()) ||
        !m_gpu_image_wh_wrapper.allocate(m_image_wh.size()) ||
        !m_gpu_lidar2img_wrapper.allocate(m_lidar2img.size())) {
        LOG(ERROR) << "[ERROR] Failed to allocate auxiliary data GPU memory!";
        status_ = false;
        return status_;
    }
    LOG(INFO) << "[INFO] Auxiliary data GPU memory pre-allocated successfully";

    // 将辅助数据复制到GPU内存
    std::vector<int32_t> expanded;
    expanded.reserve(6 * 4 * 2);
    for (int cam = 0; cam < 6; ++cam) {
      for (int lvl = 0; lvl < 4; ++lvl) {
        expanded.push_back(m_spatial_shapes[lvl * 2 + 0]);
        expanded.push_back(m_spatial_shapes[lvl * 2 + 1]);
      }
    }
    m_gpu_spatial_shapes_wrapper.cudaMemUpdateWrap(expanded);
    m_gpu_level_start_index_wrapper.cudaMemUpdateWrap(std::vector<int32_t>(m_level_start_index.begin(), m_level_start_index.end()));
    m_gpu_instance_feature_wrapper.cudaMemUpdateWrap(m_instance_feature);
    m_gpu_anchor_wrapper.cudaMemUpdateWrap(m_anchor);
    m_gpu_time_interval_wrapper.cudaMemUpdateWrap(m_time_interval);
    m_gpu_image_wh_wrapper.cudaMemUpdateWrap(m_image_wh);
    m_gpu_lidar2img_wrapper.cudaMemUpdateWrap(m_lidar2img);

    LOG(INFO) << "[INFO] Auxiliary data copied to GPU memory";
    
    // 初始化GPU版本的InstanceBank
    instance_bank_gpu_ = std::make_unique<InstanceBankGPU>(m_taskConfig);
    has_previous_frame_ = false;
    current_timestamp_ = 0.0;
    previous_timestamp_ = 0.0;
    
    // 初始化坐标变换矩阵为单位矩阵
    current_global_to_lidar_mat_ = Eigen::Matrix<double, 4, 4>::Identity();
    previous_global_to_lidar_mat_ = Eigen::Matrix<double, 4, 4>::Identity();
    
    status_ = true;
    LOG(INFO) << "[INFO] SparseBEV module initialized successfully with GPU InstanceBank";
    return status_;
}

/**
 * @brief 加载辅助数据
 */
void SparseBEV::loadAuxiliaryData() 
{
    // 从TaskConfig获取空间形状数据
    m_spatial_shapes.clear();
    for (int i = 0; i < m_taskConfig.model_cfg_params().sparse4d_extract_feat_spatial_shapes_ld_size(); ++i) {
        m_spatial_shapes.push_back(m_taskConfig.model_cfg_params().sparse4d_extract_feat_spatial_shapes_ld(i));
    }
    
    // 从TaskConfig获取层级起始索引
    m_level_start_index.clear();
    for (int i = 0; i < m_taskConfig.model_cfg_params().sparse4d_extract_feat_level_start_index_size(); ++i) {
        m_level_start_index.push_back(m_taskConfig.model_cfg_params().sparse4d_extract_feat_level_start_index(i));
    }
    
    // 从TaskConfig获取实例特征大小并初始化
    size_t instance_feature_size = m_taskConfig.instance_bank_params().num_querys() * 
                                  m_taskConfig.model_cfg_params().embedfeat_dims();
    m_instance_feature.resize(instance_feature_size, 0.0f);
    
    // 从锚点文件加载锚点数据
    m_anchor.clear();
    std::string anchor_path = m_taskConfig.instance_bank_params().instance_bank_anchor_path();
    if (!anchor_path.empty()) {
        size_t expected_anchor_size = m_taskConfig.instance_bank_params().num_querys() * 
                                     m_taskConfig.instance_bank_params().query_dims();
        
        std::ifstream file(anchor_path, std::ios::binary);
        if (file.is_open()) {
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);
            
            size_t expected_file_size = expected_anchor_size * sizeof(float);
            if (file_size == expected_file_size) {
                m_anchor.resize(expected_anchor_size);
                file.read(reinterpret_cast<char*>(m_anchor.data()), file_size);
                LOG(INFO) << "[INFO] Successfully loaded anchor data from: " << anchor_path;
            } else {
                LOG(ERROR) << "[ERROR] Anchor file size mismatch! Expected: " << expected_file_size 
                          << " bytes, Actual: " << file_size << " bytes";
                m_anchor.resize(expected_anchor_size, 0.0f);
            }
            file.close();
        } else {
            LOG(ERROR) << "[ERROR] Failed to open anchor file: " << anchor_path;
            m_anchor.resize(expected_anchor_size, 0.0f);
        }
    } else {
        LOG(WARNING) << "[WARNING] No anchor file path specified, using default values";
        size_t expected_anchor_size = m_taskConfig.instance_bank_params().num_querys() * 
                                     m_taskConfig.instance_bank_params().query_dims();
        m_anchor.resize(expected_anchor_size, 0.0f);
    }
    
    // 从TaskConfig获取时间间隔
    m_time_interval = {m_taskConfig.instance_bank_params().default_time_interval()};
    
    // 从TaskConfig获取图像宽高（为6个相机分别保存）
    m_image_wh.resize(12); // 6个相机 * 2个值（宽高）
    for (int i = 0; i < 6; ++i) {
        m_image_wh[i * 2] = static_cast<float>(m_taskConfig.preprocessor_params().model_input_img_w());
        m_image_wh[i * 2 + 1] = static_cast<float>(m_taskConfig.preprocessor_params().model_input_img_h());
    }
    
    // 激光雷达到图像变换矩阵大小
    size_t lidar2img_size = m_taskConfig.preprocessor_params().num_cams() * 4 * 4;
    m_lidar2img.resize(lidar2img_size, 0.0f);
    
    // 初始化默认的单位矩阵作为标定参数
    for (int i = 0; i < m_taskConfig.preprocessor_params().num_cams(); ++i) {
        m_lidar2img[i * 16 + 0] = 1.0f;  // [0,0]
        m_lidar2img[i * 16 + 5] = 1.0f;  // [1,1]
        m_lidar2img[i * 16 + 10] = 1.0f; // [2,2]
        m_lidar2img[i * 16 + 15] = 1.0f; // [3,3]
    }
    
    LOG(INFO) << "[INFO] Auxiliary data loaded from TaskConfig";
}

/**
 * @brief 设置输入数据
 * @param input 输入数据指针
 */
void SparseBEV::setInput(void* input) {
    if (input == nullptr) {
        LOG(ERROR) << "[ERROR] Input pointer is null!";
        return;
    }
    
    // 尝试解析为SparseBEVInputWrapper（新格式）
    auto* wrapper = static_cast<sparsebev::SparseBEVInputWrapper*>(input);
    if (wrapper != nullptr && wrapper->isValid()) {
        const auto* input_data = wrapper->getData();
        if (input_data != nullptr) {
            setInputDataFromWrapper(*input_data);
            return;
        }
    }
    
    // 尝试解析为CudaWrapper<float>（旧格式，向后兼容）
    auto* float_wrapper = static_cast<CudaWrapper<float>*>(input);
    if (float_wrapper != nullptr) {
        auto float_data = float_wrapper->cudaMemcpyD2HResWrap();
        m_float_input_wrapper = CudaWrapper<float>(float_data);
        LOG(INFO) << "[INFO] Set input data using CudaWrapper<float> format";
        return;
    }
    
    LOG(ERROR) << "[ERROR] Unsupported input data format!";
}

/**
 * @brief 执行推理
 */
void SparseBEV::execute() 
{
    if (!status_) {
        LOG(ERROR) << "[ERROR] Module not initialized!";
        return;
    }
    
    LOG(INFO) << "[INFO] Starting SparseBEV inference with GPU InstanceBank...";
    
    // 添加输入数据验证
    if (!m_float_input_wrapper.isValid()) {
        LOG(ERROR) << "[ERROR] Input wrapper is not valid!";
        return;
    }
    
    LOG(INFO) << "[INFO] Input wrapper size: " << m_float_input_wrapper.getSize();
    
    // 验证TensorRT引擎
    if (!m_extract_feat_engine || !m_head1st_engine || !m_head2nd_engine) {
        LOG(ERROR) << "[ERROR] TensorRT engines not properly initialized!";
        return;
    }
    
    LOG(INFO) << "[INFO] TensorRT engines validation passed";
    
    // 验证输出缓冲区
    if (!m_float_features_wrapper.isValid()) {
        LOG(ERROR) << "[ERROR] Features wrapper is not valid!";
        return;
    }
    
    LOG(INFO) << "[INFO] Features wrapper size: " << m_float_features_wrapper.getSize();
    
    // 创建CUDA流
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        LOG(ERROR) << "[ERROR] Failed to create CUDA stream!";
        return;
    }
    
    LOG(INFO) << "[INFO] CUDA stream created successfully";
    
    Status result = Status::kSuccess;
    
    // 声明时间戳变量
    int64_t feature_extraction_start, feature_extraction_end, feature_extraction_time;
    int64_t first_frame_start, first_frame_end, first_frame_time;
    int64_t second_frame_start, second_frame_end, second_frame_time;
    int64_t instance_bank_start, instance_bank_end, instance_bank_time;
    
    try {
        // 初始化
        CudaWrapper<int32_t> track_ids;
        
        // 步骤1：特征提取
        LOG(INFO) << "[INFO] Step 1: Feature extraction";
        
        feature_extraction_start = GetTimeStamp();
        result = extractFeatures(m_float_input_wrapper, stream, m_float_features_wrapper);
        feature_extraction_end = GetTimeStamp();
        feature_extraction_time = feature_extraction_end - feature_extraction_start;
        LOG(INFO) << "[INFO] Feature extraction completed in " << feature_extraction_time << "ms";
        
        if (result != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Feature extraction failed!";
            cudaStreamDestroy(stream);
            return;
        }

        // 步骤2：获取temporal信息（使用GPU版本的InstanceBank）
        instance_bank_start = GetTimeStamp();
        LOG(INFO) << "[INFO] Step 2: Getting temporal information from GPU InstanceBank";
        
        // 直接使用GPU版本的InstanceBank，无需数据传输
        auto [gpu_instance_feature, gpu_anchor, gpu_cached_feature, gpu_cached_anchor, 
              time_interval, mask, gpu_cached_track_ids] = 
            instance_bank_gpu_->get(current_timestamp_, current_global_to_lidar_mat_, is_first_frame_, stream);
        
        instance_bank_end = GetTimeStamp();
        instance_bank_time = instance_bank_end - instance_bank_start;
        LOG(INFO) << "[INFO] GPU InstanceBank completed in " << instance_bank_time << "ms";
        
        // 重要：将GPU版本的InstanceBank返回的数据赋值给相应的GPU内存包装器
        // 这样后续的推理就能使用这些数据了
        LOG(INFO) << "[INFO] Copying GPU InstanceBank data to GPU memory wrappers for inference...";
        
        // 拷贝实例特征数据
        cudaMemcpyAsync(m_gpu_instance_feature_wrapper.getCudaPtr(), 
                        gpu_instance_feature.getCudaPtr(),
                        gpu_instance_feature.getSize() * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        // 拷贝锚点数据
        cudaMemcpyAsync(m_gpu_anchor_wrapper.getCudaPtr(), 
                        gpu_anchor.getCudaPtr(),
                        gpu_anchor.getSize() * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        // 对于第二帧推理需要的时序数据
        if (!is_first_frame_ && has_previous_frame_) {
            // 拷贝缓存的GPU数据到时序数据包装器
            cudaMemcpyAsync(m_float_temp_instance_feature_wrapper.getCudaPtr(), 
                            gpu_cached_feature.getCudaPtr(),
                            gpu_cached_feature.getSize() * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            
            cudaMemcpyAsync(m_float_temp_anchor_wrapper.getCudaPtr(), 
                            gpu_cached_anchor.getCudaPtr(),
                            gpu_cached_anchor.getSize() * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            
            // 将mask值转换为向量并赋值
            std::vector<int32_t> mask_vector = {static_cast<int32_t>(mask)};
            m_float_mask_wrapper.cudaMemUpdateWrap(mask_vector);
        }
        
        // 更新时间间隔
        m_time_interval[0] = time_interval;
        m_gpu_time_interval_wrapper.cudaMemUpdateWrap(m_time_interval);
        
        LOG(INFO) << "[INFO] GPU InstanceBank data successfully assigned to GPU memory wrappers";
        
        // 步骤3：根据是否为第一帧选择不同的推理路径
        if (is_first_frame_ || !has_previous_frame_) {
            LOG(INFO) << "[INFO] Step 3: First frame head inference";
            
            first_frame_start = GetTimeStamp();
            
            // 使用GPU上的数据直接进行第一帧头部推理
            result = headFirstFrame(m_float_features_wrapper, stream,
                                  m_float_pred_instance_feature_wrapper,
                                  m_float_pred_anchor_wrapper,
                                  m_float_pred_class_score_wrapper,
                                  m_float_pred_quality_score_wrapper);
                                      
            if (result == Status::kSuccess) {
                first_frame_end = GetTimeStamp();
                first_frame_time = first_frame_end - first_frame_start;
                LOG(INFO) << "[INFO] First frame inference completed in " << first_frame_time << "ms, caching results...";
                
                // 直接在GPU上缓存结果到InstanceBank
                instance_bank_gpu_->cache(m_float_pred_instance_feature_wrapper,
                                             m_float_pred_anchor_wrapper,
                                             m_float_pred_class_score_wrapper,
                                             is_first_frame_, stream);
                
                // 从GPU版本的InstanceBank获取跟踪ID
                Status status = instance_bank_gpu_->getTrackId(is_first_frame_,m_int32_temp_track_id_wrapper, stream);
                if (status != kSuccess) {
                    LOG(ERROR) << "[ERROR] Failed to get track IDs";
                }
                
                // 标记第一帧已完成
                is_first_frame_ = false;
                has_previous_frame_ = true;
                LOG(INFO) << "[INFO] First frame results cached to GPU InstanceBank";
                LOG(INFO) << "[INFO] Track IDs from first frame assigned to temp wrapper for second frame inference";
            }
        } else {
            LOG(INFO) << "[INFO] Step 3: Second frame head inference";
            
            second_frame_start = GetTimeStamp();
            
            // 使用GPU上的数据直接进行第二帧头部推理
            // 注意：m_int32_temp_track_id_wrapper现在包含了第一帧推理产生的跟踪ID
            result = headSecondFrame(m_float_features_wrapper, stream,
                                    m_float_pred_instance_feature_wrapper,
                                    m_float_pred_anchor_wrapper,
                                    m_float_pred_class_score_wrapper,
                                    m_float_pred_quality_score_wrapper,
                                    m_int32_pred_track_id_wrapper);
                                    
            if (result == Status::kSuccess) {
                second_frame_end = GetTimeStamp();
                second_frame_time = second_frame_end - second_frame_start;
                LOG(INFO) << "[INFO] Second frame inference completed in " << second_frame_time << "ms, caching results...";
                
                // 直接在GPU上缓存结果到InstanceBank
                instance_bank_gpu_->cache(m_float_pred_instance_feature_wrapper,
                                             m_float_pred_anchor_wrapper,
                                             m_float_pred_class_score_wrapper,
                                             is_first_frame_, stream);
                
                // 获取跟踪ID
                Status status = instance_bank_gpu_->getTrackId(is_first_frame_, m_int32_temp_track_id_wrapper, stream);
                if (status != kSuccess) {
                    LOG(ERROR) << "[ERROR] Failed to get track IDs";
                }
                
                LOG(INFO) << "[INFO] Second frame results cached to GPU InstanceBank";
            }
        }
        
        if (result != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Frame inference failed!";
            cudaStreamDestroy(stream);
            return;
        }
        
        // 步骤4：转换输出格式
        LOG(INFO) << "[INFO] Step 4: Converting output format";
        
        // 从GPU获取数据并转换格式
        auto pred_instance_feature = m_float_pred_instance_feature_wrapper.cudaMemcpyD2HResWrap();
        auto pred_anchor = m_float_pred_anchor_wrapper.cudaMemcpyD2HResWrap();
        auto pred_class_score = m_float_pred_class_score_wrapper.cudaMemcpyD2HResWrap();
        auto pred_quality_score = m_float_pred_quality_score_wrapper.cudaMemcpyD2HResWrap();
        auto track_ids_cpu = track_ids.cudaMemcpyD2HResWrap();
        
        convertToOutputFormat(pred_instance_feature, pred_anchor, pred_class_score, 
                             pred_quality_score, track_ids_cpu, m_raw_result);
        
        // 更新时间戳
        previous_timestamp_ = current_timestamp_;
        previous_global_to_lidar_mat_ = current_global_to_lidar_mat_;
        
        // 计算总体推理耗时
        int64_t total_inference_time = 0;
        if (is_first_frame_ || !has_previous_frame_) {
            total_inference_time = feature_extraction_time + first_frame_time;
        } else {
            total_inference_time = feature_extraction_time + second_frame_time;
        }
        
        LOG(INFO) << "[INFO] ========== Inference Time Summary ==========";
        LOG(INFO) << "[INFO] Feature extraction time: " << feature_extraction_time << "ms";
        LOG(INFO) << "[INFO] GPU InstanceBank time: " << instance_bank_time << "ms";
        if (is_first_frame_ || !has_previous_frame_) {
            LOG(INFO) << "[INFO] First frame inference time: " << first_frame_time << "ms";
        } else {
            LOG(INFO) << "[INFO] Second frame inference time: " << second_frame_time << "ms";
        }
        LOG(INFO) << "[INFO] Total inference time: " << total_inference_time << "ms";
        LOG(INFO) << "[INFO] ===========================================";
        
        LOG(INFO) << "[INFO] SparseBEV inference completed successfully with GPU InstanceBank";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] Exception during inference: " << e.what();
        result = Status::kInferenceErr;
    }
    
    // 清理CUDA流
    cudaStreamDestroy(stream);
}

/**
 * @brief 从输入包装器设置输入数据
 * @param input_data 输入数据结构
 * @return 设置是否成功
 */
bool SparseBEV::setInputDataFromWrapper(const sparsebev::SparseBEVInputData& input_data)
{
    // 设置图像特征
    if (!setImageFeature(input_data.image_feature)) {
        LOG(ERROR) << "[ERROR] Failed to set image feature";
        return false;
    }
    
    // 保存前一帧的时间戳和坐标变换矩阵
    previous_timestamp_ = current_timestamp_;
    previous_global_to_lidar_mat_ = current_global_to_lidar_mat_;
    
    // 设置当前时间戳信息
    current_timestamp_ = input_data.time_interval.time_interval ? 
        input_data.time_interval.time_interval->cudaMemcpyD2HResWrap()[0] : 0.0;
    LOG(INFO) << "setInputDataFromWrapper() current_timestamp_ : " << current_timestamp_;
    
    // 设置坐标变换矩阵
    if (input_data.coordinate_transform.data_valid) {
        current_global_to_lidar_mat_ = input_data.coordinate_transform.global_to_lidar_mat;
        LOG(INFO) << "[INFO] Set coordinate transform matrix from input data";
    } else {
        if (!has_previous_frame_) {
            current_global_to_lidar_mat_ = Eigen::Matrix<double, 4, 4>::Identity();
            LOG(ERROR) << "[ERROR] No coordinate transform provided for first frame, using identity matrix";
        } else {
            LOG(ERROR) << "[ERROR] No coordinate transform provided, using previous frame matrix";
        }
    }
    
    // 设置时间间隔
    if (!setTimeInterval(input_data.time_interval)) {
        LOG(ERROR) << "[ERROR] Failed to set time interval";
        return false;
    }
    
    // 设置图像标定（包括lidar2img数据）
    if (!setImageCalibration(input_data.image_calibration)) {
        LOG(ERROR) << "[ERROR] Failed to set image calibration";
        return false;
    }
    
    // 标记已有前一帧数据
    has_previous_frame_ = true;
    
    LOG(INFO) << "[INFO] Successfully set input data from wrapper";
    LOG(INFO) << "[INFO] Previous timestamp: " << previous_timestamp_ << "ms";
    LOG(INFO) << "[INFO] Current timestamp: " << current_timestamp_ << "ms";
    LOG(INFO) << "[INFO] Lidar2img data size: " << m_lidar2img.size() << " floats";

    return true;
}

/**
 * @brief 设置图像特征
 */
bool SparseBEV::setImageFeature(const sparsebev::ImageFeature& image_data)
{
    if (!image_data.data_valid || image_data.feature == nullptr) {
        LOG(ERROR) << "[ERROR] Image data is not valid or feature is null!";
        return false;
    }
    
    LOG(INFO) << "[INFO] Setting image feature...";
    
    auto float_data = image_data.feature->cudaMemcpyD2HResWrap();
    m_float_input_wrapper = CudaWrapper<float>(float_data);
    
    LOG(INFO) << "[INFO] Image feature set successfully";
    return true;
}

/**
 * @brief 设置时间间隔
 */
bool SparseBEV::setTimeInterval(const sparsebev::TimeInterval& time_interval) 
{
    // 计算时间间隔
    if (has_previous_frame_) {
        double time_diff = std::abs(current_timestamp_ - previous_timestamp_);
        m_time_interval[0] = static_cast<float>(time_diff / 1000.0);
        
        LOG(INFO) << "[INFO] Calculated time interval: " << m_time_interval[0] 
                  << "s (from " << previous_timestamp_ << "ms to " << current_timestamp_ << "ms)";
    } else {
        m_time_interval[0] = m_taskConfig.instance_bank_params().default_time_interval();
        LOG(INFO) << "[INFO] First frame, using default time interval: " << m_time_interval[0] << "s";
    }
    
    return true;
}

/**
 * @brief 设置图像标定信息
 */
bool SparseBEV::setImageCalibration(const sparsebev::ImageCalibration& image_calibration) 
{
    if (!image_calibration.data_valid) {
        LOG(WARNING) << "[WARNING] Image calibration data is not valid, using default values";
        // 使用默认的单位矩阵作为标定参数
        for (int i = 0; i < m_taskConfig.preprocessor_params().num_cams(); ++i) {
            m_lidar2img[i * 16 + 0] = 1.0f;  // [0,0]
            m_lidar2img[i * 16 + 5] = 1.0f;  // [1,1]
            m_lidar2img[i * 16 + 10] = 1.0f; // [2,2]
            m_lidar2img[i * 16 + 15] = 1.0f; // [3,3]
        }
        
        // 更新GPU内存包装器
        m_gpu_lidar2img_wrapper.cudaMemUpdateWrap(m_lidar2img);
        LOG(INFO) << "[INFO] Updated GPU lidar2img wrapper with default identity matrices";
        
        return true;
    }
    
    // 从输入数据中获取lidar2img矩阵
    if (image_calibration.lidar2img && image_calibration.lidar2img->getSize() > 0) {
        auto lidar2img_data = image_calibration.lidar2img->cudaMemcpyD2HResWrap();
        
        // 检查数据大小是否匹配预期
        size_t expected_size = m_taskConfig.preprocessor_params().num_cams() * 4 * 4;
        if (lidar2img_data.size() != expected_size) {
            LOG(ERROR) << "[ERROR] Lidar2img data size mismatch. Expected: " 
                       << expected_size << " floats, got: " << lidar2img_data.size() << " floats";
            return false;
        }
        
        // 复制数据到m_lidar2img
        m_lidar2img.assign(lidar2img_data.begin(), lidar2img_data.end());
        
        // 同时更新GPU内存包装器
        m_gpu_lidar2img_wrapper.cudaMemUpdateWrap(m_lidar2img);
        
        LOG(INFO) << "[INFO] Successfully set lidar2img data from input calibration";
        LOG(INFO) << "[INFO] Lidar2img data size: " << m_lidar2img.size() << " floats";
        
        return true;
    } else {
        LOG(WARNING) << "[WARNING] No valid lidar2img data in image calibration, using default values";
        // 使用默认的单位矩阵作为标定参数
        for (int i = 0; i < m_taskConfig.preprocessor_params().num_cams(); ++i) {
            m_lidar2img[i * 16 + 0] = 1.0f;  // [0,0]
            m_lidar2img[i * 16 + 5] = 1.0f;  // [1,1]
            m_lidar2img[i * 16 + 10] = 1.0f; // [2,2]
            m_lidar2img[i * 16 + 15] = 1.0f; // [3,3]
        }
        
        // 更新GPU内存包装器
        m_gpu_lidar2img_wrapper.cudaMemUpdateWrap(m_lidar2img);
        LOG(INFO) << "[INFO] Updated GPU lidar2img wrapper with default identity matrices";
        
        return true;
    }
}

/**
 * @brief 特征提取推理
 */
Status SparseBEV::extractFeatures(const CudaWrapper<float>& input_imgs, 
                                  const cudaStream_t& stream,
                                  CudaWrapper<float>& output_features) 
{
    if (m_extract_feat_engine == nullptr) {
        LOG(ERROR) << "[ERROR] Extract feature engine is null!";
        return Status::kInferenceErr;
    }
    
    // 准备输入输出缓冲区
    std::vector<void*> input_buffers = {const_cast<void*>(static_cast<const void*>(input_imgs.getCudaPtr()))};
    std::vector<void*> output_buffers = {static_cast<void*>(output_features.getCudaPtr())};
    
    // 执行推理
    bool success = m_extract_feat_engine->infer(input_buffers.data(), output_buffers.data(), stream);
    
    if (success) {
        LOG(INFO) << "[INFO] Feature extraction completed successfully";
        return Status::kSuccess;
    } else {
        LOG(ERROR) << "[ERROR] Feature extraction failed";
        return Status::kInferenceErr;
    }
}

/**
 * @brief 第一帧头部推理
 */
Status SparseBEV::headFirstFrame(const CudaWrapper<float>& features,
                                 const cudaStream_t& stream,
                                 CudaWrapper<float>& pred_instance_feature,
                                 CudaWrapper<float>& pred_anchor,
                                 CudaWrapper<float>& pred_class_score,
                                 CudaWrapper<float>& pred_quality_score) 
{
    if (m_head1st_engine == nullptr) {
        LOG(ERROR) << "[ERROR] Head1st engine is null!";
        return Status::kInferenceErr;
    }
    
    // 准备输入数据
    std::vector<void*> input_buffers = {
        const_cast<void*>(static_cast<const void*>(features.getCudaPtr())),
        static_cast<void*>(m_gpu_spatial_shapes_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_level_start_index_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_instance_feature_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_anchor_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_time_interval_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_image_wh_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_lidar2img_wrapper.getCudaPtr())
    };
    
    // 准备输出缓冲区
    std::vector<void*> output_buffers = {
        static_cast<void*>(m_float_tmp_outs0_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_tmp_outs1_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_tmp_outs2_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_tmp_outs3_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_tmp_outs4_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_tmp_outs5_wrapper.getCudaPtr()),
        static_cast<void*>(pred_instance_feature.getCudaPtr()),
        static_cast<void*>(pred_anchor.getCudaPtr()),
        static_cast<void*>(pred_class_score.getCudaPtr()),
        static_cast<void*>(pred_quality_score.getCudaPtr())
    };
    
    // 执行推理
    bool success = m_head1st_engine->infer(input_buffers.data(), output_buffers.data(), stream);
    
    if (success) {
        LOG(INFO) << "[INFO] First frame head inference completed successfully";
        return Status::kSuccess;
    } else {
        LOG(ERROR) << "[ERROR] First frame head inference failed";
        return Status::kInferenceErr;
    }
}

/**
 * @brief 第二帧头部推理
 */
Status SparseBEV::headSecondFrame(const CudaWrapper<float>& features,
                                  const cudaStream_t& stream,
                                  CudaWrapper<float>& pred_instance_feature,
                                  CudaWrapper<float>& pred_anchor,
                                  CudaWrapper<float>& pred_class_score,
                                  CudaWrapper<float>& pred_quality_score,
                                  CudaWrapper<int32_t>& pred_track_id)
{
    if (m_head2nd_engine == nullptr) {
        LOG(ERROR) << "[ERROR] Head2nd engine is null!";
        return Status::kInferenceErr;
    }
    
    // 准备输入数据 - 使用正确的变量名
    std::vector<void*> input_buffers = {
        const_cast<void*>(static_cast<const void*>(features.getCudaPtr())),
        static_cast<void*>(m_gpu_spatial_shapes_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_level_start_index_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_instance_feature_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_anchor_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_time_interval_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_temp_instance_feature_wrapper.getCudaPtr()),  // 修正变量名
        static_cast<void*>(m_float_temp_anchor_wrapper.getCudaPtr()),           // 修正变量名
        static_cast<void*>(m_float_mask_wrapper.getCudaPtr()),                  // 修正变量名
        static_cast<void*>(m_int32_temp_track_id_wrapper.getCudaPtr()),         // 修正变量名
        static_cast<void*>(m_gpu_image_wh_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_lidar2img_wrapper.getCudaPtr())
    };
    
    // 准备输出缓冲区
    std::vector<void*> output_buffers = {
        static_cast<void*>(m_float_tmp_outs0_wrapper.getCudaPtr()),
        static_cast<void*>(pred_track_id.getCudaPtr()),
        static_cast<void*>(m_float_tmp_outs1_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_tmp_outs2_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_tmp_outs3_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_tmp_outs4_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_tmp_outs5_wrapper.getCudaPtr()),
        static_cast<void*>(pred_instance_feature.getCudaPtr()),
        static_cast<void*>(pred_anchor.getCudaPtr()),
        static_cast<void*>(pred_class_score.getCudaPtr()),
        static_cast<void*>(pred_quality_score.getCudaPtr())
    };
    
    // 执行推理
    bool success = m_head2nd_engine->infer(input_buffers.data(), output_buffers.data(), stream);
    
    if (success) {
        LOG(INFO) << "[INFO] Second frame head inference completed successfully";
        return Status::kSuccess;
    } else {
        LOG(ERROR) << "[ERROR] Second frame head inference failed";
        return Status::kInferenceErr;
    }
}

/**
 * @brief 转换输出格式
 */
void SparseBEV::convertToOutputFormat(const std::vector<float>& pred_instance_feature,
                                     const std::vector<float>& pred_anchor,
                                     const std::vector<float>& pred_class_score,
                                     const std::vector<float>& pred_quality_score,
                                     const std::vector<int32_t>& track_ids,
                                     RawInferenceResult& raw_result) 
{
    // 创建CudaWrapper并复制数据
    raw_result.pred_instance_feature = std::make_shared<CudaWrapper<float>>(pred_instance_feature);
    raw_result.pred_anchor = std::make_shared<CudaWrapper<float>>(pred_anchor);
    raw_result.pred_class_score = std::make_shared<CudaWrapper<float>>(pred_class_score);
    raw_result.pred_quality_score = std::make_shared<CudaWrapper<float>>(pred_quality_score);
    
    // 处理跟踪ID
    if (!track_ids.empty()) {
        raw_result.pred_track_id = std::make_shared<CudaWrapper<int32_t>>(track_ids);
    }
    
    // 设置元数据
    raw_result.num_objects = pred_instance_feature.size() / m_taskConfig.model_cfg_params().embedfeat_dims();
    raw_result.num_classes = m_taskConfig.model_cfg_params().num_classes();
    raw_result.is_first_frame = is_first_frame_;
    
    LOG(INFO) << "[INFO] Converted output format with " << raw_result.num_objects << " objects";
}

/**
 * @brief 获取原始推理结果
 */
RawInferenceResult SparseBEV::getRawInferenceResult() const 
{
    return m_raw_result;
}

/**
 * @brief 获取输出数据
 */
void* SparseBEV::getOutput() 
{
    return static_cast<void*>(&m_raw_result);
}

/**
 * @brief 设置当前样本索引
 */
void SparseBEV::setCurrentSampleIndex(int sample_index) {
    current_sample_index_ = sample_index;
    LOG(INFO) << "setCurrentSampleIndex() current_sample_index_ : " << current_sample_index_;
}