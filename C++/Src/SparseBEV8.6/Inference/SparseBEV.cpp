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

    // 重置内存池
    m_memory_pool.reset();
    
    // 解析任务配置
    m_taskConfig = *static_cast<const sparsebev::TaskConfig*>(p_pAlgParam);
    // 强制使用全精度，取消半精度支持
    use_half_precision_ = false;
    
    LOG(INFO) << "[INFO] Task config loaded, forced to use full precision (FP32)";
    
    // 创建TensorRT引擎
    LOG(INFO) << "[INFO] Creating TensorRT engines...";
    
    // 特征提取引擎
    LOG(INFO) << "[DEBUG] Loading extract feature engine from: " << m_taskConfig.extract_feat_engine().engine_path();
    LOG(INFO) << "[DEBUG] Extract feature engine input names:";
    for (const auto& name : m_taskConfig.extract_feat_engine().input_names()) {
        LOG(INFO) << "[DEBUG]   " << name;
    }
    LOG(INFO) << "[DEBUG] Extract feature engine output names:";
    for (const auto& name : m_taskConfig.extract_feat_engine().output_names()) {
        LOG(INFO) << "[DEBUG]   " << name;
    }
    
    m_extract_feat_engine = std::make_shared<TensorRT>(
        m_taskConfig.extract_feat_engine().engine_path(),
        "",  // 无插件路径
        std::vector<std::string>(m_taskConfig.extract_feat_engine().input_names().begin(), 
                                m_taskConfig.extract_feat_engine().input_names().end()),
        std::vector<std::string>(m_taskConfig.extract_feat_engine().output_names().begin(), 
                                m_taskConfig.extract_feat_engine().output_names().end()));
    
    if (m_extract_feat_engine) {
        LOG(INFO) << "[DEBUG] Extract feature engine loaded successfully";
    } else {
        LOG(ERROR) << "[ERROR] Failed to load extract feature engine";
    }

    // 第一帧头部引擎
    m_head1st_engine = std::make_shared<TensorRT>(
        m_taskConfig.head1st_engine().engine_path(),
        m_taskConfig.head1st_engine().plugin_path(),
        std::vector<std::string>(m_taskConfig.head1st_engine().input_names().begin(), 
                                m_taskConfig.head1st_engine().input_names().end()),
        std::vector<std::string>(m_taskConfig.head1st_engine().output_names().begin(), 
                                m_taskConfig.head1st_engine().output_names().end()));

    // 添加详细的引擎信息调试
    LOG(INFO) << "[DEBUG] ========== Head1st Engine Info ==========";
    LOG(INFO) << "[DEBUG] Engine path: " << m_taskConfig.head1st_engine().engine_path();
    LOG(INFO) << "[DEBUG] Plugin path: " << m_taskConfig.head1st_engine().plugin_path();
    
    LOG(INFO) << "[DEBUG] Input names:";
    for (const auto& name : m_taskConfig.head1st_engine().input_names()) {
        LOG(INFO) << "[DEBUG]   " << name;
    }
    
    LOG(INFO) << "[DEBUG] Output names:";
    for (const auto& name : m_taskConfig.head1st_engine().output_names()) {
        LOG(INFO) << "[DEBUG]   " << name;
    }
    
    // 检查输出名称是否与我们的缓冲区顺序匹配
    std::vector<std::string> expected_output_names = {
       "tmp_outs0", "tmp_outs1", "tmp_outs2", "tmp_outs3", "tmp_outs4", "tmp_outs5", "pred_instance_feature", "pred_anchor", "pred_class_score", "pred_quality_score"
    };
    
    LOG(INFO) << "[DEBUG] Expected output names:";
    for (const auto& name : expected_output_names) {
        LOG(INFO) << "[DEBUG]   " << name;
    }
    
    // 检查前4个输出名称是否匹配
    bool names_match = true;
    for (size_t i = 0; i < 4 && i < m_taskConfig.head1st_engine().output_names().size(); ++i) {
        if (m_taskConfig.head1st_engine().output_names(i) != expected_output_names[i]) {
            LOG(ERROR) << "[ERROR] Output name mismatch at index " << i 
                       << ": expected '" << expected_output_names[i] 
                       << "', got '" << m_taskConfig.head1st_engine().output_names(i) << "'";
            names_match = false;
        }
    }
    
    if (names_match) {
        LOG(INFO) << "[DEBUG] Output names match expected order";
    } else {
        LOG(ERROR) << "[ERROR] Output names do not match expected order!";
    }

    // 第二帧头部引擎
    m_head2nd_engine = std::make_shared<TensorRT>(
        m_taskConfig.head2nd_engine().engine_path(),
        m_taskConfig.head2nd_engine().plugin_path(),
        std::vector<std::string>(m_taskConfig.head2nd_engine().input_names().begin(), 
                                m_taskConfig.head2nd_engine().input_names().end()),
        std::vector<std::string>(m_taskConfig.head2nd_engine().output_names().begin(), 
                                m_taskConfig.head2nd_engine().output_names().end()));

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
    size_t class_score_size = 1 * m_taskConfig.instance_bank_params().num_querys() * m_taskConfig.model_cfg_params().num_classes(); // 假设10个类别
    size_t quality_score_size = 1 * m_taskConfig.instance_bank_params().num_querys() * 2; // 质量分数维度
    size_t track_id_size = 1 * m_taskConfig.instance_bank_params().num_querys();

    // 根据精度分配GPU内存 - 强制使用float精度
        if (!m_float_input_wrapper.allocate(input_size) ||
            !m_float_features_wrapper.allocate(feature_size) ||
            !m_float_pred_instance_feature_wrapper.allocate(pred_size) ||
            !m_float_pred_anchor_wrapper.allocate(anchor_size) ||
            !m_float_pred_class_score_wrapper.allocate(class_score_size) ||
            !m_float_pred_quality_score_wrapper.allocate(quality_score_size) ||
            !m_int32_pred_track_id_wrapper.allocate(track_id_size) ||  // 改为int32_t
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

    // 分配辅助数据的GPU内存
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

    // 将辅助数据复制到GPU内存
    // 原始: m_spatial_shapes.size()==8, 形如 [H0,W0,H1,W1,H2,W2,H3,W3]
    std::vector<int32_t> expanded;
    expanded.reserve(6 * 4 * 2);
    for (int cam = 0; cam < 6; ++cam) {
      for (int lvl = 0; lvl < 4; ++lvl) {
        expanded.push_back(m_spatial_shapes[lvl * 2 + 0]);
        expanded.push_back(m_spatial_shapes[lvl * 2 + 1]);
      }
    }
    // 用 expanded 来 allocate/cudaMemUpdateWrap，确保 GPU 侧是 48 个元素
    m_gpu_spatial_shapes_wrapper.cudaMemUpdateWrap(expanded);
    m_gpu_level_start_index_wrapper.cudaMemUpdateWrap(std::vector<int32_t>(m_level_start_index.begin(), m_level_start_index.end()));
    m_gpu_instance_feature_wrapper.cudaMemUpdateWrap(m_instance_feature);
    m_gpu_anchor_wrapper.cudaMemUpdateWrap(m_anchor);
    m_gpu_time_interval_wrapper.cudaMemUpdateWrap(m_time_interval);
    m_gpu_image_wh_wrapper.cudaMemUpdateWrap(m_image_wh);
    m_gpu_lidar2img_wrapper.cudaMemUpdateWrap(m_lidar2img);

    LOG(INFO) << "[INFO] Auxiliary data copied to GPU memory";
    
    // 初始化InstanceBank
    instance_bank_ = std::make_unique<InstanceBank>(m_taskConfig);
    has_previous_frame_ = false;
    current_timestamp_ = 0.0;
    previous_timestamp_ = 0.0;
    
    // 初始化坐标变换矩阵为单位矩阵
    current_global_to_lidar_mat_ = Eigen::Matrix<double, 4, 4>::Identity();
    previous_global_to_lidar_mat_ = Eigen::Matrix<double, 4, 4>::Identity();
    
    status_ = true;
    LOG(INFO) << "[INFO] SparseBEV module initialized successfully";
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
        // 计算期望的锚点数据大小
        size_t expected_anchor_size = m_taskConfig.instance_bank_params().num_querys() * 
                                     m_taskConfig.instance_bank_params().query_dims();
        
        // 读取二进制文件
        std::ifstream file(anchor_path, std::ios::binary);
        if (file.is_open()) {
            // 获取文件大小
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);
            
            // 验证文件大小是否正确（float32 = 4 bytes）
            size_t expected_file_size = expected_anchor_size * sizeof(float);
            if (file_size == expected_file_size) {
                m_anchor.resize(expected_anchor_size);
                file.read(reinterpret_cast<char*>(m_anchor.data()), file_size);
                LOG(INFO) << "[INFO] Successfully loaded anchor data from: " << anchor_path;
                LOG(INFO) << "[INFO] Anchor data size: " << m_anchor.size() << " floats";
            } else {
                LOG(ERROR) << "[ERROR] Anchor file size mismatch! Expected: " << expected_file_size 
                          << " bytes, Actual: " << file_size << " bytes";
                // 使用默认锚点数据
                m_anchor.resize(expected_anchor_size, 0.0f);
            }
            file.close();
        } else {
            LOG(ERROR) << "[ERROR] Failed to open anchor file: " << anchor_path;
            // 使用默认锚点数据
            m_anchor.resize(expected_anchor_size, 0.0f);
        }
    } else {
        LOG(WARNING) << "[WARNING] No anchor file path specified, using default values";
        // 使用默认锚点数据
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
    
    // 激光雷达到图像变换矩阵大小（需要根据实际数据加载）每个相机4x4变换矩阵
    size_t lidar2img_size = m_taskConfig.preprocessor_params().num_cams() * 4 * 4;
    m_lidar2img.resize(lidar2img_size, 0.0f);
    
    // 初始化默认的单位矩阵作为标定参数（实际数据将在setImageCalibration中设置）
    for (int i = 0; i < m_taskConfig.preprocessor_params().num_cams(); ++i) {
        m_lidar2img[i * 16 + 0] = 1.0f;  // [0,0]
        m_lidar2img[i * 16 + 5] = 1.0f;  // [1,1]
        m_lidar2img[i * 16 + 10] = 1.0f; // [2,2]
        m_lidar2img[i * 16 + 15] = 1.0f; // [3,3]
    }
    
    LOG(INFO) << "[INFO] Lidar2img matrices initialized with default values (will be updated from input data)";
    
    LOG(INFO) << "[INFO] Auxiliary data loaded from TaskConfig";
    LOG(INFO) << "[INFO] Spatial shapes size: " << m_spatial_shapes.size();
    LOG(INFO) << "[INFO] Level start index size: " << m_level_start_index.size();
    LOG(INFO) << "[INFO] Instance feature size: " << m_instance_feature.size();
    LOG(INFO) << "[INFO] Anchor size: " << m_anchor.size();
    LOG(INFO) << "[INFO] Lidar2img size: " << m_lidar2img.size();
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
        // 使用新的数据结构
        const auto* input_data = wrapper->getData();
        if (input_data != nullptr) {
            setInputDataFromWrapper(*input_data);
            return;
        }
    }
    
    // 尝试解析为CudaWrapper<float>（旧格式，向后兼容）
    auto* float_wrapper = static_cast<CudaWrapper<float>*>(input);
    if (float_wrapper != nullptr) {
        // 直接使用float精度数据
            auto float_data = float_wrapper->cudaMemcpyD2HResWrap();
            m_float_input_wrapper = CudaWrapper<float>(float_data);
        LOG(INFO) << "[INFO] Set input data using CudaWrapper<float> format";
        return;
    }
    
    // 尝试解析为CudaWrapper<half>（旧格式，向后兼容）
    auto* half_wrapper = static_cast<CudaWrapper<half>*>(input);
    if (half_wrapper != nullptr) {
            // 将half转换为float精度
            auto half_data = half_wrapper->cudaMemcpyD2HResWrap();
            std::vector<float> float_data(half_data.begin(), half_data.end());
            m_float_input_wrapper = CudaWrapper<float>(float_data);
        LOG(INFO) << "[INFO] Set input data using legacy CudaWrapper<half> format (converted to float)";
        return;
    }
    
    LOG(ERROR) << "[ERROR] Unsupported input data format!";
    LOG(ERROR) << "[ERROR] Expected SparseBEVInputWrapper, CudaWrapper<float>, or CudaWrapper<half>";
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
    
    LOG(INFO) << "[INFO] Starting SparseBEV inference...";
    
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
    
    try {
        // 步骤0：保存中间变量数据（当run_status为False时）
        if (!m_taskConfig.run_status()) {
            savePreprocessedData();
            saveTimeIntervalData();
            saveImageWhData();
            saveLidar2imgData();
            saveSpatialShapesData();
            saveLevelStartIndexData();
            
            // 添加调试信息
            LOG(INFO) << "[DEBUG] Image WH size: " << m_image_wh.size();
            LOG(INFO) << "[DEBUG] Lidar2img size: " << m_lidar2img.size();
            LOG(INFO) << "[DEBUG] Time interval: " << m_time_interval[0];
        }
        
        // 初始化
        std::vector<int32_t> track_ids;

        // 步骤1：特征提取
        LOG(INFO) << "[INFO] Step 1: Feature extraction";
        LOG(INFO) << "[INFO] About to call extractFeatures...";
        // 强制使用float精度进行特征提取
        result = extractFeatures(m_float_input_wrapper, stream, m_float_features_wrapper);
        
        LOG(INFO) << "[INFO] extractFeatures completed with result: " << static_cast<int>(result);
        
        if (result != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Feature extraction failed!";
            cudaStreamDestroy(stream);
            return;
        }
        
        // 当run_status为False时，保存特征提取结果
        if (!m_taskConfig.run_status()) {
            saveFeatureExtractionResults();
            
        }
        
        // 步骤2：获取temporal信息
        LOG(INFO) << "[INFO] Step 2: Getting temporal information: current_timestamp_ : " << current_timestamp_;
        auto [instance_feature, anchor, cached_feature, cached_anchor, time_interval, mask, cached_track_ids] = 
            instance_bank_->get(current_timestamp_, current_global_to_lidar_mat_, is_first_frame_);
        
        // 将获取的instance_feature和anchor复制到GPU内存包装器中用于推理
        m_gpu_instance_feature_wrapper.cudaMemUpdateWrap(instance_feature);
        m_gpu_anchor_wrapper.cudaMemUpdateWrap(anchor);
        m_float_temp_instance_feature_wrapper.cudaMemUpdateWrap(cached_feature);
        m_float_temp_anchor_wrapper.cudaMemUpdateWrap(cached_anchor);
        // 将单个mask值转换为向量
        LOG(INFO) << "mask : " << mask;
        std::vector<int32_t> mask_vector = {static_cast<int32_t>(mask)};
        m_float_mask_wrapper.cudaMemUpdateWrap(mask_vector);
        // 更新当前帧的时间间隔
        m_time_interval[0] = time_interval;
        // 将更新后的时间间隔复制到GPU内存
        m_gpu_time_interval_wrapper.cudaMemUpdateWrap(m_time_interval);
        
        if(!m_taskConfig.run_status()) {
            m_instance_feature = instance_feature;
            saveInstanceFeatureData();
            saveAnchorData();
        }
        
        LOG(INFO) << "[INFO] Updated GPU instance_feature size: " << instance_feature.size() << " floats";
        LOG(INFO) << "[INFO] Updated GPU anchor size: " << anchor.size() << " floats";
        LOG(INFO) << "[INFO] Updated time interval: " << m_time_interval[0] << "s";
        
        // 步骤3：根据是否为第一帧选择不同的推理路径
        if (is_first_frame_ || !has_previous_frame_) {
            LOG(INFO) << "[INFO] Step 3: First frame head inference";
            // 强制使用float精度进行第一帧头部推理
                result = headFirstFrame(m_float_features_wrapper, stream,
                                      m_float_pred_instance_feature_wrapper,
                                      m_float_pred_anchor_wrapper,
                                      m_float_pred_class_score_wrapper,
                                      m_float_pred_quality_score_wrapper);
                                      
            if (result == Status::kSuccess) {
                LOG(INFO) << "[INFO] First frame inference completed, caching results...";
                
                // 将结果缓存到InstanceBank
                auto pred_feature = m_float_pred_instance_feature_wrapper.cudaMemcpyD2HResWrap();
                auto pred_anchor = m_float_pred_anchor_wrapper.cudaMemcpyD2HResWrap();
                auto pred_class = m_float_pred_class_score_wrapper.cudaMemcpyD2HResWrap();
                
                instance_bank_->cache(pred_feature, pred_anchor, pred_class, is_first_frame_);
            
                // 标记第一帧已完成
                is_first_frame_ = false;
                has_previous_frame_ = true;

                // 从InstanceBank获取跟踪ID
                track_ids = instance_bank_->getTrackId(track_ids);
                LOG(INFO) << "track_ids : " << track_ids.size();
                m_int32_temp_track_id_wrapper.cudaMemUpdateWrap(track_ids);

                // 保存InstanceBank缓存数据
                instance_bank_->saveInstanceBankData(0);  // sample_0
                
                LOG(INFO) << "[INFO] First frame results cached to InstanceBank";
            }
        } else {
            LOG(INFO) << "[INFO] Step 3: Second frame head inference";
            // 强制使用float精度进行第二帧头部推理
            result = headSecondFrame(m_float_features_wrapper, stream,
                                    m_float_pred_instance_feature_wrapper,
                                    m_float_pred_anchor_wrapper,
                                    m_float_pred_class_score_wrapper,
                                    m_float_pred_quality_score_wrapper,
                                    m_int32_pred_track_id_wrapper);  // 改为int32_t
                                    
            if (result == Status::kSuccess) {
                LOG(INFO) << "[INFO] Second frame inference completed, caching results...";
                
                // 将结果缓存到InstanceBank  
                auto pred_feature = m_float_pred_instance_feature_wrapper.cudaMemcpyD2HResWrap();
                auto pred_anchor = m_float_pred_anchor_wrapper.cudaMemcpyD2HResWrap();
                auto pred_class = m_float_pred_class_score_wrapper.cudaMemcpyD2HResWrap();
                auto pred_track_id = m_int32_pred_track_id_wrapper.cudaMemcpyD2HResWrap();  // 改为int32_t
                
                instance_bank_->cache(pred_feature, pred_anchor, pred_class, is_first_frame_);
                
                // 直接使用int32_t类型，无需类型转换
                track_ids = instance_bank_->getTrackId(pred_track_id);
                
                LOG(INFO) << "[INFO] Second frame results cached to InstanceBank";
            }
        }
        
        if (result != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Frame inference failed!";
            cudaStreamDestroy(stream);
            return;
        }
        
        // // 打印跟踪ID信息
        // LOG(INFO) << "[INFO] Retrieved track_ids from InstanceBank, size: " << track_ids.size();
        // if (!track_ids.empty()) {
        //     LOG(INFO) << "[INFO] First 10 track_ids: ";
        //     for (size_t i = 0; i < std::min(size_t(10), track_ids.size()); ++i) {
        //         LOG(INFO) << "  [" << i << "]: " << track_ids[i];
        //     }
        // }
        
        // 步骤4：转换输出格式
        LOG(INFO) << "[INFO] Step 4: Converting output format";
        // 强制使用float精度进行输出格式转换
            convertToOutputFormat(
                m_float_pred_instance_feature_wrapper.cudaMemcpyD2HResWrap(),
                m_float_pred_anchor_wrapper.cudaMemcpyD2HResWrap(),
                m_float_pred_class_score_wrapper.cudaMemcpyD2HResWrap(),
                m_float_pred_quality_score_wrapper.cudaMemcpyD2HResWrap(),
                track_ids,
                m_raw_result
            );
        
        // 步骤5：保存推理结果数据（当run_status为False时）
        if (!m_taskConfig.run_status()) {
            LOG(INFO) << "[INFO] Step 5: Saving inference results";
            
            // 验证推理结果
                auto pred_feature = m_float_pred_instance_feature_wrapper.cudaMemcpyD2HResWrap();
                auto pred_anchor = m_float_pred_anchor_wrapper.cudaMemcpyD2HResWrap();
                auto pred_class = m_float_pred_class_score_wrapper.cudaMemcpyD2HResWrap();
                auto pred_quality = m_float_pred_quality_score_wrapper.cudaMemcpyD2HResWrap();
                
                LOG(INFO) << "[DEBUG] Pred feature size: " << pred_feature.size();
                LOG(INFO) << "[DEBUG] Pred anchor size: " << pred_anchor.size();
                LOG(INFO) << "[DEBUG] Pred class size: " << pred_class.size();
                LOG(INFO) << "[DEBUG] Pred quality size: " << pred_quality.size();
            
            // 保存推理结果数据
            savePredInstanceFeatureData();
            savePredAnchorData();
            savePredClassScoreData();
            savePredQualityScoreData();
            savePredTrackIdData();
            
            // 保存tmp_outs0~5数据
            saveTmpOuts0Data();
            saveTmpOuts1Data();
            saveTmpOuts2Data();
            saveTmpOuts3Data();
            saveTmpOuts4Data();
            saveTmpOuts5Data();
            
            // 如果是第二帧，保存时序数据
            if (!is_first_frame_) {
                saveTempInstanceFeatureData();
                saveTempAnchorData();
                saveMaskData();
                saveTrackIdData();
            }
        }
        
        // 更新时间戳
        previous_timestamp_ = current_timestamp_;
        previous_global_to_lidar_mat_ = current_global_to_lidar_mat_;
        
        LOG(INFO) << "[INFO] SparseBEV inference completed successfully";
        
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
        
        // 打印变换矩阵（调试用）
        LOG(INFO) << "[INFO] Current global_to_lidar matrix:";
        for (int i = 0; i < 4; ++i) {
            std::stringstream ss;
            ss << "  Row " << i << ": ";
            for (int j = 0; j < 4; ++j) {
                ss << std::fixed << std::setprecision(6) 
                   << current_global_to_lidar_mat_(i, j) << " ";
            }
            LOG(INFO) << ss.str();
        }
    } else {
        // 如果没有提供坐标变换矩阵，第一帧使用单位矩阵，后续帧保持前一帧的值
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
    
    // 强制使用float精度设置输入数据
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
        // 如果有前一帧，计算实际的时间间隔
        double time_diff = std::abs(current_timestamp_ - previous_timestamp_);
        // 将毫秒转换为秒
        m_time_interval[0] = static_cast<float>(time_diff / 1000.0);
        
        LOG(INFO) << "[INFO] Calculated time interval: " << m_time_interval[0] 
                  << "s (from " << previous_timestamp_ << "ms to " << current_timestamp_ << "ms)";
    } else {
        // 第一帧使用默认时间间隔
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
        LOG(INFO) << "[INFO] Updated GPU lidar2img wrapper with default identity matrices (no valid data)";
        
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
        LOG(INFO) << "[INFO] GPU lidar2img wrapper updated with new data";
        
        // 打印第一个相机的矩阵（调试用）
        LOG(INFO) << "[INFO] Camera 0 lidar2img matrix from input:";
        for (int row = 0; row < 4; ++row) {
            std::stringstream ss;
            ss << "  Row " << row << ": ";
            for (int col = 0; col < 4; ++col) {
                ss << std::fixed << std::setprecision(6) 
                   << m_lidar2img[row * 4 + col] << " ";
            }
            LOG(INFO) << ss.str();
        }
        
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
        LOG(INFO) << "[INFO] Updated GPU lidar2img wrapper with default identity matrices (no valid data)";
        
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
    
    // 准备输入输出缓冲区 - 使用分离的输入输出buffer
    std::vector<void*> input_buffers = {const_cast<void*>(static_cast<const void*>(input_imgs.getCudaPtr()))};
    std::vector<void*> output_buffers = {static_cast<void*>(output_features.getCudaPtr())};
    
    // 执行推理
    LOG(INFO) << "[DEBUG] Starting feature extraction inference (float)...";
    LOG(INFO) << "[DEBUG] Input buffer size: " << input_buffers.size();
    LOG(INFO) << "[DEBUG] Output buffer size: " << output_buffers.size();
    LOG(INFO) << "[DEBUG] Input data size: " << input_imgs.getSize();
    LOG(INFO) << "[DEBUG] Output data size: " << output_features.getSize();
    
    bool success = m_extract_feat_engine->infer(input_buffers.data(), output_buffers.data(), stream);
    
    if (success) {
        LOG(INFO) << "[INFO] Feature extraction completed successfully";
        
        // 添加调试信息：检查输出数据的统计信息
        auto output_data = output_features.cudaMemcpyD2HResWrap();
        if (!output_data.empty()) {
            float min_val = output_data[0];
            float max_val = output_data[0];
            float sum = 0.0f;
            
            for (size_t i = 0; i < std::min(output_data.size(), size_t(1000)); ++i) {
                float val = output_data[i];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
            }
            
            LOG(INFO) << "[DEBUG] Feature extraction output stats:";
            LOG(INFO) << "[DEBUG]   Min value: " << min_val;
            LOG(INFO) << "[DEBUG]   Max value: " << max_val;
            LOG(INFO) << "[DEBUG]   Average (first 1000): " << (sum / std::min(output_data.size(), size_t(1000)));
            LOG(INFO) << "[DEBUG]   Total elements: " << output_data.size();
        } else {
            LOG(ERROR) << "[ERROR] Feature extraction output is empty!";
        }
        
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
    
    // 添加详细的调试信息
    LOG(INFO) << "[DEBUG] ========== HeadFirstFrame Debug Info ==========";
    LOG(INFO) << "[DEBUG] Input features size: " << features.getSize();
    LOG(INFO) << "[DEBUG] Output pred_instance_feature size: " << pred_instance_feature.getSize();
    LOG(INFO) << "[DEBUG] Output pred_anchor size: " << pred_anchor.getSize();
    LOG(INFO) << "[DEBUG] Output pred_class_score size: " << pred_class_score.getSize();
    LOG(INFO) << "[DEBUG] Output pred_quality_score size: " << pred_quality_score.getSize();
    
    // 检查输入数据的前几个值
    auto input_data = features.cudaMemcpyD2HResWrap();
    // LOG(INFO) << "[DEBUG] Input features first 5 values: " 
    //           << input_data[0] << ", " << input_data[1] << ", " << input_data[2] 
    //           << ", " << input_data[3] << ", " << input_data[4];
    
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
    
    
    // 准备输出缓冲区 - 修复顺序以匹配TensorRT引擎的实际输出绑定
    std::vector<void*> output_buffers = {
        static_cast<void*>(m_float_tmp_outs0_wrapper.getCudaPtr()),  // binding 8
        static_cast<void*>(m_float_tmp_outs1_wrapper.getCudaPtr()),  // binding 9
        static_cast<void*>(m_float_tmp_outs2_wrapper.getCudaPtr()),  // binding 10
        static_cast<void*>(m_float_tmp_outs3_wrapper.getCudaPtr()),  // binding 11
        static_cast<void*>(m_float_tmp_outs4_wrapper.getCudaPtr()),  // binding 12
        static_cast<void*>(m_float_tmp_outs5_wrapper.getCudaPtr()),  // binding 13
        static_cast<void*>(pred_instance_feature.getCudaPtr()),       // binding 14
        static_cast<void*>(pred_anchor.getCudaPtr()),                // binding 15
        static_cast<void*>(pred_class_score.getCudaPtr()),           // binding 16
        static_cast<void*>(pred_quality_score.getCudaPtr())          // binding 17
    };
    
    LOG(INFO) << "[DEBUG] Input buffers count: " << input_buffers.size();
    LOG(INFO) << "[DEBUG] Output buffers count: " << output_buffers.size();
    
    // 在执行infer前保存input_buffers的各种数据
    if (!m_taskConfig.run_status()) {
        LOG(INFO) << "[DEBUG] Saving input buffer data before inference...";
        
        // 保存输入特征数据
        auto input_features = features.cudaMemcpyD2HResWrap();
        saveInputFeaturesData(input_features);
        
        // 保存空间形状数据
        auto spatial_shapes = m_gpu_spatial_shapes_wrapper.cudaMemcpyD2HResWrap();
        saveInputSpatialShapesData(spatial_shapes);
        
        // 保存层级起始索引数据
        auto level_start_index = m_gpu_level_start_index_wrapper.cudaMemcpyD2HResWrap();
        saveInputLevelStartIndexData(level_start_index);
        
        // 保存实例特征数据
        auto instance_feature = m_gpu_instance_feature_wrapper.cudaMemcpyD2HResWrap();
        saveInputInstanceFeatureData(instance_feature);
        
        // 保存锚点数据
        auto anchor = m_gpu_anchor_wrapper.cudaMemcpyD2HResWrap();
        saveInputAnchorData(anchor);
        
        // 保存时间间隔数据
        auto time_interval = m_gpu_time_interval_wrapper.cudaMemcpyD2HResWrap();
        saveInputTimeIntervalData(time_interval);
        
        // 保存图像宽高数据
        auto image_wh = m_gpu_image_wh_wrapper.cudaMemcpyD2HResWrap();
        saveInputImageWhData(image_wh);
        
        // 保存Lidar2img变换矩阵数据
        auto lidar2img = m_gpu_lidar2img_wrapper.cudaMemcpyD2HResWrap();
        saveInputLidar2imgData(lidar2img);
        
        LOG(INFO) << "[DEBUG] Input buffer data saved successfully";
    }
    
    // 执行推理
    bool success = m_head1st_engine->infer(input_buffers.data(), output_buffers.data(), stream);
    
    if (success) {
        LOG(INFO) << "[INFO] First frame head inference completed successfully";
        
        // 检查输出数据
        auto pred_feature_data = pred_instance_feature.cudaMemcpyD2HResWrap();
        auto pred_anchor_data = pred_anchor.cudaMemcpyD2HResWrap();
        auto pred_class_data = pred_class_score.cudaMemcpyD2HResWrap();
        auto pred_quality_data = pred_quality_score.cudaMemcpyD2HResWrap();
        
        LOG(INFO) << "[DEBUG] ========== Output Data Analysis ==========";
        LOG(INFO) << "[DEBUG] Pred feature data size: " << pred_feature_data.size();
        LOG(INFO) << "[DEBUG] Pred anchor data size: " << pred_anchor_data.size();
        LOG(INFO) << "[DEBUG] Pred class data size: " << pred_class_data.size();
        LOG(INFO) << "[DEBUG] Pred quality data size: " << pred_quality_data.size();
        
        // 检查是否有非零值
        bool has_nonzero = false;
        for (size_t i = 0; i < pred_feature_data.size(); ++i) {
            if (pred_feature_data[i] != 0.0f) {
                has_nonzero = true;
                break;
            }
        }
        
        if (!has_nonzero) {
            LOG(ERROR) << "[ERROR] All pred_feature values are zero!";
        } else {
            LOG(INFO) << "[DEBUG] Found non-zero values in pred_feature";
        }
        
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
                                  CudaWrapper<int32_t>& pred_track_id)  // 改为int32_t
{
    if (m_head2nd_engine == nullptr) {
        LOG(ERROR) << "[ERROR] Head2nd engine is null!";
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
        static_cast<void*>(m_float_temp_instance_feature_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_temp_anchor_wrapper.getCudaPtr()),
        static_cast<void*>(m_float_mask_wrapper.getCudaPtr()),
        static_cast<void*>(m_int32_temp_track_id_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_image_wh_wrapper.getCudaPtr()),
        static_cast<void*>(m_gpu_lidar2img_wrapper.getCudaPtr())
    };
    
    // 在执行infer前保存input_buffers的各种数据
    if (!m_taskConfig.run_status()) {
        LOG(INFO) << "[DEBUG] Saving input buffer data before second frame inference...";
        
        // 保存输入特征数据
        auto input_features = features.cudaMemcpyD2HResWrap();
        saveInputFeaturesData(input_features);
        
        // 保存空间形状数据
        auto spatial_shapes = m_gpu_spatial_shapes_wrapper.cudaMemcpyD2HResWrap();
        saveInputSpatialShapesData(spatial_shapes);
        
        // 保存层级起始索引数据
        auto level_start_index = m_gpu_level_start_index_wrapper.cudaMemcpyD2HResWrap();
        saveInputLevelStartIndexData(level_start_index);
        
        // 保存实例特征数据
        auto instance_feature = m_gpu_instance_feature_wrapper.cudaMemcpyD2HResWrap();
        saveInputInstanceFeatureData(instance_feature);
        
        // 保存锚点数据
        auto anchor = m_gpu_anchor_wrapper.cudaMemcpyD2HResWrap();
        saveInputAnchorData(anchor);
        
        // 保存时间间隔数据
        auto time_interval = m_gpu_time_interval_wrapper.cudaMemcpyD2HResWrap();
        saveInputTimeIntervalData(time_interval);
        
        // 保存临时实例特征数据
        auto temp_instance_feature = m_float_temp_instance_feature_wrapper.cudaMemcpyD2HResWrap();
        saveTempInstanceFeatureData();
        
        // 保存临时锚点数据
        auto temp_anchor = m_float_temp_anchor_wrapper.cudaMemcpyD2HResWrap();
        saveTempAnchorData();
        
        // 保存掩码数据
        auto mask = m_float_mask_wrapper.cudaMemcpyD2HResWrap();
        saveMaskData();
        
        // 保存临时跟踪ID数据
        auto temp_track_id = m_int32_temp_track_id_wrapper.cudaMemcpyD2HResWrap();
        saveTrackIdData();
        
        // 保存图像宽高数据
        auto image_wh = m_gpu_image_wh_wrapper.cudaMemcpyD2HResWrap();
        saveInputImageWhData(image_wh);
        
        // 保存Lidar2img变换矩阵数据
        auto lidar2img = m_gpu_lidar2img_wrapper.cudaMemcpyD2HResWrap();
        saveInputLidar2imgData(lidar2img);
        
        LOG(INFO) << "[DEBUG] Second frame input buffer data saved successfully";
    }
    
    // 准备输出缓冲区 - 修复顺序以匹配TensorRT引擎的实际输出绑定
    std::vector<void*> output_buffers = {
        static_cast<void*>(m_float_tmp_outs0_wrapper.getCudaPtr()),  // binding 12
        static_cast<void*>(pred_track_id.getCudaPtr()),               // binding 13
        static_cast<void*>(m_float_tmp_outs1_wrapper.getCudaPtr()),  // binding 14
        static_cast<void*>(m_float_tmp_outs2_wrapper.getCudaPtr()),  // binding 15
        static_cast<void*>(m_float_tmp_outs3_wrapper.getCudaPtr()),  // binding 16
        static_cast<void*>(m_float_tmp_outs4_wrapper.getCudaPtr()),  // binding 17
        static_cast<void*>(m_float_tmp_outs5_wrapper.getCudaPtr()),  // binding 18
        static_cast<void*>(pred_instance_feature.getCudaPtr()),       // binding 19
        static_cast<void*>(pred_anchor.getCudaPtr()),                // binding 20
        static_cast<void*>(pred_class_score.getCudaPtr()),           // binding 21
        static_cast<void*>(pred_quality_score.getCudaPtr())          // binding 22
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
 * @brief 转换输出格式（float版本）
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
        // 直接使用int32_t类型，与RawInferenceResult.h中的定义保持一致
        raw_result.pred_track_id = std::make_shared<CudaWrapper<int32_t>>(track_ids);
    }
    
    // 设置元数据
    raw_result.num_objects = pred_instance_feature.size() / m_taskConfig.model_cfg_params().embedfeat_dims();
    raw_result.num_classes = m_taskConfig.model_cfg_params().num_classes();
    raw_result.is_first_frame = is_first_frame_;
    
    // 调试：打印pred_class_score得分最高的前10个object信息
    {
        const int num_objects = raw_result.num_objects;
        const int num_classes = raw_result.num_classes;
        
        // 计算每个object在类别维的最大分数
        std::vector<std::pair<float, size_t>> object_scores;
        object_scores.reserve(num_objects);
        
        for (int i = 0; i < num_objects; ++i) {
            const size_t base = i * num_classes;
            float max_score = -std::numeric_limits<float>::infinity();
            int best_class = 0;
            
            // 在类别维找最大分数
            for (int c = 0; c < num_classes; ++c) {
                float score = pred_class_score[base + c];
                if (score > max_score) {
                    max_score = score;
                    best_class = c;
                }
            }
            object_scores.emplace_back(max_score, i);
        }
        
        // 按分数排序，取前10
        std::sort(object_scores.begin(), object_scores.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        const int show_count = std::min(10, num_objects);
        LOG(INFO) << "[DEBUG] Top-" << show_count << " objects by class score:";
        
        for (int k = 0; k < show_count; ++k) {
            const size_t obj_idx = object_scores[k].second;
            const float score = object_scores[k].first;
            
            // 解码对应的anchor信息
            const size_t anchor_offset = obj_idx * 11; // anchor_dims = 11
            const float x = pred_anchor[anchor_offset + 0];
            const float y = pred_anchor[anchor_offset + 1];
            const float z = pred_anchor[anchor_offset + 2];
            const float w = std::exp(pred_anchor[anchor_offset + 3]);
            const float l = std::exp(pred_anchor[anchor_offset + 4]);
            const float h = std::exp(pred_anchor[anchor_offset + 5]);
            const float sin_yaw = pred_anchor[anchor_offset + 6];
            const float cos_yaw = pred_anchor[anchor_offset + 7];
            const float yaw = std::atan2(sin_yaw, cos_yaw);
            
            LOG(INFO) << "[DEBUG] Object " << k << ": idx=" << obj_idx 
                      << ", score=" << score 
                      << ", xyz=(" << x << "," << y << "," << z << ")"
                      << ", wlh=(" << w << "," << l << "," << h << ")"
                      << ", yaw=" << yaw
                      << ", track_id=" << track_ids[k];
        }
    }
    
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

// 内存池管理实现
std::shared_ptr<CudaWrapper<float>> SparseBEV::MemoryPool::getFloatBuffer(size_t size) {
    // 查找可用的缓冲区
    for (auto& buffer : float_buffers) {
        if (buffer && buffer->getSize() >= size) {
            return buffer;
        }
    }
    
    // 创建新的缓冲区
    auto new_buffer = std::make_shared<CudaWrapper<float>>();
    if (new_buffer->allocate(size)) {
        float_buffers.push_back(new_buffer);
        return new_buffer;
    }
    
    LOG(ERROR) << "[ERROR] Failed to allocate float buffer of size: " << size;
    return nullptr;
}



std::shared_ptr<CudaWrapper<int32_t>> SparseBEV::MemoryPool::getInt32Buffer(size_t size) {
    // 查找可用的缓冲区
    for (auto& buffer : int32_buffers) {
        if (buffer && buffer->getSize() >= size) {
            return buffer;
        }
    }
    
    // 创建新的缓冲区
    auto new_buffer = std::make_shared<CudaWrapper<int32_t>>();
    if (new_buffer->allocate(size)) {
        int32_buffers.push_back(new_buffer);
        return new_buffer;
    }
    
    LOG(ERROR) << "[ERROR] Failed to allocate int32 buffer of size: " << size;
    return nullptr;
}

void SparseBEV::MemoryPool::reset() {
    float_buffers.clear();
    int32_buffers.clear();
    LOG(INFO) << "[INFO] Memory pool reset completed";
}

/**
 * @brief 保存特征提取结果为bin文件
 * 当run_status为false时，将特征提取engine的输出结果保存到Output/val_bin/目录
 */
void SparseBEV::saveFeatureExtractionResults() {
    try {
        // 创建保存目录
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        // 强制使用float精度获取特征提取结果
        std::vector<float> feature_data = m_float_features_wrapper.cudaMemcpyD2HResWrap();
        
        // 生成文件名 - 使用当前样本索引，与Python脚本格式保持一致
        // 特征数据形状: [1, 89760, 256] - [batch_size, total_spatial_points, feature_dim]
        std::string shape_str = "1*89760*256";
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_feature_" + 
                              shape_str + "_float32.bin";
        
        // 保存数据到bin文件
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(feature_data.data()), 
                         feature_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 特征提取结果已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存特征提取数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存预处理数据为bin文件
 * 当run_status为false时，将输入图像数据保存到Output/val_bin/目录
 */
void SparseBEV::savePreprocessedData() {
    try {
        // 创建保存目录
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        // 获取当前时间戳作为文件名的一部分
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        
        // 强制使用float精度获取输入图像数据
        std::vector<float> image_data = m_float_input_wrapper.cudaMemcpyD2HResWrap();
        
        // 从TaskConfig获取图像形状
        int batch_size = 1;
        int num_cams = m_taskConfig.preprocessor_params().num_cams();
        int channels = m_taskConfig.preprocessor_params().model_input_img_c();
        int height = m_taskConfig.preprocessor_params().model_input_img_h();
        int width = m_taskConfig.preprocessor_params().model_input_img_w();
        
        // 生成文件名 - 使用当前样本索引，与Python脚本格式保持一致
        std::string shape_str = std::to_string(batch_size) + "*" + 
                               std::to_string(num_cams) + "*" + 
                               std::to_string(channels) + "*" + 
                               std::to_string(height) + "*" + 
                               std::to_string(width);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_imgs_" + 
                              shape_str + "_float32.bin";
        
        // 保存数据到bin文件
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(image_data.data()), 
                         image_data.size() * sizeof(float));
            outfile.close();
            
            LOG(INFO) << "[INFO] Image data saved to: " << filename;
            LOG(INFO) << "[INFO] Image shape: " << shape_str;
            LOG(INFO) << "[INFO] Total elements: " << image_data.size();
            LOG(INFO) << "[INFO] File size: " << (image_data.size() * sizeof(float)) << " bytes";
        } else {
            LOG(ERROR) << "[ERROR] Failed to open file for writing: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] Exception while saving image data: " << e.what();
    }
}

void SparseBEV::setCurrentSampleIndex(int sample_index) {
    current_sample_index_ = sample_index;
    LOG(INFO) << "setCurrentSampleIndex() current_sample_index_ : " << current_sample_index_;
}

/**
 * @brief 保存时间间隔数据
 */
void SparseBEV::saveTimeIntervalData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_time_interval_1_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(m_time_interval.data()), 
                         m_time_interval.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 时间间隔数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存时间间隔数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存图像宽高数据
 */
void SparseBEV::saveImageWhData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_image_wh_1*6*2_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(m_image_wh.data()), 
                         m_image_wh.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 图像宽高数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存图像宽高数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存Lidar2img变换矩阵数据
 */
void SparseBEV::saveLidar2imgData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_lidar2img_1*6*4*4_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(m_lidar2img.data()), 
                         m_lidar2img.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] Lidar2img变换矩阵数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存Lidar2img变换矩阵数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存锚点数据
 */
void SparseBEV::saveAnchorData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_anchor_1*900*11_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(m_anchor.data()), 
                         m_anchor.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 锚点数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存锚点数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存空间形状数据
 */
void SparseBEV::saveSpatialShapesData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_spatial_shapes_6*4*2_int32.bin";
        
        // 重塑数据为正确的形状 [6, 4, 2]
        std::vector<int32_t> reshaped_data;
        if (m_spatial_shapes.size() == 8) {
            // 原始数据是 [8]，需要重塑为 [6, 4, 2]
            reshaped_data.resize(48); // 6*4*2 = 48
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 4; ++j) {
                    reshaped_data[i * 8 + j * 2] = m_spatial_shapes[j * 2];
                    reshaped_data[i * 8 + j * 2 + 1] = m_spatial_shapes[j * 2 + 1];
                }
            }
        } else {
            reshaped_data = m_spatial_shapes;
        }
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(reshaped_data.data()), 
                         reshaped_data.size() * sizeof(int32_t));
            outfile.close();
            LOG(INFO) << "[INFO] 空间形状数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存空间形状数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存层级起始索引数据
 */
void SparseBEV::saveLevelStartIndexData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_level_start_index_6*4_int32.bin";
        
        // 重塑数据为正确的形状 [6, 4]
        std::vector<int32_t> reshaped_data;
        if (m_level_start_index.size() == 8) {
            // 原始数据是 [8]，需要重塑为 [6, 4]
            reshaped_data.resize(24); // 6*4 = 24
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 4; ++j) {
                    reshaped_data[i * 4 + j] = m_level_start_index[j];
                }
            }
        } else {
            reshaped_data = m_level_start_index;
        }
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(reshaped_data.data()), 
                         reshaped_data.size() * sizeof(int32_t));
            outfile.close();
            LOG(INFO) << "[INFO] 层级起始索引数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存层级起始索引数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存实例特征数据
 */
void SparseBEV::saveInstanceFeatureData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_instance_feature_1*900*256_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(m_instance_feature.data()), 
                         m_instance_feature.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 实例特征数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存实例特征数据时发生异常: " << e.what();
    }
}



/**
 * @brief 保存预测实例特征数据
 */
void SparseBEV::savePredInstanceFeatureData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_pred_instance_feature_1*900*256_float32.bin";
        
        // 强制使用float精度获取预测实例特征数据
        std::vector<float> pred_data = m_float_pred_instance_feature_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(pred_data.data()), 
                         pred_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 预测实例特征数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存预测实例特征数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存预测锚点数据
 */
void SparseBEV::savePredAnchorData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_pred_anchor_1*900*11_float32.bin";
        
        // 强制使用float精度获取预测锚点数据
        std::vector<float> pred_data = m_float_pred_anchor_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(pred_data.data()), 
                         pred_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 预测锚点数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存预测锚点数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存预测分类得分数据
 */
void SparseBEV::savePredClassScoreData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_pred_class_score_1*900*10_float32.bin";
        
        // 强制使用float精度获取预测分类得分数据
        std::vector<float> pred_data = m_float_pred_class_score_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(pred_data.data()), 
                         pred_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 预测分类得分数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存预测分类得分数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存预测质量得分数据
 */
void SparseBEV::savePredQualityScoreData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_pred_quality_score_1*900*2_float32.bin";
        
        // 强制使用float精度获取预测质量得分数据
        std::vector<float> pred_data = m_float_pred_quality_score_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(pred_data.data()), 
                         pred_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 预测质量得分数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存预测质量得分数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存预测跟踪ID数据
 */
void SparseBEV::savePredTrackIdData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_pred_track_id_1*900_int32.bin";
        
        // 直接获取int32_t类型数据，无需类型转换
        std::vector<int32_t> pred_data = m_int32_pred_track_id_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(pred_data.data()), 
                         pred_data.size() * sizeof(int32_t));
            outfile.close();
            LOG(INFO) << "[INFO] 预测跟踪ID数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存预测跟踪ID数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存临时实例特征数据
 */
void SparseBEV::saveTempInstanceFeatureData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_temp_instance_feature_1*600*256_float32.bin";
        
        // 强制使用float精度获取临时实例特征数据
        std::vector<float> temp_data = m_float_temp_instance_feature_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(temp_data.data()), 
                         temp_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 临时实例特征数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存临时实例特征数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存临时锚点数据
 */
void SparseBEV::saveTempAnchorData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_temp_anchor_1*600*11_float32.bin";
        
        // 强制使用float精度获取临时锚点数据
        std::vector<float> temp_data = m_float_temp_anchor_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(temp_data.data()), 
                         temp_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 临时锚点数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存临时锚点数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存掩码数据
 */
void SparseBEV::saveMaskData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_mask_1_int32.bin";
        
        // 强制使用float精度获取掩码数据
        std::vector<int32_t> mask_data = m_float_mask_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(mask_data.data()), 
                         mask_data.size() * sizeof(int32_t));
            outfile.close();
            LOG(INFO) << "[INFO] 掩码数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存掩码数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存跟踪ID数据
 */
void SparseBEV::saveTrackIdData() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_track_id_1*900_int32.bin";
        
        // 使用int32_t类型的临时跟踪ID包装器
        std::vector<int32_t> track_id_data = m_int32_temp_track_id_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(track_id_data.data()), 
                         track_id_data.size() * sizeof(int32_t));
            outfile.close();
            LOG(INFO) << "[INFO] 跟踪ID数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存跟踪ID数据时发生异常: " << e.what();
    }
}

void SparseBEV::setInputData(void* input) 
{
    if (!input) {
        LOG(ERROR) << "[ERROR] Input data is null!";
        return;
    }
    
    // 尝试转换为SparseBEVInputWrapper
    auto* input_wrapper = static_cast<sparsebev::SparseBEVInputWrapper*>(input);
    if (input_wrapper != nullptr) {
        // 直接使用float精度数据
        auto input_data = input_wrapper->getInputData();
        if (input_data && input_data->image_feature.data_valid && input_data->image_feature.feature) {
            // 重新分配内存并复制数据
            auto feature_data = input_data->image_feature.feature->cudaMemcpyD2HResWrap();
            m_float_input_wrapper = CudaWrapper<float>(feature_data);
            LOG(INFO) << "[INFO] Set input data using SparseBEVInputWrapper format";
            return;
        }
    }
    
    // 尝试转换为CudaWrapper<float>
    auto* float_wrapper = static_cast<CudaWrapper<float>*>(input);
    if (float_wrapper != nullptr) {
        // 直接使用float精度数据
        auto float_data = float_wrapper->cudaMemcpyD2HResWrap();
        m_float_input_wrapper = CudaWrapper<float>(float_data);
        LOG(INFO) << "[INFO] Set input data using CudaWrapper<float> format";
        return;
    }
    
    // 尝试解析为CudaWrapper<half>（旧格式，向后兼容）
    auto* half_wrapper = static_cast<CudaWrapper<half>*>(input);
    if (half_wrapper != nullptr) {
        // 将half转换为float精度
        auto half_data = half_wrapper->cudaMemcpyD2HResWrap();
        std::vector<float> float_data(half_data.begin(), half_data.end());
        m_float_input_wrapper = CudaWrapper<float>(float_data);
        LOG(INFO) << "[INFO] Set input data using legacy CudaWrapper<half> format (converted to float)";
        return;
    }
    
    LOG(ERROR) << "[ERROR] Unsupported input data format!";
    LOG(ERROR) << "[ERROR] Expected SparseBEVInputWrapper, CudaWrapper<float>, or CudaWrapper<half>";
}

/**
 * @brief 保存tmp_outs0数据
 */
void SparseBEV::saveTmpOuts0Data() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_tmp_outs0_1*900*512_float32.bin";
        
        // 获取tmp_outs0数据
        std::vector<float> tmp_data = m_float_tmp_outs0_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(tmp_data.data()), 
                         tmp_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] tmp_outs0数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存tmp_outs0数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存tmp_outs1数据
 */
void SparseBEV::saveTmpOuts1Data() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_tmp_outs1_1*900*512_float32.bin";
        
        // 获取tmp_outs1数据
        std::vector<float> tmp_data = m_float_tmp_outs1_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(tmp_data.data()), 
                         tmp_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] tmp_outs1数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存tmp_outs1数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存tmp_outs2数据
 */
void SparseBEV::saveTmpOuts2Data() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_tmp_outs2_1*900*512_float32.bin";
        
        // 获取tmp_outs2数据
        std::vector<float> tmp_data = m_float_tmp_outs2_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(tmp_data.data()), 
                         tmp_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] tmp_outs2数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存tmp_outs2数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存tmp_outs3数据
 */
void SparseBEV::saveTmpOuts3Data() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_tmp_outs3_1*900*512_float32.bin";
        
        // 获取tmp_outs3数据
        std::vector<float> tmp_data = m_float_tmp_outs3_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(tmp_data.data()), 
                         tmp_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] tmp_outs3数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存tmp_outs3数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存tmp_outs4数据
 */
void SparseBEV::saveTmpOuts4Data() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_tmp_outs4_1*900*512_float32.bin";
        
        // 获取tmp_outs4数据
        std::vector<float> tmp_data = m_float_tmp_outs4_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(tmp_data.data()), 
                         tmp_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] tmp_outs4数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存tmp_outs4数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存tmp_outs5数据
 */
void SparseBEV::saveTmpOuts5Data() {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_tmp_outs5_1*900*512_float32.bin";
        
        // 获取tmp_outs5数据
        std::vector<float> tmp_data = m_float_tmp_outs5_wrapper.cudaMemcpyD2HResWrap();
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(tmp_data.data()), 
                         tmp_data.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] tmp_outs5数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存tmp_outs5数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存输入特征数据
 */
void SparseBEV::saveInputFeaturesData(const std::vector<float>& input_features) {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_input_features_1*89760*256_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(input_features.data()), 
                         input_features.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 输入特征数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存输入特征数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存输入空间形状数据
 */
void SparseBEV::saveInputSpatialShapesData(const std::vector<int32_t>& spatial_shapes) {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_input_spatial_shapes_6*4*2_int32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(spatial_shapes.data()), 
                         spatial_shapes.size() * sizeof(int32_t));
            outfile.close();
            LOG(INFO) << "[INFO] 输入空间形状数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存输入空间形状数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存输入层级起始索引数据
 */
void SparseBEV::saveInputLevelStartIndexData(const std::vector<int32_t>& level_start_index) {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_input_level_start_index_6*4_int32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(level_start_index.data()), 
                         level_start_index.size() * sizeof(int32_t));
            outfile.close();
            LOG(INFO) << "[INFO] 输入层级起始索引数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存输入层级起始索引数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存输入实例特征数据
 */
void SparseBEV::saveInputInstanceFeatureData(const std::vector<float>& instance_feature) {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_input_instance_feature_1*900*256_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(instance_feature.data()), 
                         instance_feature.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 输入实例特征数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存输入实例特征数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存输入锚点数据
 */
void SparseBEV::saveInputAnchorData(const std::vector<float>& anchor) {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_input_anchor_1*900*11_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(anchor.data()), 
                         anchor.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 输入锚点数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存输入锚点数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存输入时间间隔数据
 */
void SparseBEV::saveInputTimeIntervalData(const std::vector<float>& time_interval) {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_input_time_interval_1_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(time_interval.data()), 
                         time_interval.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 输入时间间隔数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存输入时间间隔数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存输入图像宽高数据
 */
void SparseBEV::saveInputImageWhData(const std::vector<float>& image_wh) {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_input_image_wh_1*6*2_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(image_wh.data()), 
                         image_wh.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 输入图像宽高数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存输入图像宽高数据时发生异常: " << e.what();
    }
}

/**
 * @brief 保存输入Lidar2img变换矩阵数据
 */
void SparseBEV::saveInputLidar2imgData(const std::vector<float>& lidar2img) {
    try {
        std::string save_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin/";
        std::filesystem::create_directories(save_dir);
        
        std::string filename = save_dir + "sample_" + std::to_string(current_sample_index_) + "_input_lidar2img_1*6*4*4_float32.bin";
        
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(lidar2img.data()), 
                         lidar2img.size() * sizeof(float));
            outfile.close();
            LOG(INFO) << "[INFO] 输入Lidar2img变换矩阵数据已保存到: " << filename;
        } else {
            LOG(ERROR) << "[ERROR] 无法打开文件进行写入: " << filename;
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] 保存输入Lidar2img变换矩阵数据时发生异常: " << e.what();
    }
}