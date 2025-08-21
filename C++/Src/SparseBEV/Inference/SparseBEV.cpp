#include "SparseBEV.h"
#include <fstream>
#include <iostream>

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
    // 解析任务配置参数
    if (p_pAlgParam != nullptr) {
        m_taskConfig = *static_cast<sparsebev::TaskConfig*>(p_pAlgParam);
        use_half_precision_ = m_taskConfig.use_half_precision();
        
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

        // 第一帧头部引擎
        m_head1st_engine = std::make_shared<TensorRT>(
            m_taskConfig.head1st_engine().engine_path(),
            m_taskConfig.head1st_engine().plugin_path(),
            std::vector<std::string>(m_taskConfig.head1st_engine().input_names().begin(), 
                                    m_taskConfig.head1st_engine().input_names().end()),
            std::vector<std::string>(m_taskConfig.head1st_engine().output_names().begin(), 
                                    m_taskConfig.head1st_engine().output_names().end()));

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

        // 根据精度分配GPU内存
        if (use_half_precision_) {
            if (!m_half_input_wrapper.allocate(input_size) ||
                !m_half_features_wrapper.allocate(feature_size) ||
                !m_half_pred_instance_feature_wrapper.allocate(pred_size) ||
                !m_half_pred_anchor_wrapper.allocate(anchor_size) ||
                !m_half_pred_class_score_wrapper.allocate(class_score_size) ||
                !m_half_pred_quality_score_wrapper.allocate(quality_score_size) ||
                !m_half_pred_track_id_wrapper.allocate(track_id_size)) {
                LOG(ERROR) << "[ERROR] Failed to allocate half precision GPU memory!";
                status_ = false;
                return status_;
            }
            LOG(INFO) << "[INFO] Pre-allocated half precision GPU memory";
        } else {
            if (!m_float_input_wrapper.allocate(input_size) ||
                !m_float_features_wrapper.allocate(feature_size) ||
                !m_float_pred_instance_feature_wrapper.allocate(pred_size) ||
                !m_float_pred_anchor_wrapper.allocate(anchor_size) ||
                !m_float_pred_class_score_wrapper.allocate(class_score_size) ||
                !m_float_pred_quality_score_wrapper.allocate(quality_score_size) ||
                !m_float_pred_track_id_wrapper.allocate(track_id_size)) {
                LOG(ERROR) << "[ERROR] Failed to allocate float precision GPU memory!";
                status_ = false;
                return status_;
            }
            LOG(INFO) << "[INFO] Pre-allocated float precision GPU memory";
        }

        // 加载辅助数据
        loadAuxiliaryData();
        
        // 初始化InstanceBank
        instance_bank_ = std::make_unique<InstanceBank>(m_taskConfig);
        has_previous_frame_ = false;
        current_timestamp_ = 0.0;
        previous_timestamp_ = 0.0;
        
        // 初始化坐标变换矩阵为单位矩阵
        current_global_to_lidar_mat_ = Eigen::Matrix<double, 4, 4>::Identity();
        previous_global_to_lidar_mat_ = Eigen::Matrix<double, 4, 4>::Identity();
        
        // 初始化缓存数据
        size_t instance_feature_size = m_taskConfig.instance_bank_params().num_querys() * 
                                      m_taskConfig.model_cfg_params().embedfeat_dims();
        size_t anchor_size = m_taskConfig.instance_bank_params().num_querys() * 
                            m_taskConfig.instance_bank_params().query_dims();
        size_t confidence_size = m_taskConfig.instance_bank_params().num_querys() * 
                                m_taskConfig.model_cfg_params().num_classes();
        size_t track_id_size = m_taskConfig.instance_bank_params().num_querys();
        
        cached_instance_feature_.resize(instance_feature_size, 0.0f);
        cached_anchor_.resize(anchor_size, 0.0f);
        cached_confidence_.resize(confidence_size, 0.0f);
        cached_track_ids_.resize(track_id_size, -1);
        
        status_ = true;
        LOG(INFO) << "[INFO] SparseBEV module initialized successfully";
    } else {
        LOG(ERROR) << "[ERROR] Failed to get algorithm parameters!";
        status_ = false;
    }
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
    
    // 从TaskConfig获取图像宽高
    m_image_wh = {static_cast<float>(m_taskConfig.preprocessor_params().model_input_img_w()),
                  static_cast<float>(m_taskConfig.preprocessor_params().model_input_img_h())};
    
    // 激光雷达到图像变换矩阵大小（需要根据实际数据加载）每个相机4x4变换矩阵
    size_t lidar2img_size = m_taskConfig.preprocessor_params().num_cams() * 4 * 4;
    m_lidar2img.resize(lidar2img_size, 0.0f);
    
    // 加载标定参数并计算变换后的lidar2img矩阵
    if (!m_taskConfig.homography_path().empty()) {
        if (!loadHomographyCalibration(m_taskConfig.homography_path())) {
            LOG(WARNING) << "[WARNING] Failed to load homography calibration, using default values";
            // 使用默认的单位矩阵作为标定参数
            for (int i = 0; i < m_taskConfig.preprocessor_params().num_cams(); ++i) {
                m_lidar2img[i * 16 + 0] = 1.0f;  // [0,0]
                m_lidar2img[i * 16 + 5] = 1.0f;  // [1,1]
                m_lidar2img[i * 16 + 10] = 1.0f; // [2,2]
                m_lidar2img[i * 16 + 15] = 1.0f; // [3,3]
            }
        } else {
            // 成功加载标定参数后，计算图像变换并更新lidar2img矩阵
            computeTransformedLidar2img();
        }
    } else {
        LOG(WARNING) << "[WARNING] No homography_path specified, using default values";
        // 使用默认的单位矩阵作为标定参数
        for (int i = 0; i < m_taskConfig.preprocessor_params().num_cams(); ++i) {
            m_lidar2img[i * 16 + 0] = 1.0f;  // [0,0]
            m_lidar2img[i * 16 + 5] = 1.0f;  // [1,1]
            m_lidar2img[i * 16 + 10] = 1.0f; // [2,2]
            m_lidar2img[i * 16 + 15] = 1.0f; // [3,3]
        }
    }
    
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
        if (use_half_precision_) {
            // 将float转换为half精度
            auto float_data = float_wrapper->cudaMemcpyD2HResWrap();
            std::vector<half> half_data(float_data.begin(), float_data.end());
            m_half_input_wrapper = CudaWrapper<half>(half_data);
        } else {
            m_float_input_wrapper = *float_wrapper;
        }
        LOG(INFO) << "[INFO] Set input data using legacy CudaWrapper format";
        return;
    }
    
    // 尝试解析为CudaWrapper<half>（旧格式，向后兼容）
    auto* half_wrapper = static_cast<CudaWrapper<half>*>(input);
    if (half_wrapper != nullptr) {
        if (use_half_precision_) {
            m_half_input_wrapper = *half_wrapper;
        } else {
            // 将half转换为float精度
            auto half_data = half_wrapper->cudaMemcpyD2HResWrap();
            std::vector<float> float_data(half_data.begin(), half_data.end());
            m_float_input_wrapper = CudaWrapper<float>(float_data);
        }
        LOG(INFO) << "[INFO] Set input data using legacy CudaWrapper<half> format";
        return;
    }
    
    LOG(ERROR) << "[ERROR] Unsupported input data format!";
    LOG(ERROR) << "[ERROR] Expected SparseBEVInputWrapper, CudaWrapper<float>, or CudaWrapper<half>";
}

/**
 * @brief 加载标定参数
 * @param calib_path 标定文件路径
 * @return 加载是否成功
 */
bool SparseBEV::loadHomographyCalibration(const std::string& calib_path) 
{
    std::ifstream calib_file(calib_path, std::ios::binary);
    if (!calib_file.is_open()) {
        LOG(ERROR) << "[ERROR] Failed to open homography calibration file: " << calib_path;
        return false;
    }
    
    // 获取文件大小
    calib_file.seekg(0, std::ios::end);
    size_t file_size = calib_file.tellg();
    calib_file.seekg(0, std::ios::beg);
    
    // 计算float数量
    size_t num_floats = file_size / sizeof(float);
    
    // 检查数据大小是否匹配预期
    size_t expected_size = m_taskConfig.preprocessor_params().num_cams() * 4 * 4;
    if (num_floats != expected_size) {
        LOG(ERROR) << "[ERROR] Homography calibration file size mismatch. Expected: " 
                   << expected_size << " floats, got: " << num_floats << " floats";
        return false;
    }
    
    // 读取标定参数
    m_lidar2img.resize(num_floats);
    calib_file.read(reinterpret_cast<char*>(m_lidar2img.data()), file_size);
    
    if (calib_file.fail()) {
        LOG(ERROR) << "[ERROR] Failed to read homography calibration data from file: " << calib_path;
        return false;
    }
    
    LOG(INFO) << "[INFO] Successfully loaded homography calibration from: " << calib_path;
    LOG(INFO) << "[INFO] Calibration data size: " << num_floats << " floats";
    
    // 打印标定参数信息（调试用）
    for (int i = 0; i < m_taskConfig.preprocessor_params().num_cams(); ++i) {
        VLOG(1) << "[DEBUG] Camera " << i << " homography matrix:";
        for (int row = 0; row < 4; ++row) {
            std::stringstream ss;
            ss << "  Row " << row << ": ";
            for (int col = 0; col < 4; ++col) {
                ss << std::fixed << std::setprecision(6) 
                   << m_lidar2img[i * 16 + row * 4 + col] << " ";
            }
            VLOG(1) << ss.str();
        }
    }
    
    return true;
}

/**
 * @brief 计算变换后的lidar2img矩阵
 */
void SparseBEV::computeTransformedLidar2img() {
    // 获取预处理参数
    float resize = m_taskConfig.preprocessor_params().resize_ratio();
    uint32_t crop_height = m_taskConfig.preprocessor_params().crop_height();
    uint32_t crop_width = m_taskConfig.preprocessor_params().crop_width();
    
    LOG(INFO) << "[INFO] Computing transformed lidar2img matrices with parameters:";
    LOG(INFO) << "[INFO]   resize_ratio: " << resize;
    LOG(INFO) << "[INFO]   crop_height: " << crop_height;
    LOG(INFO) << "[INFO]   crop_width: " << crop_width;
    
    // 计算图像变换矩阵（与Python中的_img_transform方法对应）
    // 变换矩阵 = 缩放矩阵 @ 裁剪矩阵
    // 注意：这里使用简化的变换，如果需要翻转和旋转，可以进一步扩展
    
    // 创建变换矩阵（4x4）
    std::vector<float> transform_matrix(16, 0.0f);
    
    // 设置单位矩阵
    transform_matrix[0] = 1.0f;   // [0,0]
    transform_matrix[5] = 1.0f;   // [1,1]
    transform_matrix[10] = 1.0f;  // [2,2]
    transform_matrix[15] = 1.0f;  // [3,3]
    
    // 1. 缩放变换
    transform_matrix[0] = resize;  // [0,0] = resize_ratio
    transform_matrix[5] = resize;  // [1,1] = resize_ratio
    
    // 2. 裁剪变换（平移）
    transform_matrix[3] = -static_cast<float>(crop_width);   // [0,3] = -crop_width
    transform_matrix[7] = -static_cast<float>(crop_height);  // [1,3] = -crop_height
    
    LOG(INFO) << "[INFO] Image transform matrix:";
    LOG(INFO) << "[INFO]   [" << transform_matrix[0] << ", " << transform_matrix[1] << ", " << transform_matrix[2] << ", " << transform_matrix[3] << "]";
    LOG(INFO) << "[INFO]   [" << transform_matrix[4] << ", " << transform_matrix[5] << ", " << transform_matrix[6] << ", " << transform_matrix[7] << "]";
    LOG(INFO) << "[INFO]   [" << transform_matrix[8] << ", " << transform_matrix[9] << ", " << transform_matrix[10] << ", " << transform_matrix[11] << "]";
    LOG(INFO) << "[INFO]   [" << transform_matrix[12] << ", " << transform_matrix[13] << ", " << transform_matrix[14] << ", " << transform_matrix[15] << "]";
    
    // 保存原始lidar2img矩阵（用于调试）
    std::vector<float> original_lidar2img = m_lidar2img;
    
    // 应用变换到所有相机的lidar2img矩阵
    int num_cams = m_taskConfig.preprocessor_params().num_cams();
    for (int cam_idx = 0; cam_idx < num_cams; ++cam_idx) {
        // 提取当前相机的原始lidar2img矩阵
        std::vector<float> lidar2img_matrix(16);
        for (int i = 0; i < 16; ++i) {
            lidar2img_matrix[i] = original_lidar2img[cam_idx * 16 + i];
        }
        
        // 应用变换：new_lidar2img = transform_matrix @ original_lidar2img
        std::vector<float> transformed_matrix(16, 0.0f);
        
        // 矩阵乘法：4x4 @ 4x4
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 4; ++k) {
                    transformed_matrix[i * 4 + j] += transform_matrix[i * 4 + k] * lidar2img_matrix[k * 4 + j];
                }
            }
        }
        
        // 将变换后的矩阵存储回数组
        for (int i = 0; i < 16; ++i) {
            m_lidar2img[cam_idx * 16 + i] = transformed_matrix[i];
        }
        
        LOG(DEBUG) << "[DEBUG] Camera " << cam_idx << " lidar2img matrix updated";
        
        // 打印变换前后的矩阵（调试用）
        if (cam_idx == 0) {  // 只打印第一个相机的矩阵
            LOG(INFO) << "[INFO] Camera 0 - Original lidar2img matrix:";
            for (int row = 0; row < 4; ++row) {
                std::stringstream ss;
                ss << "  Row " << row << ": ";
                for (int col = 0; col < 4; ++col) {
                    ss << std::fixed << std::setprecision(6) 
                       << original_lidar2img[cam_idx * 16 + row * 4 + col] << " ";
                }
                LOG(INFO) << ss.str();
            }
            
            LOG(INFO) << "[INFO] Camera 0 - Transformed lidar2img matrix:";
            for (int row = 0; row < 4; ++row) {
                std::stringstream ss;
                ss << "  Row " << row << ": ";
                for (int col = 0; col < 4; ++col) {
                    ss << std::fixed << std::setprecision(6) 
                       << m_lidar2img[cam_idx * 16 + row * 4 + col] << " ";
                }
                LOG(INFO) << ss.str();
            }
        }
    }
    
    LOG(INFO) << "[INFO] Successfully computed transformed lidar2img matrices for " << num_cams << " cameras";
}