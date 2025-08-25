#include "PostProcess.h"

// 注册模块
REGISTER_MODULE("SparseBEV", PostProcessor, PostProcessor)

/**
 * @brief 析构函数
 */
PostProcessor::~PostProcessor() {
    // 清理资源
    // m_gpuNMS.reset(); // 已删除
}

/**
 * @brief 初始化模块
 * @param p_pAlgParam 算法参数指针
 * @return 初始化是否成功
 */
bool PostProcessor::init(void* p_pAlgParam) {
    if (!p_pAlgParam) {
        LOG(ERROR) << "[ERROR] Invalid config pointer";
        return false;
    }
    
    m_taskConfig = *static_cast<const sparsebev::TaskConfig*>(p_pAlgParam);
    
    // 从配置中读取后处理参数
    m_useGPU = m_taskConfig.postprocessor_params().use_gpu_nms();
    m_gpuNMSThreshold = m_taskConfig.postprocessor_params().gpu_nms_threshold();
    m_maxOutputBoxes = m_taskConfig.postprocessor_params().max_output_boxes();
    m_confidenceThreshold = m_taskConfig.postprocessor_params().post_process_threshold();
    
    // 初始化GPU NMS（如果启用）
    if (m_useGPU) {
        if (!m_gpuNMS.initialize()) {
            LOG(WARNING) << "[WARNING] Failed to initialize GPU NMS, falling back to CPU NMS";
            m_useGPU = false;
        } else {
            LOG(INFO) << "[INFO] GPU NMS initialized successfully";
        }
    }
    
    status_ = true;
    LOG(INFO) << "[INFO] PostProcessor initialized with " 
              << (m_useGPU ? "GPU" : "CPU") << " NMS, threshold=" << m_gpuNMSThreshold
              << ", max_boxes=" << m_maxOutputBoxes
              << ", confidence_threshold=" << m_confidenceThreshold;
    
    return status_;
}

/**
 * @brief 设置GPU推理结果输入（主要接口）
 * @param raw_result GPU推理结果
 */
void PostProcessor::setInput(void* input) {
    if (!input) {
        LOG(ERROR) << "[ERROR] Invalid input pointer";
        return;
    }
    
    // 将输入转换为RawInferenceResult
    m_raw_input_result = *static_cast<const RawInferenceResult*>(input);
    
    // 记录输入信息
    LOG(INFO) << "[INFO] Set GPU inference result input with " << m_raw_input_result.getInstanceCount() << " objects";
    
    // 验证输入数据
    if (!m_raw_input_result.hasValidResults()) {
        LOG(WARNING) << "[WARNING] Raw inference result has invalid data";
    }
}

/**
 * @brief 获取输出结果
 * @return 输出结果指针
 */
void* PostProcessor::getOutput() {
    return static_cast<void*>(&m_output_result);
}

/**
 * @brief 执行后处理
 */
void PostProcessor::execute() {
    if (!status_) {
        LOG(ERROR) << "[ERROR] Module not initialized!";
        return;
    }
    
    if (!m_raw_input_result.hasValidResults()) {
        LOG(ERROR) << "[ERROR] No valid raw input result available";
        return;
    }
    
    LOG(INFO) << "[INFO] Starting post-processing with " << m_raw_input_result.getInstanceCount() << " objects";
    
    try {
        // 清空输出结果
        m_output_result.vecFrameResult().clear();
        
        // 1. 将GPU推理结果转换为3D边界框格式
        std::vector<BoundingBox3D> input_boxes = m_raw_input_result.toBoundingBox3D();
        
        if (input_boxes.empty()) {
            LOG(WARNING) << "[WARNING] No valid bounding boxes found in raw result";
            return;
        }
        
        LOG(INFO) << "[INFO] Converted " << input_boxes.size() << " bounding boxes for NMS";
        
        // 1.5. 根据置信度阈值过滤目标
        int count = 0;
        std::vector<BoundingBox3D> filtered_boxes;
        filtered_boxes.reserve(input_boxes.size());
        for (const auto& box : input_boxes) {
            if (box.confidence >= m_confidenceThreshold) {
                filtered_boxes.push_back(box);
                // std::cout << "box.confidence: " << box.confidence
                //           << " , box.label: " << box.label
                //           << " , box.track_id: " << box.track_id
                //           << " , xyz=(" << box.x << "," << box.y << "," << box.z << ")"
                //           << " , wlh=(" << box.w << "," << box.l << "," << box.h << ")"
                //           << " , yaw=" << box.yaw
                //           << ", m_confidenceThreshold: " << m_confidenceThreshold << std::endl;
            }
            else{
                count++;
            }
        }
        
        // std::cout << "count: " << count << std::endl;
        
        LOG(INFO) << "[INFO] Confidence filtering: " << input_boxes.size() 
                  << " -> " << filtered_boxes.size() << " boxes (threshold=" << m_confidenceThreshold << ")";
        
        if (filtered_boxes.empty()) {
            LOG(WARNING) << "[WARNING] No boxes passed confidence threshold";
            return;
        }
        
        // 2. 执行NMS去重
        std::vector<BoundingBox3D> nms_boxes;
        if (m_useGPU) {
            // LOG(INFO) << "[INFO] Executing GPU NMS with threshold=" << m_gpuNMSThreshold;
            int num_output = m_gpuNMS.processBatch(filtered_boxes, nms_boxes, m_gpuNMSThreshold, m_maxOutputBoxes);
            LOG(INFO) << "[INFO] GPU NMS completed, output " << num_output << " boxes"  << ", threshold=" << m_gpuNMSThreshold;
        } else {
            LOG(INFO) << "[INFO] Executing CPU NMS with threshold=" << m_gpuNMSThreshold;
            nms_boxes = cpuNMS(filtered_boxes, m_gpuNMSThreshold, m_maxOutputBoxes);
            LOG(INFO) << "[INFO] CPU NMS completed, output " << nms_boxes.size() << " boxes";
        }
        
        // 3. 转换为最终输出格式
        std::vector<CObjectResult> objects = convertRawToObjectResult(nms_boxes);
        
        // 4. 构建输出结果
        CFrameResult output_frame;
        for (const auto& obj : objects) {
            output_frame.vecObjectResult().push_back(obj);
        }
        
        // 将输出帧添加到结果中
        m_output_result.vecFrameResult().push_back(output_frame);
        
        LOG(INFO) << "[INFO] Post-processing completed, final result has " << objects.size() << " objects";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] Exception during post-processing: " << e.what();
    }
}

/**
 * @brief 从GPU推理结果转换为CObjectResult
 * @param raw_result GPU推理结果
 * @return CObjectResult列表
 */
std::vector<CObjectResult> PostProcessor::convertRawToObjectResult(const RawInferenceResult& raw_result) {
    std::vector<CObjectResult> objects;
    
    if (!raw_result.isValid()) {
        return objects;
    }
    
    // 转换为3D边界框格式
    std::vector<BoundingBox3D> boxes = raw_result.toBoundingBox3D();
    
    // 转换为CObjectResult格式
    return convertRawToObjectResult(boxes);
}

std::vector<CObjectResult> PostProcessor::convertRawToObjectResult(const std::vector<BoundingBox3D>& boxes) {
    std::vector<CObjectResult> objects;
    objects.reserve(boxes.size());
    for (const auto& box : boxes) {
        CObjectResult obj;
        obj.x() = box.x;
        obj.y() = box.y;
        obj.z() = box.z;
        obj.l() = box.l;
        obj.w() = box.w;
        obj.h() = box.h;
        obj.yaw() = box.yaw;
        obj.label() = box.label;
        obj.confidence() = box.confidence;
        obj.trackid() = box.track_id;
        objects.push_back(obj);
    }
    return objects;
}

void PostProcessor::setCurrentSampleIndex(int sample_index) {
    current_sample_index_ = sample_index;
}

/**
 * @brief 移除低置信度目标
 * @param objects 目标列表
 * @param confidence_threshold 置信度阈值
 */
void PostProcessor::removeLowConfidenceObjects(std::vector<CObjectResult>& objects, 
                                              float confidence_threshold) 
{
    objects.erase(
        std::remove_if(objects.begin(), objects.end(),
                      [confidence_threshold](const CObjectResult& obj) {
                          return obj.confidence() < confidence_threshold;
                      }),
        objects.end()
    );
}

/**
 * @brief 按置信度排序
 * @param objects 目标列表
 */
void PostProcessor::sortByConfidence(std::vector<CObjectResult>& objects) 
{
    std::sort(objects.begin(), objects.end(),
              [](const CObjectResult& a, const CObjectResult& b) {
                  return a.confidence() > b.confidence();
              });
}

/**
 * @brief 结果优化
 * @param objects 目标列表
 */
void PostProcessor::optimizeResults(std::vector<CObjectResult>& objects) 
{
    // 1. 按置信度排序
    sortByConfidence(objects);
    
    // 2. 可以添加其他优化逻辑，比如：
    // - 边界框平滑
    // - 速度预测
    // - 类别一致性检查
    // - 物理约束检查
    
    LOG(INFO) << "[INFO] Results optimized, " << objects.size() << " objects remaining";
}

/**
 * @brief CPU NMS实现
 * @param boxes 输入边界框列表
 * @param iou_threshold IoU阈值
 * @param max_output_boxes 最大输出框数量
 * @return 去重后的边界框列表
 */
std::vector<BoundingBox3D> PostProcessor::cpuNMS(const std::vector<BoundingBox3D>& boxes, 
                                                 float iou_threshold, 
                                                 int max_output_boxes) {
    std::vector<BoundingBox3D> result;
    
    if (boxes.empty()) {
        return result;
    }
    
    // 按置信度排序（降序）
    std::vector<BoundingBox3D> sorted_boxes = boxes;
    std::sort(sorted_boxes.begin(), sorted_boxes.end(), 
              [](const BoundingBox3D& a, const BoundingBox3D& b) {
                  return a.confidence > b.confidence;
              });
    
    // 标记被抑制的检测框
    std::vector<bool> suppressed(sorted_boxes.size(), false);
    
    // 执行NMS
    for (size_t i = 0; i < sorted_boxes.size(); ++i) {
        if (suppressed[i]) continue;
        
        // 添加当前检测框到结果中
        result.push_back(sorted_boxes[i]);
        
        // 如果达到最大输出数量，停止
        if (result.size() >= max_output_boxes) break;
        
        // 抑制与当前检测框IoU大于阈值的检测框
        for (size_t j = i + 1; j < sorted_boxes.size(); ++j) {
            if (suppressed[j]) continue;
            
            // 只对相同类别的检测框进行NMS
            if (sorted_boxes[i].label == sorted_boxes[j].label) {
                float iou = calculate3DIoU(sorted_boxes[i], sorted_boxes[j]);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    return result;
}

float PostProcessor::calculate3DIoU(const BoundingBox3D& box1, const BoundingBox3D& box2) {
    // 简化的3D IoU计算（基于2D投影）
    // 计算中心点距离
    float dx = box1.x - box2.x;
    float dy = box1.y - box2.y;
    float distance = sqrtf(dx * dx + dy * dy);
    
    // 计算边界框对角线长度的一半作为阈值
    float threshold1 = sqrtf(box1.l * box1.l + box1.w * box1.w) / 2.0f;
    float threshold2 = sqrtf(box2.l * box2.l + box2.w * box2.w) / 2.0f;
    float overlap_threshold = (threshold1 + threshold2) * 0.5f;
    
    // 如果距离太远，IoU为0
    if (distance > overlap_threshold) {
        return 0.0f;
    }
    
    // 简化的IoU计算
    float overlap_ratio = 1.0f - (distance / overlap_threshold);
    return std::max(0.0f, overlap_ratio);
}
