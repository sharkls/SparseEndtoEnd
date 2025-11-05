#include "postprocessor.hpp"
#include <algorithm>

namespace sparse4d{
namespace postprocessor{

class PostprocessorImplement : public Postprocessor{
public:
    virtual ~PostprocessorImplement() = default;

    Status init(const TaskConfig &param) override;

    Status forward(const common::HeadOutput& head_output, 
                   const cudaStream_t& stream,
                   CAlgResult& output_result) override;

private:
    // CPU NMS相关方法
    std::vector<BoundingBox3D> cpu_nms(const std::vector<BoundingBox3D>& boxes, 
                                        float iou_threshold, 
                                        int max_output_boxes);
    float calculate_3d_iou(const BoundingBox3D& box1, const BoundingBox3D& box2);

    std::vector<BoundingBox3D> to_bounding_box_3d(const common::HeadOutput& head_output);
           
    void convert_to_alg_result(const std::vector<BoundingBox3D>& boxes, CAlgResult& output_result);
           
    // GPU版本的方法 - 直接处理原始数据
    Status process_gpu_direct(const common::HeadOutput& head_output,
                            const cudaStream_t& stream,
                            CAlgResult& output_result);
    
    // CPU版本的方法 - 复制到CPU后处理
    Status process_cpu_direct(const common::HeadOutput& head_output,
                             const cudaStream_t& stream,
                             CAlgResult& output_result);

    TaskConfig m_task_config;
    PostprocessorParams m_postprocessor_params;
};

Status PostprocessorImplement::init(const TaskConfig &param)
{
    LOG(INFO) << "[INFO] Sparse4D::PostprocessorImplement::init start";

    m_task_config = param;
    m_postprocessor_params.use_gpu_nms = m_task_config.postprocessor_params().use_gpu_nms();
    // map proto fields -> local params
    m_postprocessor_params.nms_threshold = m_task_config.postprocessor_params().gpu_nms_threshold();
    m_postprocessor_params.max_output_boxes = static_cast<int>(m_task_config.postprocessor_params().max_output_boxes());
    // use post_process_threshold as confidence threshold for filtering
    m_postprocessor_params.confidence_threshold = m_task_config.postprocessor_params().post_process_threshold();

    // 检查GPU NMS是否可用（如果启用）
    if (m_postprocessor_params.use_gpu_nms) {
        if (!is_gpu_nms_available()) {
            LOG(WARNING) << "[WARNING] GPU NMS not available, falling back to CPU NMS";
            m_postprocessor_params.use_gpu_nms = false;
        } else {
            LOG(INFO) << "[INFO] GPU NMS is available";
        }
    }

    LOG(INFO) << "[INFO] Sparse4D::PostprocessorImplement::init end";
    return Status::kSuccess;
}

Status PostprocessorImplement::forward(const common::HeadOutput& head_output, 
                                       const cudaStream_t& stream,
                                       CAlgResult& output_result)
{
    // LOG(INFO) << "[INFO] Sparse4D::PostprocessorImplement::forward start";

    // 根据配置选择处理路径
    if (m_postprocessor_params.use_gpu_nms) {
        // LOG(INFO) << "[INFO] Using GPU direct processing path";
        return process_gpu_direct(head_output, stream, output_result);
    } else {
        // LOG(INFO) << "[INFO] Using CPU direct processing path";
        return process_cpu_direct(head_output, stream, output_result);
    }
}


Status PostprocessorImplement::process_gpu_direct(const common::HeadOutput& head_output,
                                                 const cudaStream_t& stream,
                                                 CAlgResult& output_result)
{
    // LOG(INFO) << "[INFO] Sparse4D::PostprocessorImplement::process_gpu_direct start";

    // RAII内存管理
    BoundingBox3D* d_output_boxes = nullptr;
    int* d_output_count = nullptr;
    
    auto cleanup = [&]() {
        if (d_output_boxes) cudaFree(d_output_boxes);
        if (d_output_count) cudaFree(d_output_count);
    };

    try {
        // 检查输入数据
        if (!head_output.pred_anchor.isValid() || !head_output.pred_class_score.isValid() || 
            !head_output.pred_quality_score.isValid()) {
            LOG(WARNING) << "[WARNING] Empty input data";
            // produce empty frame result
            output_result.vecFrameResult(std::vector<CFrameResult>{});
            return Status::kSuccess;
        }

        // 计算目标数量
        int anchor_dim = 11;  // x, y, z, l, w, h, yaw, vx, vy, vz, time
        int num_objects = static_cast<int>(head_output.pred_anchor.getSize() / anchor_dim);
        int num_classes = m_task_config.model_cfg_params().num_classes();
        
        if (num_objects <= 0) {
            LOG(WARNING) << "[WARNING] No valid objects found";
            output_result.vecFrameResult(std::vector<CFrameResult>{});
            return Status::kSuccess;
        }
        
        // LOG(INFO) << "[INFO] Processing " << num_objects << " objects on GPU";

        // 分配输出内存
        cudaError_t err = cudaMalloc(&d_output_boxes, m_postprocessor_params.max_output_boxes * sizeof(BoundingBox3D));
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for output boxes: " << cudaGetErrorString(err);
            return Status::kPostprocessorForwardErr;
        }
        
        err = cudaMalloc(&d_output_count, sizeof(int));
        if (err != cudaSuccess) {
            LOG(ERROR) << "[ERROR] Failed to allocate GPU memory for output count: " << cudaGetErrorString(err);
            cleanup();
            return Status::kPostprocessorForwardErr;
        }
        
        // 直接在GPU上处理原始数据并执行NMS
        int num_output = gpu_nms_direct(
            head_output.pred_anchor.getCudaPtr(),
            head_output.pred_class_score.getCudaPtr(),
            head_output.pred_quality_score.getCudaPtr(),
            head_output.pred_track_ids.getCudaPtr(),
            num_objects,
            num_classes,
            d_output_boxes,
            d_output_count,
            m_postprocessor_params.confidence_threshold,
            m_postprocessor_params.nms_threshold,
            m_postprocessor_params.max_output_boxes,
            stream
        );
        
        if (num_output < 0) {
            LOG(ERROR) << "[ERROR] GPU NMS failed";
            cleanup();
            return Status::kPostprocessorForwardErr;
        }
        
        // LOG(INFO) << "[INFO] GPU NMS processed " << num_output << " boxes";

        // 将GPU结果复制到CPU并转换为CAlgResult格式
        if (num_output > 0) {
            std::vector<BoundingBox3D> host_boxes(num_output);
            cudaMemcpyAsync(host_boxes.data(), d_output_boxes, 
                           num_output * sizeof(BoundingBox3D),
                           cudaMemcpyDeviceToHost, stream);
            
            // 等待复制完成
            cudaStreamSynchronize(stream);
            
            // 转换为CAlgResult格式
            convert_to_alg_result(host_boxes, output_result);
        } else {
            output_result.vecFrameResult(std::vector<CFrameResult>{});
        }
        
        // 清理GPU内存
        cleanup();

        // LOG(INFO) << "[INFO] GPU direct processing completed, output " << num_output << " boxes";
        return Status::kSuccess;

    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] Exception in GPU direct processing: " << e.what();
        cleanup();
        return Status::kPostprocessorForwardErr;
    } catch (...) {
        LOG(ERROR) << "[ERROR] Unknown exception in GPU direct processing";
        cleanup();
        return Status::kPostprocessorForwardErr;
    }
}

Status PostprocessorImplement::process_cpu_direct(const common::HeadOutput& head_output,
                                                const cudaStream_t& stream,
                                                CAlgResult& output_result)
{
    LOG(INFO) << "[INFO] Sparse4D::PostprocessorImplement::process_cpu_direct start";

    try {
        // 1. 将GPU数据复制到CPU
        std::vector<BoundingBox3D> boxes = to_bounding_box_3d(head_output);

        if (boxes.empty()) {
            LOG(WARNING) << "[WARNING] No valid bounding boxes found";
            output_result.vecFrameResult(std::vector<CFrameResult>{});
            return Status::kSuccess;
        }

        LOG(INFO) << "[INFO] Converted " << boxes.size() << " bounding boxes on CPU";

        // 2. 使用CPU NMS
        LOG(INFO) << "[INFO] Using CPU NMS";
        std::vector<BoundingBox3D> nms_result = cpu_nms(boxes, 
                                                        m_postprocessor_params.nms_threshold, 
                                                        m_postprocessor_params.max_output_boxes);

        // 3. 将结果转换为CAlgResult格式
        convert_to_alg_result(nms_result, output_result);

        LOG(INFO) << "[INFO] CPU direct processing completed, output " << nms_result.size() << " boxes";
        return Status::kSuccess;

    } catch (const std::exception& e) {
        LOG(ERROR) << "[ERROR] Exception in CPU direct processing: " << e.what();
        return Status::kPostprocessorForwardErr;
    } catch (...) {
        LOG(ERROR) << "[ERROR] Unknown exception in CPU direct processing";
        return Status::kPostprocessorForwardErr;
    }
}

std::vector<BoundingBox3D> PostprocessorImplement::to_bounding_box_3d(const common::HeadOutput& head_output)
{
    std::vector<BoundingBox3D> boxes;
    
    if (!head_output.pred_anchor.isValid() || !head_output.pred_class_score.isValid() || !head_output.pred_quality_score.isValid()) {
        return boxes;
    }
    
    // 假设数据格式：每个目标有11个锚点值 + 类别分数 + 质量分数
    int anchor_dim = 11;  // x, y, z, l, w, h, yaw, vx, vy, vz, time
    int num_objects = static_cast<int>(head_output.pred_anchor.getSize() / anchor_dim);
    
    boxes.reserve(num_objects);
    
    for (int i = 0; i < num_objects; ++i) {
        BoundingBox3D box;
        
        // 从锚点中获取位置和尺寸 (1×900×11 格式)
        // 索引: [batch_idx][instance_idx][anchor_dim]
        size_t anchor_offset = 0 * static_cast<size_t>(num_objects) * anchor_dim + static_cast<size_t>(i) * anchor_dim;
        // 将GPU数据拷贝到CPU缓存
        const std::vector<float> host_anchor = head_output.pred_anchor.cudaMemcpyD2HResWrap();
        const std::vector<float> host_cls = head_output.pred_class_score.cudaMemcpyD2HResWrap();
        const std::vector<float> host_q = head_output.pred_quality_score.cudaMemcpyD2HResWrap();
        const std::vector<int32_t> host_track = head_output.pred_track_ids.cudaMemcpyD2HResWrap();

        // 辅助函数：sigmoid
        auto sigmoid_f = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        
        if (anchor_offset + 7 < host_anchor.size()) {  // 需要访问到anchor[7]
            // 位置信息
            box.x = host_anchor[anchor_offset + 0];  // 中心点x
            box.y = host_anchor[anchor_offset + 1];  // 中心点y
            box.z = host_anchor[anchor_offset + 2];  // 中心点z
            
            // 尺寸信息：模型输出的是log值，需要应用exp变换
            box.w = std::exp(host_anchor[anchor_offset + 3]);  // exp(log(w)) = w
            box.l = std::exp(host_anchor[anchor_offset + 4]);  // exp(log(l)) = l
            box.h = std::exp(host_anchor[anchor_offset + 5]);  // exp(log(h)) = h
            
            // 航向角：从anchor[6]读取sin(yaw)，从anchor[7]读取cos(yaw)
            const float sin_yaw = host_anchor[anchor_offset + 6];
            const float cos_yaw = host_anchor[anchor_offset + 7];
            box.yaw = std::atan2(sin_yaw, cos_yaw);
            
            // 验证航向角是否合理
            if (std::isnan(box.yaw) || std::isinf(box.yaw)) {
                box.yaw = 0.0f;  // 设置为默认值
            }
        } else {
            // 如果数据不足，跳过这个对象
            continue;
        }
        
        // 从类别分数中获取最高置信度的类别
        // 1. 对class_score应用sigmoid并找到最大值
        int class_offset = i * m_task_config.model_cfg_params().num_classes();
        float max_score = -1e10f;  // 初始化为很小的值
        int best_class = 0;
        
        for (int c = 0; c < m_task_config.model_cfg_params().num_classes(); ++c) {
            if (class_offset + c < static_cast<int>(host_cls.size())) {
                float sigmoid_score = sigmoid_f(host_cls[class_offset + c]);
                if (sigmoid_score > max_score) {
                    max_score = sigmoid_score;
                    best_class = c;
                }
            }
        }
        
        // 2. 获取quality_score（格式为[1, 900, 2]，取第一个维度作为centerness）
        // quality_score的实际格式：每个对象有2个值，索引为 i * 2 + 0 和 i * 2 + 1
        int quality_dims = 2;
        int quality_offset = i * quality_dims;
        float centerness_raw = 0.0f;
        if (quality_offset + 0 < static_cast<int>(host_q.size())) {
            centerness_raw = host_q[quality_offset + 0];  // 取第一个维度（centerness）
        }
        float centerness = sigmoid_f(centerness_raw);  // 应用sigmoid
        
        // 3. 计算最终置信度：centerness * max_class_score
        box.confidence = centerness * max_score;  // 综合置信度
        box.label = best_class;
        box.index = i;
        
        // 设置跟踪ID
        if (i < static_cast<int>(host_track.size())) {
            // 跟踪ID现在是int32_t类型，直接转换为int
            box.track_id = static_cast<int>(host_track[i]);
        } else {
            box.track_id = -1;  // 默认值
        }
        
        // 过滤低置信度的检测框
        if (box.confidence > m_postprocessor_params.confidence_threshold) {
            boxes.push_back(box);
        }
    }
    
    return boxes;
}

/**
 * @brief CPU NMS实现
 * @param boxes 输入边界框列表
 * @param iou_threshold IoU阈值
 * @param max_output_boxes 最大输出框数量
 * @return 去重后的边界框列表
 */
std::vector<BoundingBox3D> PostprocessorImplement::cpu_nms(const std::vector<BoundingBox3D>& boxes, 
                                                 float iou_threshold, 
                                                 int max_output_boxes) 
{
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
                float iou = calculate_3d_iou(sorted_boxes[i], sorted_boxes[j]);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    return result;
}

float PostprocessorImplement::calculate_3d_iou(const BoundingBox3D& box1, const BoundingBox3D& box2) 
{
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

/**
 * @brief 将BoundingBox3D结果转换为CAlgResult格式
 * @param boxes 输入边界框列表
 * @param output_result 输出结果
 */
void PostprocessorImplement::convert_to_alg_result(const std::vector<BoundingBox3D>& boxes, 
                                                  CAlgResult& output_result) 
{
    // 根据 CAlgResult/CFrameResult 接口填充
    CFrameResult frame;
    std::vector<CObjectResult> detections;
    detections.reserve(boxes.size());
    for (const auto& box : boxes) {
        CObjectResult obj;
        obj.x(box.x);
        obj.y(box.y);
        obj.z(box.z);
        obj.l(box.l);
        obj.w(box.w);
        obj.h(box.h);
        obj.yaw(box.yaw);
        obj.confidence(box.confidence);
        obj.label(static_cast<uint8_t>(box.label));
        obj.trackid(static_cast<int32_t>(box.track_id));
        detections.push_back(std::move(obj));
    }
    frame.vecObjectResult(std::move(detections));
    std::vector<CFrameResult> frames;
    frames.push_back(std::move(frame));
    output_result.vecFrameResult(std::move(frames));
}


// 工厂函数实现
std::shared_ptr<Postprocessor> create_postprocessor(const TaskConfig &param) {
    auto instance = std::make_shared<PostprocessorImplement>();
    Status status = instance->init(param);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Postprocessor init failed";
        return nullptr;
    }
    return instance;
}
}//namespace postprocessor
}//namespace sparse4d