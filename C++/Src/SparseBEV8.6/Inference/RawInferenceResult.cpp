#include "RawInferenceResult.h"
#include "../Postprocess/gpu_nms.h"

// 从RawInferenceResult创建GPUNMSInput（仅在需要时进行GPU到CPU拷贝）
GPUNMSInput GPUNMSInput::fromRawResult(const RawInferenceResult& raw_result) 
{
    GPUNMSInput input;
    
    if (!raw_result.isValid()) {
        return input;
    }
    
    input.num_objects = raw_result.num_objects;
    input.num_classes = raw_result.num_classes;
    
    // 从GPU内存拷贝到CPU内存（仅在需要时）
    if (raw_result.pred_instance_feature) {
        input.instance_features = raw_result.pred_instance_feature->cudaMemcpyD2HResWrap();
    }
    
    if (raw_result.pred_anchor) {
        input.anchors = raw_result.pred_anchor->cudaMemcpyD2HResWrap();
    }
    
    if (raw_result.pred_class_score) {
        input.class_scores = raw_result.pred_class_score->cudaMemcpyD2HResWrap();
    }
    
    if (raw_result.pred_quality_score) {
        input.quality_scores = raw_result.pred_quality_score->cudaMemcpyD2HResWrap();
    }
    
    if (raw_result.pred_track_id) {
        input.track_ids = raw_result.pred_track_id->cudaMemcpyD2HResWrap();
        std::cout << "[GPUNMSInput][DEBUG] Retrieved track_ids from raw_result, size: " << input.track_ids.size() << std::endl;
        if (!input.track_ids.empty()) {
            std::cout << "[GPUNMSInput][DEBUG] First 10 track_ids: ";
            for (size_t i = 0; i < std::min(size_t(10), input.track_ids.size()); ++i) {
                std::cout << input.track_ids[i] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "[GPUNMSInput][DEBUG] raw_result.pred_track_id is null" << std::endl;
    }
    
    return input;
}

// 转换为BoundingBox3D格式（用于GPU NMS）
std::vector<BoundingBox3D> GPUNMSInput::toBoundingBox3D() const {
    std::vector<BoundingBox3D> boxes;
    
    if (anchors.empty() || class_scores.empty() || quality_scores.empty()) {
        return boxes;
    }
    
    // 假设数据格式：每个目标有11个锚点值 + 类别分数 + 质量分数
    int anchor_dim = 11;  // x, y, z, l, w, h, yaw, vx, vy, vz, time
    int num_objects = anchors.size() / anchor_dim;
    
    boxes.reserve(num_objects);
    
    for (int i = 0; i < num_objects; ++i) {
        BoundingBox3D box;
        
        // 从锚点中获取位置和尺寸 (1×900×11 格式)
        // 索引: [batch_idx][instance_idx][anchor_dim]
        size_t anchor_offset = 0 * num_objects * anchor_dim + i * anchor_dim;
        if (anchor_offset + 6 < anchors.size()) {
            box.x = anchors[anchor_offset + 0];  // 中心点x
            box.y = anchors[anchor_offset + 1];  // 中心点y
            box.z = anchors[anchor_offset + 2];  // 中心点z
            box.w = anchors[anchor_offset + 3];  // 宽度
            box.l = anchors[anchor_offset + 4];  // 长度
            box.h = anchors[anchor_offset + 5];  // 高度
            
            // 处理航向角：确保在[-π, π]范围内
            float raw_yaw = anchors[anchor_offset + 6];
            box.yaw = std::atan2(std::sin(raw_yaw), std::cos(raw_yaw));
            
            // 验证航向角是否合理
            if (std::isnan(box.yaw) || std::isinf(box.yaw)) {
                box.yaw = 0.0f;  // 设置为默认值
            }
        }
        
        // 从类别分数中获取最高置信度的类别
        int class_offset = i * num_classes;
        float max_score = 0.0f;
        int best_class = 0;
        
        for (int c = 0; c < num_classes; ++c) {
            if (class_scores[class_offset + c] > max_score) {
                max_score = class_scores[class_offset + c];
                best_class = c;
            }
        }
        
        // 使用质量分数作为置信度
        float quality_score = (i < quality_scores.size()) ? quality_scores[i] : 0.0f;
        box.confidence = quality_score * max_score;  // 综合置信度
        box.label = best_class;
        box.index = i;
        
        // 设置跟踪ID
        if (i < track_ids.size()) {
            // 跟踪ID现在是int32_t类型，直接转换为int
            box.track_id = static_cast<int>(track_ids[i]);
        } else {
            box.track_id = -1;  // 默认值
        }
        std::cout << "[BoundingBox3D][DEBUG] Box[" << i << "] track_id: " << box.track_id << ", confidence: " << box.confidence << std::endl;
        
        // 过滤低置信度的检测框
        if (box.confidence > 0.1f) {  // 可配置的阈值
            boxes.push_back(box);
        }
    }
    
    return boxes;
} 