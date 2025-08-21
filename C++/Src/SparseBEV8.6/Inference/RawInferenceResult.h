#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <algorithm>  // 为std::sort提供支持
#include <numeric>    // 为std::iota提供支持
#include "../../../Include/Common/Utils/CudaWrapper.h"
#include "../Postprocess/gpu_nms.h"
#include "log.h"

/**
 * @brief 原始推理结果结构，保持GPU内存格式
 * 
 * 这个结构用于存储TensorRT推理的原始输出，避免不必要的数据转换
 */
struct RawInferenceResult 
{
    // GPU内存中的推理结果
    std::shared_ptr<CudaWrapper<float>> pred_instance_feature;  // 实例特征
    std::shared_ptr<CudaWrapper<float>> pred_anchor;           // 锚点预测
    std::shared_ptr<CudaWrapper<float>> pred_class_score;      // 分类得分
    std::shared_ptr<CudaWrapper<float>> pred_quality_score;    // 质量得分
    std::shared_ptr<CudaWrapper<int32_t>> pred_track_id;         // 跟踪ID

    // 元数据
    int num_objects;                    // 检测到的目标数量
    int num_classes;                    // 类别数量
    bool is_first_frame;                // 是否为第一帧

    RawInferenceResult() : num_objects(0), num_classes(10), is_first_frame(true) {}

    /**
     * @brief 计算GPU内存总大小
     * @return GPU内存大小（字节）
     */
    size_t getGPUMemorySize() const {
        size_t total_size = 0;
        if (pred_instance_feature) total_size += pred_instance_feature->getSize() * sizeof(float);
        if (pred_anchor) total_size += pred_anchor->getSize() * sizeof(float);
        if (pred_class_score) total_size += pred_class_score->getSize() * sizeof(float);
        if (pred_quality_score) total_size += pred_quality_score->getSize() * sizeof(float);
        if (pred_track_id) total_size += pred_track_id->getSize() * sizeof(int32_t);
        return total_size;
    }

    /**
     * @brief 检查是否有有效的推理结果
     * @return true如果有有效结果
     */
    bool hasValidResults() const {
        return pred_instance_feature && pred_instance_feature->isValid() &&
               pred_anchor && pred_anchor->isValid() &&
               pred_class_score && pred_class_score->isValid() &&
               pred_quality_score && pred_quality_score->isValid();
    }

    /**
     * @brief 检查是否有效（兼容性方法）
     * @return true如果有有效结果
     */
    bool isValid() const {
        return hasValidResults();
    }

    /**
     * @brief 获取预测的实例数量
     * @return 实例数量
     */
    size_t getInstanceCount() const {
        // 使用元数据中的 num_objects，而不是张量大小
        return num_objects;
    }

    /**
     * @brief 转换为3D边界框格式（用于后处理）
     * @return 3D边界框向量
     */
    std::vector<BoundingBox3D> toBoundingBox3D() const {
        std::vector<BoundingBox3D> boxes;
        
        if (!hasValidResults()) {
            return boxes;
        }
        
        // 从GPU内存复制数据到CPU
        std::vector<float> anchors = pred_anchor->cudaMemcpyD2HResWrap();
        std::vector<float> class_scores = pred_class_score->cudaMemcpyD2HResWrap();
        std::vector<float> quality_scores = pred_quality_score->cudaMemcpyD2HResWrap();
        
        std::vector<int32_t> track_ids;
        const bool squeeze_cls = (pred_track_id && pred_track_id->isValid());
        if (squeeze_cls) {
            track_ids = pred_track_id->cudaMemcpyD2HResWrap();
        }
        
        const size_t num_instances = getInstanceCount();
        if (num_instances == 0) {
            return boxes;
        }
        
        // 根据单元测试确认的数据格式：
        // pred_anchor: 1×900×11 (batch_size=1, num_instances=900, anchor_dims=11)
        // pred_class_score: 1×900×10 (batch_size=1, num_instances=900, num_classes=10)
        // pred_quality_score: 1×900×2 (batch_size=1, num_instances=900, quality_dims=2)
        
        const size_t batch_size = 1;
        const size_t anchor_dims = 11;  // X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ
        const size_t quality_dims = 2;
        const int num_cls = num_classes; // 与decoder一致

        // 校验大小
        const size_t expected_anchors_size = batch_size * num_instances * anchor_dims;
        const size_t expected_scores_size  = batch_size * num_instances * num_cls;
        const size_t expected_quality_size = batch_size * num_instances * quality_dims;
        
        if (anchors.size() != expected_anchors_size || 
            class_scores.size() != expected_scores_size ||
            quality_scores.size() != expected_quality_size) {
            LOG(ERROR) << "数据大小不匹配: anchors=" << anchors.size() 
                       << " expected=" << expected_anchors_size
                       << ", scores=" << class_scores.size() 
                       << " expected=" << expected_scores_size
                       << ", quality=" << quality_scores.size() 
                       << " expected=" << expected_quality_size;
            return boxes;
        }
        
        // 1) 对分类logits先做sigmoid
        std::vector<float> cls_sigmoid(class_scores.size());
        for (size_t i = 0; i < class_scores.size(); ++i) {
            cls_sigmoid[i] = 1.0f / (1.0f + std::exp(-class_scores[i]));
        }

        // 2) 若有track_id，先在类维上max并记录argmax（squeeze_cls）
        std::vector<float> squeezed_scores;       // (N,)
        std::vector<int>   squeezed_cls_ids;      // (N,)
        if (squeeze_cls) {
            squeezed_scores.resize(num_instances);
            squeezed_cls_ids.resize(num_instances);
        for (size_t i = 0; i < num_instances; ++i) {
                float best = -std::numeric_limits<float>::infinity();
                int   best_c = 0;
                const size_t base = i * num_cls;
                for (int c = 0; c < num_cls; ++c) {
                    float s = cls_sigmoid[base + c];
                    if (s > best) {
                        best = s;
                        best_c = c;
                    }
                }
                squeezed_scores[i] = best;
                squeezed_cls_ids[i] = best_c;
            }
        }

        // 3) topk
        const int num_output = 300;
        std::vector<size_t> topk_indices;  // 实例索引
        std::vector<int>    topk_class_ids;
        std::vector<float>  topk_scores;

        if (squeeze_cls) {
            // 对(N,)做topk
            std::vector<std::pair<float, size_t>> scored_idx;
            scored_idx.reserve(num_instances);
            for (size_t i = 0; i < num_instances; ++i) {
                scored_idx.emplace_back(squeezed_scores[i], i);
            }
            std::sort(scored_idx.begin(), scored_idx.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

            const size_t K = std::min<size_t>(num_output, scored_idx.size());
            topk_indices.resize(K);
            topk_class_ids.resize(K);
            topk_scores.resize(K);
            for (size_t k = 0; k < K; ++k) {
                size_t inst = scored_idx[k].second;
                topk_indices[k] = inst;
                topk_class_ids[k] = squeezed_cls_ids[inst];
                topk_scores[k] = scored_idx[k].first;
            }
        } else {
            // 对(N*C,)做topk
            std::vector<std::pair<float, size_t>> scored_flat_idx;
            scored_flat_idx.reserve(num_instances * static_cast<size_t>(num_cls));
            for (size_t i = 0; i < num_instances; ++i) {
                const size_t base = i * num_cls;
                for (int c = 0; c < num_cls; ++c) {
                    scored_flat_idx.emplace_back(cls_sigmoid[base + c], base + c);
                }
            }
            std::sort(scored_flat_idx.begin(), scored_flat_idx.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

            const size_t K = std::min<size_t>(num_output, scored_flat_idx.size());
            topk_indices.resize(K);
            topk_class_ids.resize(K);
            topk_scores.resize(K);
            for (size_t k = 0; k < K; ++k) {
                size_t flat = scored_flat_idx[k].second;
                size_t inst = flat / num_cls;
                int    cid  = static_cast<int>(flat % num_cls);
                topk_indices[k] = inst;
                topk_class_ids[k] = cid;
                topk_scores[k] = scored_flat_idx[k].first;
            }
        }

        // 统计：乘以centerness之前的Top-10分数
        {
            const size_t Nprint = std::min<size_t>(topk_scores.size(), 10);
            if (Nprint > 0) {
                float pre_sum = 0.0f;
                float pre_max = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < Nprint; ++i) {
                    pre_sum += topk_scores[i];
                    if (topk_scores[i] > pre_max) pre_max = topk_scores[i];
                }
                float pre_mean = pre_sum / static_cast<float>(Nprint);
                LOG(INFO) << "[DecodeStat] pre_score_mean(top10)=" << pre_mean
                          << ", pre_score_max(top10)=" << pre_max;
            }
        }

        // 4) 应用centerness并重排（CNS=质量的第0通道）
        if (!quality_scores.empty()) {
            auto sigmoid_f = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

            // 先取对应实例的centerness
            std::vector<float> centerness_sel(topk_indices.size());
            for (size_t k = 0; k < topk_indices.size(); ++k) {
                size_t i = topk_indices[k];
                const size_t qoff = i * quality_dims; // CNS=0
                float cns = quality_scores[qoff + 0];
                centerness_sel[k] = sigmoid_f(cns);
            }

            // 统计：centerness的Top-10均值/最大值（与当前排序对应的前10）
            {
                const size_t Nprint = std::min<size_t>(centerness_sel.size(), 10);
                if (Nprint > 0) {
                    float cns_sum = 0.0f;
                    float cns_max = -std::numeric_limits<float>::infinity();
                    for (size_t i = 0; i < Nprint; ++i) {
                        cns_sum += centerness_sel[i];
                        if (centerness_sel[i] > cns_max) cns_max = centerness_sel[i];
                    }
                    float cns_mean = cns_sum / static_cast<float>(Nprint);
                    LOG(INFO) << "[DecodeStat] centerness_mean(top10)=" << cns_mean
                              << ", centerness_max(top10)=" << cns_max;
                }
            }

            // 乘以centerness并按新scores降序重排indices/class_ids/scores
            for (size_t k = 0; k < topk_scores.size(); ++k) {
                topk_scores[k] *= centerness_sel[k];
            }
            std::vector<size_t> order(topk_scores.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](size_t a, size_t b) { return topk_scores[a] > topk_scores[b]; });

            // 应用重排（分别重排三组向量，避免类型技巧）
            if (!order.empty()) {
                std::vector<size_t> topk_indices_tmp(order.size());
                std::vector<int>    topk_class_ids_tmp(order.size());
                std::vector<float>  topk_scores_tmp(order.size());
                for (size_t r = 0; r < order.size(); ++r) {
                    size_t idx = order[r];
                    topk_indices_tmp[r]   = topk_indices[idx];
                    topk_class_ids_tmp[r] = topk_class_ids[idx];
                    topk_scores_tmp[r]    = topk_scores[idx];
                }
                topk_indices.swap(topk_indices_tmp);
                topk_class_ids.swap(topk_class_ids_tmp);
                topk_scores.swap(topk_scores_tmp);
            }

            // 统计：乘以centerness后的Top-10分数
            {
                const size_t Nprint = std::min<size_t>(topk_scores.size(), 10);
                if (Nprint > 0) {
                    float post_sum = 0.0f;
                    float post_max = -std::numeric_limits<float>::infinity();
                    for (size_t i = 0; i < Nprint; ++i) {
                        post_sum += topk_scores[i];
                        if (topk_scores[i] > post_max) post_max = topk_scores[i];
                    }
                    float post_mean = post_sum / static_cast<float>(Nprint);
                    LOG(INFO) << "[DecodeStat] post_score_mean(top10)=" << post_mean
                              << ", post_score_max(top10)=" << post_max;
                }
            }
        }

        // 5) 解码box并构建输出（严格按decoder的decode_box）
        boxes.reserve(topk_indices.size());
        for (size_t k = 0; k < topk_indices.size(); ++k) {
            const size_t i = topk_indices[k];
            const size_t aoff = i * anchor_dims;

            BoundingBox3D box;
            // 先复制/解码
            const float x = anchors[aoff + 0];
            const float y = anchors[aoff + 1];
            const float z = anchors[aoff + 2];
            const float W = std::exp(anchors[aoff + 3]);
            const float L = std::exp(anchors[aoff + 4]);
            const float H = std::exp(anchors[aoff + 5]);
            const float sin_yaw = anchors[aoff + 6];
            const float cos_yaw = anchors[aoff + 7];
            const float yaw = std::atan2(sin_yaw, cos_yaw);

            box.x = x; box.y = y; box.z = z;
            // 注意：结构体语义为 l(长度)、w(宽度)、h，与Python中 [W,L,H] 一致
            box.w = W; box.l = L; box.h = H;
            box.yaw = yaw;

            // 置信度/类别/track_id
            box.confidence = topk_scores[k];
            box.label = topk_class_ids[k];
            if (squeeze_cls) {
                if (i < track_ids.size()) box.track_id = static_cast<int>(track_ids[i]);
                else box.track_id = -1;
            } else {
                box.track_id = -1;
            }
            box.index = static_cast<int>(k);

            // 完全对齐decoder：不做分数/几何阈值过滤
            boxes.push_back(box);
        }

        LOG(INFO) << "解析了 " << num_instances << " 个实例，选择Top-" << boxes.size()
                  << "，有效检测框 " << boxes.size() << " 个";
        return boxes;
    }
};

// 用于GPU NMS的中间数据结构
struct GPUNMSInput {
    std::vector<float> instance_features;    // 实例特征
    std::vector<float> anchors;              // 锚点
    std::vector<float> class_scores;         // 类别分数
    std::vector<float> quality_scores;       // 质量分数
    std::vector<int32_t> track_ids;          // 跟踪ID
    
    int num_objects;                         // 目标数量
    int num_classes;                         // 类别数量
    
    GPUNMSInput() : num_objects(0), num_classes(10) {}
    
    // 从RawInferenceResult创建（仅在需要时进行GPU到CPU拷贝）
    static GPUNMSInput fromRawResult(const RawInferenceResult& raw_result);
    
    // 转换为BoundingBox3D格式（用于GPU NMS）
    std::vector<BoundingBox3D> toBoundingBox3D() const;
}; 