#include "sparse4d_impl.hpp"
#include "log.h"
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cuda_fp16.h>

namespace sparse4d {
namespace bev {

template <typename T>
Sparse4DImpl<T>::Sparse4DImpl() {
    cudaStreamCreate(&stream_);
}

template <typename T>
Sparse4DImpl<T>::~Sparse4DImpl() {
    if (stream_) cudaStreamDestroy(stream_);
}

template <typename T>
bool load_bin_file(const std::string& file_path, std::vector<T>& buffer) {
    std::ifstream ifs(file_path, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    
    // Check size matches (optional, but good for safety)
    if (size % sizeof(T) != 0) return false;
    
    size_t num_elements = size / sizeof(T);
    buffer.resize(num_elements);
    
    if (!ifs.read(reinterpret_cast<char*>(buffer.data()), size)) return false;
    return true;
}

template <typename T>
bool Sparse4DImpl<T>::init(const TaskConfig& config, const std::string& exe_path) {
    config_ = config;
    LOG(INFO) << "Initializing Unified Sparse4D...";

    // 1. Initialize Components
    LOG(INFO) << "Init Preprocessor...";
    preprocessor_ = std::make_unique<PreprocessorImpl<T>>();
    if (!preprocessor_->init(config)) {
        LOG(ERROR) << "Failed to init Preprocessor";
        return false;
    }

    LOG(INFO) << "Init InstanceBank...";
    instance_bank_ = std::make_unique<InstanceBank<T>>();
    if (!instance_bank_->init(config)) {
        LOG(ERROR) << "Failed to init InstanceBank";
        return false;
    }

    LOG(INFO) << "Init Postprocessor...";
    postprocessor_ = std::make_unique<PostprocessorImpl<T>>();
    if (!postprocessor_->init(config)) {
        LOG(ERROR) << "Failed to init Postprocessor";
        return false;
    }

    // 2. Initialize Engines
    LOG(INFO) << "Init Engines...";
    std::vector<std::string> plugins;
    if (!config.model_cfg_params().multiview_multiscale_deformable_attention_aggregation_path().empty()) {
        plugins.push_back(config.model_cfg_params().multiview_multiscale_deformable_attention_aggregation_path());
    }

    backbone_ = std::make_unique<EngineWrapper>();
    // if (!backbone_->init(config.backbone_engine().engine_path(), plugins)) return false;
    // // Check precision: T=float checks for FP32 engine, T=half checks for FP16
    // if (!backbone_->check_precision<T>()) return false;

    head1_ = std::make_unique<EngineWrapper>();
    if (!head1_->init(config.head1st_engine().engine_path(), plugins)) return false;

    head2_ = std::make_unique<EngineWrapper>();
    if (!head2_->init(config.head2nd_engine().engine_path(), plugins)) return false;

    // 3. Allocate Memory & Setup Bindings
    if (!init_memory()) return false;

    // 4. Load Aux Data
    if (!init_aux_data()) return false;

    return true;
}

template <typename T>
bool Sparse4DImpl<T>::init_memory() {
    // A. Preprocessor Out (Backbone In)
    auto img_shape = backbone_->get_binding_shape(backbone_->get_binding_index("img"));
    size_t img_vol = 1;
    for(int i=0; i<img_shape.nbDims; ++i) img_vol *= img_shape.d[i];
    input_imgs_.allocate(img_vol);

    // B. Backbone Out (Head In)
    auto feat_shape = backbone_->get_binding_shape(backbone_->get_binding_index("feature"));
    size_t feat_vol = 1;
    for(int i=0; i<feat_shape.nbDims; ++i) feat_vol *= feat_shape.d[i];
    feature_maps_.allocate(feat_vol);

    // C. Head Outputs (Assuming Head1 and Head2 have similar output shapes)
    // Pred Feature: [900, 256]
    pred_instance_feature_.allocate(900 * 256);
    pred_anchor_.allocate(900 * 11);
    pred_class_score_.allocate(900 * 10);
    pred_quality_score_.allocate(900 * 2);
    pred_track_id_.allocate(900); // Only head2 has this, head1 ignores it

    // D. Aux inputs (Need to load from file or set up constants)
    // For now, allocating dummy sizes, resized in init_aux_data
    spatial_shapes_.allocate(100); 
    level_start_index_.allocate(100);
    lidar2img_.allocate(6 * 4 * 4);
    image_wh_.allocate(6 * 2);

    // E. Setup Bindings Vectors
    auto setup_bindings = [&](EngineWrapper* engine, std::vector<void*>& bindings) {
        int n = engine->get_num_bindings();
        bindings.resize(n);
        for(int i=0; i<n; ++i) {
            std::string name = engine->get_binding_name(i);
            
            // Map name to buffer
            if (name == "img") bindings[i] = input_imgs_.getCudaPtr();
            else if (name == "feature") bindings[i] = feature_maps_.getCudaPtr();
            else if (name == "instance_feature") {
                // If input, it's from InstanceBank (temp) or init
                bindings[i] = nullptr; // Set dynamically
            }
            // Outputs
            else if (name == "pred_instance_feature") bindings[i] = pred_instance_feature_.getCudaPtr();
            else if (name == "pred_anchor") bindings[i] = pred_anchor_.getCudaPtr();
            else if (name == "pred_class_score") bindings[i] = pred_class_score_.getCudaPtr();
            else if (name == "pred_quality_score") bindings[i] = pred_quality_score_.getCudaPtr();
            else if (name == "pred_track_id") bindings[i] = pred_track_id_.getCudaPtr();
            else {
                // Aux inputs
                if (name == "spatial_shapes") bindings[i] = spatial_shapes_.getCudaPtr();
                else if (name == "level_start_index") bindings[i] = level_start_index_.getCudaPtr();
                else if (name == "lidar2img") bindings[i] = lidar2img_.getCudaPtr();
                else if (name == "image_wh") bindings[i] = image_wh_.getCudaPtr();
                // Instance Bank specific inputs
                else if (name == "temp_instance_feature") bindings[i] = instance_bank_->get_temp_features().getCudaPtr();
                else if (name == "temp_anchor") bindings[i] = instance_bank_->get_temp_anchors().getCudaPtr();
                else if (name == "mask") bindings[i] = instance_bank_->get_mask().getCudaPtr();
                else if (name == "track_id") bindings[i] = instance_bank_->get_track_ids().getCudaPtr();
                else if (name == "time_interval") bindings[i] = instance_bank_->get_time_interval().getCudaPtr();
                // Head1 specific inputs might be "anchor" or "instance_feature" -> handled dynamically
                else if (name == "anchor") {
                     bindings[i] = nullptr; // Set dynamically
                }
                else {
                    LOG(WARNING) << "Unknown binding name: " << name;
                }
            }
        }
    };

    setup_bindings(backbone_.get(), backbone_bindings_);
    setup_bindings(head1_.get(), head1_bindings_);
    setup_bindings(head2_.get(), head2_bindings_);

    return true;
}

template <typename T>
bool Sparse4DImpl<T>::init_aux_data() {
    // Load spatial_shapes and level_start_index from config
    std::vector<int32_t> spatial_shapes;
    for (int i = 0; i < config_.model_cfg_params().sparse4d_extract_feat_spatial_shapes_ld_size(); ++i) {
        spatial_shapes.push_back(config_.model_cfg_params().sparse4d_extract_feat_spatial_shapes_ld(i));
    }
    
    // Expand spatial_shapes [num_levels * 2] -> [num_cams * num_levels * 2]
    int num_cams = config_.preprocessor_params().num_cams();
    int num_levels = 4;
    std::vector<int32_t> spatial_shapes_expanded;
    spatial_shapes_expanded.reserve(num_cams * num_levels * 2);
    
    for (int cam = 0; cam < num_cams; ++cam) {
        for (int lvl = 0; lvl < num_levels; ++lvl) {
            spatial_shapes_expanded.push_back(spatial_shapes[lvl * 2 + 0]);
            spatial_shapes_expanded.push_back(spatial_shapes[lvl * 2 + 1]);
        }
    }
    
    spatial_shapes_.allocate(spatial_shapes_expanded.size());
    spatial_shapes_.cudaMemUpdateWrap(spatial_shapes_expanded);

    // level_start_index
    std::vector<int32_t> level_start_index;
    for (int i = 0; i < config_.model_cfg_params().sparse4d_extract_feat_level_start_index_size(); ++i) {
        level_start_index.push_back(config_.model_cfg_params().sparse4d_extract_feat_level_start_index(i));
    }
    level_start_index_.allocate(level_start_index.size());
    level_start_index_.cudaMemUpdateWrap(level_start_index);

    // Initial Anchor & Instance Feature (Load from bin file)
    std::string anchor_path = config_.instance_bank_params().instance_bank_anchor_path();
    if (!anchor_path.empty()) {
        std::vector<float> anchors_f;
        // Need a simple host loader for float
        // But load_bin_file is templated on T. 
        // We need to load as float first, then cast to T.
        // Let's create a float buffer.
        std::ifstream ifs(anchor_path, std::ios::binary | std::ios::ate);
        if (ifs) {
            std::streamsize size = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            size_t num = size / sizeof(float);
            anchors_f.resize(num);
            ifs.read(reinterpret_cast<char*>(anchors_f.data()), size);
            
            // Convert to T
            std::vector<T> anchors_t(num);
            for(size_t i=0; i<num; ++i) anchors_t[i] = (T)anchors_f[i];
            
            init_anchor_.allocate(num);
            init_anchor_.cudaMemUpdateWrap(anchors_t);
        } else {
            LOG(ERROR) << "Failed to load anchor file: " << anchor_path;
        }
    }
    
    // Init instance feature (zeros)
    size_t feat_dim = config_.model_cfg_params().embedfeat_dims();
    size_t num_queries = config_.instance_bank_params().num_querys();
    init_instance_feature_.allocate(num_queries * feat_dim);
    init_instance_feature_.cudaMemSetWrap(T(0.0f)); // Fixed T(0.0f) cast

    return true;
}

template <typename T>
void Sparse4DImpl<T>::forward(const CTimeMatchSrcData* src_data, CAlgResult& result) {
    if (!src_data) return;

    // 1. Preprocessing
    if (!preprocessor_->forward(src_data, stream_, input_imgs_)) {
        LOG(ERROR) << "Preprocessing failed";
        return;
    }

    // 2. Backbone Inference
    if (!backbone_->forward(backbone_bindings_, stream_)) {
        LOG(ERROR) << "Backbone inference failed";
        return;
    }

    // 3. Head Inference (Recurrent)
    double current_timestamp = (double)src_data->lTimeStamp() * 1e-6; // us to s

    Eigen::Matrix4d lidar_to_global = Eigen::Matrix4d::Identity();
    const auto& matrix_data = src_data->transform_info().lidar2global_matrix();
    if (matrix_data.size() == 16) {
        for (int i = 0; i < 16; ++i) {
            lidar_to_global(i / 4, i % 4) = matrix_data[i];
        }
    }

    // Project history
    instance_bank_->project_anchors(current_timestamp, lidar_to_global, stream_);

    // Decide Head1 or Head2
    bool first_frame = instance_bank_->is_first_frame();
    EngineWrapper* head_engine = first_frame ? head1_.get() : head2_.get();
    std::vector<void*>& head_bindings = first_frame ? head1_bindings_ : head2_bindings_;

    // Update Bindings for Instance Inputs
    int binding_idx_feat = head_engine->get_binding_index("instance_feature");
    int binding_idx_anchor = head_engine->get_binding_index("anchor");
    
    if (binding_idx_feat != -1) {
        if (first_frame) {
            head_bindings[binding_idx_feat] = init_instance_feature_.getCudaPtr();
        } else {
            head_bindings[binding_idx_feat] = instance_bank_->get_temp_features().getCudaPtr();
        }
    }
    
    if (binding_idx_anchor != -1) {
        if (first_frame) {
            head_bindings[binding_idx_anchor] = init_anchor_.getCudaPtr();
        } else {
            head_bindings[binding_idx_anchor] = instance_bank_->get_temp_anchors().getCudaPtr();
        }
    }

    // Execute Head
    if (!head_engine->forward(head_bindings, stream_)) {
         LOG(ERROR) << "Head inference failed";
         return;
    }

    // 4. Update Instance Bank
    instance_bank_->update(pred_instance_feature_, pred_anchor_, pred_class_score_, pred_track_id_, stream_);

    // 5. Postprocessing
    if (!postprocessor_->forward(pred_anchor_, pred_class_score_, pred_quality_score_, pred_track_id_, stream_, result)) {
        LOG(ERROR) << "Postprocessing failed";
        return;
    }
}

template <typename T>
bool Sparse4DImpl<T>::update_params(void* param) {
    return true;
}

// Explicit Instantiation
template class Sparse4DImpl<float>;
// For __half, include cuda_fp16.h which is done.
// But we might need to ensure __half is treated as a valid type for template instantiation.
// If compiling with C++ compiler (not nvcc), __half is usually struct __half { unsigned short __x; };
// So it should work.
template class Sparse4DImpl<__half>;

} // namespace bev
} // namespace sparse4d
