#include "../include/sparse4d_bev.hpp"
#include "sparse4d_impl.hpp"
#include "log.h" 
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace sparse4d {
namespace bev {

CoreImplement::CoreImplement() = default;
CoreImplement::~CoreImplement() = default;

bool CoreImplement::initAlgorithm(const std::string exe_path, const AlgCallback& alg_cb, void* hd) {
    LOG(INFO) << "Initializing Sparse4D BEV Unified Core...";
    alg_cb_ = alg_cb;
    user_handle_ = hd;
    
    // 1. Load Configuration (Mock logic based on typical Sparse4D setup)
    // Fix: Correct casing for config file and path relative to executable
    // Note: exe_path is typically the directory containing the executable.
    // If running from Output, exe_path should be Output.
    // Configs are at Output/Configs/Alg/Sparse4d.conf
    std::string config_path = exe_path + "/Configs/Alg/Sparse4d.conf";
    
    // Check if config file exists, if not, try looking one level up (if running from build dir but exe_path is source?)
    // Or if running from Output but config is in ../Configs? 
    // Based on ls -R: Output/Configs/Alg/Sparse4d.conf exists.
    // If exe_path is Output, then Output/Configs/Alg/Sparse4d.conf is correct.
    // But the error said: Failed to open: .../C++/Configs/Alg/Sparse4d.conf
    // This implies exe_path was passed as .../C++
    
    // Let's add a fallback or check file existence.
    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        // Fallback: assume exe_path might need "/Output" appended if it points to root
        std::string config_path_fallback = exe_path + "/Output/Configs/Alg/Sparse4d.conf";
        ifs.open(config_path_fallback);
        if (ifs.is_open()) {
            config_path = config_path_fallback;
            LOG(INFO) << "Found config at fallback path: " << config_path;
        } else {
             LOG(ERROR) << "Failed to open config file: " << config_path;
             return false;
        }
    }
    
    TaskConfig config;
    
    google::protobuf::io::IstreamInputStream iis(&ifs);
    if (!google::protobuf::TextFormat::Parse(&iis, &config)) {
        LOG(ERROR) << "Failed to parse config file: " << config_path;
        return false;
    }

    // 2. Determine Precision based on Config
    // Default to FP32. Check precision_type if available, otherwise fallback to use_half_precision
    bool use_fp16 = false; 
    
    // Check new precision_type field first
    if (!config.precision_type().empty()) {
        if (config.precision_type() == "fp16") {
            use_fp16 = true;
        } else if (config.precision_type() == "fp32") {
            use_fp16 = false;
        } else {
            LOG(WARNING) << "Unknown precision_type: " << config.precision_type() << ", defaulting to FP32";
        }
    } else {
        // Fallback to legacy boolean flag if precision_type is not set
        if (config.use_half_precision()) {
            use_fp16 = true;
        }
    }

    if (use_fp16) {
        impl_ = std::make_unique<Sparse4DImpl<half>>();
        LOG(INFO) << "Selected Precision: FP16";
    } else {
        impl_ = std::make_unique<Sparse4DImpl<float>>();
        LOG(INFO) << "Selected Precision: FP32";
    }

    return impl_->init(config, exe_path);
}

bool CoreImplement::update(void* p_pParam) {
    if (impl_) {
        return impl_->update_params(p_pParam);
    }
    return false;
}

void CoreImplement::runAlgorithm(void* p_pSrcData) {
    if (!impl_) return;
    
    CAlgResult result;
    auto* src_data = static_cast<CTimeMatchSrcData*>(p_pSrcData);
    impl_->forward(src_data, result);
    
    // Callback with result
    if (alg_cb_) {
        alg_cb_(result, user_handle_);
    }
}

} // namespace bev
} // namespace sparse4d

extern "C" {
    ICore* CreateCoreObj(const std::string& p_strExePath) {
        return new sparse4d::bev::CoreImplement();
    }
}

