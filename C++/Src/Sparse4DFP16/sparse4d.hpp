

#ifndef __SPARSE4D_HPP__
#define __SPARSE4D_HPP__

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <cuda_fp16.h>

// 包含 protobuf 生成的 TaskConfig 定义
#include "Sparse4D_conf.pb.h"
#include "CTimeMatchSrcData.h"
#include "CAlgResult.h"

#include "./sparse4d/backbone.hpp"
#include "./sparse4d/first_head.hpp"
#include "./sparse4d/second_head.hpp"
#include "./sparse4d/instance_bank.hpp"
#include "./postprocessor/postprocessor.hpp"
#include "./preprocessor/img_preprocessor.hpp"
#include "./common/context.hpp"
#include "../../Include/Interface/ExportSparse4D.h"
#include <cuda_runtime.h>

namespace sparse4d{
namespace core{
class CoreImplement : public ICore{
    public:
        virtual ~CoreImplement();

        bool initAlgorithm(const std::string exe_path,  const AlgCallback& alg_cb, void* hd) override;

        bool update(void* p_pParam) override;

        void runAlgorithm(void* p_pSrcData) override;

    private:

        // 初始化
        bool init(const TaskConfig &param);

        CAlgResult forward(const CTimeMatchSrcData *raw_data, void *stream);

        void update(const float *lidar2camera, void *stream = nullptr);

        void free_excess_memory();

        CAlgResult forward_only(const CTimeMatchSrcData *raw_data, void *stream);
        
        // // 模板化的 forward_only 实现
        // template<typename FloatType>
        // CAlgResult forward_only_impl(const CTimeMatchSrcData *raw_data, 
        //                              void *stream,
        //                              common::PipelineContext<FloatType>& pipeline_context,
        //                              common::HeadOutput<FloatType>& head_output);

        CAlgResult forward_timer(const CTimeMatchSrcData *raw_data, void *stream);
        
        // // 模板化的 forward_timer 实现
        // template<typename FloatType>
        // CAlgResult forward_timer_impl(const CTimeMatchSrcData *raw_data, 
        //                              void *stream,
        //                              common::PipelineContext<FloatType>& pipeline_context,
        //                              common::HeadOutput<FloatType>& head_output);

        bool loadAuxiliaryData();
        
        // // 模板化的辅助函数，用于根据精度类型加载辅助数据
        // template<typename FloatType>
        // bool loadAuxiliaryDataImpl(common::PipelineContext<FloatType>& pipeline_context,
        //                           common::HeadOutput<FloatType>& head_output);
        
        /**
         * @brief 预热推理：多次调用以触发 CUDA Graph 捕获
         * @return 成功返回 true，失败返回 false
         * @note TensorRT 8.0+ 需要 2-3 次调用 enqueueV2 才能捕获 CUDA Graph
         */
        bool warmupInference();

        std::shared_ptr<preprocessor::Preprocessor> preprocessor_;
        std::shared_ptr<instance_bank::InstanceBank> instance_bank_;
        std::shared_ptr<backbone::Backbone> backbone_;
        std::shared_ptr<first_head::FirstHead> head1_;
        std::shared_ptr<second_head::SecondHead> head2_;
        std::shared_ptr<postprocessor::Postprocessor> postprocessor_;
        TaskConfig param_;

        // 运行全局变量
        bool is_first_frame_ = true;  // 初始化为 true，第一次调用时使用 head1
        common::PipelineContext pipeline_context_;
        common::HeadOutput head_output_;
        bool enable_timer_ = false;

        // CUDA Stream 用于 CUDA Graph 优化
        cudaStream_t inference_stream_ = nullptr;  // 固定的推理 stream，用于启用 CUDA Graph

        // 外部接口适配所需
        std::string exe_path_;
        AlgCallback alg_cb_;
        void* user_handle_ = nullptr;
};
    
}  // namespace core
}  // namespace sparse4d
#endif // __SPARSE4D_HPP__