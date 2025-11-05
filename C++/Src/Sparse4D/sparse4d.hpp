

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

namespace sparse4d{
namespace core{
class CoreImplement : public ICore{
    public:
        virtual ~CoreImplement() = default;

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

        CAlgResult forward_timer(const CTimeMatchSrcData *raw_data, void *stream);

        bool loadAuxiliaryData();

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

        // 外部接口适配所需
        std::string exe_path_;
        AlgCallback alg_cb_;
        void* user_handle_ = nullptr;
};
    
}  // namespace core
}  // namespace sparse4d
#endif // __SPARSE4D_HPP__