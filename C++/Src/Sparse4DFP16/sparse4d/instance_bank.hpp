#ifndef __SPARSE4D_INSTANCE_BANK_HPP__
#define __SPARSE4D_INSTANCE_BANK_HPP__

#include "../common/context.hpp"
#include "Sparse4D_conf.pb.h"
#include "log.h"
#include "CTimeMatchSrcData.h"

#include "../../../Include/Common/Core/GlobalContext.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"

// 使用条件编译避免重复包含Eigen
#ifndef EIGEN_DENSE_INCLUDED
#define EIGEN_DENSE_INCLUDED
#include <Eigen/Dense>
#endif

namespace sparse4d{
namespace instance_bank{
    class InstanceBank{
    public:
        virtual ~InstanceBank() = default;
        
        virtual Status init(const TaskConfig &param) = 0;

        /// @brief 重置实例银行状态
        virtual Status reset() = 0;

        virtual Status get(const CTimeMatchSrcData *raw_data, const bool& is_first_frame, cudaStream_t stream, common::PipelineContext& pipeline_context) = 0;

        virtual Status cache(common::HeadOutput& head_output, const bool& is_first_frame, cudaStream_t stream = 0) = 0;

        virtual Status getTrackId(common::HeadOutput& head_output, const bool& is_first_frame, cudaStream_t stream = 0) = 0;
    };

    std::shared_ptr<InstanceBank> create_instance_bank(const TaskConfig &param);
}//namespace instance_bank
}//namespace sparse4d
#endif //__SPARSE4D_INSTANCE_BANK_HPP__