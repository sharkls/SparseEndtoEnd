#ifndef __SPARSE4D_FIRST_HEAD_HPP__
#define __SPARSE4D_FIRST_HEAD_HPP__

#include "Sparse4D_conf.pb.h"
#include "../common/context.hpp"
#include "../../../Include/Common/Core/GlobalContext.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"

namespace sparse4d{
namespace first_head{
    class FirstHead{
    public:
        virtual ~FirstHead() = default;

        virtual Status init(const TaskConfig &param) = 0;

        virtual Status forward(common::PipelineContext& pipeline_context,
                                const cudaStream_t& stream,
                                common::HeadOutput& head_output) = 0;
    };

    std::shared_ptr<FirstHead> create_first_head(const TaskConfig &param);

}//namespace first_head
}//namespace sparse4d
#endif // __SPARSE4D_FIRST_HEAD_HPP__