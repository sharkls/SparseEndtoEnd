#ifndef __SPARSE4D_SECOND_HEAD_HPP__
#define __SPARSE4D_SECOND_HEAD_HPP__

#include "Sparse4D_conf.pb.h"
#include "../common/context.hpp"
#include "../../../Include/Common/Core/GlobalContext.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"
#include "../../Common/TensorRT/TensorRT.h"
#include "log.h"

namespace sparse4d{
namespace second_head{
    class SecondHead{
    public:
        virtual ~SecondHead() = default;

        virtual Status init(const TaskConfig &param) = 0;

        virtual Status forward(common::PipelineContext& pipeline_context,
                                const cudaStream_t& stream, 
                                common::HeadOutput& head_output) = 0;
    };

    std::shared_ptr<SecondHead> create_second_head(const TaskConfig &param);

}//namespace second_head
}//namespace sparse4d
#endif // __SPARSE4D_SECOND_HEAD_HPP__