#ifndef __SPARSE4D_BACKBONE_HPP__
#define __SPARSE4D_BACKBONE_HPP__

#include "log.h"
#include "Sparse4D_conf.pb.h"
#include "../common/context.hpp"
#include "../../../Include/Common/Core/GlobalContext.h"
#include "../../../Include/Common/Utils/CudaWrapper.h"

namespace sparse4d{
namespace backbone{

class Backbone{
public:
    virtual ~Backbone() = default;
    virtual Status init(const TaskConfig &param) = 0;
    
    virtual Status forward(common::PipelineContext& pipeline_context,
                            const cudaStream_t& stream) = 0;
};

std::shared_ptr<Backbone> create_backbone(const TaskConfig &param);

}//namespace backbone
}//namespace sparse4d
#endif //__SPARSE4D_BACKBONE_HPP__