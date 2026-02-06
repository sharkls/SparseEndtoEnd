

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
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace sparse4d{
namespace core{

// 1. 定义单帧上下文，包含所有该帧需要的显存资源
struct FrameContext {
    int frame_id = -1;
    common::PipelineContext pipeline_context; // 输入特征、中间变量
    common::HeadOutput head_output;           // 输出结果
    
    // 同步原语
    cudaEvent_t event_backbone_done = nullptr; // 标记 Backbone 完成
    cudaEvent_t event_all_done = nullptr;      // 标记整帧完成
    
    // 构造/析构中分配和释放显存及Event
    bool init(const TaskConfig& param);
    void free();
};

class CoreImplement : public ICore{
    public:
        virtual ~CoreImplement();

        bool initAlgorithm(const std::string exe_path,  const AlgCallback& alg_cb, void* hd) override;

        bool update(void* p_pParam) override;

        void runAlgorithm(void* p_pSrcData) override;

    private:

        // 初始化
        bool init(const TaskConfig &param);

        // 临时保留旧接口签名，内部实现将改为从资源池获取Context
        CAlgResult forward(const CTimeMatchSrcData *raw_data, void *stream);

        void update(const float *lidar2camera, void *stream = nullptr);

        void free_excess_memory();

        CAlgResult forward_only(const CTimeMatchSrcData *raw_data, void *stream, FrameContext* context);
        
        // // 模板化的 forward_only 实现
        // template<typename FloatType>
        // CAlgResult forward_only_impl(const CTimeMatchSrcData *raw_data, 
        //                              void *stream,
        //                              common::PipelineContext<FloatType>& pipeline_context,
        //                              common::HeadOutput<FloatType>& head_output);

        CAlgResult forward_timer(const CTimeMatchSrcData *raw_data, void *stream, FrameContext* context);
        
        // // 模板化的 forward_timer 实现
        // template<typename FloatType>
        // CAlgResult forward_timer_impl(const CTimeMatchSrcData *raw_data, 
        //                              void *stream,
        //                              common::PipelineContext<FloatType>& pipeline_context,
        //                              common::HeadOutput<FloatType>& head_output);

        // 辅助函数改为静态或成员函数，传入Context
        bool loadAuxiliaryData(FrameContext* context);
        
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
        // common::PipelineContext pipeline_context_; // 移除：改为资源池
        // common::HeadOutput head_output_;           // 移除：改为资源池
        bool enable_timer_ = false;

        // CUDA Stream 用于 CUDA Graph 优化
        // cudaStream_t inference_stream_ = nullptr;  // 移除：改为双流

        // 外部接口适配所需
        std::string exe_path_;
        AlgCallback alg_cb_;
        void* user_handle_ = nullptr;

        // --- 新增成员 ---
        
        // 2. 资源池管理
        std::vector<std::shared_ptr<FrameContext>> context_pool_;
        std::queue<std::shared_ptr<FrameContext>> free_contexts_; // 空闲池
        std::mutex context_mutex_;
        std::condition_variable context_cv_;

        // 3. 线程通信队列 (Step 2 使用)
        std::queue<std::shared_ptr<FrameContext>> ready_queue_; // 预处理完成，待推理
        std::mutex queue_mutex_;
        std::condition_variable queue_cv_;

        // 4. 双流相关
        cudaStream_t stream_backbone_ = nullptr;
        cudaStream_t stream_head_ = nullptr;
        cudaEvent_t event_prev_cache_done_ = nullptr; // 上一帧Cache完成事件（用于InstanceBank依赖）

        // 5. 线程句柄 (Step 2 使用)
        std::thread preprocess_thread_;
        bool stop_flag_ = false;
        
        // 内部辅助函数
        std::shared_ptr<FrameContext> get_free_context();
        void release_context(std::shared_ptr<FrameContext> context);
};
    
}  // namespace core
}  // namespace sparse4d
#endif // __SPARSE4D_HPP__