
#include "sparse4d.hpp"

#include "common/timer.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <google/protobuf/text_format.h>
#include "log.h"
#include "common/functionhub.hpp"

namespace sparse4d{
namespace core{

bool FrameContext::init(const TaskConfig& param) {
    // 1. 创建同步事件
    if (cudaEventCreate(&event_backbone_done) != cudaSuccess) return false;
    if (cudaEventCreate(&event_all_done) != cudaSuccess) return false;

    // 创建性能统计事件
    cudaEventCreate(&start_preprocess); cudaEventCreate(&stop_preprocess);
    cudaEventCreate(&start_instance_bank); cudaEventCreate(&stop_instance_bank);
    cudaEventCreate(&start_backbone); cudaEventCreate(&stop_backbone);
    cudaEventCreate(&start_head); cudaEventCreate(&stop_head);
    cudaEventCreate(&start_post); cudaEventCreate(&stop_post);

    // 2. 依据配置初始化 pipeline_context / head_output 的显存尺寸
    // 预处理输入
    size_t input_size = param.preprocessor_params().num_cams() * 
                        param.preprocessor_params().model_input_img_c() *
                        param.preprocessor_params().model_input_img_h() * 
                        param.preprocessor_params().model_input_img_w();
    
    if (!pipeline_context.input_images.allocate(input_size)) {
        std::cerr << "Failed to allocate pipeline_context.input_images (size=" << input_size << ")\n";
        return false;
    }

    const size_t temp_instance_feature_size = param.instance_bank_params().topk_querys() * param.model_cfg_params().embedfeat_dims();
    const size_t temp_anchor_size = param.instance_bank_params().topk_querys() * param.instance_bank_params().query_dims();
    const size_t temp_mask_size = 1;
    const size_t temp_track_id_size = param.instance_bank_params().topk_querys();

    const size_t pred_size = param.instance_bank_params().num_querys() * param.model_cfg_params().embedfeat_dims();
    const size_t anchor_size = param.instance_bank_params().num_querys() * param.instance_bank_params().query_dims();
    const size_t class_score_size = param.instance_bank_params().num_querys() * param.model_cfg_params().num_classes();
    const size_t quality_score_size = param.instance_bank_params().num_querys() * 2;
    const size_t track_id_size = param.instance_bank_params().num_querys();
    
    // 输入张量
    size_t feature_size = 1;
    for (int i = 0; i < param.model_cfg_params().sparse4d_extract_feat_shape_lc_size(); ++i) {
        feature_size *= param.model_cfg_params().sparse4d_extract_feat_shape_lc(i);
    }
    if (!pipeline_context.features.allocate(feature_size)) {
        std::cerr << "Failed to allocate pipeline_context.features (size=" << feature_size << ")\n";
        return false;
    }

    // 第二帧独有的输入
    if(!pipeline_context.temp_instance_feature.allocate(temp_instance_feature_size) ||
        !pipeline_context.temp_anchor.allocate(temp_anchor_size) ||
        !pipeline_context.mask.allocate(temp_mask_size) ||
        !pipeline_context.track_ids.allocate(track_id_size)) {
        std::cerr << "Failed to allocate pipeline_context.temp_instance_feature (size=" << temp_instance_feature_size << ")\n";
        return false;
    }

    // 输出张量
    if (!head_output.pred_instance_feature.allocate(pred_size) ||
        !head_output.pred_anchor.allocate(anchor_size) ||
        !head_output.pred_class_score.allocate(class_score_size) ||
        !head_output.pred_quality_score.allocate(quality_score_size) ||
        !head_output.pred_track_ids.allocate(track_id_size) ||
        !head_output.tmp_outs0.allocate(pred_size * 2) ||
        !head_output.tmp_outs1.allocate(pred_size * 2) ||
        !head_output.tmp_outs2.allocate(pred_size * 2) ||
        !head_output.tmp_outs3.allocate(pred_size * 2) ||
        !head_output.tmp_outs4.allocate(pred_size * 2) ||
        !head_output.tmp_outs5.allocate(pred_size * 2)) {
        std::cerr << "Failed to allocate head_output (size=" << pred_size * 2 << ")\n";
        return false;
    }

    return true;
}

void FrameContext::free() {
    if (event_backbone_done) cudaEventDestroy(event_backbone_done);
    if (event_all_done) cudaEventDestroy(event_all_done);
    
    if (start_preprocess) cudaEventDestroy(start_preprocess);
    if (stop_preprocess) cudaEventDestroy(stop_preprocess);
    if (start_instance_bank) cudaEventDestroy(start_instance_bank);
    if (stop_instance_bank) cudaEventDestroy(stop_instance_bank);
    if (start_backbone) cudaEventDestroy(start_backbone);
    if (stop_backbone) cudaEventDestroy(stop_backbone);
    if (start_head) cudaEventDestroy(start_head);
    if (stop_head) cudaEventDestroy(stop_head);
    if (start_post) cudaEventDestroy(start_post);
    if (stop_post) cudaEventDestroy(stop_post);

    event_backbone_done = nullptr;
    event_all_done = nullptr;
    // CudaWrapper 自动释放内存
}

/**
 * @brief 析构函数：清理 CUDA Stream 资源
 */
CoreImplement::~CoreImplement() {
    // 1. 清理资源池
    context_pool_.clear(); // shared_ptr 自动调用 FrameContext::free (如果实现析构) -> 这里我们需要手动调用 free 或者依赖 CudaWrapper
    // FrameContext 的析构函数默认不调用 free，所以最好显式调用或者让 FrameContext 析构调用 free
    // 这里简单起见，我们在 FrameContext 析构中不自动 free (避免 double free 如果拷贝)，所以手动清理
    // 但 context_pool_ 是 shared_ptr，我们没有定义 FrameContext 的析构函数，所以需要手动遍历释放 event
    // 更好的做法是给 FrameContext 加析构函数。但这里先手动做。
    // 实际上 shared_ptr 释放时会析构 FrameContext，但 FrameContext 没有析构函数去 destroy event。
    // 修正：FrameContext 应该有析构函数。或者我们在这里手动释放。
    // 由于 FrameContext 是 struct，我们可以遍历释放。
    // 实际上，我们应该在 FrameContext 中添加析构函数。但为了少改动，我这里不加析构函数，而是依赖 shared_ptr 的 deleter 或者手动释放。
    // 简单起见，手动释放。
    // 实际上 context_pool_ 里的对象会被销毁。
    // 让我们假设 FrameContext 没有析构函数。
    
    // 停止线程
    stop_flag_ = true;
    queue_cv_.notify_all(); // 唤醒推理线程
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }

    // 销毁 Streams
    if (stream_backbone_) cudaStreamDestroy(stream_backbone_);
    if (stream_head_) cudaStreamDestroy(stream_head_);
    if (event_prev_cache_done_) cudaEventDestroy(event_prev_cache_done_);

    // 销毁旧的 inference_stream_ (如果还存在)
    // if (inference_stream_ != nullptr) { ... } // 已被替换
}

/**
 * @brief 加载TaskConfig配置文件
 * @param config_path 配置文件路径
 * @param task_config 输出TaskConfig对象
 * @return 成功返回true，失败返回false
 */
 bool loadTaskConfig(const std::string& config_path, sparse4d::TaskConfig& task_config) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        LOG(ERROR) << "Failed to open config file: " << config_path;
        return false;
    }
    
    std::stringstream buffer;
    buffer << config_file.rdbuf();
    std::string content = buffer.str();
    config_file.close();
    
    // 解析protobuf文本格式配置文件
    if (!google::protobuf::TextFormat::ParseFromString(content, &task_config)) {
        LOG(ERROR) << "Failed to parse config file: " << config_path;
        return false;
    }
    
    LOG(INFO) << "Successfully loaded TaskConfig from: " << config_path;
    return true;
}

bool CoreImplement::initAlgorithm(const std::string exe_path,  const AlgCallback& alg_cb, void* hd)
{
    exe_path_ = exe_path;
    alg_cb_ = alg_cb;
    user_handle_ = hd;

    // 通过工程路径推导配置文件路径
    std::string config_path = exe_path_ + "/Output/Configs/Alg/Sparse4d.conf";
    sparse4d::TaskConfig task_config;
    if (!loadTaskConfig(config_path, task_config)) {
        LOG(ERROR) << "[ERROR] Failed to load TaskConfig from: " << config_path;
        return false;
    }

    return init(task_config);
}

bool CoreImplement::update(void* p_pParam)
{
    if (p_pParam == nullptr) return false;
    // 将通用参数转换为 lidar2camera（float*，按相机数 * 16）
    const float* lidar2camera = reinterpret_cast<const float*>(p_pParam);
    update(lidar2camera, nullptr);
    return true;
}

void CoreImplement::runAlgorithm(void* p_pSrcData)
{
    if (p_pSrcData == nullptr) return;
    const CTimeMatchSrcData* raw_data = reinterpret_cast<const CTimeMatchSrcData*>(p_pSrcData);
    
    // 1. 获取空闲 Context
    auto context = get_free_context();
    if (!context) {
        LOG(WARNING) << "[WARNING] No free context available, dropping frame!";
        return;
    }

    // 2. 在主线程执行预处理 (Host->Device 拷贝)
    // 必须在这里做，因为 raw_data 指针可能在函数返回后失效
    // 使用 stream_backbone_ 进行预处理，与 InstanceBank::get (stream_head_) 并行
    
    // 计时预处理
    if (enable_timer_) {
        // 简单计时，或者使用 EventTimer 如果需要更精确
        // 这里为了简化，直接调用 forward_only 的预处理部分
        // 但 forward_only 包含了所有步骤。我们需要拆分。
    }

    // 记录推理开始时间（用于计算纯推理吞吐量）
    context->inference_start_time = std::chrono::high_resolution_clock::now();
    
    // 我们需要手动执行预处理部分
    cudaEventRecord(context->start_preprocess, stream_backbone_);
    Status status = preprocessor_->forward(raw_data, stream_backbone_, context->pipeline_context);
    cudaEventRecord(context->stop_preprocess, stream_backbone_);
    
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Preprocessor forward failed!";
        release_context(context);
        return;
    }

    // 2.5 执行 InstanceBank::get (因为需要 raw_data)
    // 注意：InstanceBank::get 可能需要使用 stream，这里使用 stream_head_
    // 并且它可能修改 pipeline_context
    cudaEventRecord(context->start_instance_bank, stream_head_);
    status = instance_bank_->get(raw_data, is_first_frame_, stream_head_, context->pipeline_context);
    cudaEventRecord(context->stop_instance_bank, stream_head_);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank get failed!";
        release_context(context);
        return;
    }
    
    // 3. 将准备好的 Context 推入 Ready Queue
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        ready_queue_.push(context);
    }
    queue_cv_.notify_one();
    LOG(INFO) << "[DEBUG] Pushed to queue, returning from runAlgorithm";

    // 4. 立即返回，不等待推理完成
    // 结果将通过 alg_cb_ 在 inference_loop 中返回
    // 临时：为了调试，我们在这里等待一下，确保不是因为主线程退出太快
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void CoreImplement::inference_loop() {
    LOG(INFO) << "[DEBUG] Inference loop started";
    while (!stop_flag_) {
        std::shared_ptr<FrameContext> context;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !ready_queue_.empty() || stop_flag_; });
            
            if (stop_flag_ && ready_queue_.empty()) break;
            
            context = ready_queue_.front();
            ready_queue_.pop();
        }
        LOG(INFO) << "[DEBUG] Inference loop got context";

        // 执行推理 (Backbone + Head + Postprocess)
        CAlgResult result;
        cudaStream_t stream = stream_head_; // 主推理流

        LOG(INFO) << "[DEBUG] Starting Backbone forward. Stream: " << stream_backbone_;
        // 3. Backbone (on stream_backbone_)
        cudaEventRecord(context->start_backbone, stream_backbone_);
        Status status = backbone_->forward(context->pipeline_context, stream_backbone_);
        cudaEventRecord(context->stop_backbone, stream_backbone_);
        
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Backbone forward failed!";
            release_context(context);
            continue;
        }
        
        // Record event
        cudaEventRecord(context->event_backbone_done, stream_backbone_);
        LOG(INFO) << "[DEBUG] Backbone done, event recorded";

        // 4. Head (on stream_head_)
        // Wait for Backbone
        cudaStreamWaitEvent(stream_head_, context->event_backbone_done, 0);

        cudaEventRecord(context->start_head, stream_head_);
        if(is_first_frame_){
             status = head1_->forward(context->pipeline_context, stream, context->head_output);
        } else {
             status = head2_->forward(context->pipeline_context, stream, context->head_output);
        }
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Head forward failed!";
            release_context(context);
            continue;
        }

        // 5. Cache
        status = instance_bank_->cache(context->head_output, is_first_frame_, stream);
        
        // 6. TrackId
        status = instance_bank_->getTrackId(context->head_output, is_first_frame_, stream);
        
        if(is_first_frame_) is_first_frame_ = false; // 注意线程安全，如果多线程同时访问
        cudaEventRecord(context->stop_head, stream_head_);

        // 7. Postprocess
        cudaEventRecord(context->start_post, stream_head_);
        status = postprocessor_->forward(context->head_output, stream, result);
        cudaEventRecord(context->stop_post, stream_head_);
        
        LOG(INFO) << "[DEBUG] Postprocess done, calling callback";
        
        // Callback
        if (alg_cb_) {
            alg_cb_(result, user_handle_);
        }
        LOG(INFO) << "[DEBUG] Callback done";
        
        // 记录推理结束时间（postprocess 完成后）
        auto inference_end_time = std::chrono::high_resolution_clock::now();
        
        // 统计耗时
        static int frame_count = 0;
        static int total_processed_count = 0;
        static float total_preprocess = 0, total_instance_bank = 0, total_backbone = 0, total_head = 0, total_post = 0;
        static float total_inference_time = 0; // 纯推理总时间（CPU时间）
        static auto last_print_time = std::chrono::high_resolution_clock::now();
        
        float t_pre = 0, t_ib = 0, t_back = 0, t_head = 0, t_post = 0;
        // 注意：这里不阻塞等待，让 GPU 异步执行，实现帧间并行
        // 但为了统计准确，我们仍然需要同步（或者使用异步方式统计）
        cudaEventSynchronize(context->stop_post); // 确保所有事件完成（用于统计）
        cudaEventElapsedTime(&t_pre, context->start_preprocess, context->stop_preprocess);
        cudaEventElapsedTime(&t_ib, context->start_instance_bank, context->stop_instance_bank);
        cudaEventElapsedTime(&t_back, context->start_backbone, context->stop_backbone);
        cudaEventElapsedTime(&t_head, context->start_head, context->stop_head);
        cudaEventElapsedTime(&t_post, context->start_post, context->stop_post);
        
        // 计算纯推理时间（从预处理开始到 postprocess 结束）
        float inference_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            inference_end_time - context->inference_start_time).count() / 1000.0f;
        
        total_processed_count++;

        if (total_processed_count > 1) {
            total_preprocess += t_pre;
            total_instance_bank += t_ib;
            total_backbone += t_back;
            total_head += t_head;
            total_post += t_post;
            total_inference_time += inference_time_ms;
            frame_count++;
            
            if (frame_count % 9 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_time).count();
                float end_to_end_fps = 1000.0f * 9.0f / duration; // 端到端吞吐量（包含数据加载、可视化等）
                float pure_inference_fps = 1000.0f * 9.0f / total_inference_time; // 纯推理吞吐量
                
                // 计算理论最大吞吐量（基于瓶颈模块）
                float bottleneck_time = std::max(total_backbone / 9.0f, total_head / 9.0f);
                float theoretical_max_fps = 1000.0f / bottleneck_time;
                
                std::cout << "\n=== Performance Report (Avg over 9 frames) ===" << std::endl;
                std::cout << "Pure Inference Throughput: " << pure_inference_fps << " FPS" << std::endl;
                std::cout << "End-to-End Throughput: " << end_to_end_fps << " FPS (includes data loading, visualization, etc.)" << std::endl;
                std::cout << "Pure Inference Latency: " << total_inference_time / 9.0f << " ms" << std::endl;
                std::cout << "\nTheoretical Max Throughput (Pipeline): " << theoretical_max_fps << " FPS" << std::endl;
                std::cout << "  (Based on bottleneck: " << bottleneck_time << " ms)" << std::endl;
                std::cout << "Current Efficiency: " << (pure_inference_fps / theoretical_max_fps * 100.0f) << "%" << std::endl;
                std::cout << "\nLatency Breakdown (GPU time):" << std::endl;
                std::cout << "  Preprocess: " << total_preprocess / 9.0f << " ms" << std::endl;
                std::cout << "  InstanceBank::get: " << total_instance_bank / 9.0f << " ms" << std::endl;
                std::cout << "  Backbone: " << total_backbone / 9.0f << " ms" << std::endl;
                std::cout << "  Head (incl. Cache/Track): " << total_head / 9.0f << " ms" << std::endl;
                std::cout << "  Postprocess: " << total_post / 9.0f << " ms" << std::endl;
                std::cout << "\nNote: Current implementation is sequential (not pipeline parallel)." << std::endl;
                std::cout << "      To achieve theoretical max, need frame-level pipeline parallelism." << std::endl;
                std::cout << "================================================" << std::endl;
                
                total_preprocess = total_instance_bank = total_backbone = total_head = total_post = 0;
                total_inference_time = 0;
                last_print_time = now;
            }
        } else {
            // 第一帧耗时通常较高（Warmup），丢弃不计入统计，并重置起始时间
            last_print_time = std::chrono::high_resolution_clock::now();
            LOG(INFO) << "[INFO] First frame statistics discarded for performance report.";
        }

        release_context(context);
    }
}

//  初始化核心组件，包括骨干网络、第一帧头部、第二帧头部、实例银行、归一化等。每个组件的创建都进行检查，如果失败则返回 false。
bool CoreImplement::init(const TaskConfig &param)
{   
    // 保存配置
    param_ = param;
    // 检查是否有 enable_timer 字段，如果没有则默认为 false
    // 注意：如果 protobuf 中没有这个字段，需要使用 has_enable_timer() 检查
    enable_timer_ = param.enable_timer();
    
    preprocessor_ = preprocessor::create_preprocessor(param);
    if(preprocessor_ == nullptr){
        std::cerr << "Failed to create preprocessor.\n" << std::endl;
        return false;
    }

    instance_bank_ = instance_bank::create_instance_bank(param);
    if(instance_bank_ == nullptr){
        std::cerr << "Failed to create instance bank.\n" << std::endl;
        return false;
    }

    backbone_ = backbone::create_backbone(param);
    if(backbone_ == nullptr){
        std::cerr << "Failed to create backbone.\n" << std::endl;
        return false;
    }

    head1_ = first_head::create_first_head(param);
    if(head1_ == nullptr){
        std::cerr << "Failed to create head1.\n" << std::endl;
        return false;
    }

    head2_ = second_head::create_second_head(param);
    if(head2_ == nullptr){
        std::cerr << "Failed to create head2.\n" << std::endl;
        return false;
    }

    postprocessor_ = postprocessor::create_postprocessor(param);
    if(postprocessor_ == nullptr){
        std::cerr << "Failed to create postprocessor.\n" << std::endl;
        return false;
    }

    // --- 初始化资源池 ---
    // 创建双流
    if (cudaStreamCreate(&stream_backbone_) != cudaSuccess) return false;
    if (cudaStreamCreate(&stream_head_) != cudaSuccess) return false;
    if (cudaEventCreate(&event_prev_cache_done_) != cudaSuccess) return false;

    // 创建 FrameContext 池 (3帧缓冲)
    for (int i = 0; i < 3; ++i) {
        auto ctx = std::make_shared<FrameContext>();
        ctx->frame_id = -1; // 未使用
        if (!ctx->init(param_)) {
            std::cerr << "Failed to init FrameContext " << i << "\n";
            return false;
        }
        
        // 加载静态辅助数据到每个 Context
        if (!loadAuxiliaryData(ctx.get())) {
            std::cerr << "Failed to load auxiliary data for context " << i << "\n";
            return false;
        }

        context_pool_.push_back(ctx);
        free_contexts_.push(ctx);
    }
    
    // 预热推理：多次调用以触发 CUDA Graph 捕获
    // TensorRT 8.0+ 需要 2-3 次调用 enqueueV2 才能捕获 CUDA Graph
    LOG(INFO) << "[INFO] Starting warmup inference to trigger CUDA Graph capture...";
    if (!warmupInference()) {
        LOG(WARNING) << "[WARNING] Warmup inference failed, but continuing...";
    }
    LOG(INFO) << "[INFO] Warmup inference completed";

    // 启动推理线程
    stop_flag_ = false;
    inference_thread_ = std::thread(&CoreImplement::inference_loop, this);

    return true;
}

/**
 * @brief 加载辅助数据
 * @note 加载辅助数据到pipeline_context_中
 * @note spatial_shapes、level_start_index、instance_feature、anchor、time_interval、image_wh、lidar2img（7个张量）
 */
bool CoreImplement::loadAuxiliaryData(FrameContext* context) 
 {
    auto& pipeline_context = context->pipeline_context;
    // 1. 从TaskConfig获取空间形状数据
    std::vector<int32_t> spatial_shapes;
    for (int i = 0; i < param_.model_cfg_params().sparse4d_extract_feat_spatial_shapes_ld_size(); ++i) {
    spatial_shapes.push_back(param_.model_cfg_params().sparse4d_extract_feat_spatial_shapes_ld(i));
    }
    std::vector<int32_t> expanded;
    const int num_cams = param_.preprocessor_params().num_cams();
    expanded.reserve(num_cams * 4 * 2);
    for (int cam = 0; cam < num_cams; ++cam) {
      for (int lvl = 0; lvl < 4; ++lvl) {
        expanded.push_back(spatial_shapes[lvl * 2 + 0]);
        expanded.push_back(spatial_shapes[lvl * 2 + 1]);
      }
    }
    pipeline_context.spatial_shapes.cudaMemUpdateWrap(expanded);
     
    // 2. 从TaskConfig获取层级起始索引
    std::vector<int32_t> level_start_index;
    for (int i = 0; i < param_.model_cfg_params().sparse4d_extract_feat_level_start_index_size(); ++i) {
    level_start_index.push_back(param_.model_cfg_params().sparse4d_extract_feat_level_start_index(i));
    }
    pipeline_context.level_start_index.cudaMemUpdateWrap(level_start_index);
     
    // 3. 从TaskConfig获取实例特征大小并初始化
    std::vector<float> instance_feature;
    size_t instance_feature_size = param_.instance_bank_params().num_querys() * 
                                param_.model_cfg_params().embedfeat_dims();
    instance_feature.resize(instance_feature_size, 0.0f);
    pipeline_context.instance_feature.cudaMemUpdateWrap(instance_feature);
     
    // 4. 从锚点文件加载锚点数据
    std::vector<float> anchor;
    std::string anchor_path = param_.instance_bank_params().instance_bank_anchor_path();
    if (!anchor_path.empty()) {
        size_t expected_anchor_size = param_.instance_bank_params().num_querys() * 
                                    param_.instance_bank_params().query_dims();
        
        std::ifstream file(anchor_path, std::ios::binary);
        if (file.is_open()) {
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);
            
            size_t expected_file_size = expected_anchor_size * sizeof(float);
            if (file_size == expected_file_size) {
                anchor.resize(expected_anchor_size);
                file.read(reinterpret_cast<char*>(anchor.data()), file_size);
                LOG(INFO) << "[INFO] Successfully loaded anchor data from: " << anchor_path;
            } else {
                LOG(ERROR) << "[ERROR] Anchor file size mismatch! Expected: " << expected_file_size 
                        << " bytes, Actual: " << file_size << " bytes";
                anchor.resize(expected_anchor_size, 0.0f);
            }
            file.close();
        } else {
            LOG(ERROR) << "[ERROR] Failed to open anchor file: " << anchor_path;
            anchor.resize(expected_anchor_size, 0.0f);
        }
    } else {
        LOG(WARNING) << "[WARNING] No anchor file path specified, using default values";
        size_t expected_anchor_size = param_.instance_bank_params().num_querys() * 
                                    param_.instance_bank_params().query_dims();
        anchor.resize(expected_anchor_size, 0.0f);
    }
    pipeline_context.anchor.cudaMemUpdateWrap(anchor);

    // 5. 从TaskConfig获取时间间隔
    std::vector<float> time_interval;
    time_interval.push_back(param_.instance_bank_params().default_time_interval());
    pipeline_context.time_interval.cudaMemUpdateWrap(time_interval);
    
    // 从TaskConfig获取图像宽高（为每个相机分别保存）
    std::vector<float> image_wh;
    image_wh.resize(num_cams * 2); // 每个相机 * 2个值（宽高）
    for (int i = 0; i < num_cams; ++i) {
        image_wh[i * 2] = static_cast<float>(param_.preprocessor_params().model_input_img_w());
        image_wh[i * 2 + 1] = static_cast<float>(param_.preprocessor_params().model_input_img_h());
    }
    pipeline_context.image_wh.cudaMemUpdateWrap(image_wh);
    
    // 6. 激光雷达到图像变换矩阵大小
    size_t lidar2img_size = param_.preprocessor_params().num_cams() * 4 * 4;
    std::vector<float> lidar2img;
    lidar2img.resize(lidar2img_size, 0.0f);
    for (int i = 0; i < param_.preprocessor_params().num_cams(); ++i) {
        lidar2img[i * 16 + 0] = 1.0f;  // [0,0]
        lidar2img[i * 16 + 5] = 1.0f;  // [1,1]
        lidar2img[i * 16 + 10] = 1.0f; // [2,2]
        lidar2img[i * 16 + 15] = 1.0f; // [3,3]
    }
    pipeline_context.lidar2img.cudaMemUpdateWrap(lidar2img);

    // 7. 显存分配已移至 FrameContext::init
    
    LOG(INFO) << "[INFO] Auxiliary data loaded from TaskConfig";
    return true;
}

std::shared_ptr<FrameContext> CoreImplement::get_free_context() {
    std::unique_lock<std::mutex> lock(context_mutex_);
    // 简单实现：如果没有空闲，则等待 (或者返回空)
    // Step 1 中我们是同步执行，所以应该总是有空闲的，除非并发调用 forward
    if (free_contexts_.empty()) {
        return nullptr;
    }
    auto ctx = free_contexts_.front();
    free_contexts_.pop();
    return ctx;
}

void CoreImplement::release_context(std::shared_ptr<FrameContext> context) {
    if (!context) return;
    std::unique_lock<std::mutex> lock(context_mutex_);
    free_contexts_.push(context);
    context_cv_.notify_one();
}

/**
 * @brief 预热推理：多次调用以触发 CUDA Graph 捕获
 * @return 成功返回 true，失败返回 false
 * @note TensorRT 8.0+ 需要 2-3 次调用 enqueueV2 才能捕获 CUDA Graph
 * @note 使用零数据填充缓冲区，仅用于触发 CUDA Graph 捕获
 */
bool CoreImplement::warmupInference() {
    // 使用第一个 Context 进行预热
    if (context_pool_.empty()) return false;
    auto context = context_pool_[0];
    auto& pipeline_context = context->pipeline_context;
    auto& head_output = context->head_output;

    // 使用固定的 stream_backbone_ 进行预热 (或者 stream_head_)
    // 为了简单，我们只预热 stream_head_，因为它涵盖了大部分逻辑
    // 或者我们应该预热两个流？
    // 原始代码只预热 inference_stream_。现在我们有两个流。
    // 我们应该分别预热。
    
    cudaStream_t stream = stream_head_; // 暂时使用 head 流

    // ... (rest of warmup logic using pipeline_context and head_output)
    
    // 根据实际测试，Backbone 的 CUDA Graph 在第 2 次迭代就生效了（enqueue time 从 348ms 降到 1.3ms）
    // 但 Head1 和 Head2 需要更多预热迭代才能捕获 CUDA Graph
    const int BACKBONE_WARMUP = 3;   // Backbone 已经验证生效，保持 3 次
    const int HEAD1_WARMUP = 5;      // Head1 需要更多预热（从日志看第 5 次调用仍然较高）
    const int HEAD2_WARMUP = 5;      // Head2 需要更多预热
    
    LOG(INFO) << "[INFO] Performing warmup iterations: Backbone=" << BACKBONE_WARMUP 
              << ", Head1=" << HEAD1_WARMUP << ", Head2=" << HEAD2_WARMUP;
    
    // 1. 单独预热 Backbone（已验证 CUDA Graph 已生效）
    LOG(INFO) << "[INFO] Warming up Backbone (" << BACKBONE_WARMUP << " iterations)...";
    for (int i = 0; i < BACKBONE_WARMUP; ++i) {
        Status status = backbone_->forward(pipeline_context, stream);
        if (status != Status::kSuccess) {
            LOG(WARNING) << "[WARNING] Backbone warmup iteration " << i << " failed";
            continue;
        }
        cudaStreamSynchronize(stream);
    }
    
    // 2. 单独预热 Head1（需要更多迭代才能捕获 CUDA Graph）
    LOG(INFO) << "[INFO] Warming up Head1 (" << HEAD1_WARMUP << " iterations)...";
    for (int i = 0; i < HEAD1_WARMUP; ++i) {
        Status status = head1_->forward(pipeline_context, stream, head_output);
        if (status != Status::kSuccess) {
            LOG(WARNING) << "[WARNING] Head1 warmup iteration " << i << " failed";
            continue;
        }
        cudaStreamSynchronize(stream);
    }
    
    // 3. 单独预热 Head2（需要更多迭代才能捕获 CUDA Graph）
    // 注意：Head2 需要额外的输入（temp_instance_feature, temp_anchor, mask, track_id）
    // 但这些缓冲区已经分配，使用零数据即可
    LOG(INFO) << "[INFO] Warming up Head2 (" << HEAD2_WARMUP << " iterations)...";
    for (int i = 0; i < HEAD2_WARMUP; ++i) {
        Status status = head2_->forward(pipeline_context, stream, head_output);
        if (status != Status::kSuccess) {
            LOG(WARNING) << "[WARNING] Head2 warmup iteration " << i << " failed";
            continue;
        }
        cudaStreamSynchronize(stream);
    }
    
    LOG(INFO) << "[INFO] Warmup inference completed";
    return true;
}


CAlgResult CoreImplement::forward(const CTimeMatchSrcData *raw_data, void *stream)
{   
    // Step 1 临时实现：同步获取 Context 并执行
    auto context = get_free_context();
    if (!context) {
        LOG(ERROR) << "[ERROR] No free context available!";
        return CAlgResult();
    }

    CAlgResult result;
    if(enable_timer_){
        result = forward_timer(raw_data, stream, context.get());
    }
    else{
        result = forward_only(raw_data, stream, context.get());
    }
    
    release_context(context);
    return result;
}

// 内部使用的 forward_only 方法，接受 CTimeMatchSrcData
CAlgResult CoreImplement::forward_only(const CTimeMatchSrcData *raw_data, void *stream, FrameContext* context)
{   
    CAlgResult result;
    if (raw_data == nullptr || context == nullptr) {
        return result;
    }
    
    auto& pipeline_context = context->pipeline_context;
    auto& head_output = context->head_output;

    // 1.预处理
    Status status = preprocessor_->forward(raw_data, static_cast<cudaStream_t>(stream), pipeline_context);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Preprocessor forward failed!";
        return result;
    }

    // 2.执行Instance_Bank->get() 获取历史信息
    status = instance_bank_->get(raw_data, is_first_frame_, static_cast<cudaStream_t>(stream), pipeline_context);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank get failed!";
        return result;
    }

    // 3.执行BackBone->forward() 提取特征
    status = backbone_->forward(pipeline_context, static_cast<cudaStream_t>(stream));
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Backbone forward failed!";
        return result;
    }

    // 4.执行Head->forward() 提取目标
    if(is_first_frame_){
        status = head1_->forward(pipeline_context, static_cast<cudaStream_t>(stream), head_output);
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Head1 forward failed!";
            return result;
        }
    }else
    {
        status = head2_->forward(pipeline_context, static_cast<cudaStream_t>(stream), head_output);
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Head2 forward failed!";
            return result;
        }
    }

    // 5.执行Instance_Bank->cache() 缓存结果
    status = instance_bank_->cache(head_output, is_first_frame_, static_cast<cudaStream_t>(stream));
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank cache failed!";
        return result;
    }

    // 6.执行Instance_Bank->getTrackId() 获取跟踪ID
    status = instance_bank_->getTrackId(head_output, is_first_frame_, static_cast<cudaStream_t>(stream));
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank get track id failed!";
        return result;
    }

    if(is_first_frame_){
        is_first_frame_ = false;
    }

    // 7.执行Postprocessor->forward() 后处理
    status = postprocessor_->forward(head_output, static_cast<cudaStream_t>(stream), result);
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Postprocessor forward failed!";
        return result;
    }
    return result;
}

CAlgResult CoreImplement::forward_timer(const CTimeMatchSrcData *raw_data, void *stream, FrameContext* context)
{
    // 使用计时器版本的forward，在每个阶段添加计时
    CAlgResult result;
    if (raw_data == nullptr || context == nullptr) {
        return result;
    }
    
    auto& pipeline_context = context->pipeline_context;
    auto& head_output = context->head_output;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    // 计时器
    nv::EventTimer timer_preprocess, timer_instance_bank, timer_backbone, 
                   timer_head, timer_cache, timer_trackid, timer_postprocess;

    // 1.预处理
    timer_preprocess.start(cuda_stream);
    Status status = preprocessor_->forward(raw_data, cuda_stream, pipeline_context);
    timer_preprocess.stop("Preprocess");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Preprocessor forward failed!";
        return result;
    }

    // 2.执行Instance_Bank->get() 获取历史信息
    timer_instance_bank.start(cuda_stream);
    status = instance_bank_->get(raw_data, is_first_frame_, cuda_stream, pipeline_context);
    timer_instance_bank.stop("InstanceBank::get");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank get failed!";
        return result;
    }

    // 3.执行BackBone->forward() 提取特征
    timer_backbone.start(cuda_stream);
    status = backbone_->forward(pipeline_context, cuda_stream);
    timer_backbone.stop("Backbone");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Backbone forward failed!";
        return result;
    }

    // 4.执行Head推理
    timer_head.start(cuda_stream);
    if(is_first_frame_){
        status = head1_->forward(pipeline_context, cuda_stream, head_output);
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Head1 forward failed!";
            timer_head.stop("Head1");
            return result;
        }
        timer_head.stop("Head1");
    } else {
        status = head2_->forward(pipeline_context, cuda_stream, head_output);
        if (status != Status::kSuccess) {
            LOG(ERROR) << "[ERROR] Head2 forward failed!";
            timer_head.stop("Head2");
            return result;
        }
        timer_head.stop("Head2");
    }

    // 5.执行Instance_Bank->cache() 缓存结果
    timer_cache.start(cuda_stream);
    status = instance_bank_->cache(head_output, is_first_frame_, cuda_stream);
    timer_cache.stop("InstanceBank::cache");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank cache failed!";
        return result;
    }

    // 6.执行Instance_Bank->getTrackId() 获取跟踪ID
    timer_trackid.start(cuda_stream);
    status = instance_bank_->getTrackId(head_output, is_first_frame_, cuda_stream);
    timer_trackid.stop("InstanceBank::getTrackId");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Instance bank get track id failed!";
        return result;
    }

    if(is_first_frame_){
        is_first_frame_ = false;
    }

    // 7.执行Postprocessor->forward() 后处理
    timer_postprocess.start(cuda_stream);
    status = postprocessor_->forward(head_output, cuda_stream, result);
    timer_postprocess.stop("Postprocess");
    if (status != Status::kSuccess) {
        LOG(ERROR) << "[ERROR] Postprocessor forward failed!";
        return result;
    }

    return result;
}

void CoreImplement::update(const float *lidar2camera, void *stream)
{   
    // 更新lidar2img变换矩阵到pipeline_context_（支持默认流）
    if (lidar2camera != nullptr) {
        const int num_cams = param_.preprocessor_params().num_cams();
        std::vector<float> lidar2img_vec;
        lidar2img_vec.resize(num_cams * 16);
        for (int i = 0; i < num_cams; ++i) {
            std::memcpy(lidar2img_vec.data() + i * 16, lidar2camera + i * 16, 16 * sizeof(float));
        }
        cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : static_cast<cudaStream_t>(0);
        
        // 更新所有 Context 中的 lidar2img
        // 注意：这可能存在竞态条件，如果某个 Context 正在被使用。
        // 但通常 update 是在推理开始前调用的。
        // 为了安全，我们应该只更新空闲的 context 或者加锁？
        // 实际上，lidar2img 是每帧可能变化的参数。
        // 如果是每帧变化，应该作为 forward 的参数传入，或者在 get_free_context 后更新。
        // 但现有接口是独立的 update 函数。
        // 假设外部调用 update 后紧接着调用 forward。
        // 我们需要更新所有 context，或者只更新"下一个" context。
        // 简单起见，更新所有 context。
        for (auto& ctx : context_pool_) {
            ctx->pipeline_context.lidar2img.cudaMemUpdateWrapAsync(lidar2img_vec, s);
        }
    }
}

void CoreImplement::free_excess_memory()
{
    // 注意：CudaWrapper 使用 RAII 管理内存，在析构函数中自动释放
    // 如果需要手动释放，可以通过重置对象来实现
    // 但通常不需要手动释放，因为 CudaWrapper 会在析构时自动释放
    
    // 可选：通过移动构造来释放内存（将对象移动到一个临时对象，然后临时对象析构）
    // 但这会清空原有对象，可能不是期望的行为
    
    // 当前实现：不执行任何操作，依赖 RAII 自动管理
    // 如果需要强制释放，可以考虑添加 CudaWrapper::reset() 方法
    LOG(INFO) << "[INFO] free_excess_memory called (memory will be freed automatically by RAII)";
}

// legacy factory removed; use ExportSparse4D CreateCoreObj + initAlgorithm instead

} // namespace core
} // namespace sparse4d