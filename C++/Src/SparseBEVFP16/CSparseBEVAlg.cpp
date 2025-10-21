/*******************************************************
 文件名：CSparseBEVAlg.cpp
 作者：sharkls
 描述：SparseBEV算法主类实现
 版本：v1.0
 日期：2025-06-18
 *******************************************************/

#include "CSparseBEVAlg.h"
#include "Preprocess/img_preprocessor.h"
#include "Inference/SparseBEV.h"
// #include "Postprocess/Postprocessor.h"
#include "../../../Include/Common/Factory/ModuleFactory.h"
#include <log.h>
#include <fstream>
#include <sstream>
#include <google/protobuf/text_format.h>
#include <iomanip>
#include <map>

// 加载指定路径的conf配置文件并将其反序列化（解析prototext文件）
bool SparseBEVConfig::loadFromFile(const std::string& path) 
{
    std::ifstream input(path);
    if (!input) {
        LOG(ERROR) << "Failed to open config file: " << path;
        return false;
    }
    std::stringstream buffer;
    buffer << input.rdbuf();
    std::string content = buffer.str();
    if (!google::protobuf::TextFormat::ParseFromString(content, &m_config)) {
        LOG(ERROR) << "Failed to parse protobuf config file: " << path;
        return false;
    }
    return true;
}

CSparseBEVAlg::CSparseBEVAlg(const std::string& exePath)
    : m_exePath(exePath)
    , m_pConfig(std::make_shared<SparseBEVConfig>())
    , m_currentInput(nullptr)
    , m_callbackHandle(nullptr)
    , current_sample_index_(0)
{
    // 初始化错误处理器
    // m_errorHandler = ErrorHandlerFactory::getInstance().createErrorHandler("SparseBEV");
}

CSparseBEVAlg::~CSparseBEVAlg()
{
    // 清理模块链
    m_moduleChain.clear();
    
    // 重置状态
    m_is_initialized = false;
    m_is_running = false;
    
    LOG(INFO) << "[INFO] CSparseBEVAlg destroyed successfully";
}

bool CSparseBEVAlg::initAlgorithm(const std::string exe_path, const AlgCallback& alg_cb, void* hd)
{   
    // 1. 检查参数
    if (exe_path.empty()) {
        LOG(ERROR) << "[ERROR] exe_path is empty";
        return false;
    }

    // 2. 保存回调函数和句柄
    m_algCallback = alg_cb;
    m_callbackHandle = hd;

    // 3. 构建配置文件路径
    std::filesystem::path exePath(m_exePath);
    LOG(INFO) << "m_exePath : " << exePath.parent_path().string();
    std::string configPath = (exePath.parent_path() / "Configs"/ "Alg" / "SparseBEV8_6Config.conf").string();

    // 4. 加载配置文件
    if (!loadConfig(configPath)) {
        LOG(ERROR) << "[ERROR] Failed to load config file: " + configPath;
        return false;
    }

    m_run_status = m_pConfig->getModuleConfig().task_config().run_status();

    // 5. 初始化模块
    if (!initModules()) {
        LOG(ERROR) << "[ERROR] Failed to initialize modules";
        return false;
    }
    
    LOG(INFO) << "CSparseBEVAlg::initAlgorithm status: successs ";
    return true;
}

void CSparseBEVAlg::runAlgorithm(void* p_pSrcData)
{
    if (!m_is_initialized.load()) {
        LOG(WARNING) << "[WARNING] Algorithm not initialized, skipping this frame";
        if (m_algCallback) {
            m_algCallback(m_currentOutput, m_callbackHandle);
        }
        return;
    }
    
    // 防止并发执行
    bool expected = false;
    if (!m_is_running.compare_exchange_strong(expected, true)) {
        LOG(WARNING) << "[WARNING] Algorithm is already running, skipping this frame";
        return;
    }
    
    LOG(INFO) << "CSparseBEVAlg::runAlgorithm status: start ";
    
    // 开始性能监控
    startTimer("total");
    
    // 0. 每次运行前重置结构体内容
    m_currentOutput = CAlgResult();
    m_currentOutput.eDataType(DATA_TYPE_BEV_RESULT);  // 设置为BEV结果类型
    m_currentOutput.eDataSourceType(DATA_SOURCE_ONLINE);  // 设置为在线数据源
    m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_BEGIN] = GetTimeStamp();

    // 1. 核验输入数据是否为空
    if (!p_pSrcData) {
        LOG(ERROR) << "[ERROR] Input data is null";
        m_is_running = false;
        if (m_algCallback) {
            m_algCallback(m_currentOutput, m_callbackHandle);
        }
        return;
    }

    // 2. 输入数据赋值
    m_currentInput = static_cast<CTimeMatchSrcData *>(p_pSrcData);
    
    // 3. 从输入数据中提取样本索引
    current_sample_index_ = static_cast<int>(m_currentInput->unFrameId());
    // LOG(INFO) << "[INFO] Processing sample " << current_sample_index_ << " (frame ID: " << m_currentInput->unFrameId() << ")";
    
    // 4. 执行模块链
    startTimer("execution");
    if (!executeModuleChain()) {
        LOG(ERROR) << "[ERROR] Failed to execute module chain";
        endTimer("execution");
        endTimer("total");
        m_is_running = false;
        if (m_algCallback) {
            m_algCallback(m_currentOutput, m_callbackHandle);
        }
        return;
    }
    endTimer("execution");

    // 5. 更新性能统计
    endTimer("total");
    updateMemoryUsage();
    // calculateDetailedDelays();  // 计算详细延时
    logPerformanceReport();
    // logDetailedPerformanceInfo();  // 记录详细性能信息

    // 6. 通过回调函数返回结果
    if (m_algCallback) {
        m_algCallback(m_currentOutput, m_callbackHandle);
    }
    
    m_is_running = false;
    LOG(INFO) << "CSparseBEVAlg::runAlgorithm status: success ";
}

// 加载配置文件
bool CSparseBEVAlg::loadConfig(const std::string& configPath)
{
    if (!m_pConfig->loadFromFile(configPath)) {
        LOG(ERROR) << "[ERROR] Failed to load config file: " + configPath;
        return false;
    }
    
    // 验证配置完整性
    if (!validateConfig()) {
        LOG(ERROR) << "[ERROR] Config validation failed";
        return false;
    }
    
    return true;
}

// 验证配置完整性
bool CSparseBEVAlg::validateConfig()
{
    const auto& config = m_pConfig->getModuleConfig();
    const auto& taskConfig = config.task_config();
    
    // 验证必要参数
    if (taskConfig.preprocessor_params().num_cams() <= 0) {
        LOG(ERROR) << "Invalid number of cameras: " << taskConfig.preprocessor_params().num_cams();
        return false;
    }
    
    if (taskConfig.model_cfg_params().embedfeat_dims() <= 0) {
        LOG(ERROR) << "Invalid embedding feature dimensions: " << taskConfig.model_cfg_params().embedfeat_dims();
        return false;
    }
    
    // 验证引擎路径
    if (taskConfig.extract_feat_engine().engine_path().empty()) {
        LOG(ERROR) << "Extract feature engine path is empty";
        return false;
    }
    
    if (taskConfig.head1st_engine().engine_path().empty()) {
        LOG(ERROR) << "Head1st engine path is empty";
        return false;
    }
    
    // 验证模块配置
    if (config.modules_config().modules_size() == 0) {
        LOG(ERROR) << "No modules configured";
        return false;
    }
    
    LOG(INFO) << "Config validation passed";
    return true;
}

// 初始化子模块
bool CSparseBEVAlg::initModules()
{   
    LOG(INFO) << "CSparseBEVAlg::initModules status: start ";
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    auto& taskConfig = m_pConfig->getModuleConfig().task_config();
    
    // 从TaskConfig中获取模块配置
    const auto& modules = m_pConfig->getModuleConfig().modules_config();
    m_moduleChain.clear();

    // 遍历所有模块，按顺序实例化
    for (const auto& mod : modules.modules()) {
        auto module = ModuleFactory::getInstance().createModule("SparseBEV", mod.name(), m_exePath);
        if (!module) {
            LOG(ERROR) << "[ERROR] Failed to create module: " + mod.name();
            return false;
        }
        
        // 传递可写指针
        if (!module->init((void*)&taskConfig)) {
            LOG(ERROR) << "[ERROR] Failed to initialize module: " + mod.name();
            return false;
        }
        
        m_moduleChain.push_back(module);
        LOG(INFO) << "[INFO] Module initialized successfully: " << mod.name();
    }
    
    m_is_initialized = true;
    LOG(INFO) << "CSparseBEVAlg::initModules status: success ";
    return true;
}

// 执行模块链
bool CSparseBEVAlg::executeModuleChain()
{
    void* currentData = static_cast<void *>(m_currentInput);

    for (auto& module : m_moduleChain) {
        std::string module_name = module->getModuleName();
        ModuleType module_type = module->getModuleType();
        
        // 设置当前样本索引
        module->setCurrentSampleIndex(current_sample_index_);
        
        // 根据模块类型设置开始时间戳
        switch (module_type) {
            case ModuleType::PRE_PROCESS:
                m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_PREPROCESS_BEGIN] = GetTimeStamp();
                // LOG(INFO) << "[INFO] BEV Preprocess module begin: " << m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_PREPROCESS_BEGIN];
                break;
            case ModuleType::INFERENCE:
                m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_INFERENCE_BEGIN] = GetTimeStamp();
                // LOG(INFO) << "[INFO] BEV Inference module begin: " << m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_INFERENCE_BEGIN];
                break;
            case ModuleType::POST_PROCESS:
                m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_POSTPROCESS_BEGIN] = GetTimeStamp();
                // LOG(INFO) << "[INFO] BEV Postprocess module begin: " << m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_POSTPROCESS_BEGIN];
                break;
            default:
                break;
        }
        
        // 开始模块计时
        startTimer(module_name);
        
        // 设置输入数据
        module->setInput(currentData);

        // 执行模块
        module->execute();
        currentData = module->getOutput();
        if (!currentData) {
            LOG(ERROR) << "[ERROR] Module execution failed: " + module_name;
            endTimer(module_name);
            return false;
        }
        
        // 结束模块计时并根据模块类型记录到相应的延时类型
        endTimer(module_name);
        
        // 根据模块类型设置结束时间戳
        switch (module_type) {
            case ModuleType::PRE_PROCESS:
                m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_PREPROCESS_END] = GetTimeStamp();
                // LOG(INFO) << "[INFO] BEV Preprocess module completed: " << m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_PREPROCESS_END];
                break;
            case ModuleType::INFERENCE:
                m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_INFERENCE_END] = GetTimeStamp();
                // LOG(INFO) << "[INFO] BEV Inference module completed: " << m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_INFERENCE_END];
                break;
            case ModuleType::POST_PROCESS:
                m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_POSTPROCESS_END] = GetTimeStamp();
                // LOG(INFO) << "[INFO] BEV Postprocess module completed: " << m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_POSTPROCESS_END];
                break;
            default:
                LOG(INFO) << "[INFO] Unknown module type completed: " << module_name;
                break;
        }
    }

    // 直接获取最终结果，避免多次转换
    if (auto* resultPtr = static_cast<CAlgResult*>(currentData)) {
        // 保存时间戳信息
        auto saved_timestamps = m_currentOutput.mapTimeStamp();
        auto saved_delays = m_currentOutput.mapDelay();
        auto saved_fps = m_currentOutput.mapFps();
        
        // 更新结果数据
        m_currentOutput = *resultPtr;
        
        // 恢复时间戳信息
        m_currentOutput.mapTimeStamp() = saved_timestamps;
        m_currentOutput.mapDelay() = saved_delays;
        m_currentOutput.mapFps() = saved_fps;
    } else if (auto* rawResultPtr = static_cast<RawInferenceResult*>(currentData)) {
        // 如果最后一个模块输出RawInferenceResult，直接转换
        convertRawResultToAlgResult(*rawResultPtr, m_currentOutput);
    } else {
        LOG(ERROR) << "[ERROR] Unknown output type from last module";
        return false;
    }
    
    // 耗时统计
    m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_END] = GetTimeStamp();
    m_currentOutput.mapDelay()[DELAY_TYPE_BEV] = GetTimeStamp() - m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_BEGIN];
    m_currentOutput.mapDelay()[DELAY_TYPE_BEV_PREPROCESS] = m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_PREPROCESS_END] - m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_PREPROCESS_BEGIN];
    m_currentOutput.mapDelay()[DELAY_TYPE_BEV_INFERENCE] = m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_INFERENCE_END] - m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_INFERENCE_BEGIN];
    m_currentOutput.mapDelay()[DELAY_TYPE_BEV_POSTPROCESS] = m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_POSTPROCESS_END] - m_currentOutput.mapTimeStamp()[TIMESTAMP_BEV_POSTPROCESS_BEGIN];   
    // 结果穿透
    if (!m_currentOutput.vecFrameResult().empty()) {
        m_currentOutput.lTimeStamp() = m_currentInput->vecVideoSrcData()[0].lTimeStamp();
        m_currentOutput.vecFrameResult()[0].lTimeStamp() = m_currentInput->vecVideoSrcData()[0].lTimeStamp();
        m_currentOutput.vecFrameResult()[0].eDataType(DATA_TYPE_BEV_RESULT);  // 设置为BEV结果类型
        m_currentOutput.vecFrameResult()[0].eDataSourceType(DATA_SOURCE_ONLINE);  // 设置为在线数据源
    }

    return true;
}

/**
 * @brief 将RawInferenceResult转换为CAlgResult
 * @param raw_result 原始推理结果
 * @param alg_result 算法结果输出
 */
void CSparseBEVAlg::convertRawResultToAlgResult(const RawInferenceResult& raw_result, CAlgResult& alg_result)
{
    // 创建帧结果
    CFrameResult frame_result;
    frame_result.lTimeStamp() = m_currentInput->vecVideoSrcData()[0].lTimeStamp();
    frame_result.eDataType(DATA_TYPE_BEV_RESULT);  // 设置为BEV结果类型
    frame_result.eDataSourceType(DATA_SOURCE_ONLINE);  // 设置为在线数据源
    
    // 从RawInferenceResult提取检测结果
    if (raw_result.hasValidResults()) {
        // 这里需要根据实际的推理结果格式进行转换
        // 暂时创建空的检测结果
        std::vector<CObjectResult> detections;
        
        // 如果有推理结果，进行后处理转换为CObjectResult
        // 这部分需要根据具体的后处理逻辑实现
        
        frame_result.vecObjectResult(detections);
    }
    
    // 设置帧结果
    std::vector<CFrameResult> frame_results = {frame_result};
    alg_result.vecFrameResult(frame_results);
    
    LOG(INFO) << "Converted RawInferenceResult to CAlgResult";
}

// 性能监控计时器 - 使用CDataBase结构
std::map<std::string, int64_t> g_timer_start_times;

void CSparseBEVAlg::startTimer(const std::string& stage) {
    g_timer_start_times[stage] = GetTimeStamp();
    
    // 记录阶段开始日志
    if (stage == "ImagePreprocessor") {
        LOG(INFO) << "[PERFORMANCE] BEV Preprocess stage started";
    } else if (stage == "SparseBEV") {
        LOG(INFO) << "[PERFORMANCE] BEV Inference stage started";
    } else if (stage == "PostProcessor") {
        LOG(INFO) << "[PERFORMANCE] BEV Postprocess stage started";
    } else if (stage == "total") {
        LOG(INFO) << "[PERFORMANCE] BEV Total pipeline started";
    }
}

void CSparseBEVAlg::endTimer(const std::string& stage) {
    auto it = g_timer_start_times.find(stage);
    if (it != g_timer_start_times.end()) {
        int64_t duration = GetTimeStamp() - it->second;
        
        // 将性能数据记录到CAlgResult中，根据模块名称映射到相应的延时类型
        if (stage == "ImagePreprocessor" || stage == "preprocess") {
            m_currentOutput.mapDelay()[DELAY_TYPE_BEV_PREPROCESS] = duration;
            LOG(INFO) << "[PERFORMANCE] BEV Preprocess stage completed in " << duration << " ms";
        } else if (stage == "SparseBEV" || stage == "inference") {
            m_currentOutput.mapDelay()[DELAY_TYPE_BEV_INFERENCE] = duration;
            LOG(INFO) << "[PERFORMANCE] BEV Inference stage completed in " << duration << " ms";
        } else if (stage == "PostProcessor" || stage == "postprocess") {
            m_currentOutput.mapDelay()[DELAY_TYPE_BEV_POSTPROCESS] = duration;
            LOG(INFO) << "[PERFORMANCE] BEV Postprocess stage completed in " << duration << " ms";
        } else if (stage == "total") {
            m_currentOutput.mapDelay()[DELAY_TYPE_BEV] = duration;
            LOG(INFO) << "[PERFORMANCE] BEV Total pipeline completed in " << duration << " ms";
        }
        
        g_timer_start_times.erase(it);
    }
}

void CSparseBEVAlg::updateMemoryUsage() {
    // 这里可以实现实际的内存使用统计
    // 暂时使用估算值
    // m_performance_metrics.memory_usage_mb = 512; // 示例值
}

void CSparseBEVAlg::logPerformanceReport() {
    // 计算FPS
    auto total_delay = m_currentOutput.mapDelay().find(DELAY_TYPE_BEV);
    if (total_delay != m_currentOutput.mapDelay().end() && total_delay->second > 0) {
        double fps = 1000.0 / total_delay->second;
        m_currentOutput.mapFps()[FPS_BEV] = fps;
        LOG(INFO) << "[PERFORMANCE] Current FPS: " << std::fixed << std::setprecision(2) << fps;
    }
    
    // 记录各阶段延时
    LOG(INFO) << "[PERFORMANCE] === BEV Performance Summary ===";
    for (const auto& delay : m_currentOutput.mapDelay()) {
        switch (delay.first) {
            case DELAY_TYPE_BEV_PREPROCESS:
                LOG(INFO) << "[PERFORMANCE] BEV Preprocess: " << delay.second << " ms";
                break;
            case DELAY_TYPE_BEV_INFERENCE:
                LOG(INFO) << "[PERFORMANCE] BEV Inference: " << delay.second << " ms";
                break;
            case DELAY_TYPE_BEV_POSTPROCESS:
                LOG(INFO) << "[PERFORMANCE] BEV Postprocess: " << delay.second << " ms";
                break;
            case DELAY_TYPE_BEV:
                LOG(INFO) << "[PERFORMANCE] BEV Total: " << delay.second << " ms";
                break;
            default:
                break;
        }
    }
    LOG(INFO) << "[PERFORMANCE] ================================";
}

// 计算各阶段详细延时
void CSparseBEVAlg::calculateDetailedDelays() {
    // 计算BEV预处理延时
    auto preprocess_begin = m_currentOutput.mapTimeStamp().find(TIMESTAMP_BEV_PREPROCESS_BEGIN);
    auto preprocess_end = m_currentOutput.mapTimeStamp().find(TIMESTAMP_BEV_PREPROCESS_END);
    if (preprocess_begin != m_currentOutput.mapTimeStamp().end() && 
        preprocess_end != m_currentOutput.mapTimeStamp().end()) {
        int64_t preprocess_delay = preprocess_end->second - preprocess_begin->second;
        m_currentOutput.mapDelay()[DELAY_TYPE_BEV_PREPROCESS] = preprocess_delay;
    }
    
    // 计算BEV推理延时
    auto inference_begin = m_currentOutput.mapTimeStamp().find(TIMESTAMP_BEV_INFERENCE_BEGIN);
    auto inference_end = m_currentOutput.mapTimeStamp().find(TIMESTAMP_BEV_INFERENCE_END);
    if (inference_begin != m_currentOutput.mapTimeStamp().end() && 
        inference_end != m_currentOutput.mapTimeStamp().end()) {
        int64_t inference_delay = inference_end->second - inference_begin->second;
        m_currentOutput.mapDelay()[DELAY_TYPE_BEV_INFERENCE] = inference_delay;
    }
    
    // 计算BEV后处理延时
    auto postprocess_begin = m_currentOutput.mapTimeStamp().find(TIMESTAMP_BEV_POSTPROCESS_BEGIN);
    auto postprocess_end = m_currentOutput.mapTimeStamp().find(TIMESTAMP_BEV_POSTPROCESS_END);
    if (postprocess_begin != m_currentOutput.mapTimeStamp().end() && 
        postprocess_end != m_currentOutput.mapTimeStamp().end()) {
        int64_t postprocess_delay = postprocess_end->second - postprocess_begin->second;
        m_currentOutput.mapDelay()[DELAY_TYPE_BEV_POSTPROCESS] = postprocess_delay;
    }
    
    // 计算BEV总延时
    auto bev_begin = m_currentOutput.mapTimeStamp().find(TIMESTAMP_BEV_BEGIN);
    auto bev_end = m_currentOutput.mapTimeStamp().find(TIMESTAMP_BEV_END);
    if (bev_begin != m_currentOutput.mapTimeStamp().end() && 
        bev_end != m_currentOutput.mapTimeStamp().end()) {
        int64_t bev_total_delay = bev_end->second - bev_begin->second;
        m_currentOutput.mapDelay()[DELAY_TYPE_BEV] = bev_total_delay;
    }
}

void CSparseBEVAlg::logDetailedPerformanceInfo() const {
    LOG(INFO) << "[DETAILED_PERFORMANCE] === BEV Detailed Performance Info ===";
    
    // // 记录各阶段时间戳
    // for (const auto& timestamp : m_currentOutput.mapTimeStamp()) {
    //     switch (timestamp.first) {
    //         case TIMESTAMP_BEV_BEGIN:
    //             LOG(INFO) << "[DETAILED_PERFORMANCE] BEV Begin: " << timestamp.second << " ms";
    //             break;
    //         case TIMESTAMP_BEV_PREPROCESS_BEGIN:
    //             LOG(INFO) << "[DETAILED_PERFORMANCE] BEV Preprocess Begin: " << timestamp.second << " ms";
    //             break;
    //         case TIMESTAMP_BEV_PREPROCESS_END:
    //             LOG(INFO) << "[DETAILED_PERFORMANCE] BEV Preprocess End: " << timestamp.second << " ms";
    //             break;
    //         case TIMESTAMP_BEV_INFERENCE_BEGIN:
    //             LOG(INFO) << "[DETAILED_PERFORMANCE] BEV Inference Begin: " << timestamp.second << " ms";
    //             break;
    //         case TIMESTAMP_BEV_INFERENCE_END:
    //             LOG(INFO) << "[DETAILED_PERFORMANCE] BEV Inference End: " << timestamp.second << " ms";
    //             break;
    //         case TIMESTAMP_BEV_POSTPROCESS_BEGIN:
    //             LOG(INFO) << "[DETAILED_PERFORMANCE] BEV Postprocess Begin: " << timestamp.second << " ms";
    //             break;
    //         case TIMESTAMP_BEV_POSTPROCESS_END:
    //             LOG(INFO) << "[DETAILED_PERFORMANCE] BEV Postprocess End: " << timestamp.second << " ms";
    //             break;
    //         case TIMESTAMP_BEV_END:
    //             LOG(INFO) << "[DETAILED_PERFORMANCE] BEV End: " << timestamp.second << " ms";
    //             break;
    //         default:
    //             break;
    //     }
    // }
    
    // 记录各阶段延时
    for (const auto& delay : m_currentOutput.mapDelay()) {
        switch (delay.first) {
            case DELAY_TYPE_BEV_PREPROCESS:
                LOG(INFO) << "[DETAILED_PERFORMANCE] BEV Preprocess Delay: " << delay.second << " ms";
                break;
            case DELAY_TYPE_BEV_INFERENCE:
                LOG(INFO) << "[DETAILED_PERFORMANCE] BEV Inference Delay: " << delay.second << " ms";
                break;
            case DELAY_TYPE_BEV_POSTPROCESS:
                LOG(INFO) << "[DETAILED_PERFORMANCE] BEV Postprocess Delay: " << delay.second << " ms";
                break;
            case DELAY_TYPE_BEV:
                LOG(INFO) << "[DETAILED_PERFORMANCE] BEV Total Delay: " << delay.second << " ms";
                break;
            default:
                break;
        }
    }
    
    LOG(INFO) << "[DETAILED_PERFORMANCE] ======================================";
}