/*******************************************************
 文件名：CSparseBEVAlg.h
 作者：sharkls
 描述：SparseBEV算法主类，负责协调各个子模块的运行
 版本：v1.0
 日期：2025-06-18
 *******************************************************/

#ifndef __SPARSEBEVALG_H__
#define __SPARSEBEVALG_H__

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <atomic>
#include "Log.h"
#include <google/protobuf/text_format.h>    // 解析prototext格式文本

#include "../../Include/Interface/ExportSparseBEVAlgLib.h"
#include "../../Include/Common/Core/GlobalContext.h"
#include "../../Include/Common/Interface/IBaseModule.h"
#include "../../Include/Common/Interface/IAlgorithmConfig.h"
#include "../../Include/Common/Factory/ModuleFactory.h"
#include "../../Src/Common/Core/FunctionHub.h"


#include "SparseEnd2EndConfig_conf.pb.h"
#include "AlgorithmConfig_conf.pb.h"
#include "CAlgResult.h"
#include "CTimeMatchSrcData.h"
#include "RawInferenceResult.h"

class SparseBEVConfig : public IAlgorithmConfig {
public:
    bool loadFromFile(const std::string& path) override;
    const google::protobuf::Message* getConfigMessage() const override { return &m_config; }
    sparsebev::ModuleConfig& getModuleConfig() { return m_config; }
private:
    sparsebev::ModuleConfig m_config;
};

class CSparseBEVAlg : public ISparseBEVAlg {
public:
    CSparseBEVAlg(const std::string& exePath);
    ~CSparseBEVAlg() override;

    // 实现ISparseEnd2EndAlg接口
    bool initAlgorithm(const std::string exe_path, const AlgCallback& alg_cb, void* hd) override;
    void runAlgorithm(void* p_pSrcData) override;

private:
    // 加载配置文件
    bool loadConfig(const std::string& configPath);
    
    // 验证配置完整性
    bool validateConfig();
    
    // 创建并初始化模块
    bool initModules();
    
    // 执行模块链
    bool executeModuleChain();

    // 转换原始结果到算法结果
    void convertRawResultToAlgResult(const RawInferenceResult& raw_result, CAlgResult& alg_result);

private:
    std::string m_exePath;                                    // 工程路径
    std::shared_ptr<SparseBEVConfig> m_pConfig;               // 配置对象
    std::vector<std::shared_ptr<IBaseModule>> m_moduleChain;  // 模块执行链
    AlgCallback m_algCallback;                                // 算法回调函数
    void* m_callbackHandle;                                   // 回调函数句柄
    CTimeMatchSrcData* m_currentInput;                       // 当前输入数据
    CAlgResult m_currentOutput;                               // 当前输出数据

    // 线程安全保护
    mutable std::mutex m_mutex;                               // 互斥锁
    std::atomic<bool> m_is_initialized{false};                // 初始化状态
    std::atomic<bool> m_is_running{false};                    // 运行状态

    // 离线测试配置
    bool m_run_status{false};
    
    // 当前样本索引
    int current_sample_index_;

    // 性能监控方法
    void startTimer(const std::string& stage);
    void endTimer(const std::string& stage);
    void updateMemoryUsage();
    void logPerformanceReport();
    
    // 获取性能统计信息
    const CAlgResult& getCurrentOutput() const { return m_currentOutput; }
    
    // 计算各阶段详细延时
    void calculateDetailedDelays();
    void logDetailedPerformanceInfo() const;
}; 

#endif // __SPARSEBEVALG_H__