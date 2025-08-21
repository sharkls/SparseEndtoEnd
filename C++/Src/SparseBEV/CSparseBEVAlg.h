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
#include "log.h"
#include <google/protobuf/text_format.h>    // 解析prototext格式文本
#include <opencv2/opencv.hpp>

#include "../../Include/Interface/ExportSparseBEVAlgLib.h"
#include "../../Include/Common/GlobalContext.h"
#include "../Src/Common/IBaseModule.h"
#include "../Src/Common/AlgorithmConfig.h"
#include "../Src/Common/ModuleFactory.h"
#include "../Src/Common/FunctionHub.h"


#include "SparseBEV_conf.pb.h"
#include "AlgorithmConfig_conf.pb.h"
#include "CAlgResult.h"
#include "CMultiModalSrcData.h"

class SparseBEVConfig : public AlgorithmConfig {
public:
    bool loadFromFile(const std::string& path) override;
    const google::protobuf::Message* getConfigMessage() const override { return &m_config; }
    sparsebev::TaskConfig& getTaskConfig() { return m_config; }
private:
    sparsebev::TaskConfig m_config;
};

class CSparseBEVAlg : public ISparseBEVAlg {
public:
    CSparseBEVAlg(const std::string& exePath);
    ~CSparseBEVAlg() override;

    // 实现ISparseBEVAlg接口
    bool initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd) override;
    void runAlgorithm(void* p_pSrcData) override;

private:
    // 加载配置文件
    bool loadConfig(const std::string& configPath);
    
    // 创建并初始化模块
    bool initModules();
    
    // 执行模块链
    bool executeModuleChain();

    // 可视化检测结果
    void visualizationResult();

private:
    std::string m_exePath;                                    // 工程路径
    std::shared_ptr<SparseBEVConfig> m_pConfig;               // 配置对象
    std::vector<std::shared_ptr<IBaseModule>> m_moduleChain;  // 模块执行链
    AlgCallback m_algCallback;                                // 算法回调函数
    void* m_callbackHandle;                                   // 回调函数句柄
    CMultiModalSrcData* m_currentInput;                       // 当前输入数据
    CAlgResult m_currentOutput;                               // 当前输出数据

    // 离线测试配置
    bool m_run_status{false};
}; 

#endif // __SPARSEBEVALG_H__