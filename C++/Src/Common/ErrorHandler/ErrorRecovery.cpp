/*******************************************************
 文件：ErrorRecovery.cpp
 作者：sharkls
 描述：错误恢复策略类实现
 版本：v1.0
 日期：2025-01-15
 *******************************************************/

#include "../../Include/Common/ErrorHandler/ErrorRecovery.h"
#include "../../Include/Common/Core/GlobalContext.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>

ErrorRecovery::ErrorRecovery() 
    : m_default_strategy(RecoveryStrategy::NONE)
    , m_default_config()
{
    // 设置默认恢复策略
    setupDefaultRecoveryStrategies();
}

void ErrorRecovery::setupDefaultRecoveryStrategies() {
    // 算法相关错误 - 重试策略
    setRecoveryStrategy(ErrorCode::ALGORITHM_INFERENCE_FAILED, RecoveryStrategy::RETRY);
    setRetryCount(ErrorCode::ALGORITHM_INFERENCE_FAILED, 3);
    setRecoveryTimeout(ErrorCode::ALGORITHM_INFERENCE_FAILED, 10000);
    setRetryInterval(ErrorCode::ALGORITHM_INFERENCE_FAILED, 1000);
    
    // 内存错误 - 重置策略
    setRecoveryStrategy(ErrorCode::MEMORY_ALLOCATION_FAILED, RecoveryStrategy::RESET);
    setRecoveryTimeout(ErrorCode::MEMORY_ALLOCATION_FAILED, 5000);
    
    // 配置错误 - 不恢复
    setRecoveryStrategy(ErrorCode::CONFIG_LOAD_ERROR, RecoveryStrategy::NONE);
    
    // CUDA错误 - 重试策略
    setRecoveryStrategy(ErrorCode::CUDA_MEMORY_ALLOCATION_FAILED, RecoveryStrategy::RETRY);
    setRetryCount(ErrorCode::CUDA_MEMORY_ALLOCATION_FAILED, 2);
    setRecoveryTimeout(ErrorCode::CUDA_MEMORY_ALLOCATION_FAILED, 5000);
    
    // TensorRT错误 - 重试策略
    setRecoveryStrategy(ErrorCode::TENSORRT_INFERENCE_FAILED, RecoveryStrategy::RETRY);
    setRetryCount(ErrorCode::TENSORRT_INFERENCE_FAILED, 2);
    setRecoveryTimeout(ErrorCode::TENSORRT_INFERENCE_FAILED, 8000);
    
    // 传感器错误 - 重试策略
    setRecoveryStrategy(ErrorCode::SENSOR_TIMEOUT, RecoveryStrategy::RETRY);
    setRetryCount(ErrorCode::SENSOR_TIMEOUT, 5);
    setRecoveryTimeout(ErrorCode::SENSOR_TIMEOUT, 15000);
}

void ErrorRecovery::setRecoveryStrategy(ErrorCode code, RecoveryStrategy strategy) {
    m_recovery_configs[code].strategy = strategy;
}

void ErrorRecovery::setRetryCount(ErrorCode code, int max_retries) {
    m_recovery_configs[code].max_retries = max_retries;
}

void ErrorRecovery::setRecoveryTimeout(ErrorCode code, int timeout_ms) {
    m_recovery_configs[code].timeout_ms = timeout_ms;
}

void ErrorRecovery::setRetryInterval(ErrorCode code, int interval_ms) {
    m_recovery_configs[code].retry_interval_ms = interval_ms;
}

void ErrorRecovery::setCustomRecoveryFunc(ErrorCode code, std::function<bool()> func) {
    m_recovery_configs[code].custom_recovery_func = func;
    m_recovery_configs[code].strategy = RecoveryStrategy::CUSTOM;
}

void ErrorRecovery::setRecoveryConfig(ErrorCode code, const RecoveryConfig& config) {
    m_recovery_configs[code] = config;
}

RecoveryResult ErrorRecovery::executeRecovery(ErrorCode code, const std::string& module_name) {
    return executeRecovery(code, module_name, {});
}

RecoveryResult ErrorRecovery::executeRecovery(ErrorCode code, const std::string& module_name, 
                                             const std::map<std::string, std::string>& context) {
    RecoveryResult result;
    auto start_time = std::chrono::steady_clock::now();
    
    // 获取恢复配置
    RecoveryConfig config = getRecoveryConfig(code);
    
    // 检查是否可以恢复
    if (!canRecover(code)) {
        result.message = "Error cannot be recovered: " + ErrorCodeToString(code);
        result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        return result;
    }
    
    // 根据策略执行恢复
    switch (config.strategy) {
        case RecoveryStrategy::RETRY:
            result = executeRetryStrategy(code, module_name, config);
            break;
        case RecoveryStrategy::RESTART:
            result = executeRestartStrategy(code, module_name, config);
            break;
        case RecoveryStrategy::FALLBACK:
            result = executeFallbackStrategy(code, module_name, config);
            break;
        case RecoveryStrategy::RESET:
            result = executeResetStrategy(code, module_name, config);
            break;
        case RecoveryStrategy::CUSTOM:
            result = executeCustomStrategy(code, module_name, config);
            break;
        case RecoveryStrategy::NONE:
        default:
            result.message = "No recovery strategy configured";
            break;
    }
    
    // 更新恢复统计
    if (result.success) {
        m_recovery_counts[code]++;
    }
    
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    
    return result;
}

bool ErrorRecovery::canRecover(ErrorCode code) const {
    auto it = m_recovery_configs.find(code);
    if (it != m_recovery_configs.end()) {
        return it->second.strategy != RecoveryStrategy::NONE;
    }
    return m_default_config.strategy != RecoveryStrategy::NONE;
}

RecoveryConfig ErrorRecovery::getRecoveryConfig(ErrorCode code) const {
    auto it = m_recovery_configs.find(code);
    if (it != m_recovery_configs.end()) {
        return it->second;
    }
    return m_default_config;
}

std::map<ErrorCode, int> ErrorRecovery::getRecoveryCounts() const {
    return m_recovery_counts;
}

void ErrorRecovery::resetRecoveryCounts() {
    m_recovery_counts.clear();
}

void ErrorRecovery::setDefaultRecoveryStrategy(RecoveryStrategy strategy) {
    m_default_strategy = strategy;
    m_default_config.strategy = strategy;
}

RecoveryStrategy ErrorRecovery::getDefaultRecoveryStrategy() const {
    return m_default_strategy;
}

RecoveryResult ErrorRecovery::executeRetryStrategy(ErrorCode code, const std::string& module_name, 
                                                  const RecoveryConfig& config) {
    RecoveryResult result;
    
    for (int retry = 0; retry <= config.max_retries; ++retry) {
        auto start_time = std::chrono::steady_clock::now();
        
        // 模拟恢复操作（实际应用中这里会调用具体的恢复逻辑）
        bool recovery_success = simulateRecovery(code, module_name, retry);
        
        if (recovery_success) {
            result.success = true;
            result.retry_count = retry;
            result.message = "Recovery successful after " + std::to_string(retry) + " retries";
            return result;
        }
        
        // 检查超时
        if (isTimeout(start_time, config.timeout_ms)) {
            result.message = "Recovery timeout after " + std::to_string(retry) + " retries";
            break;
        }
        
        // 等待重试间隔
        if (retry < config.max_retries) {
            waitForInterval(config.retry_interval_ms);
        }
    }
    
    result.success = false;
    result.retry_count = config.max_retries;
    result.message = "Recovery failed after " + std::to_string(config.max_retries) + " retries";
    
    return result;
}

RecoveryResult ErrorRecovery::executeRestartStrategy(ErrorCode code, const std::string& module_name, 
                                                    const RecoveryConfig& config) {
    RecoveryResult result;
    
    auto start_time = std::chrono::steady_clock::now();
    
    // 模拟重启操作
    bool restart_success = simulateRestart(module_name);
    
    if (restart_success) {
        result.success = true;
        result.message = "Module restart successful";
    } else {
        result.success = false;
        result.message = "Module restart failed";
    }
    
    return result;
}

RecoveryResult ErrorRecovery::executeFallbackStrategy(ErrorCode code, const std::string& module_name, 
                                                     const RecoveryConfig& config) {
    RecoveryResult result;
    
    // 模拟降级处理
    bool fallback_success = simulateFallback(code, module_name);
    
    if (fallback_success) {
        result.success = true;
        result.message = "Fallback processing successful";
    } else {
        result.success = false;
        result.message = "Fallback processing failed";
    }
    
    return result;
}

RecoveryResult ErrorRecovery::executeResetStrategy(ErrorCode code, const std::string& module_name, 
                                                  const RecoveryConfig& config) {
    RecoveryResult result;
    
    // 模拟重置操作
    bool reset_success = simulateReset(module_name);
    
    if (reset_success) {
        result.success = true;
        result.message = "Module reset successful";
    } else {
        result.success = false;
        result.message = "Module reset failed";
    }
    
    return result;
}

RecoveryResult ErrorRecovery::executeCustomStrategy(ErrorCode code, const std::string& module_name, 
                                                   const RecoveryConfig& config) {
    RecoveryResult result;
    
    if (config.custom_recovery_func) {
        bool custom_success = config.custom_recovery_func();
        
        if (custom_success) {
            result.success = true;
            result.message = "Custom recovery successful";
        } else {
            result.success = false;
            result.message = "Custom recovery failed";
        }
    } else {
        result.success = false;
        result.message = "No custom recovery function provided";
    }
    
    return result;
}

void ErrorRecovery::waitForInterval(int interval_ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
}

bool ErrorRecovery::isTimeout(const std::chrono::steady_clock::time_point& start_time, int timeout_ms) const {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    return elapsed.count() >= timeout_ms;
}

// 模拟恢复操作的辅助函数
bool ErrorRecovery::simulateRecovery(ErrorCode code, const std::string& module_name, int retry_count) {
    // 这里应该实现具体的恢复逻辑
    // 目前只是简单的模拟
    
    // 对于某些错误，前几次重试可能失败，最后一次成功
    if (code == ErrorCode::ALGORITHM_INFERENCE_FAILED) {
        return retry_count >= 2; // 第3次重试成功
    }
    
    // 对于其他错误，随机成功
    return (retry_count % 2) == 0;
}

bool ErrorRecovery::simulateRestart(const std::string& module_name) {
    // 模拟重启操作
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return true; // 假设重启总是成功
}

bool ErrorRecovery::simulateFallback(ErrorCode code, const std::string& module_name) {
    // 模拟降级处理
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return true; // 假设降级总是成功
}

bool ErrorRecovery::simulateReset(const std::string& module_name) {
    // 模拟重置操作
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return true; // 假设重置总是成功
} 