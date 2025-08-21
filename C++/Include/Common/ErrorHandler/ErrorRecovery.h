/*******************************************************
 文件：ErrorRecovery.h
 作者：sharkls
 描述：错误恢复策略类定义
 版本：v1.0
 日期：2025-01-15
 *******************************************************/
#ifndef __ERROR_RECOVERY_H__
#define __ERROR_RECOVERY_H__

#include <string>
#include <map>
#include <functional>
#include <chrono>
#include "GlobalContext.h"

// 恢复策略类型
enum class RecoveryStrategy : uint8_t {
    NONE = 0,           // 不恢复
    RETRY = 1,          // 重试
    RESTART = 2,        // 重启模块
    FALLBACK = 3,       // 降级处理
    RESET = 4,          // 重置状态
    CUSTOM = 5          // 自定义策略
};

// 恢复策略配置
struct RecoveryConfig {
    RecoveryStrategy strategy;         // 恢复策略
    int max_retries;                   // 最大重试次数
    int timeout_ms;                    // 超时时间（毫秒）
    int retry_interval_ms;             // 重试间隔（毫秒）
    std::function<bool()> custom_recovery_func; // 自定义恢复函数
    
    RecoveryConfig() : strategy(RecoveryStrategy::NONE), max_retries(0), 
                      timeout_ms(5000), retry_interval_ms(1000) {}
};

// 恢复结果
struct RecoveryResult {
    bool success;                       // 是否成功
    int retry_count;                    // 实际重试次数
    std::string message;                // 恢复消息
    std::chrono::milliseconds duration; // 恢复耗时
    
    RecoveryResult() : success(false), retry_count(0), duration(0) {}
};

// 错误恢复策略类
class ErrorRecovery {
public:
    ErrorRecovery();
    ~ErrorRecovery() = default;
    
    // ================== 策略配置 ==================
    // 设置恢复策略
    void setRecoveryStrategy(ErrorCode code, RecoveryStrategy strategy);
    
    // 设置重试次数
    void setRetryCount(ErrorCode code, int max_retries);
    
    // 设置超时时间
    void setRecoveryTimeout(ErrorCode code, int timeout_ms);
    
    // 设置重试间隔
    void setRetryInterval(ErrorCode code, int interval_ms);
    
    // 设置自定义恢复函数
    void setCustomRecoveryFunc(ErrorCode code, std::function<bool()> func);
    
    // 批量设置恢复配置
    void setRecoveryConfig(ErrorCode code, const RecoveryConfig& config);
    
    // ================== 恢复执行 ==================
    // 执行恢复
    RecoveryResult executeRecovery(ErrorCode code, const std::string& module_name);
    
    // 执行恢复（带自定义上下文）
    RecoveryResult executeRecovery(ErrorCode code, const std::string& module_name, 
                                  const std::map<std::string, std::string>& context);
    
    // 检查是否可以恢复
    bool canRecover(ErrorCode code) const;
    
    // 获取恢复配置
    RecoveryConfig getRecoveryConfig(ErrorCode code) const;
    
    // ================== 恢复统计 ==================
    // 获取恢复统计
    std::map<ErrorCode, int> getRecoveryCounts() const;
    
    // 重置恢复统计
    void resetRecoveryCounts();
    
    // ================== 默认策略 ==================
    // 设置默认恢复策略
    void setDefaultRecoveryStrategy(RecoveryStrategy strategy);
    
    // 获取默认恢复策略
    RecoveryStrategy getDefaultRecoveryStrategy() const;
    
private:
    // 执行重试策略
    RecoveryResult executeRetryStrategy(ErrorCode code, const std::string& module_name, 
                                       const RecoveryConfig& config);
    
    // 执行重启策略
    RecoveryResult executeRestartStrategy(ErrorCode code, const std::string& module_name, 
                                         const RecoveryConfig& config);
    
    // 执行降级策略
    RecoveryResult executeFallbackStrategy(ErrorCode code, const std::string& module_name, 
                                          const RecoveryConfig& config);
    
    // 执行重置策略
    RecoveryResult executeResetStrategy(ErrorCode code, const std::string& module_name, 
                                       const RecoveryConfig& config);
    
    // 执行自定义策略
    RecoveryResult executeCustomStrategy(ErrorCode code, const std::string& module_name, 
                                        const RecoveryConfig& config);
    
    // 等待指定时间
    void waitForInterval(int interval_ms);
    
    // 检查超时
    bool isTimeout(const std::chrono::steady_clock::time_point& start_time, int timeout_ms) const;

private:
    std::map<ErrorCode, RecoveryConfig> m_recovery_configs;   // 恢复配置
    std::map<ErrorCode, int> m_recovery_counts;               // 恢复次数统计
    RecoveryStrategy m_default_strategy;                       // 默认恢复策略
    RecoveryConfig m_default_config;                          // 默认配置
};

#endif // __ERROR_RECOVERY_H__ 