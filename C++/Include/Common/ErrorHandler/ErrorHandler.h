/*******************************************************
 文件：ErrorHandler.h
 作者：sharkls
 描述：错误处理主类定义
 版本：v1.0
 日期：2025-01-15
 *******************************************************/
#ifndef __ERROR_HANDLER_H__
#define __ERROR_HANDLER_H__

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include "IErrorHandler.h"
#include "ErrorRecovery.h"
#include "ErrorMonitor.h"
#include "ErrorHandlerConfigParser.h"

// 错误处理配置
struct ErrorHandlerConfig {
    bool enable_error_handling;                 // 是否启用错误处理
    bool enable_error_recovery;                 // 是否启用错误恢复
    bool enable_error_monitoring;               // 是否启用错误监控
    size_t max_error_history;                   // 最大错误历史数量
    int log_level;                              // 日志级别
    std::string module_name;                    // 模块名称
    
    ErrorHandlerConfig() : enable_error_handling(true), enable_error_recovery(true),
                          enable_error_monitoring(true), max_error_history(1000), 
                          log_level(1), module_name("") {}
};

// 错误处理主类
class ErrorHandler : public IErrorHandler {
public:
    explicit ErrorHandler(const std::string& module_name = "");
    ~ErrorHandler() override = default;
    
    // ================== 初始化配置 ==================
    // 初始化错误处理器
    bool initialize(const ErrorHandlerConfig& config);
    
    // 从配置文件初始化
    bool initializeFromConfig(const std::string& config_path);
    
    // 从配置数据初始化
    bool initializeFromConfigData(const ErrorHandlerConfigData& config_data);
    
    // 获取配置
    ErrorHandlerConfig getConfig() const;
    
    // 获取配置数据
    ErrorHandlerConfigData getConfigData() const;
    
    // ================== 错误报告 ==================
    // 报告错误
    void reportError(ErrorCode code, const std::string& message, 
                    const std::string& module_name, const std::string& context = "") override;
    
    // 报告错误（带上下文）
    void reportError(ErrorCode code, const std::string& message, 
                    const std::string& module_name, const ErrorContext& context) override;
    
    // 报告错误（带严重程度）
    void reportError(ErrorCode code, const std::string& message, 
                    const std::string& module_name, ErrorSeverity severity,
                    const std::string& context = "") override;
    
    // ================== 错误恢复 ==================
    // 尝试恢复错误
    bool tryRecovery(ErrorCode code, const std::string& module_name) override;
    
    // 尝试恢复错误（带重试次数）
    bool tryRecovery(ErrorCode code, const std::string& module_name, int max_retries) override;
    
    // 检查是否可以恢复
    bool canRecover(ErrorCode code) const override;
    
    // ================== 错误查询 ==================
    // 检查是否有错误
    bool hasError() const override;
    
    // 获取最后一个错误
    ErrorInfo getLastError() const override;
    
    // 获取错误历史
    std::vector<ErrorInfo> getErrorHistory() const override;
    
    // 获取指定模块的错误历史
    std::vector<ErrorInfo> getErrorHistory(const std::string& module_name) const override;
    
    // 获取指定错误码的历史
    std::vector<ErrorInfo> getErrorHistory(ErrorCode code) const override;
    
    // ================== 错误统计 ==================
    // 获取错误统计信息
    ErrorStatistics getErrorStatistics() const override;
    
    // 获取指定模块的错误统计
    ErrorStatistics getErrorStatistics(const std::string& module_name) const override;
    
    // 重置错误统计
    void resetErrorStatistics() override;
    
    // ================== 错误监控 ==================
    // 设置错误阈值
    void setErrorThreshold(ErrorCode code, int threshold) override;
    
    // 设置模块错误阈值
    void setModuleErrorThreshold(const std::string& module_name, int threshold) override;
    
    // 设置告警回调
    void setAlertCallback(std::function<void(const ErrorInfo&)> callback) override;
    
    // 检查是否超过阈值
    bool isThresholdExceeded(ErrorCode code) const override;
    
    // 检查模块是否超过阈值
    bool isModuleThresholdExceeded(const std::string& module_name) const override;
    
    // ================== 错误管理 ==================
    // 清除错误历史
    void clearErrorHistory() override;
    
    // 清除指定模块的错误历史
    void clearErrorHistory(const std::string& module_name) override;
    
    // 设置最大错误历史数量
    void setMaxErrorHistory(size_t max_count) override;
    
    // ================== 配置管理 ==================
    // 启用/禁用错误处理
    void setEnabled(bool enabled) override;
    
    // 检查是否启用
    bool isEnabled() const override;
    
    // 设置日志级别
    void setLogLevel(int level) override;
    
    // 获取日志级别
    int getLogLevel() const override;
    
    // ================== 高级功能 ==================
    // 获取错误恢复器
    std::shared_ptr<ErrorRecovery> getErrorRecovery() const;
    
    // 获取错误监控器
    std::shared_ptr<ErrorMonitor> getErrorMonitor() const;
    
    // 生成错误报告
    std::string generateErrorReport() const;
    
    // 导出错误数据
    bool exportErrorData(const std::string& file_path) const;
    
    // 导入错误数据
    bool importErrorData(const std::string& file_path);
    
private:
    // 记录错误到历史
    void recordError(const ErrorInfo& error);
    
    // 记录错误到日志
    void logError(const ErrorInfo& error, ErrorSeverity severity = ErrorSeverity::MEDIUM);
    
    // 检查是否需要触发告警
    void checkAndTriggerAlert(const ErrorInfo& error);
    
    // 清理过期错误历史
    void cleanupExpiredHistory();
    
    // 获取当前时间戳
    int64_t getCurrentTimestamp() const;
    
    // 获取错误严重程度
    ErrorSeverity getErrorSeverity(ErrorCode code) const;
    
    // 获取错误分类
    ErrorCategory getErrorCategory(ErrorCode code) const;

private:
    mutable std::mutex m_mutex;                                    // 线程安全锁
    ErrorHandlerConfig m_config;                                   // 配置
    ErrorHandlerConfigData m_config_data;                          // 配置数据  
    std::vector<ErrorInfo> m_error_history;                        // 错误历史
    ErrorInfo m_last_error;                                        // 最后一个错误
    std::shared_ptr<ErrorRecovery> m_error_recovery;               // 错误恢复器
    std::shared_ptr<ErrorMonitor> m_error_monitor;                 // 错误监控器
    std::function<void(const ErrorInfo&)> m_alert_callback;        // 告警回调
    bool m_enabled;                                                // 是否启用
    int m_log_level;                                               // 日志级别
};

#endif // __ERROR_HANDLER_H__ 