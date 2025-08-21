/*******************************************************
 文件：IErrorHandler.h
 作者：sharkls
 描述：错误处理接口定义
 版本：v1.0
 日期：2025-01-15
 *******************************************************/
#ifndef __IERROR_HANDLER_H__
#define __IERROR_HANDLER_H__

#include <string>
#include <vector>
#include <functional>
#include "ErrorTypes.h"

// 错误处理接口
class IErrorHandler {
public:
    virtual ~IErrorHandler() = default;
    
    // ================== 错误报告 ==================
    // 报告错误
    virtual void reportError(ErrorCode code, const std::string& message, 
                           const std::string& module_name, const std::string& context = "") = 0;
    
    // 报告错误（带上下文）
    virtual void reportError(ErrorCode code, const std::string& message, 
                           const std::string& module_name, const ErrorContext& context) = 0;
    
    // 报告错误（带严重程度）
    virtual void reportError(ErrorCode code, const std::string& message, 
                           const std::string& module_name, ErrorSeverity severity,
                           const std::string& context = "") = 0;
    
    // ================== 错误恢复 ==================
    // 尝试恢复错误
    virtual bool tryRecovery(ErrorCode code, const std::string& module_name) = 0;
    
    // 尝试恢复错误（带重试次数）
    virtual bool tryRecovery(ErrorCode code, const std::string& module_name, int max_retries) = 0;
    
    // 检查是否可以恢复
    virtual bool canRecover(ErrorCode code) const = 0;
    
    // ================== 错误查询 ==================
    // 检查是否有错误
    virtual bool hasError() const = 0;
    
    // 获取最后一个错误
    virtual ErrorInfo getLastError() const = 0;
    
    // 获取错误历史
    virtual std::vector<ErrorInfo> getErrorHistory() const = 0;
    
    // 获取指定模块的错误历史
    virtual std::vector<ErrorInfo> getErrorHistory(const std::string& module_name) const = 0;
    
    // 获取指定错误码的历史
    virtual std::vector<ErrorInfo> getErrorHistory(ErrorCode code) const = 0;
    
    // ================== 错误统计 ==================
    // 获取错误统计信息
    virtual ErrorStatistics getErrorStatistics() const = 0;
    
    // 获取指定模块的错误统计
    virtual ErrorStatistics getErrorStatistics(const std::string& module_name) const = 0;
    
    // 重置错误统计
    virtual void resetErrorStatistics() = 0;
    
    // ================== 错误监控 ==================
    // 设置错误阈值
    virtual void setErrorThreshold(ErrorCode code, int threshold) = 0;
    
    // 设置模块错误阈值
    virtual void setModuleErrorThreshold(const std::string& module_name, int threshold) = 0;
    
    // 设置告警回调
    virtual void setAlertCallback(std::function<void(const ErrorInfo&)> callback) = 0;
    
    // 检查是否超过阈值
    virtual bool isThresholdExceeded(ErrorCode code) const = 0;
    
    // 检查模块是否超过阈值
    virtual bool isModuleThresholdExceeded(const std::string& module_name) const = 0;
    
    // ================== 错误管理 ==================
    // 清除错误历史
    virtual void clearErrorHistory() = 0;
    
    // 清除指定模块的错误历史
    virtual void clearErrorHistory(const std::string& module_name) = 0;
    
    // 设置最大错误历史数量
    virtual void setMaxErrorHistory(size_t max_count) = 0;
    
    // ================== 配置管理 ==================
    // 启用/禁用错误处理
    virtual void setEnabled(bool enabled) = 0;
    
    // 检查是否启用
    virtual bool isEnabled() const = 0;
    
    // 设置日志级别
    virtual void setLogLevel(int level) = 0;
    
    // 获取日志级别
    virtual int getLogLevel() const = 0;
};

#endif // __IERROR_HANDLER_H__ 