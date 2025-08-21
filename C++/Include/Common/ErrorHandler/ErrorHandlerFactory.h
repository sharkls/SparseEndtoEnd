/*******************************************************
 文件：ErrorHandlerFactory.h
 作者：sharkls
 描述：错误处理器工厂类定义
 版本：v1.0
 日期：2025-01-15
 *******************************************************/
#ifndef __ERROR_HANDLER_FACTORY_H__
#define __ERROR_HANDLER_FACTORY_H__

#include <string>
#include <map>
#include <memory>
#include <mutex>
#include "ErrorHandler.h"

// 错误处理器工厂类
class ErrorHandlerFactory {
public:
    // 获取单例实例
    static ErrorHandlerFactory& getInstance();
    
    // 获取错误处理器
    std::shared_ptr<IErrorHandler> getErrorHandler(const std::string& module_name);
    
    // 注册错误处理器
    void registerErrorHandler(const std::string& module_name, 
                            std::shared_ptr<IErrorHandler> handler);
    
    // 创建错误处理器
    std::shared_ptr<IErrorHandler> createErrorHandler(const std::string& module_name,
                                                     const ErrorHandlerConfig& config = ErrorHandlerConfig());
    
    // 从配置文件创建错误处理器
    std::shared_ptr<IErrorHandler> createErrorHandlerFromConfig(const std::string& module_name,
                                                               const std::string& config_path);
    
    // 移除错误处理器
    void removeErrorHandler(const std::string& module_name);
    
    // 检查错误处理器是否存在
    bool hasErrorHandler(const std::string& module_name) const;
    
    // 获取所有错误处理器
    std::map<std::string, std::shared_ptr<IErrorHandler>> getAllErrorHandlers() const;
    
    // 获取错误处理器数量
    size_t getErrorHandlerCount() const;
    
    // 清理所有错误处理器
    void clearAllErrorHandlers();
    
    // 设置全局配置
    void setGlobalConfig(const ErrorHandlerConfig& config);
    
    // 获取全局配置
    ErrorHandlerConfig getGlobalConfig() const;
    
    // 设置全局告警回调
    void setGlobalAlertCallback(std::function<void(const ErrorInfo&)> callback);
    
    // 获取全局告警回调
    std::function<void(const ErrorInfo&)> getGlobalAlertCallback() const;
    
    // 初始化工厂
    bool initialize(const std::string& global_config_path = "");
    
    // 关闭工厂
    void shutdown();
    
    // 生成全局错误报告
    std::string generateGlobalErrorReport() const;
    
    // 导出全局错误数据
    bool exportGlobalErrorData(const std::string& file_path) const;
    
    // 重置所有错误统计
    void resetAllErrorStatistics();
    
    // 获取全局错误统计
    ErrorStatistics getGlobalErrorStatistics() const;
    
private:
    // 私有构造函数（单例模式）
    ErrorHandlerFactory();
    
    // 禁用拷贝构造和赋值
    ErrorHandlerFactory(const ErrorHandlerFactory&) = delete;
    ErrorHandlerFactory& operator=(const ErrorHandlerFactory&) = delete;
    
    // 加载全局配置
    bool loadGlobalConfig(const std::string& config_path);
    
    // 应用全局配置到错误处理器
    void applyGlobalConfig(std::shared_ptr<IErrorHandler> handler);
    
    // 创建默认错误处理器
    std::shared_ptr<IErrorHandler> createDefaultErrorHandler(const std::string& module_name);

private:
    mutable std::mutex m_mutex;                                    // 线程安全锁
    std::map<std::string, std::shared_ptr<IErrorHandler>> m_handlers; // 错误处理器映射
    ErrorHandlerConfig m_global_config;                            // 全局配置
    std::function<void(const ErrorInfo&)> m_global_alert_callback; // 全局告警回调
    bool m_initialized;                                            // 是否已初始化
};

// 便捷宏定义
#define GET_ERROR_HANDLER(module_name) \
    ErrorHandlerFactory::getInstance().getErrorHandler(module_name)

#define CREATE_ERROR_HANDLER(module_name) \
    ErrorHandlerFactory::getInstance().createErrorHandler(module_name)

#define CREATE_ERROR_HANDLER_WITH_CONFIG(module_name, config) \
    ErrorHandlerFactory::getInstance().createErrorHandler(module_name, config)

#endif // __ERROR_HANDLER_FACTORY_H__ 