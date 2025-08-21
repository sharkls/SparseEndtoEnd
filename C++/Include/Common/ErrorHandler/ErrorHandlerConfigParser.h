/*******************************************************
 文件：ErrorHandlerConfigParser.h
 作者：sharkls
 描述：错误处理配置文件解析器
 版本：v1.0
 日期：2025-01-15
 *******************************************************/
#ifndef __ERROR_HANDLER_CONFIG_PARSER_H__
#define __ERROR_HANDLER_CONFIG_PARSER_H__

#include <string>
#include <map>
#include <vector>
#include <memory>
#include "ErrorTypes.h"
#include "ErrorRecovery.h"
#include "ErrorMonitor.h"

// 前向声明
namespace errorhandler {
    class ErrorHandlerConfig;
    class GlobalConfig;
    class RecoveryStrategies;
    class MonitorConfig;
    class ErrorThresholds;
    class AlertConfig;
    class LogConfig;
    class PerformanceConfig;
}

// 错误处理配置结构
struct ErrorHandlerConfigData {
    // 全局配置
    struct GlobalConfig {
        bool enable_error_handling = true;
        bool enable_error_recovery = true;
        bool enable_error_monitoring = true;
        size_t max_error_history = 1000;
        int log_level = 1;
    } global_config;

    // 恢复策略配置
    std::map<ErrorCode, RecoveryConfig> recovery_strategies;

    // 监控配置
    struct MonitorConfig {
        bool enable_monitoring = true;
        bool enable_alerts = true;
        int alert_threshold = 10;
        int time_window_ms = 60000;
        std::vector<std::string> alert_channels;
        int max_alert_history = 100;
    } monitor_config;

    // 错误阈值配置
    std::map<ErrorCode, int> error_thresholds;
    std::map<std::string, int> module_thresholds;

    // 告警配置
    struct AlertConfig {
        std::map<ErrorCode, ErrorSeverity> alert_levels;
        std::map<ErrorCode, std::string> message_templates;
    } alert_config;

    // 日志配置
    struct LogConfig {
        std::string error_log_level = "ERROR";
        std::string warning_log_level = "WARNING";
        std::string info_log_level = "INFO";
        std::string debug_log_level = "DEBUG";
        std::string log_format = "[{timestamp}] [{level}] [{module_name}] {message} (ErrorCode: {error_code})";
        bool log_file_enabled = true;
        std::string log_file_path = "logs/error_handler.log";
        int log_file_max_size_mb = 100;
        int log_file_max_files = 10;
    } log_config;

    // 性能配置
    struct PerformanceConfig {
        int statistics_interval_ms = 60000;
        bool enable_performance_reporting = true;
        int performance_report_interval_ms = 300000;
        bool enable_memory_monitoring = true;
        int memory_threshold_mb = 1024;
    } performance_config;
};

// 错误处理配置解析器
class ErrorHandlerConfigParser {
public:
    // 从protobuf文本文件解析配置
    static bool parseConfigFile(const std::string& config_path, ErrorHandlerConfigData& config);
    
    // 从protobuf消息解析配置
    static bool parseProtobufMessage(const errorhandler::ErrorHandlerConfig& pb_config, 
                                    ErrorHandlerConfigData& config);
    
    // 验证配置有效性
    static bool validateConfig(const ErrorHandlerConfigData& config);
    
    // 获取默认配置
    static ErrorHandlerConfigData getDefaultConfig();
    
    // 合并配置（全局配置 + 模块特定配置）
    static ErrorHandlerConfigData mergeConfigs(const ErrorHandlerConfigData& global_config, 
                                              const ErrorHandlerConfigData& module_config);

private:
    // 解析全局配置
    static bool parseGlobalConfig(const errorhandler::GlobalConfig& pb_config, 
                                 ErrorHandlerConfigData::GlobalConfig& config);
    
    // 解析恢复策略
    static bool parseRecoveryStrategies(const errorhandler::RecoveryStrategies& pb_config, 
                                       std::map<ErrorCode, RecoveryConfig>& strategies);
    
    // 解析监控配置
    static bool parseMonitorConfig(const errorhandler::MonitorConfig& pb_config, 
                                  ErrorHandlerConfigData::MonitorConfig& config);
    
    // 解析错误阈值
    static bool parseErrorThresholds(const errorhandler::ErrorThresholds& pb_config, 
                                    std::map<ErrorCode, int>& thresholds,
                                    std::map<std::string, int>& module_thresholds);
    
    // 解析告警配置
    static bool parseAlertConfig(const errorhandler::AlertConfig& pb_config, 
                                ErrorHandlerConfigData::AlertConfig& config);
    
    // 解析日志配置
    static bool parseLogConfig(const errorhandler::LogConfig& pb_config, 
                              ErrorHandlerConfigData::LogConfig& config);
    
    // 解析性能配置
    static bool parsePerformanceConfig(const errorhandler::PerformanceConfig& pb_config, 
                                      ErrorHandlerConfigData::PerformanceConfig& config);
    
    // 字符串转ErrorCode
    static ErrorCode stringToErrorCode(const std::string& str);
    
    // 字符串转ErrorSeverity
    static ErrorSeverity stringToErrorSeverity(const std::string& str);
    
    // 字符串转RecoveryStrategy
    static RecoveryStrategy stringToRecoveryStrategy(const std::string& str);
    
    // protobuf枚举转RecoveryStrategy
    static RecoveryStrategy pbStrategyToRecoveryStrategy(int pb_strategy);
    
    // protobuf枚举转ErrorSeverity
    static ErrorSeverity pbSeverityToErrorSeverity(int pb_severity);
};

#endif // __ERROR_HANDLER_CONFIG_PARSER_H__ 