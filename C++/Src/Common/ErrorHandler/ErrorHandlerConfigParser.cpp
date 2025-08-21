/*******************************************************
 文件：ErrorHandlerConfigParser.cpp
 作者：sharkls
 描述：错误处理配置文件解析器实现
 版本：v1.0
 日期：2025-01-15
 *******************************************************/

#include "../../Include/Common/ErrorHandler/ErrorHandlerConfigParser.h"
#include "../../../Submodules/Protoser/param/ErrorHandlerConfig_conf.pb.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <google/protobuf/text_format.h>

bool ErrorHandlerConfigParser::parseConfigFile(const std::string& config_path, ErrorHandlerConfigData& config) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << config_path << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    errorhandler::ErrorHandlerConfig pb_config;
    if (!google::protobuf::TextFormat::ParseFromString(content, &pb_config)) {
        std::cerr << "Failed to parse protobuf config file: " << config_path << std::endl;
        return false;
    }
    
    return parseProtobufMessage(pb_config, config);
}

bool ErrorHandlerConfigParser::parseProtobufMessage(const errorhandler::ErrorHandlerConfig& pb_config, 
                                                   ErrorHandlerConfigData& config) {
    // 解析全局配置
    if (!parseGlobalConfig(pb_config.global_config(), config.global_config)) {
        return false;
    }
    
    // 解析恢复策略
    if (!parseRecoveryStrategies(pb_config.recovery_strategies(), config.recovery_strategies)) {
        return false;
    }
    
    // 解析监控配置
    if (!parseMonitorConfig(pb_config.monitor_config(), config.monitor_config)) {
        return false;
    }
    
    // 解析错误阈值
    if (!parseErrorThresholds(pb_config.error_thresholds(), config.error_thresholds, config.module_thresholds)) {
        return false;
    }
    
    // 解析告警配置
    if (!parseAlertConfig(pb_config.alert_config(), config.alert_config)) {
        return false;
    }
    
    // 解析日志配置
    if (!parseLogConfig(pb_config.log_config(), config.log_config)) {
        return false;
    }
    
    // 解析性能配置
    if (!parsePerformanceConfig(pb_config.performance_config(), config.performance_config)) {
        return false;
    }
    
    return true;
}

bool ErrorHandlerConfigParser::parseGlobalConfig(const errorhandler::GlobalConfig& pb_config, 
                                                ErrorHandlerConfigData::GlobalConfig& config) {
    config.enable_error_handling = pb_config.enable_error_handling();
    config.enable_error_recovery = pb_config.enable_error_recovery();
    config.enable_error_monitoring = pb_config.enable_error_monitoring();
    config.max_error_history = pb_config.max_error_history();
    config.log_level = pb_config.log_level();
    return true;
}

bool ErrorHandlerConfigParser::parseRecoveryStrategies(const errorhandler::RecoveryStrategies& pb_config, 
                                                      std::map<ErrorCode, RecoveryConfig>& strategies) {
    for (const auto& entry : pb_config.strategies()) {
        ErrorCode error_code = stringToErrorCode(entry.error_code());
        RecoveryConfig recovery_config;
        
        const auto& strategy = entry.strategy();
        recovery_config.strategy = pbStrategyToRecoveryStrategy(strategy.strategy());
        recovery_config.max_retries = strategy.max_retries();
        recovery_config.timeout_ms = strategy.timeout_ms();
        recovery_config.retry_interval_ms = strategy.retry_interval_ms();
        recovery_config.custom_recovery_func = strategy.custom_recovery_func();
        
        strategies[error_code] = recovery_config;
    }
    return true;
}

bool ErrorHandlerConfigParser::parseMonitorConfig(const errorhandler::MonitorConfig& pb_config, 
                                                 ErrorHandlerConfigData::MonitorConfig& config) {
    config.enable_monitoring = pb_config.enable_monitoring();
    config.enable_alerts = pb_config.enable_alerts();
    config.alert_threshold = pb_config.alert_threshold();
    config.time_window_ms = pb_config.time_window_ms();
    config.max_alert_history = pb_config.max_alert_history();
    
    // 解析告警通道
    config.alert_channels.clear();
    for (const auto& channel : pb_config.alert_channels()) {
        config.alert_channels.push_back(channel);
    }
    
    return true;
}

bool ErrorHandlerConfigParser::parseErrorThresholds(const errorhandler::ErrorThresholds& pb_config, 
                                                   std::map<ErrorCode, int>& thresholds,
                                                   std::map<std::string, int>& module_thresholds) {
    // 解析错误码阈值
    for (const auto& entry : pb_config.error_code_thresholds()) {
        ErrorCode error_code = stringToErrorCode(entry.error_code());
        thresholds[error_code] = entry.threshold();
    }
    
    // 解析模块阈值
    for (const auto& entry : pb_config.module_thresholds()) {
        module_thresholds[entry.module_name()] = entry.threshold();
    }
    
    return true;
}

bool ErrorHandlerConfigParser::parseAlertConfig(const errorhandler::AlertConfig& pb_config, 
                                               ErrorHandlerConfigData::AlertConfig& config) {
    // 解析告警级别
    for (const auto& entry : pb_config.alert_levels()) {
        ErrorCode error_code = stringToErrorCode(entry.error_code());
        ErrorSeverity severity = pbSeverityToErrorSeverity(entry.severity());
        config.alert_levels[error_code] = severity;
    }
    
    // 解析消息模板
    for (const auto& entry : pb_config.message_templates().templates()) {
        ErrorCode error_code = stringToErrorCode(entry.error_code());
        config.message_templates[error_code] = entry.template();
    }
    
    return true;
}

bool ErrorHandlerConfigParser::parseLogConfig(const errorhandler::LogConfig& pb_config, 
                                             ErrorHandlerConfigData::LogConfig& config) {
    config.error_log_level = pb_config.error_log_level();
    config.warning_log_level = pb_config.warning_log_level();
    config.info_log_level = pb_config.info_log_level();
    config.debug_log_level = pb_config.debug_log_level();
    config.log_format = pb_config.log_format();
    config.log_file_enabled = pb_config.log_file_enabled();
    config.log_file_path = pb_config.log_file_path();
    config.log_file_max_size_mb = pb_config.log_file_max_size_mb();
    config.log_file_max_files = pb_config.log_file_max_files();
    return true;
}

bool ErrorHandlerConfigParser::parsePerformanceConfig(const errorhandler::PerformanceConfig& pb_config, 
                                                     ErrorHandlerConfigData::PerformanceConfig& config) {
    config.statistics_interval_ms = pb_config.statistics_interval_ms();
    config.enable_performance_reporting = pb_config.enable_performance_reporting();
    config.performance_report_interval_ms = pb_config.performance_report_interval_ms();
    config.enable_memory_monitoring = pb_config.enable_memory_monitoring();
    config.memory_threshold_mb = pb_config.memory_threshold_mb();
    return true;
}

ErrorCode ErrorHandlerConfigParser::stringToErrorCode(const std::string& str) {
    if (str == "ALGORITHM_INFERENCE_FAILED") return ErrorCode::ALGORITHM_INFERENCE_FAILED;
    if (str == "MEMORY_ALLOCATION_FAILED") return ErrorCode::MEMORY_ALLOCATION_FAILED;
    if (str == "CONFIG_FILE_NOT_FOUND") return ErrorCode::CONFIG_FILE_NOT_FOUND;
    if (str == "CUDA_MEMORY_ALLOCATION_FAILED") return ErrorCode::CUDA_MEMORY_ALLOCATION_FAILED;
    if (str == "TENSORRT_INFERENCE_FAILED") return ErrorCode::TENSORRT_INFERENCE_FAILED;
    if (str == "SENSOR_TIMEOUT") return ErrorCode::SENSOR_TIMEOUT;
    if (str == "INVALID_PARAMETER") return ErrorCode::INVALID_PARAMETER;
    if (str == "NULL_POINTER") return ErrorCode::NULL_POINTER;
    if (str == "ALGORITHM_NOT_INITIALIZED") return ErrorCode::ALGORITHM_NOT_INITIALIZED;
    if (str == "CONFIG_PARAMETER_INVALID") return ErrorCode::CONFIG_PARAMETER_INVALID;
    if (str == "CONFIG_LOAD_ERROR") return ErrorCode::CONFIG_LOAD_ERROR;
    return ErrorCode::UNKNOWN_ERROR;
}

ErrorSeverity ErrorHandlerConfigParser::stringToErrorSeverity(const std::string& str) {
    if (str == "DEBUG") return ErrorSeverity::DEBUG;
    if (str == "INFO") return ErrorSeverity::INFO;
    if (str == "WARNING") return ErrorSeverity::WARNING;
    if (str == "ERROR") return ErrorSeverity::ERROR;
    if (str == "CRITICAL") return ErrorSeverity::CRITICAL;
    return ErrorSeverity::ERROR;
}

RecoveryStrategy ErrorHandlerConfigParser::stringToRecoveryStrategy(const std::string& str) {
    if (str == "NONE") return RecoveryStrategy::NONE;
    if (str == "RETRY") return RecoveryStrategy::RETRY;
    if (str == "RESTART") return RecoveryStrategy::RESTART;
    if (str == "FALLBACK") return RecoveryStrategy::FALLBACK;
    if (str == "RESET") return RecoveryStrategy::RESET;
    if (str == "CUSTOM") return RecoveryStrategy::CUSTOM;
    return RecoveryStrategy::NONE;
}

RecoveryStrategy ErrorHandlerConfigParser::pbStrategyToRecoveryStrategy(int pb_strategy) {
    switch (pb_strategy) {
        case 0: return RecoveryStrategy::NONE;
        case 1: return RecoveryStrategy::RETRY;
        case 2: return RecoveryStrategy::RESTART;
        case 3: return RecoveryStrategy::FALLBACK;
        case 4: return RecoveryStrategy::RESET;
        case 5: return RecoveryStrategy::CUSTOM;
        default: return RecoveryStrategy::NONE;
    }
}

ErrorSeverity ErrorHandlerConfigParser::pbSeverityToErrorSeverity(int pb_severity) {
    switch (pb_severity) {
        case 0: return ErrorSeverity::DEBUG;
        case 1: return ErrorSeverity::INFO;
        case 2: return ErrorSeverity::WARNING;
        case 3: return ErrorSeverity::ERROR;
        case 4: return ErrorSeverity::CRITICAL;
        default: return ErrorSeverity::ERROR;
    }
}

bool ErrorHandlerConfigParser::validateConfig(const ErrorHandlerConfigData& config) {
    // 基本验证
    if (config.global_config.max_error_history == 0) {
        std::cerr << "Invalid max_error_history: must be greater than 0" << std::endl;
        return false;
    }
    
    if (config.monitor_config.alert_threshold <= 0) {
        std::cerr << "Invalid alert_threshold: must be greater than 0" << std::endl;
        return false;
    }
    
    if (config.monitor_config.time_window_ms <= 0) {
        std::cerr << "Invalid time_window_ms: must be greater than 0" << std::endl;
        return false;
    }
    
    return true;
}

ErrorHandlerConfigData ErrorHandlerConfigParser::getDefaultConfig() {
    ErrorHandlerConfigData config;
    // 使用默认值（已在结构体定义中设置）
    return config;
}

ErrorHandlerConfigData ErrorHandlerConfigParser::mergeConfigs(const ErrorHandlerConfigData& global_config, 
                                                             const ErrorHandlerConfigData& module_config) {
    ErrorHandlerConfigData merged_config = global_config;
    
    // 合并恢复策略（模块配置优先）
    for (const auto& pair : module_config.recovery_strategies) {
        merged_config.recovery_strategies[pair.first] = pair.second;
    }
    
    // 合并错误阈值（模块配置优先）
    for (const auto& pair : module_config.error_thresholds) {
        merged_config.error_thresholds[pair.first] = pair.second;
    }
    
    for (const auto& pair : module_config.module_thresholds) {
        merged_config.module_thresholds[pair.first] = pair.second;
    }
    
    // 合并告警配置（模块配置优先）
    for (const auto& pair : module_config.alert_config.alert_levels) {
        merged_config.alert_config.alert_levels[pair.first] = pair.second;
    }
    
    for (const auto& pair : module_config.alert_config.message_templates) {
        merged_config.alert_config.message_templates[pair.first] = pair.second;
    }
    
    return merged_config;
} 