/*******************************************************
 文件：ErrorMonitor.h
 作者：sharkls
 描述：错误监控类定义
 版本：v1.0
 日期：2025-01-15
 *******************************************************/
#ifndef __ERROR_MONITOR_H__
#define __ERROR_MONITOR_H__

#include <string>
#include <map>
#include <vector>
#include <functional>
#include <chrono>
#include <mutex>
#include "ErrorTypes.h"

// 告警级别
enum class AlertLevel : uint8_t {
    INFO = 0,       // 信息
    WARNING = 1,    // 警告
    ERROR = 2,      // 错误
    CRITICAL = 3    // 严重
};

// 告警信息
struct AlertInfo {
    AlertLevel level;                           // 告警级别
    std::string message;                        // 告警消息
    std::string module_name;                    // 模块名称
    ErrorCode error_code;                       // 错误码
    int64_t timestamp;                          // 时间戳
    std::map<std::string, std::string> context; // 上下文信息
    
    AlertInfo() : level(AlertLevel::INFO), error_code(ErrorCode::UNKNOWN_ERROR), timestamp(0) {}
};

// 监控配置
struct MonitorConfig {
    bool enable_monitoring;                     // 是否启用监控
    bool enable_alerts;                         // 是否启用告警
    int alert_threshold;                        // 告警阈值
    int time_window_ms;                         // 时间窗口（毫秒）
    std::vector<std::string> alert_channels;    // 告警通道
    std::function<void(const AlertInfo&)> alert_callback; // 告警回调
    
    MonitorConfig() : enable_monitoring(true), enable_alerts(true), 
                     alert_threshold(10), time_window_ms(60000) {}
};

// 错误监控类
class ErrorMonitor {
public:
    ErrorMonitor();
    ~ErrorMonitor() = default;
    
    // ================== 监控配置 ==================
    // 设置监控配置
    void setMonitorConfig(const MonitorConfig& config);
    
    // 获取监控配置
    MonitorConfig getMonitorConfig() const;
    
    // 启用/禁用监控
    void setMonitoringEnabled(bool enabled);
    
    // 启用/禁用告警
    void setAlertsEnabled(bool enabled);
    
    // 设置告警阈值
    void setAlertThreshold(int threshold);
    
    // 设置时间窗口
    void setTimeWindow(int window_ms);
    
    // 设置告警回调
    void setAlertCallback(std::function<void(const AlertInfo&)> callback);
    
    // ================== 错误阈值管理 ==================
    // 设置错误码阈值
    void setErrorThreshold(ErrorCode code, int threshold);
    
    // 设置模块错误阈值
    void setModuleErrorThreshold(const std::string& module_name, int threshold);
    
    // 获取错误码阈值
    int getErrorThreshold(ErrorCode code) const;
    
    // 获取模块错误阈值
    int getModuleErrorThreshold(const std::string& module_name) const;
    
    // 检查是否超过阈值
    bool isThresholdExceeded(ErrorCode code) const;
    
    // 检查模块是否超过阈值
    bool isModuleThresholdExceeded(const std::string& module_name) const;
    
    // ================== 错误统计更新 ==================
    // 更新错误统计
    void updateStatistics(const ErrorInfo& error);
    
    // 更新恢复统计
    void updateRecoveryStatistics(ErrorCode code, bool success);
    
    // 获取错误统计
    ErrorStatistics getStatistics() const;
    
    // 获取指定模块的统计
    ErrorStatistics getStatistics(const std::string& module_name) const;
    
    // 重置统计
    void resetStatistics();
    
    // ================== 告警管理 ==================
    // 触发告警
    void triggerAlert(AlertLevel level, const std::string& message, 
                     const std::string& module_name, ErrorCode error_code,
                     const std::map<std::string, std::string>& context = {});
    
    // 获取告警历史
    std::vector<AlertInfo> getAlertHistory() const;
    
    // 清除告警历史
    void clearAlertHistory();
    
    // 设置最大告警历史数量
    void setMaxAlertHistory(size_t max_count);
    
    // ================== 实时监控 ==================
    // 检查错误频率
    bool isErrorFrequencyHigh(ErrorCode code, int time_window_ms = 60000) const;
    
    // 检查模块错误频率
    bool isModuleErrorFrequencyHigh(const std::string& module_name, int time_window_ms = 60000) const;
    
    // 获取错误频率
    double getErrorFrequency(ErrorCode code, int time_window_ms = 60000) const;
    
    // 获取模块错误频率
    double getModuleErrorFrequency(const std::string& module_name, int time_window_ms = 60000) const;
    
    // ================== 报告生成 ==================
    // 生成错误报告
    std::string generateErrorReport() const;
    
    // 生成模块错误报告
    std::string generateModuleErrorReport(const std::string& module_name) const;
    
    // 生成告警报告
    std::string generateAlertReport() const;
    
    // 导出统计数据
    void exportStatistics(const std::string& file_path) const;
    
private:
    // 检查是否需要触发告警
    bool shouldTriggerAlert(ErrorCode code) const;
    
    // 检查模块是否需要触发告警
    bool shouldTriggerModuleAlert(const std::string& module_name) const;
    
    // 计算错误频率
    double calculateErrorFrequency(const std::vector<int64_t>& timestamps, int time_window_ms) const;
    
    // 清理过期数据
    void cleanupExpiredData();
    
    // 发送告警
    void sendAlert(const AlertInfo& alert);

private:
    mutable std::mutex m_mutex;                                    // 线程安全锁
    MonitorConfig m_config;                                        // 监控配置
    ErrorStatistics m_statistics;                                  // 错误统计
    std::map<std::string, ErrorStatistics> m_module_statistics;   // 模块错误统计
    std::map<ErrorCode, int> m_error_thresholds;                   // 错误码阈值
    std::map<std::string, int> m_module_thresholds;                // 模块错误阈值
    std::map<ErrorCode, std::vector<int64_t>> m_error_timestamps;  // 错误时间戳
    std::map<std::string, std::vector<int64_t>> m_module_timestamps; // 模块错误时间戳
    std::vector<AlertInfo> m_alert_history;                        // 告警历史
    size_t m_max_alert_history;                                    // 最大告警历史数量
    std::chrono::steady_clock::time_point m_last_cleanup;          // 上次清理时间
};

#endif // __ERROR_MONITOR_H__ 