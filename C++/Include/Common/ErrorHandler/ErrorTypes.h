/*******************************************************
 文件：ErrorTypes.h
 作者：sharkls
 描述：错误处理相关类型定义
 版本：v1.0
 日期：2025-01-15
 *******************************************************/
#ifndef __ERROR_TYPES_H__
#define __ERROR_TYPES_H__

#include <string>
#include <map>
#include <vector>
#include <chrono>
#include "GlobalContext.h"

// 错误信息结构
struct ErrorInfo {
    ErrorCode code;                    // 错误码
    std::string message;               // 错误消息
    std::string module_name;           // 模块名称
    std::string context;               // 错误上下文
    int64_t timestamp;                 // 时间戳
    std::string stack_trace;           // 堆栈跟踪（可选）
    
    ErrorInfo() : code(ErrorCode::UNKNOWN_ERROR), timestamp(0) {}
    
    ErrorInfo(ErrorCode c, const std::string& msg, const std::string& module = "", 
              const std::string& ctx = "", int64_t ts = 0)
        : code(c), message(msg), module_name(module), context(ctx), timestamp(ts) {}
};

// 错误统计结构
struct ErrorStatistics {
    std::map<ErrorCode, int> error_counts;                    // 各错误码出现次数
    std::map<std::string, int> module_error_counts;           // 各模块错误次数
    std::map<ErrorCode, int> recovery_counts;                 // 各错误码恢复次数
    int64_t total_errors;                                      // 总错误数
    int64_t total_recovered_errors;                           // 总恢复数
    double recovery_rate;                                      // 恢复率
    std::chrono::system_clock::time_point first_error_time;   // 首次错误时间
    std::chrono::system_clock::time_point last_error_time;    // 最后错误时间
    std::chrono::system_clock::time_point last_recovery_time; // 最后恢复时间
    
    ErrorStatistics() : total_errors(0), total_recovered_errors(0), recovery_rate(0.0) {}
};

// 错误严重程度
enum class ErrorSeverity : uint8_t {
    LOW = 0,        // 低严重程度
    MEDIUM = 1,     // 中等严重程度
    HIGH = 2,       // 高严重程度
    CRITICAL = 3    // 严重错误
};

// 错误分类
enum class ErrorCategory : uint8_t {
    SYSTEM = 0,         // 系统错误
    ALGORITHM = 1,      // 算法错误
    DATA = 2,           // 数据错误
    CONFIGURATION = 3,  // 配置错误
    NETWORK = 4,        // 网络错误
    HARDWARE = 5,       // 硬件错误
    USER = 6,           // 用户错误
    UNKNOWN = 7         // 未知错误
};

// 错误上下文信息
struct ErrorContext {
    std::string function_name;         // 函数名
    std::string file_name;             // 文件名
    int line_number;                   // 行号
    std::string thread_id;             // 线程ID
    std::map<std::string, std::string> additional_info; // 额外信息
    
    ErrorContext() : line_number(0) {}
};

#endif // __ERROR_TYPES_H__ 