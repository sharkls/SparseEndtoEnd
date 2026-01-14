#ifndef LOG_H
#define LOG_H

/**
 * @file log.h
 * @brief glog 日志库包装头文件
 * 
 * 此文件提供对 glog 日志库的统一访问接口。
 * 代码中使用 #include "log.h" 即可使用 glog 的 LOG(INFO) 等宏。
 * 
 * 使用示例:
 *   LOG(INFO) << "信息日志";
 *   LOG(WARNING) << "警告日志";
 *   LOG(ERROR) << "错误日志";
 *   TINFO << "带模块名的信息日志";
 */

// glog 0.8.0 需要在使用前定义 GLOG_USE_GLOG_EXPORT
#define GLOG_USE_GLOG_EXPORT

#include <glog/logging.h>
#include <glog/log_severity.h>

// 兼容旧代码的自定义宏定义
#ifndef MOUDLE_NAME
#define MODULE_NAME GetName().c_str()
#endif

#define LEFT_BRACKET "["
#define RIGHT_BRACKET "]"

#define DEBUG_MODULE(module) VLOG(4) << LEFT_BRACKET << module << RIGHT_BRACKET << "[DEBUG] "

#ifndef LOG_MODULE_STREAM
#define LOG_MODULE_STREAM(log_severity) \
        LOG_MODULE_STREAM_##log_severity
#endif

#ifndef LOG_MODULE
#define LOG_MODULE(module, log_severity) \
        LOG_MODULE_STREAM(log_severity)(module)
#endif

#define LOG_MODULE_STREAM_INFO(module)                                  \
        google::LogMessage(__FILE__, __LINE__, google::INFO).stream()   \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define LOG_MODULE_STREAM_WARN(module)                                      \
        google::LogMessage(__FILE__, __LINE__, google::WARNING).stream()    \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define LOG_MODULE_STREAM_ERROR(module)                                 \
        google::LogMessage(__FILE__, __LINE__, google::ERROR).stream()  \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define LOG_MODULE_STREAM_FATAL(module)                                 \
        google::LogMessage(__FILE__, __LINE__, google::FATAL).stream()  \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define TDEBUG  DEBUG_MODULE(MODULE_NAME)
#define TINFO   LOG_MODULE(MODULE_NAME, INFO)
#define TWARN   LOG_MODULE(MODULE_NAME, WARN)
#define TERROR  LOG_MODULE(MODULE_NAME, ERROR)
#define TFATAL  LOG_MODULE(MODULE_NAME, FATAL)

#endif // LOG_H
