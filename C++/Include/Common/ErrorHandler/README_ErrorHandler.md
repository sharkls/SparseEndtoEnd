# 错误处理模块使用说明

## 概述

错误处理模块是一个独立的公共模块，为各个算法模块提供统一的错误处理、恢复和监控功能。该模块采用配置驱动的方式，支持多种错误恢复策略和监控机制。

## 架构设计

### 核心组件

1. **ErrorTypes.h** - 错误类型定义
2. **IErrorHandler.h** - 错误处理接口
3. **ErrorHandler.h** - 错误处理主类
4. **ErrorRecovery.h** - 错误恢复策略
5. **ErrorMonitor.h** - 错误监控
6. **ErrorHandlerFactory.h** - 错误处理器工厂

### 错误处理流程

```
错误发生 → 错误报告 → 错误监控 → 错误恢复 → 结果反馈
    ↓         ↓         ↓         ↓         ↓
  检测错误   记录错误   统计错误   执行恢复   返回结果
```

## 使用方法

### 1. 基本使用

```cpp
#include "Include/Common/ErrorHandlerFactory.h"

class MyAlgorithm {
private:
    std::shared_ptr<IErrorHandler> m_errorHandler;
    
public:
    MyAlgorithm() {
        // 创建错误处理器
        m_errorHandler = ErrorHandlerFactory::getInstance().createErrorHandler("MyAlgorithm");
    }
    
    bool process() {
        // 业务逻辑
        if (!someOperation()) {
            // 报告错误
            m_errorHandler->reportError(ErrorCode::ALGORITHM_INFERENCE_FAILED, 
                                      "Operation failed", "MyAlgorithm");
            return false;
        }
        
        // 尝试恢复
        if (m_errorHandler->hasError()) {
            return m_errorHandler->tryRecovery(ErrorCode::ALGORITHM_INFERENCE_FAILED, "MyAlgorithm");
        }
        
        return true;
    }
};
```

### 2. 使用便捷宏

```cpp
// 获取错误处理器
auto handler = GET_ERROR_HANDLER("MyModule");

// 创建错误处理器
auto handler = CREATE_ERROR_HANDLER("MyModule");

// 创建带配置的错误处理器
ErrorHandlerConfig config;
config.enable_error_recovery = true;
auto handler = CREATE_ERROR_HANDLER_WITH_CONFIG("MyModule", config);
```

### 3. 错误恢复策略

```cpp
// 设置重试策略
auto recovery = handler->getErrorRecovery();
recovery->setRecoveryStrategy(ErrorCode::ALGORITHM_INFERENCE_FAILED, RecoveryStrategy::RETRY);
recovery->setRetryCount(ErrorCode::ALGORITHM_INFERENCE_FAILED, 3);
recovery->setRecoveryTimeout(ErrorCode::ALGORITHM_INFERENCE_FAILED, 10000);

// 设置自定义恢复函数
recovery->setCustomRecoveryFunc(ErrorCode::CUSTOM_ERROR, []() {
    // 自定义恢复逻辑
    return performCustomRecovery();
});
```

### 4. 错误监控

```cpp
// 设置错误阈值
handler->setErrorThreshold(ErrorCode::ALGORITHM_INFERENCE_FAILED, 5);

// 设置告警回调
handler->setAlertCallback([](const ErrorInfo& error) {
    std::cout << "Alert: " << error.message << std::endl;
});

// 检查是否超过阈值
if (handler->isThresholdExceeded(ErrorCode::ALGORITHM_INFERENCE_FAILED)) {
    // 处理告警
}
```

## 配置说明

### 配置文件格式

错误处理模块支持通过配置文件进行配置，配置文件采用类似JSON的格式：

```yaml
# 全局配置
global_config {
    enable_error_handling: true
    enable_error_recovery: true
    enable_error_monitoring: true
    max_error_history: 1000
    log_level: 1
}

# 错误恢复策略配置
recovery_strategies {
    ALGORITHM_INFERENCE_FAILED {
        strategy: RETRY
        max_retries: 3
        timeout_ms: 10000
        retry_interval_ms: 1000
    }
}
```

### 支持的恢复策略

1. **NONE** - 不恢复
2. **RETRY** - 重试策略
3. **RESTART** - 重启模块
4. **FALLBACK** - 降级处理
5. **RESET** - 重置状态
6. **CUSTOM** - 自定义策略

## 错误码说明

### 通用错误码 (0-999)
- `SUCCESS` - 成功
- `INVALID_PARAMETER` - 无效参数
- `NULL_POINTER` - 空指针
- `MEMORY_ALLOCATION_FAILED` - 内存分配失败
- `TIMEOUT` - 超时

### 算法相关错误码 (1000-1999)
- `ALGORITHM_NOT_INITIALIZED` - 算法未初始化
- `ALGORITHM_INFERENCE_FAILED` - 算法推理失败
- `ALGORITHM_PREPROCESS_FAILED` - 算法预处理失败
- `ALGORITHM_POSTPROCESS_FAILED` - 算法后处理失败

### 图像处理相关错误码 (2000-2999)
- `IMAGE_INVALID_FORMAT` - 图像格式无效
- `IMAGE_INVALID_SIZE` - 图像尺寸无效
- `IMAGE_DECODE_FAILED` - 图像解码失败

### CUDA相关错误码 (3000-3999)
- `CUDA_INITIALIZATION_FAILED` - CUDA初始化失败
- `CUDA_MEMORY_ALLOCATION_FAILED` - CUDA内存分配失败
- `CUDA_KERNEL_LAUNCH_FAILED` - CUDA核函数启动失败

### TensorRT相关错误码 (4000-4999)
- `TENSORRT_INITIALIZATION_FAILED` - TensorRT初始化失败
- `TENSORRT_INFERENCE_FAILED` - TensorRT推理失败
- `TENSORRT_ENGINE_LOAD_FAILED` - TensorRT引擎加载失败

## 性能监控

### 错误统计

```cpp
// 获取错误统计
auto stats = handler->getErrorStatistics();
std::cout << "Total errors: " << stats.total_errors << std::endl;
std::cout << "Recovery rate: " << stats.recovery_rate << std::endl;

// 获取指定模块的统计
auto moduleStats = handler->getErrorStatistics("MyModule");
```

### 错误报告

```cpp
// 生成错误报告
std::string report = handler->generateErrorReport();
std::cout << report << std::endl;

// 导出错误数据
handler->exportErrorData("error_data.json");
```

## 最佳实践

### 1. 错误处理原则

- **及时报告**：错误发生后立即报告
- **分类处理**：根据错误类型选择合适的处理策略
- **记录详细**：记录足够的错误上下文信息
- **优雅降级**：提供降级处理机制

### 2. 配置管理

- **环境适配**：根据运行环境调整配置
- **动态配置**：支持运行时配置更新
- **配置验证**：确保配置参数的有效性

### 3. 监控告警

- **阈值设置**：合理设置错误阈值
- **告警分级**：根据错误严重程度分级告警
- **统计分析**：定期分析错误统计信息

### 4. 性能优化

- **异步处理**：错误处理不应阻塞主流程
- **资源管理**：合理管理错误处理相关资源
- **缓存机制**：缓存常用的错误处理结果

## 扩展开发

### 1. 添加新的错误码

在 `GlobalContext.h` 中的 `ErrorCode` 枚举中添加新的错误码：

```cpp
enum class ErrorCode : uint32_t {
    // 现有错误码...
    
    // 新增错误码
    MY_CUSTOM_ERROR = 10000,
    MY_ANOTHER_ERROR = 10001,
    
    // 最大错误码
    ERROR_CODE_MAX = 65535
};
```

### 2. 添加新的恢复策略

在 `ErrorRecovery.h` 中的 `RecoveryStrategy` 枚举中添加新的策略：

```cpp
enum class RecoveryStrategy : uint8_t {
    // 现有策略...
    
    // 新增策略
    MY_CUSTOM_STRATEGY = 6,
    
    // 最大策略数
    RECOVERY_STRATEGY_MAX = 7
};
```

### 3. 自定义错误处理器

继承 `IErrorHandler` 接口实现自定义错误处理器：

```cpp
class MyCustomErrorHandler : public IErrorHandler {
public:
    // 实现接口方法...
    
private:
    // 自定义成员...
};
```

## 故障排除

### 常见问题

1. **错误处理器创建失败**
   - 检查模块名称是否有效
   - 确认错误处理模块已正确初始化

2. **错误恢复不生效**
   - 检查恢复策略配置
   - 确认错误码是否支持恢复

3. **监控告警不触发**
   - 检查阈值设置
   - 确认告警回调函数正确设置

4. **性能影响**
   - 检查错误处理是否在主流程中执行
   - 考虑使用异步错误处理

### 调试技巧

1. **启用详细日志**
   ```cpp
   handler->setLogLevel(3); // DEBUG级别
   ```

2. **检查错误历史**
   ```cpp
   auto history = handler->getErrorHistory();
   for (const auto& error : history) {
       std::cout << error.message << std::endl;
   }
   ```

3. **导出错误数据**
   ```cpp
   handler->exportErrorData("debug_errors.json");
   ```

## 版本历史

- **v1.0** (2025-01-15)
  - 初始版本
  - 基础错误处理功能
  - 错误恢复策略
  - 错误监控和告警
  - 配置驱动设计 