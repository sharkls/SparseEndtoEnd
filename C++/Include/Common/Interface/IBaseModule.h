/*******************************************************
 文件名：IBaseModule.h
 作者：sharkls
 描述：基础模块接口实现，用于基础模块的运行及结果数据处理
 版本：v1.0
 日期：2025-06-18
 *******************************************************/

#ifndef __IBASEMODULE_H__
#define __IBASEMODULE_H__

#include <string>
#include <memory>
#include <vector>

// 模块类型枚举
enum class ModuleType {
    PRE_PROCESS,    // 前处理模块
    INFERENCE,      // 推理模块
    POST_PROCESS    // 后处理模块
};

class IBaseModule {
public:
    IBaseModule(std::string exe_path) : m_exe_path(exe_path) {}
    virtual ~IBaseModule() = default;

    // 获取模块名称
    virtual std::string getModuleName() const = 0;
    
    // 获取模块类型
    virtual ModuleType getModuleType() const = 0;

    // 初始化模块，返回是否成功
    virtual bool init(void* p_pAlgParam) = 0;

    // 执行模块功能，返回执行结果
    virtual void execute() = 0;

    // 设置输入数据
    virtual void setInput(void* input) = 0;

    // 获取输出数据
    virtual void* getOutput() = 0;
    
    // 设置当前样本索引
    virtual void setCurrentSampleIndex(int sample_index) = 0;

    // 禁用拷贝构造和赋值操作
    IBaseModule(const IBaseModule&) = delete;
    IBaseModule& operator=(const IBaseModule&) = delete;

protected:
    std::string m_exe_path;
};

// 模块工厂基类
class IModuleFactory {
public:
    virtual ~IModuleFactory() = default;
    virtual std::shared_ptr<IBaseModule> createModule(const std::string& moduleName) = 0;
};

#endif // __IBASEMODULE_H__
