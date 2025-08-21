/*******************************************************
 文件名：ModuleFactory.h
 作者：sharkls
 描述：算法配置基类，用于算法配置的加载及获取
 版本：v1.0
 日期：2025-06-18
 *******************************************************/

#ifndef __MODULEFACTORY_H__
#define __MODULEFACTORY_H__

#include <string>
#include <memory>
#include <map>
#include <functional>
#include "IBaseModule.h"

using ModuleCreator = std::function<std::shared_ptr<IBaseModule>(const std::string& exe_path)>;

class ModuleFactory {
public:
    static ModuleFactory& getInstance();

    // 注册模块，按任务和模块名区分
    void registerModule(const std::string& task, const std::string& moduleName, ModuleCreator creator);
    // 创建模块实例
    std::shared_ptr<IBaseModule> createModule(const std::string& task, const std::string& moduleName, const std::string& exe_path);

private:
    ModuleFactory() = default;
    ~ModuleFactory() = default;
    ModuleFactory(const ModuleFactory&) = delete;
    ModuleFactory& operator=(const ModuleFactory&) = delete;

    std::map<std::string, std::map<std::string, ModuleCreator>> m_taskModuleCreators;
};

#define REGISTER_MODULE(task, moduleName, className) \
    static bool moduleName##_registered = []() { \
        ModuleFactory::getInstance().registerModule(task, #moduleName, \
            [](const std::string& exe_path) { return std::make_shared<className>(exe_path); }); \
        return true; \
    }(); 

#endif // __MODULEFACTORY_H__