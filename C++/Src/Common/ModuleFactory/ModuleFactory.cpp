#include "ModuleFactory.h"

ModuleFactory& ModuleFactory::getInstance() {
    static ModuleFactory instance;
    return instance;
}

void ModuleFactory::registerModule(const std::string& task, const std::string& moduleName, ModuleCreator creator) {
    m_taskModuleCreators[task][moduleName] = creator;
}

std::shared_ptr<IBaseModule> ModuleFactory::createModule(const std::string& task, const std::string& moduleName, const std::string& exe_path) {
    auto taskIt = m_taskModuleCreators.find(task);
    if (taskIt != m_taskModuleCreators.end()) {
        auto& moduleMap = taskIt->second;
        auto it = moduleMap.find(moduleName);
        if (it != moduleMap.end()) {
            return it->second(exe_path);
        }
    }
    return nullptr;
} 