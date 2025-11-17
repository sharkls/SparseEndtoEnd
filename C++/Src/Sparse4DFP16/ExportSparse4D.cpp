#include "ExportSparse4D.h"
#include "sparse4d.hpp"

extern "C" __attribute__ ((visibility("default"))) ICore* CreateCoreObj(const std::string& p_strExePath)
{
    (void)p_strExePath; // 路径在 initAlgorithm 中处理
    return new sparse4d::core::CoreImplement();
}


