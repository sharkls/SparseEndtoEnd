#include "ExportSparseBEVAlgLib.h"
#include "CSparseBEVAlg.h"

extern "C" __attribute__ ((visibility("default"))) ISparseBEVAlg* CreateSparseBEVAlgObj(const std::string& p_strExePath)
{
    return new CSparseBEVAlg(p_strExePath);
}