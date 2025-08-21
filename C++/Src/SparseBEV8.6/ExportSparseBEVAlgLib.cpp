#include "ExportSparseBEVAlgLib.h"
#include "CSparseBEVAlg.h"

extern "C" __attribute__ ((visibility("default"))) ISparseBEVAlg* CreateSparseBEV8_6AlgObj(const std::string& p_strExePath)
{
    return new CSparseBEVAlg(p_strExePath);
}