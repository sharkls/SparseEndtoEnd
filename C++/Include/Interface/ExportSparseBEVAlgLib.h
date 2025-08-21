/*******************************************************
 文件名：ExportSparseEnd2EndAlgLib.h
 作者：sharkls
 描述：SparseEnd2End算法库的算法接口类导出函数头文件
 版本：v1.0
 日期：2025-06-17
 *******************************************************/
#ifndef __EXPORT_SPARSEBEVALG_LIB_H__
#define __EXPORT_SPARSEBEVALG_LIB_H__

#include <string>
#include "../Common/Core/GlobalContext.h"

struct ISparseBEVAlg
{
    ISparseBEVAlg(){};
    virtual ~ISparseBEVAlg(){};

    //初始化算法接口对象，内部主要处理只需初始化一次的操作，比如模型加载之类的，成功返回true，失败返回false
    virtual bool initAlgorithm(const std::string exe_path,  const AlgCallback& alg_cb, void* hd)   = 0;

    //执行算法函数，传入原始数据体，算法执行成功返回处理后的数据或者检测结果（由算法类型而定），失败返回nullptr
    virtual void runAlgorithm(void* p_pSrcData)  = 0;
};

extern "C" __attribute__ ((visibility("default"))) ISparseBEVAlg*   CreateSparseBEVAlgObj(const std::string& p_strExePath);

// SparseBEV8.6版本导出函数
extern "C" __attribute__ ((visibility("default"))) ISparseBEVAlg*   CreateSparseBEV8_6AlgObj(const std::string& p_strExePath);

#endif // __EXPORT_SPARSEBEVALG_LIB_H__