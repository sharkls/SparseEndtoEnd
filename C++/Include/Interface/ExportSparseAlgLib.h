/*******************************************************
 文件名：ExportSparseEnd2EndAlgLib.h
 作者：sharkls
 描述：SparseEnd2End算法库的算法接口类导出函数头文件
 版本：v1.0
 日期：2025-06-17
 *******************************************************/
#pragma once
#include <string>
#include "GlobalContext.h"

struct ISparseEnd2EndAlg
{
    ISparseEnd2EndAlg(){};
    virtual ~ISparseEnd2EndAlg(){};

    //初始化算法接口对象，内部主要处理只需初始化一次的操作，比如模型加载之类的，成功返回true，失败返回false
    virtual bool initAlgorithm(const std::string exe_path,  const AlgCallback& alg_cb, void* hd)   = 0;

    //执行算法函数，传入原始数据体，算法执行成功返回处理后的数据或者检测结果（由算法类型而定），失败返回nullptr
    virtual void runAlgorithm(void* p_pSrcData)  = 0;
};

extern "C" __attribute__ ((visibility("default"))) ISparseEnd2EndAlg*   CreateSparseEnd2EndAlgObj(const std::string& p_strExePath);