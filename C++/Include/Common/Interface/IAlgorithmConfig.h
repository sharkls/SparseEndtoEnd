/*******************************************************
 文件名：IAlgorithmConfig.h
 作者：sharkls
 描述：算法配置基类，用于算法配置的加载及获取
 版本：v1.0
 日期：2025-06-18
 *******************************************************/

#ifndef __ALGORITHMCONFIG_H__
#define __ALGORITHMCONFIG_H__

#include <string>
#include <google/protobuf/message.h>

class IAlgorithmConfig {
public:
    virtual ~IAlgorithmConfig() = default;
    // 加载配置文件（可重载支持不同格式）
    virtual bool loadFromFile(const std::string& path) = 0;
    // 获取protobuf配置对象
    virtual const google::protobuf::Message* getConfigMessage() const = 0;
}; 

#endif // __ALGORITHMCONFIG_H__