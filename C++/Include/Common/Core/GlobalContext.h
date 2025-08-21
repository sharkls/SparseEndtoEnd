/*******************************************************
 文件：GlobalContext.h
 作者：sharkls
 描述：智能车全局上下文
 版本：v2.0
 日期：2025-05-15
 *******************************************************/
#ifndef __GLOBAL_CONTEXT_H__
#define __GLOBAL_CONTEXT_H__

#include <stdint.h>
#include <string>
#include <functional>
#include "CAlgResult.h"

// 算法接收的回调函数定义
using AlgCallback = std::function<void(const CAlgResult&, void*)>;

enum class ObstacleType : std::uint8_t {
  CAR = 0U,
  TRCUK = 1U,
  CONSTRUCTION_VEHICLE = 2U,
  BUS = 3U,
  TRAILER = 4U,
  BARRIER = 5U,
  MOTORCYCLE = 6U,
  BICYCLE = 7U,
  PEDESTRIAN = 8U,
  TRAFFIC_CONE = 9U,
  OBSTACLETYPE_Max = 10U,
  OBSTACLETYPE_INVALID = 255U,
};

enum class CameraFrameID : std::uint8_t {
  CAM_FRONT = 0U,
  CAM_FRONT_RIGHT = 1U,
  CAM_FRONT_LEFT = 2U,
  CAM_BACK = 3U,
  CAM_BACK_LEFT = 4U,
  CAM_BACK_RIGHT = 5U,
  CameraFrameID_Max = 6U,
  CameraFrameID_INVALID = 255U,
};

// ================== 状态码 ==================
enum Status : uint8_t
{
    kSuccess = 0,                    // 成功
    kInvalidInput,                   // 无效输入
    kMemoryErr,                      // 内存错误
    kInferenceErr,                   // 推理错误
    kImgPreprocesSizeErr,            // 图像预处理尺寸错误
    kImgPreprocesFormatErr,          // 图像预处理格式错误
    kImgPreprocesCudaErr,            // 图像预处理CUDA错误
    kImgPreprocesParamErr,           // 图像预处理参数错误
    kImgPreprocesMaxErr,             // 图像预处理最大错误码

    kImgPreprocessLaunchErr = 2U,

    kBackboneInferErr = 20U,

    kHead1stInferErr = 30U,
    kHead2ndInferErr = 31U,

    kDecoderErr = 50U
};

// ================== 时间戳类型 ==================
enum TIMESTAMP_TYPE : uint8_t
{
    // 图像数据像相关
    TIMESTAMP_IMAGE_EXPOSURE,      // 图像曝光时间戳
    TIMESTAMP_IMAGE_ARRIVE,        // 图像落地（接收）时间戳
    TIMESTAMP_IMAGE,               // 通用图像时间戳

    // 毫米波相关
    TIMESTAMP_RADAR,            // 毫米波时间戳

    // 激光雷达数据相关
    TIMESTAMP_LIDAR_EXPOSURE,      // 激光雷达曝光时间戳
    TIMESTAMP_LIDAR_ARRIVE,        // 激光雷达落地（接收）时间戳
    TIMESTAMP_LIDAR,               // 通用激光雷达时间戳

    // 时间匹配相关
    TIMESTAMP_TIME_MATCH,       // 用于时间匹配的时间戳

    // 融合跟踪算法相关
    TIMESTAMP_FUSIONTRACKING_BEGIN,     // 进入融合跟踪算法前时间戳
    TIMESTAMP_FUSIONTRACKING_END,       // 融合跟踪算法后时间戳

    // 点云检测算法相关
    TIMESTAMP_POINTCLOUD_DETECTION_BEGIN,     // 进入点云检测算法前时间戳
    TIMESTAMP_POINTCLOUD_DETECTION_END,       // 点云检测算法后时间戳

    // 图像检测算法相关
    TIMESTAMP_IMAGEDETECTION_BEGIN,     // 进入图像检测算法前时间戳
    TIMESTAMP_IMAGEDETECTION_END,       // 图像检测算法后时间戳

    // 跟踪算法相关
    TIMESTAMP_TRACKING_BEGIN,     // 进入跟踪算法前时间戳
    TIMESTAMP_TRACKING_END,       // 跟踪算法后时间戳

    // 毫米波雷达检测算法相关
    TIMESTAMP_RADARDETECTION_BEGIN,     // 进入毫米波雷达检测算法前时间戳
    TIMESTAMP_RADARDETECTION_END,       // 毫米波雷达检测算法后时间戳

    // BEV感知算法相关
    TIMESTAMP_BEV_BEGIN,               // BEV感知算法总开始时间戳
    TIMESTAMP_BEV_PREPROCESS_BEGIN,     // 进入BEV感知算法预处理前时间戳
    TIMESTAMP_BEV_PREPROCESS_END,       // BEV感知算法预处理后时间戳
    TIMESTAMP_BEV_INFERENCE_BEGIN,     // 进入BEV感知算法推理前时间戳
    TIMESTAMP_BEV_INFERENCE_END,       // BEV感知算法推理后时间戳
    TIMESTAMP_BEV_POSTPROCESS_BEGIN,     // 进入BEV感知算法后处理前时间戳
    TIMESTAMP_BEV_POSTPROCESS_END,       // BEV感知算法后处理后时间戳
    TIMESTAMP_BEV_END,                 // BEV感知算法总结束时间戳

    TIMESTAMP_TYPE_MAX          // 时间戳类型总个数
};

// ================== 帧率类型 ==================
enum EFPSType : uint8_t
{
    FPS_IMAGE_EXPOSURE,         // 图像曝光帧率
    FPS_IMAGE_ARRIVE,           // 图像落地帧率
    FPS_RADAR,                  // 毫米波数据帧率
    FPS_LIDAR_EXPOSURE,         // 激光雷达曝光帧率
    FPS_LIDAR_ARRIVE,           // 激光雷达落地帧率
    FPS_TIME_MATCH,             // 时间匹配帧率
    FPS_FUSIONTRACKING,         // 融合跟踪算法帧率
    FPS_POINTCLOUD_DETECTION,   // 点云检测算法帧率
    FPS_IMAGEDETECTION,         // 图像检测算法帧率
    FPS_TRACKING,               // 跟踪算法帧率
    FPS_RADARDETECTION,         // 毫米波雷达检测算法帧率
    FPS_BEV,                    // BEV感知算法帧率
    FPS_TYPE_MAX
};

// ================== 延时类型 ==================
enum EDelayType : uint8_t
{
    DELAY_TYPE_IMAGE_EXPOSURE_TO_ARRIVE,    // 图像曝光到落地延时
    DELAY_TYPE_LIDAR_EXPOSURE_TO_ARRIVE,    // 激光雷达曝光到落地延时
    DELAY_TYPE_RADAR,                       // 毫米波数据处理延时
    DELAY_TYPE_TIME_MATCH,                  // 时间匹配耗时
    DELAY_TYPE_FUSIONTRACKING,              // 融合跟踪算法耗时
    DELAY_TYPE_POINTCLOUD_DETECTION,        // 点云检测算法耗时
    DELAY_TYPE_IMAGEDETECTION,              // 图像检测算法耗时
    DELAY_TYPE_TRACKING,                    // 跟踪算法耗时
    DELAY_TYPE_RADARDETECTION,              // 毫米波雷达检测算法耗时

    DELAY_TYPE_BEV_PREPROCESS,              // BEV感知算法预处理耗时
    DELAY_TYPE_BEV_INFERENCE,               // BEV感知算法推理耗时
    DELAY_TYPE_BEV_POSTPROCESS,             // BEV感知算法后处理耗时
    DELAY_TYPE_BEV,                         // BEV感知算法总耗时

    DELAY_TYPE_PIPELINE_ALL,                // 全流程总延时
    DELAY_TYPE_MAX
};

// ================== 数据来源类型 ==================
enum DATA_SOURCE_TYPE : uint8_t
{
    DATA_SOURCE_ONLINE,    // 在线
    DATA_SOURCE_OFFLINE,   // 离线
    DATA_SOURCE_SIM,       // 仿真/回放
    DATA_SOURCE_TYPE_MAX
};

// ================== 数据类型 ==================
enum EDataType : uint8_t
{
    DATA_TYPE_IMAGE,                        // 图像数据
    DATA_TYPE_RADAR,                        // 毫米波数据
    DATA_TYPE_LIDAR,                        // 激光雷达数据
    DATA_TYPE_TIME_MATCHED,                 // 时间匹配后的多源数据
    DATA_TYPE_FUSIONTRACKING_RESULT,        // 融合跟踪算法结果
    DATA_TYPE_POINTCLOUD_DETECTION_RESULT,  // 点云检测算法结果
    DATA_TYPE_IMAGEDETECTION_RESULT,        // 图像检测算法结果
    DATA_TYPE_TRACKING_RESULT,              // 跟踪算法结果
    DATA_TYPE_RADARDETECTION_RESULT,        // 毫米波雷达检测算法结果

    // BEV 感知
    DATA_TYPE_BEV_PREPROCESS_RESULT,        // BEV感知算法预处理结果
    DATA_TYPE_BEV_INFERENCE_RESULT,         // BEV感知算法推理结果
    DATA_TYPE_BEV_POSTPROCESS_RESULT,       // BEV感知算法后处理结果
    DATA_TYPE_BEV_RESULT,                   // BEV感知算法总结果
    DATA_TYPE_POSEALG_RESULT,               // 姿态估计算法结果
    DATA_TYPE_MAX
};


#endif  // __GLOBAL_CONTEXT_H__