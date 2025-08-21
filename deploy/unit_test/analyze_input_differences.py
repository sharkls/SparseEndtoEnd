#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import logging
from tool.utils.logger import logger_wrapper

def analyze_input_differences():
    """分析第二帧推理输入数据的差异"""
    logger, _, _ = logger_wrapper("", False)
    logger.setLevel(logging.INFO)
    
    prefix = "script/tutorial/asset/"
    
    # 分析Sample 1的数据（第二帧）
    sample_id = 1
    
    # 加载输入数据
    feature = np.fromfile(
        f"{prefix}sample_{sample_id}_feature_1*89760*256_float32.bin",
        dtype=np.float32,
    ).reshape(1, 89760, 256)
    
    spatial_shapes = np.fromfile(
        f"{prefix}sample_{sample_id}_spatial_shapes_6*4*2_int32.bin", dtype=np.int32
    ).reshape(6, 4, 2)
    
    level_start_index = np.fromfile(
        f"{prefix}sample_{sample_id}_level_start_index_6*4_int32.bin",
        dtype=np.int32,
    ).reshape(6, 4)
    
    instance_feature = np.fromfile(
        f"{prefix}sample_{sample_id}_instance_feature_1*900*256_float32.bin",
        dtype=np.float32,
    ).reshape(1, 900, 256)
    
    anchor = np.fromfile(
        f"{prefix}sample_{sample_id}_anchor_1*900*11_float32.bin", dtype=np.float32
    ).reshape(1, 900, 11)
    
    time_interval = np.fromfile(
        f"{prefix}sample_{sample_id}_time_interval_1_float32.bin",
        dtype=np.float32,
    ).reshape(1)
    
    temp_instance_feature = np.fromfile(
        f"{prefix}sample_{sample_id}_temp_instance_feature_1*600*256_float32.bin",
        dtype=np.float32,
    ).reshape(1, 600, 256)
    
    temp_anchor = np.fromfile(
        f"{prefix}sample_{sample_id}_temp_anchor_1*600*11_float32.bin",
        dtype=np.float32,
    ).reshape(1, 600, 11)
    
    mask = np.fromfile(
        f"{prefix}sample_{sample_id}_mask_1_int32.bin",
        dtype=np.int32,
    ).reshape(1)
    
    track_id = np.fromfile(
        f"{prefix}sample_{sample_id}_track_id_1*900_int32.bin",
        dtype=np.int32,
    ).reshape(1, 900)
    
    image_wh = np.fromfile(
        f"{prefix}sample_{sample_id}_image_wh_1*6*2_float32.bin",
        dtype=np.float32,
    ).reshape(1, 6, 2)
    
    lidar2img = np.fromfile(
        f"{prefix}sample_{sample_id}_lidar2img_1*6*4*4_float32.bin",
        dtype=np.float32,
    ).reshape(1, 6, 4, 4)
    
    # 分析关键输入数据
    logger.info("=== 第二帧推理输入数据分析 ===")
    
    # 1. 分析instance_feature（当前帧实例特征）
    logger.info(f"instance_feature - 全零检查: {np.all(instance_feature == 0)}")
    logger.info(f"instance_feature - 形状: {instance_feature.shape}")
    logger.info(f"instance_feature - 统计: min={instance_feature.min()}, max={instance_feature.max()}, mean={instance_feature.mean()}")
    
    # 2. 分析anchor（当前帧锚点）
    logger.info(f"anchor - 形状: {anchor.shape}")
    logger.info(f"anchor - 统计: min={anchor.min()}, max={anchor.max()}, mean={anchor.mean()}")
    
    # 3. 分析temp_instance_feature（temporal实例特征）
    logger.info(f"temp_instance_feature - 形状: {temp_instance_feature.shape}")
    logger.info(f"temp_instance_feature - 统计: min={temp_instance_feature.min()}, max={temp_instance_feature.max()}, mean={temp_instance_feature.mean()}")
    
    # 4. 分析temp_anchor（temporal锚点）
    logger.info(f"temp_anchor - 形状: {temp_anchor.shape}")
    logger.info(f"temp_anchor - 统计: min={temp_anchor.min()}, max={temp_anchor.max()}, mean={temp_anchor.mean()}")
    
    # 5. 分析track_id
    logger.info(f"track_id - 形状: {track_id.shape}")
    logger.info(f"track_id - 统计: min={track_id.min()}, max={track_id.max()}, mean={track_id.mean()}")
    logger.info(f"track_id - 非-1元素数量: {np.sum(track_id != -1)}")
    
    # 6. 分析mask
    logger.info(f"mask - 形状: {mask.shape}")
    logger.info(f"mask - 值: {mask}")
    
    # 7. 分析time_interval
    logger.info(f"time_interval - 形状: {time_interval.shape}")
    logger.info(f"time_interval - 值: {time_interval}")
    
    # 8. 检查数据一致性
    logger.info("\n=== 数据一致性检查 ===")
    
    # 检查instance_feature是否全零（这可能是问题所在）
    if np.all(instance_feature == 0):
        logger.warning("⚠️  WARNING: instance_feature 全为零！这可能导致推理结果不正确。")
        logger.warning("   第二帧推理应该使用当前帧的实例特征，而不是全零。")
    
    # 检查temp_instance_feature是否有有效数据
    if np.all(temp_instance_feature == 0):
        logger.warning("⚠️  WARNING: temp_instance_feature 全为零！")
    else:
        logger.info("✅ temp_instance_feature 包含有效数据")
    
    # 检查temp_anchor是否有有效数据
    if np.all(temp_anchor == 0):
        logger.warning("⚠️  WARNING: temp_anchor 全为零！")
    else:
        logger.info("✅ temp_anchor 包含有效数据")
    
    # 9. 分析可能的问题
    logger.info("\n=== 问题分析 ===")
    
    if np.all(instance_feature == 0):
        logger.error("❌ 主要问题: instance_feature 全为零")
        logger.error("   第二帧推理需要当前帧的实例特征，但数据生成时可能没有正确设置")
        logger.error("   这会导致模型无法正确处理当前帧的信息")
    
    # 检查track_id的有效性
    valid_track_ids = track_id[track_id != -1]
    if len(valid_track_ids) == 0:
        logger.warning("⚠️  WARNING: 没有有效的track_id")
    else:
        logger.info(f"✅ 有效track_id数量: {len(valid_track_ids)}")
    
    return {
        'instance_feature_zero': np.all(instance_feature == 0),
        'temp_feature_valid': not np.all(temp_instance_feature == 0),
        'temp_anchor_valid': not np.all(temp_anchor == 0),
        'track_id_valid': len(valid_track_ids) > 0
    }

if __name__ == "__main__":
    analyze_input_differences() 