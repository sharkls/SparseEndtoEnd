#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import logging
from tool.utils.logger import logger_wrapper

def debug_onnx_trt_structure():
    """对比ONNX和TensorRT的结构差异"""
    logger, _, _ = logger_wrapper("", False)
    logger.setLevel(logging.INFO)
    
    # 加载第二帧的输入数据
    prefix = "script/tutorial/asset/"
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
        f"{prefix}sample_{sample_id}_temp_anchor_1*600*11_float32.bin", dtype=np.float32
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
    
    logger.info("=== 输入数据检查 ===")
    inputs = {
        "feature": feature,
        "spatial_shapes": spatial_shapes,
        "level_start_index": level_start_index,
        "instance_feature": instance_feature,
        "anchor": anchor,
        "time_interval": time_interval,
        "temp_instance_feature": temp_instance_feature,
        "temp_anchor": temp_anchor,
        "mask": mask,
        "track_id": track_id,
        "image_wh": image_wh,
        "lidar2img": lidar2img,
    }
    
    for name, data in inputs.items():
        logger.info(f"  {name}: shape={data.shape}, dtype={data.dtype}, min={data.min()}, max={data.max()}")
    
    # 检查mask的数据类型
    logger.info(f"mask数据类型: {mask.dtype}")
    logger.info(f"mask值: {mask}")
    
    # 检查track_id的数据
    valid_track_ids = track_id[track_id != -1]
    logger.info(f"有效track_id数量: {len(valid_track_ids)}")
    logger.info(f"track_id范围: {track_id.min()} ~ {track_id.max()}")
    
    # 检查第二帧特有的输入
    logger.info("=== 第二帧特有输入检查 ===")
    logger.info(f"temp_instance_feature - 非零元素: {np.count_nonzero(temp_instance_feature)}")
    logger.info(f"temp_anchor - 非零元素: {np.count_nonzero(temp_anchor)}")
    logger.info(f"mask - 值: {mask}")
    logger.info(f"track_id - 有效ID数量: {len(valid_track_ids)}")

if __name__ == "__main__":
    debug_onnx_trt_structure() 