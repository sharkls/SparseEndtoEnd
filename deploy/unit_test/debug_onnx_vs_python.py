#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import logging
from tool.utils.logger import logger_wrapper
from modules.sparse4d_detector import Sparse4D
from tool.utils.config import read_cfg
from tool.runner.checkpoint import load_checkpoint
from tool.runner.fp16_utils import wrap_fp16_model
from tool.trainer.utils import set_random_seed

def build_module(cfg, default_args=None):
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)

def debug_onnx_vs_python():
    """对比ONNX导出和Python推理的差异"""
    logger, _, _ = logger_wrapper("", False)
    logger.setLevel(logging.INFO)
    
    # 设置随机种子
    set_random_seed(seed=1, deterministic=True)
    
    # 加载配置和模型
    cfg = read_cfg("dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py")
    model = build_module(cfg["model"])
    load_checkpoint(model, "ckpt/sparse4dv3_r50.pth", map_location="cpu")
    model.eval().cuda()
    
    # 加载第二帧的输入数据
    prefix = "script/tutorial/asset/"
    sample_id = 1
    
    # 加载输入数据
    feature = np.fromfile(
        f"{prefix}sample_{sample_id}_feature_1*89760*256_float32.bin",
        dtype=np.float32,
    ).reshape(1, 89760, 256)
    feature = torch.from_numpy(feature).cuda()
    
    spatial_shapes = np.fromfile(
        f"{prefix}sample_{sample_id}_spatial_shapes_6*4*2_int32.bin", dtype=np.int32
    ).reshape(6, 4, 2)
    spatial_shapes = torch.from_numpy(spatial_shapes).cuda()
    
    level_start_index = np.fromfile(
        f"{prefix}sample_{sample_id}_level_start_index_6*4_int32.bin",
        dtype=np.int32,
    ).reshape(6, 4)
    level_start_index = torch.from_numpy(level_start_index).cuda()
    
    instance_feature = np.fromfile(
        f"{prefix}sample_{sample_id}_instance_feature_1*900*256_float32.bin",
        dtype=np.float32,
    ).reshape(1, 900, 256)
    instance_feature = torch.from_numpy(instance_feature).cuda()
    
    anchor = np.fromfile(
        f"{prefix}sample_{sample_id}_anchor_1*900*11_float32.bin", dtype=np.float32
    ).reshape(1, 900, 11)
    anchor = torch.from_numpy(anchor).cuda()
    
    time_interval = np.fromfile(
        f"{prefix}sample_{sample_id}_time_interval_1_float32.bin",
        dtype=np.float32,
    ).reshape(1)
    time_interval = torch.from_numpy(time_interval).cuda()
    
    temp_instance_feature = np.fromfile(
        f"{prefix}sample_{sample_id}_temp_instance_feature_1*600*256_float32.bin",
        dtype=np.float32,
    ).reshape(1, 600, 256)
    temp_instance_feature = torch.from_numpy(temp_instance_feature).cuda()
    
    temp_anchor = np.fromfile(
        f"{prefix}sample_{sample_id}_temp_anchor_1*600*11_float32.bin", dtype=np.float32
    ).reshape(1, 600, 11)
    temp_anchor = torch.from_numpy(temp_anchor).cuda()
    
    mask = np.fromfile(
        f"{prefix}sample_{sample_id}_mask_1_int32.bin",
        dtype=np.int32,
    ).reshape(1)
    mask = torch.from_numpy(mask).cuda()
    
    track_id = np.fromfile(
        f"{prefix}sample_{sample_id}_track_id_1*900_int32.bin",
        dtype=np.int32,
    ).reshape(1, 900)
    track_id = torch.from_numpy(track_id).cuda()
    
    image_wh = np.fromfile(
        f"{prefix}sample_{sample_id}_image_wh_1*6*2_float32.bin",
        dtype=np.float32,
    ).reshape(1, 6, 2)
    image_wh = torch.from_numpy(image_wh).cuda()
    
    lidar2img = np.fromfile(
        f"{prefix}sample_{sample_id}_lidar2img_1*6*4*4_float32.bin",
        dtype=np.float32,
    ).reshape(1, 6, 4, 4)
    lidar2img = torch.from_numpy(lidar2img).cuda()
    
    # 加载期望输出
    pred_instance_feature = np.fromfile(
        f"{prefix}sample_{sample_id}_pred_instance_feature_1*900*256_float32.bin",
        dtype=np.float32,
    ).reshape(1, 900, 256)
    pred_instance_feature = torch.from_numpy(pred_instance_feature).cuda()
    
    pred_anchor = np.fromfile(
        f"{prefix}sample_{sample_id}_pred_anchor_1*900*11_float32.bin",
        dtype=np.float32,
    ).reshape(1, 900, 11)
    pred_anchor = torch.from_numpy(pred_anchor).cuda()
    
    pred_class_score = np.fromfile(
        f"{prefix}sample_{sample_id}_pred_class_score_1*900*10_float32.bin",
        dtype=np.float32,
    ).reshape(1, 900, 10)
    pred_class_score = torch.from_numpy(pred_class_score).cuda()
    
    pred_quality_score = np.fromfile(
        f"{prefix}sample_{sample_id}_pred_quality_score_1*900*2_float32.bin",
        dtype=np.float32,
    ).reshape(1, 900, 2)
    pred_quality_score = torch.from_numpy(pred_quality_score).cuda()
    
    # 执行Python推理
    logger.info("=== 执行Python推理 ===")
    with torch.no_grad():
        # 直接调用head的forward方法
        head = model.head
        
        # 准备输入
        feature_maps = [feature, spatial_shapes, level_start_index]
        metas = {
            "image_wh": image_wh,
            "lidar2img": lidar2img,
        }
        
        # 执行推理
        output = head.forward(feature_maps, metas)
        
        # 提取结果
        python_cls = output["classification"][-1]  # 最后一个分类结果
        python_qt = output["quality"][-1]  # 最后一个质量结果
        python_prediction = output["prediction"][-1]  # 最后一个预测结果
        
        # 从InstanceBank获取当前状态
        ib = head.instance_bank
        python_instance_feature = ib.instance_feature  # 当前实例特征
        python_anchor = ib.anchor  # 当前锚点
        
        logger.info(f"Python推理结果:")
        logger.info(f"  instance_feature shape: {python_instance_feature.shape}")
        logger.info(f"  anchor shape: {python_anchor.shape}")
        logger.info(f"  cls shape: {python_cls.shape}")
        logger.info(f"  qt shape: {python_qt.shape}")
        
        # 计算误差
        instance_feature_error = torch.abs(python_instance_feature - pred_instance_feature).max().item()
        anchor_error = torch.abs(python_anchor - pred_anchor).max().item()
        cls_error = torch.abs(python_cls - pred_class_score).max().item()
        qt_error = torch.abs(python_qt - pred_quality_score).max().item()
        
        logger.info(f"Python vs 期望输出误差:")
        logger.info(f"  instance_feature error: {instance_feature_error}")
        logger.info(f"  anchor error: {anchor_error}")
        logger.info(f"  cls error: {cls_error}")
        logger.info(f"  qt error: {qt_error}")
        
        # 检查InstanceBank状态
        logger.info("=== InstanceBank状态 ===")
        ib = head.instance_bank
        logger.info(f"  cached_feature: {ib.cached_feature is not None}")
        logger.info(f"  cached_anchor: {ib.cached_anchor is not None}")
        logger.info(f"  confidence: {ib.confidence is not None}")
        logger.info(f"  mask: {ib.mask is not None}")
        logger.info(f"  track_id: {ib.track_id is not None}")
        
        if ib.cached_feature is not None:
            logger.info(f"  cached_feature shape: {ib.cached_feature.shape}")
            logger.info(f"  cached_anchor shape: {ib.cached_anchor.shape}")
            logger.info(f"  confidence shape: {ib.confidence.shape}")
            if ib.mask is not None:
                logger.info(f"  mask shape: {ib.mask.shape}")
            if ib.track_id is not None:
                logger.info(f"  track_id shape: {ib.track_id.shape}")

if __name__ == "__main__":
    debug_onnx_vs_python() 