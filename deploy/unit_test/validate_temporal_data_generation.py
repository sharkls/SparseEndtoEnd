#!/usr/bin/env python3
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
"""
Temporal数据生成一致性验证工具
用于验证PyTorch和TensorRT版本生成的temporal数据是否一致
"""

import os
import numpy as np
import torch
import logging
from typing import Tuple, List

def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d::%H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_first_frame_results(sample_id: int, prefix: str = "script/tutorial/asset/") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载第一帧的推理结果"""
    logger = setup_logger()
    
    # 加载预测的实例特征
    pred_instance_feature = np.fromfile(
        f"{prefix}sample_{sample_id}_pred_instance_feature_1*900*256_float32.bin",
        dtype=np.float32
    ).reshape(1, 900, 256)
    
    # 加载预测的锚点
    pred_anchor = np.fromfile(
        f"{prefix}sample_{sample_id}_pred_anchor_1*900*11_float32.bin",
        dtype=np.float32
    ).reshape(1, 900, 11)
    
    # 加载预测的分类分数
    pred_class_score = np.fromfile(
        f"{prefix}sample_{sample_id}_pred_class_score_1*900*10_float32.bin",
        dtype=np.float32
    ).reshape(1, 900, 10)
    
    logger.info(f"Loaded first frame results for sample {sample_id}")
    logger.info(f"  pred_instance_feature shape: {pred_instance_feature.shape}")
    logger.info(f"  pred_anchor shape: {pred_anchor.shape}")
    logger.info(f"  pred_class_score shape: {pred_class_score.shape}")
    
    return pred_instance_feature, pred_anchor, pred_class_score

def load_expected_temporal_data(sample_id: int, prefix: str = "script/tutorial/asset/") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载期望的temporal数据"""
    logger = setup_logger()
    
    # 加载临时实例特征
    temp_instance_feature = np.fromfile(
        f"{prefix}sample_{sample_id}_temp_instance_feature_1*600*256_float32.bin",
        dtype=np.float32
    ).reshape(1, 600, 256)
    
    # 加载临时锚点
    temp_anchor = np.fromfile(
        f"{prefix}sample_{sample_id}_temp_anchor_1*600*11_float32.bin",
        dtype=np.float32
    ).reshape(1, 600, 11)
    
    # 加载掩码
    mask = np.fromfile(
        f"{prefix}sample_{sample_id}_mask_1_int32.bin",
        dtype=np.int32
    ).reshape(1)
    
    # 加载跟踪ID
    track_id = np.fromfile(
        f"{prefix}sample_{sample_id}_track_id_1*900_int32.bin",
        dtype=np.int32
    ).reshape(1, 900)
    
    logger.info(f"Loaded expected temporal data for sample {sample_id}")
    logger.info(f"  temp_instance_feature shape: {temp_instance_feature.shape}")
    logger.info(f"  temp_anchor shape: {temp_anchor.shape}")
    logger.info(f"  mask shape: {mask.shape}, value: {mask}")
    logger.info(f"  track_id shape: {track_id.shape}")
    
    return temp_instance_feature, temp_anchor, mask, track_id

def pytorch_temporal_data_generation(
    pred_instance_feature: np.ndarray,
    pred_anchor: np.ndarray,
    pred_class_score: np.ndarray,
    num_temp_instances: int = 600,
    confidence_decay: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PyTorch版本的temporal数据生成逻辑"""
    logger = setup_logger()
    
    # 转换为PyTorch tensor
    instance_feature = torch.from_numpy(pred_instance_feature)
    anchor = torch.from_numpy(pred_anchor)
    confidence = torch.from_numpy(pred_class_score)
    
    # 计算置信度：max(dim=-1).values.sigmoid()
    confidence = confidence.max(dim=-1).values.sigmoid()  # (1, 900)
    
    logger.info(f"PyTorch confidence calculation:")
    logger.info(f"  confidence shape: {confidence.shape}")
    logger.info(f"  confidence range: [{confidence.min().item():.6f}, {confidence.max().item():.6f}]")
    logger.info(f"  confidence mean: {confidence.mean().item():.6f}")
    
    # 选择置信度最高的num_temp_instances个实例
    confidence_sorted, indices = torch.topk(confidence, num_temp_instances, dim=1)
    
    logger.info(f"TopK selection:")
    logger.info(f"  selected confidence range: [{confidence_sorted.min().item():.6f}, {confidence_sorted.max().item():.6f}]")
    logger.info(f"  selected indices range: [{indices.min().item()}, {indices.max().item()}]")
    
    # 提取对应的实例特征和锚点
    batch_size = instance_feature.shape[0]
    indices_flat = indices + torch.arange(batch_size, device=indices.device)[:, None] * instance_feature.shape[1]
    indices_flat = indices_flat.flatten()
    
    temp_instance_feature = instance_feature.flatten(end_dim=1)[indices_flat].reshape(batch_size, num_temp_instances, -1)
    temp_anchor = anchor.flatten(end_dim=1)[indices_flat].reshape(batch_size, num_temp_instances, -1)
    
    # 生成掩码（假设所有temporal实例都有效）
    mask = torch.ones(1, dtype=torch.int32)
    
    # 生成跟踪ID（简化版本）
    track_id = torch.full((1, 900), -1, dtype=torch.int32)
    track_id[0, :num_temp_instances] = torch.arange(num_temp_instances, dtype=torch.int32)
    
    logger.info(f"Generated temporal data:")
    logger.info(f"  temp_instance_feature shape: {temp_instance_feature.shape}")
    logger.info(f"  temp_anchor shape: {temp_anchor.shape}")
    logger.info(f"  mask: {mask}")
    logger.info(f"  track_id shape: {track_id.shape}")
    
    return temp_instance_feature.numpy(), temp_anchor.numpy(), mask.numpy(), track_id.numpy()

def validate_temporal_data_consistency(
    pytorch_temp_feature: np.ndarray,
    pytorch_temp_anchor: np.ndarray,
    pytorch_mask: np.ndarray,
    pytorch_track_id: np.ndarray,
    expected_temp_feature: np.ndarray,
    expected_temp_anchor: np.ndarray,
    expected_mask: np.ndarray,
    expected_track_id: np.ndarray,
    tolerance: float = 1e-6
) -> bool:
    """验证temporal数据的一致性"""
    logger = setup_logger()
    
    # 验证实例特征
    feature_diff = np.abs(pytorch_temp_feature - expected_temp_feature)
    feature_max_diff = np.max(feature_diff)
    feature_mean_diff = np.mean(feature_diff)
    
    # 验证锚点
    anchor_diff = np.abs(pytorch_temp_anchor - expected_temp_anchor)
    anchor_max_diff = np.max(anchor_diff)
    anchor_mean_diff = np.mean(anchor_diff)
    
    # 验证掩码
    mask_diff = np.abs(pytorch_mask - expected_mask)
    mask_max_diff = np.max(mask_diff)
    
    # 验证跟踪ID
    track_id_diff = np.abs(pytorch_track_id - expected_track_id)
    track_id_max_diff = np.max(track_id_diff)
    
    logger.info("Temporal data consistency validation:")
    logger.info(f"  Instance feature - Max diff: {feature_max_diff:.6f}, Mean diff: {feature_mean_diff:.6f}")
    logger.info(f"  Anchor - Max diff: {anchor_max_diff:.6f}, Mean diff: {anchor_mean_diff:.6f}")
    logger.info(f"  Mask - Max diff: {mask_max_diff}")
    logger.info(f"  Track ID - Max diff: {track_id_max_diff}")
    
    # 检查是否在容差范围内
    is_consistent = (
        feature_max_diff < tolerance and
        anchor_max_diff < tolerance and
        mask_max_diff == 0 and
        track_id_max_diff == 0
    )
    
    if is_consistent:
        logger.info("✓ Temporal data consistency validation PASSED")
    else:
        logger.error("✗ Temporal data consistency validation FAILED")
    
    return is_consistent

def main():
    """主函数"""
    logger = setup_logger()
    logger.info("Starting temporal data generation consistency validation...")
    
    # 测试多个样本
    for sample_id in range(1, 3):  # 测试样本1和2
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing sample {sample_id}")
        logger.info(f"{'='*60}")
        
        try:
            # 1. 加载第一帧结果
            pred_instance_feature, pred_anchor, pred_class_score = load_first_frame_results(sample_id)
            
            # 2. 加载期望的temporal数据
            expected_temp_feature, expected_temp_anchor, expected_mask, expected_track_id = load_expected_temporal_data(sample_id)
            
            # 3. 使用PyTorch逻辑生成temporal数据
            pytorch_temp_feature, pytorch_temp_anchor, pytorch_mask, pytorch_track_id = pytorch_temporal_data_generation(
                pred_instance_feature, pred_anchor, pred_class_score
            )
            
            # 4. 验证一致性
            is_consistent = validate_temporal_data_consistency(
                pytorch_temp_feature, pytorch_temp_anchor, pytorch_mask, pytorch_track_id,
                expected_temp_feature, expected_temp_anchor, expected_mask, expected_track_id
            )
            
            if not is_consistent:
                logger.error(f"Sample {sample_id} temporal data generation is INCONSISTENT!")
                return False
                
        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {e}")
            return False
    
    logger.info("\n" + "="*60)
    logger.info("✓ All temporal data generation consistency validations PASSED!")
    logger.info("="*60)
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 