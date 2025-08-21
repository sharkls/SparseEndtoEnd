#!/usr/bin/env python3
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
"""
重新生成正确的temporal数据
使用PyTorch逻辑重新生成second frame的temporal数据
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
    
    return pred_instance_feature, pred_anchor, pred_class_score

def generate_correct_temporal_data(
    pred_instance_feature: np.ndarray,
    pred_anchor: np.ndarray,
    pred_class_score: np.ndarray,
    num_temp_instances: int = 600,
    confidence_decay: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """使用PyTorch逻辑生成正确的temporal数据"""
    logger = setup_logger()
    
    # 转换为PyTorch tensor
    instance_feature = torch.from_numpy(pred_instance_feature)
    anchor = torch.from_numpy(pred_anchor)
    confidence = torch.from_numpy(pred_class_score)
    
    # 计算置信度：max(dim=-1).values.sigmoid()
    confidence = confidence.max(dim=-1).values.sigmoid()  # (1, 900)
    
    logger.info(f"Generated confidence - range: [{confidence.min().item():.6f}, {confidence.max().item():.6f}]")
    
    # 选择置信度最高的num_temp_instances个实例
    confidence_sorted, indices = torch.topk(confidence, num_temp_instances, dim=1)
    
    logger.info(f"Selected {num_temp_instances} instances with confidence range: [{confidence_sorted.min().item():.6f}, {confidence_sorted.max().item():.6f}]")
    
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

def save_temporal_data(
    temp_instance_feature: np.ndarray,
    temp_anchor: np.ndarray,
    mask: np.ndarray,
    track_id: np.ndarray,
    sample_id: int,
    output_prefix: str = "script/tutorial/asset/"
):
    """保存temporal数据到文件"""
    logger = setup_logger()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # 保存临时实例特征
    temp_feature_path = f"{output_prefix}sample_{sample_id}_temp_instance_feature_1*600*256_float32.bin"
    temp_instance_feature.tofile(temp_feature_path)
    logger.info(f"Saved temp_instance_feature to: {temp_feature_path}")
    
    # 保存临时锚点
    temp_anchor_path = f"{output_prefix}sample_{sample_id}_temp_anchor_1*600*11_float32.bin"
    temp_anchor.tofile(temp_anchor_path)
    logger.info(f"Saved temp_anchor to: {temp_anchor_path}")
    
    # 保存掩码
    mask_path = f"{output_prefix}sample_{sample_id}_mask_1_int32.bin"
    mask.tofile(mask_path)
    logger.info(f"Saved mask to: {mask_path}")
    
    # 保存跟踪ID
    track_id_path = f"{output_prefix}sample_{sample_id}_track_id_1*900_int32.bin"
    track_id.tofile(track_id_path)
    logger.info(f"Saved track_id to: {track_id_path}")
    
    logger.info(f"All temporal data for sample {sample_id} saved successfully!")

def main():
    """主函数"""
    logger = setup_logger()
    logger.info("Starting temporal data regeneration...")
    
    # 重新生成多个样本的temporal数据
    for sample_id in range(1, 3):  # 样本1和2
        logger.info(f"\n{'='*60}")
        logger.info(f"Regenerating temporal data for sample {sample_id}")
        logger.info(f"{'='*60}")
        
        try:
            # 1. 加载第一帧结果
            pred_instance_feature, pred_anchor, pred_class_score = load_first_frame_results(sample_id)
            
            # 2. 使用PyTorch逻辑生成正确的temporal数据
            temp_instance_feature, temp_anchor, mask, track_id = generate_correct_temporal_data(
                pred_instance_feature, pred_anchor, pred_class_score
            )
            
            # 3. 保存temporal数据
            save_temporal_data(temp_instance_feature, temp_anchor, mask, track_id, sample_id)
            
            logger.info(f"✓ Sample {sample_id} temporal data regeneration completed!")
            
        except Exception as e:
            logger.error(f"Error regenerating temporal data for sample {sample_id}: {e}")
            return False
    
    logger.info("\n" + "="*60)
    logger.info("✓ All temporal data regeneration completed successfully!")
    logger.info("="*60)
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 