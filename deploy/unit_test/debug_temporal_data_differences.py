#!/usr/bin/env python3
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
"""
Temporal数据差异调试工具
用于详细分析PyTorch和期望的temporal数据之间的差异
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
    
    return temp_instance_feature, temp_anchor, mask, track_id

def analyze_confidence_differences(pred_class_score: np.ndarray, expected_temp_feature: np.ndarray, expected_temp_anchor: np.ndarray):
    """分析置信度差异"""
    logger = setup_logger()
    
    # 转换为PyTorch tensor
    confidence = torch.from_numpy(pred_class_score)
    
    # 计算置信度：max(dim=-1).values.sigmoid()
    confidence_scores = confidence.max(dim=-1).values.sigmoid()  # (1, 900)
    
    logger.info("Confidence analysis:")
    logger.info(f"  Raw confidence range: [{confidence_scores.min().item():.6f}, {confidence_scores.max().item():.6f}]")
    logger.info(f"  Raw confidence mean: {confidence_scores.mean().item():.6f}")
    
    # 选择置信度最高的600个实例
    confidence_sorted, indices = torch.topk(confidence_scores, 600, dim=1)
    
    logger.info(f"Top 600 confidence range: [{confidence_sorted.min().item():.6f}, {confidence_sorted.max().item():.6f}]")
    logger.info(f"Top 600 indices range: [{indices.min().item()}, {indices.max().item()}]")
    
    # 检查期望的temporal数据是否来自这些索引
    # 这里需要反向工程期望的temporal数据是如何生成的
    
    # 分析期望的temporal数据的统计信息
    logger.info("Expected temporal data analysis:")
    logger.info(f"  Expected temp_feature range: [{expected_temp_feature.min():.6f}, {expected_temp_feature.max():.6f}]")
    logger.info(f"  Expected temp_anchor range: [{expected_temp_anchor.min():.6f}, {expected_temp_anchor.max():.6f}]")
    
    return confidence_scores.numpy(), indices.numpy()

def analyze_index_differences(pred_instance_feature: np.ndarray, pred_anchor: np.ndarray, 
                            expected_temp_feature: np.ndarray, expected_temp_anchor: np.ndarray,
                            confidence_scores: np.ndarray, topk_indices: np.ndarray):
    """分析索引差异"""
    logger = setup_logger()
    
    # 使用PyTorch逻辑生成temporal数据
    instance_feature = torch.from_numpy(pred_instance_feature)
    anchor = torch.from_numpy(pred_anchor)
    confidence = torch.from_numpy(confidence_scores)
    
    # 选择置信度最高的600个实例
    confidence_sorted, indices = torch.topk(confidence, 600, dim=1)
    
    # 提取对应的实例特征和锚点
    batch_size = instance_feature.shape[0]
    indices_flat = indices + torch.arange(batch_size, device=indices.device)[:, None] * instance_feature.shape[1]
    indices_flat = indices_flat.flatten()
    
    temp_instance_feature = instance_feature.flatten(end_dim=1)[indices_flat].reshape(batch_size, 600, -1)
    temp_anchor = anchor.flatten(end_dim=1)[indices_flat].reshape(batch_size, 600, -1)
    
    logger.info("Index analysis:")
    logger.info(f"  PyTorch generated temp_feature range: [{temp_instance_feature.min().item():.6f}, {temp_instance_feature.max().item():.6f}]")
    logger.info(f"  PyTorch generated temp_anchor range: [{temp_anchor.min().item():.6f}, {temp_anchor.max().item():.6f}]")
    
    # 计算差异
    feature_diff = np.abs(temp_instance_feature.numpy() - expected_temp_feature)
    anchor_diff = np.abs(temp_anchor.numpy() - expected_temp_anchor)
    
    logger.info(f"  Feature diff - Max: {feature_diff.max():.6f}, Mean: {feature_diff.mean():.6f}")
    logger.info(f"  Anchor diff - Max: {anchor_diff.max():.6f}, Mean: {anchor_diff.mean():.6f}")
    
    # 检查是否有完全匹配的实例
    feature_exact_matches = np.sum(feature_diff < 1e-6)
    anchor_exact_matches = np.sum(anchor_diff < 1e-6)
    
    logger.info(f"  Feature exact matches: {feature_exact_matches}/{feature_diff.size} ({feature_exact_matches/feature_diff.size*100:.2f}%)")
    logger.info(f"  Anchor exact matches: {anchor_exact_matches}/{anchor_diff.size} ({anchor_exact_matches/anchor_diff.size*100:.2f}%)")
    
    return temp_instance_feature.numpy(), temp_anchor.numpy()

def main():
    """主函数"""
    logger = setup_logger()
    logger.info("Starting temporal data difference analysis...")
    
    # 测试样本1
    sample_id = 1
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing sample {sample_id}")
    logger.info(f"{'='*60}")
    
    try:
        # 1. 加载第一帧结果
        pred_instance_feature, pred_anchor, pred_class_score = load_first_frame_results(sample_id)
        
        # 2. 加载期望的temporal数据
        expected_temp_feature, expected_temp_anchor, expected_mask, expected_track_id = load_expected_temporal_data(sample_id)
        
        # 3. 分析置信度差异
        confidence_scores, topk_indices = analyze_confidence_differences(
            pred_class_score, expected_temp_feature, expected_temp_anchor
        )
        
        # 4. 分析索引差异
        pytorch_temp_feature, pytorch_temp_anchor = analyze_index_differences(
            pred_instance_feature, pred_anchor, expected_temp_feature, expected_temp_anchor,
            confidence_scores, topk_indices
        )
        
        logger.info("\n" + "="*60)
        logger.info("Analysis completed!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 