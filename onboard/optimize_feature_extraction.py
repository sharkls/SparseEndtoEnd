#!/usr/bin/env python3
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
"""
特征提取精度优化脚本
用于分析和优化特征提取阶段的误差
"""

import numpy as np
import os
import sys
import argparse
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

def load_bin_file(file_path: str, shape: Tuple[int, ...]) -> np.ndarray:
    """加载二进制文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    data = np.fromfile(file_path, dtype=np.float32)
    return data.reshape(shape)

def calculate_error_metrics(pred: np.ndarray, expected: np.ndarray) -> Dict[str, float]:
    """计算误差指标"""
    if pred.shape != expected.shape:
        raise ValueError(f"形状不匹配: pred={pred.shape}, expected={expected.shape}")
    
    # 绝对误差
    abs_error = np.abs(pred - expected)
    
    # 相对误差
    relative_error = np.abs(pred - expected) / (np.abs(expected) + 1e-8)
    
    # 统计指标
    metrics = {
        'max_abs_error': np.max(abs_error),
        'mean_abs_error': np.mean(abs_error),
        'std_abs_error': np.std(abs_error),
        'max_relative_error': np.max(relative_error),
        'mean_relative_error': np.mean(relative_error),
        'error_over_threshold_ratio': np.sum(abs_error > 0.1) / abs_error.size,
        'error_over_threshold_01_ratio': np.sum(abs_error > 0.01) / abs_error.size,
    }
    
    return metrics

def analyze_error_distribution(pred: np.ndarray, expected: np.ndarray, 
                            save_dir: str = "error_analysis") -> None:
    """分析误差分布"""
    os.makedirs(save_dir, exist_ok=True)
    
    abs_error = np.abs(pred - expected)
    relative_error = np.abs(pred - expected) / (np.abs(expected) + 1e-8)
    
    # 创建误差分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绝对误差直方图
    axes[0, 0].hist(abs_error.flatten(), bins=100, alpha=0.7, color='blue')
    axes[0, 0].set_title('绝对误差分布')
    axes[0, 0].set_xlabel('绝对误差')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].axvline(np.mean(abs_error), color='red', linestyle='--', 
                       label=f'均值: {np.mean(abs_error):.6f}')
    axes[0, 0].legend()
    
    # 相对误差直方图
    axes[0, 1].hist(relative_error.flatten(), bins=100, alpha=0.7, color='green')
    axes[0, 1].set_title('相对误差分布')
    axes[0, 1].set_xlabel('相对误差')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].axvline(np.mean(relative_error), color='red', linestyle='--',
                       label=f'均值: {np.mean(relative_error):.6f}')
    axes[0, 1].legend()
    
    # 误差热力图（前1000个元素）
    sample_size = min(1000, abs_error.size)
    sample_error = abs_error.flatten()[:sample_size].reshape(-1, 1)
    im = axes[1, 0].imshow(sample_error.T, aspect='auto', cmap='hot')
    axes[1, 0].set_title(f'误差热力图 (前{sample_size}个元素)')
    axes[1, 0].set_xlabel('元素索引')
    axes[1, 0].set_ylabel('通道')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 误差累积分布
    sorted_error = np.sort(abs_error.flatten())
    cumulative = np.arange(1, len(sorted_error) + 1) / len(sorted_error)
    axes[1, 1].plot(sorted_error, cumulative)
    axes[1, 1].set_title('误差累积分布')
    axes[1, 1].set_xlabel('绝对误差')
    axes[1, 1].set_ylabel('累积概率')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"误差分布图已保存到: {os.path.join(save_dir, 'error_distribution.png')}")

def compare_feature_statistics(pred: np.ndarray, expected: np.ndarray) -> None:
    """比较特征统计信息"""
    print("\n=== 特征统计信息比较 ===")
    print(f"预测特征 - 最大值: {np.max(pred):.6f}, 最小值: {np.min(pred):.6f}, "
          f"均值: {np.mean(pred):.6f}, 标准差: {np.std(pred):.6f}")
    print(f"预期特征 - 最大值: {np.max(expected):.6f}, 最小值: {np.min(expected):.6f}, "
          f"均值: {np.mean(expected):.6f}, 标准差: {np.std(expected):.6f}")
    
    # 计算相关系数
    correlation = np.corrcoef(pred.flatten(), expected.flatten())[0, 1]
    print(f"相关系数: {correlation:.6f}")

def suggest_optimization_strategies(metrics: Dict[str, float]) -> List[str]:
    """根据误差指标建议优化策略"""
    suggestions = []
    
    if metrics['error_over_threshold_ratio'] > 0.01:
        suggestions.append("1. 归一化参数不匹配 - 检查Python和C++端的归一化参数")
        suggestions.append("2. 颜色空间转换问题 - 确保BGR到RGB转换一致")
    
    if metrics['max_abs_error'] > 1.0:
        suggestions.append("3. TensorRT精度设置 - 考虑使用FP32而不是FP16")
        suggestions.append("4. 数值精度问题 - 检查浮点数计算精度")
    
    if metrics['mean_relative_error'] > 0.001:
        suggestions.append("5. 预处理步骤差异 - 检查resize、crop等操作")
        suggestions.append("6. 数据范围问题 - 检查输入数据范围是否一致")
    
    if not suggestions:
        suggestions.append("当前精度已经很好，无需进一步优化")
    
    return suggestions

def main():
    parser = argparse.ArgumentParser(description="特征提取精度优化分析")
    parser.add_argument("--pred_file", type=str, required=True,
                       help="预测结果文件路径")
    parser.add_argument("--expected_file", type=str, required=True,
                       help="预期结果文件路径")
    parser.add_argument("--shape", type=str, default="1,89760,256",
                       help="数据形状 (逗号分隔)")
    parser.add_argument("--save_dir", type=str, default="error_analysis",
                       help="分析结果保存目录")
    
    args = parser.parse_args()
    
    # 解析形状
    shape = tuple(map(int, args.shape.split(',')))
    
    print("=== 特征提取精度分析 ===")
    print(f"预测文件: {args.pred_file}")
    print(f"预期文件: {args.expected_file}")
    print(f"数据形状: {shape}")
    
    try:
        # 加载数据
        pred = load_bin_file(args.pred_file, shape)
        expected = load_bin_file(args.expected_file, shape)
        
        print(f"数据加载成功 - 形状: {pred.shape}")
        
        # 计算误差指标
        metrics = calculate_error_metrics(pred, expected)
        
        print("\n=== 误差指标 ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.6f}")
        
        # 比较统计信息
        compare_feature_statistics(pred, expected)
        
        # 分析误差分布
        analyze_error_distribution(pred, expected, args.save_dir)
        
        # 建议优化策略
        suggestions = suggest_optimization_strategies(metrics)
        print("\n=== 优化建议 ===")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
        # 保存详细报告
        report_file = os.path.join(args.save_dir, "analysis_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 特征提取精度分析报告 ===\n")
            f.write(f"预测文件: {args.pred_file}\n")
            f.write(f"预期文件: {args.expected_file}\n")
            f.write(f"数据形状: {shape}\n\n")
            
            f.write("=== 误差指标 ===\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6f}\n")
            
            f.write("\n=== 优化建议 ===\n")
            for i, suggestion in enumerate(suggestions, 1):
                f.write(f"{i}. {suggestion}\n")
        
        print(f"\n详细报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 