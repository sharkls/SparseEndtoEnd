#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较两个pred_instance_feature文件的脚本
统计各列的最大误差
"""

import numpy as np
import os
import sys

def load_bin_file(file_path, shape):
    """
    加载二进制文件
    
    Args:
        file_path: 文件路径
        shape: 数据形状 (batch_size, num_instances, feature_dim)
    
    Returns:
        numpy数组
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return None
    
    try:
        # 读取二进制文件
        data = np.fromfile(file_path, dtype=np.float32)
        
        # 重塑为指定形状
        if len(data) != np.prod(shape):
            print(f"错误: 文件大小不匹配. 期望: {np.prod(shape)}, 实际: {len(data)}")
            return None
        
        data = data.reshape(shape)
        print(f"成功加载文件: {file_path}")
        print(f"  形状: {data.shape}")
        print(f"  数据类型: {data.dtype}")
        print(f"  数值范围: [{data.min():.6f}, {data.max():.6f}]")
        
        return data
    except Exception as e:
        print(f"错误: 加载文件失败: {e}")
        return None

def compare_features(data1, data2, tolerance=1e-6):
    """
    比较两个特征数据并统计误差
    
    Args:
        data1: 第一个数据
        data2: 第二个数据
        tolerance: 容差，小于此值的差异被认为是数值误差
    
    Returns:
        误差统计信息
    """
    if data1.shape != data2.shape:
        print(f"错误: 数据形状不匹配. data1: {data1.shape}, data2: {data2.shape}")
        return None
    
    print(f"\n=== 数据比较 ===")
    print(f"数据形状: {data1.shape}")
    
    # 计算绝对误差
    abs_diff = np.abs(data1 - data2)
    
    # 计算相对误差（避免除零）
    relative_diff = np.zeros_like(abs_diff)
    mask = (np.abs(data2) > tolerance)
    relative_diff[mask] = abs_diff[mask] / np.abs(data2[mask])
    
    # 统计信息
    stats = {
        'max_abs_error': np.max(abs_diff),
        'mean_abs_error': np.mean(abs_diff),
        'std_abs_error': np.std(abs_diff),
        'max_relative_error': np.max(relative_diff),
        'mean_relative_error': np.mean(relative_diff),
        'std_relative_error': np.std(relative_diff),
        'num_different_elements': np.sum(abs_diff > tolerance),
        'total_elements': abs_diff.size,
        'percentage_different': np.sum(abs_diff > tolerance) / abs_diff.size * 100
    }
    
    # 按列统计最大误差
    if len(data1.shape) == 3:  # (batch, instances, features)
        batch_size, num_instances, feature_dim = data1.shape
        
        print(f"\n=== 按特征维度统计最大误差 ===")
        print(f"特征维度: {feature_dim}")
        
        column_max_errors = []
        for i in range(feature_dim):
            col_diff = abs_diff[:, :, i]
            max_error = np.max(col_diff)
            mean_error = np.mean(col_diff)
            column_max_errors.append({
                'feature_dim': i,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        column_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        print(f"\n前10个最大误差的特征维度:")
        for i, col_stats in enumerate(column_max_errors[:10]):
            print(f"  特征维度 {col_stats['feature_dim']:3d}: "
                  f"最大误差 = {col_stats['max_error']:.6f}, "
                  f"平均误差 = {col_stats['mean_error']:.6f}")
        
        # 按特征点统计最大误差
        print(f"\n=== 按特征点统计最大误差 ===")
        print(f"特征点数量: {num_instances}")
        
        feature_max_errors = []
        for i in range(num_instances):
            feature_diff = abs_diff[:, i, :]
            max_error = np.max(feature_diff)
            mean_error = np.mean(feature_diff)
            feature_max_errors.append({
                'feature_point': i,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        feature_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        print(f"\n前10个最大误差的特征点:")
        for i, feat_stats in enumerate(feature_max_errors[:10]):
            print(f"  特征点 {feat_stats['feature_point']:5d}: "
                  f"最大误差 = {feat_stats['max_error']:.6f}, "
                  f"平均误差 = {feat_stats['mean_error']:.6f}")
    
    return stats, column_max_errors, feature_max_errors

def main():
    # 文件路径
    file1_path = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_0_feature_1*89760*256_float32.bin"
    file2_path = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_feature_1*89760*256_float32.bin"
    
    # 数据形状
    shape = (1, 89760, 256)  # (batch_size, num_features, feature_dim)
    
    print("=== Feature 比较脚本 ===")
    print(f"文件1: {file1_path}")
    print(f"文件2: {file2_path}")
    print(f"期望形状: {shape}")
    
    # 加载数据
    print(f"\n=== 加载数据 ===")
    data1 = load_bin_file(file1_path, shape)
    data2 = load_bin_file(file2_path, shape)
    
    if data1 is None or data2 is None:
        print("错误: 无法加载数据文件")
        return
    
    # 比较数据
    stats, column_max_errors, feature_max_errors = compare_features(data1, data2)
    
    if stats is None:
        return
    
    # 打印总体统计
    print(f"\n=== 总体统计 ===")
    print(f"最大绝对误差: {stats['max_abs_error']:.6f}")
    print(f"平均绝对误差: {stats['mean_abs_error']:.6f}")
    print(f"绝对误差标准差: {stats['std_abs_error']:.6f}")
    print(f"最大相对误差: {stats['max_relative_error']:.6f}")
    print(f"平均相对误差: {stats['mean_relative_error']:.6f}")
    print(f"相对误差标准差: {stats['std_relative_error']:.6f}")
    print(f"不同元素数量: {stats['num_different_elements']}")
    print(f"总元素数量: {stats['total_elements']}")
    print(f"不同元素百分比: {stats['percentage_different']:.2f}%")
    
    # 判断是否一致
    tolerance = 20
    if stats['max_abs_error'] < tolerance:
        print(f"\n✓ 数据完全一致 (最大误差 < {tolerance})")
    else:
        print(f"\n✗ 数据存在差异 (最大误差 >= {tolerance})")
    
    # 保存详细结果到文件
    output_file = "feature_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Feature 比较结果 ===\n")
        f.write(f"文件1: {file1_path}\n")
        f.write(f"文件2: {file2_path}\n")
        f.write(f"数据形状: {shape}\n\n")
        
        f.write("=== 总体统计 ===\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\n=== 前20个最大误差的特征维度 ===\n")
        for i, col_stats in enumerate(column_max_errors[:20]):
            f.write(f"特征维度 {col_stats['feature_dim']:3d}: "
                   f"最大误差 = {col_stats['max_error']:.6f}, "
                   f"平均误差 = {col_stats['mean_error']:.6f}\n")
        
        f.write(f"\n=== 前20个最大误差的特征点 ===\n")
        for i, feat_stats in enumerate(feature_max_errors[:20]):
            f.write(f"特征点 {feat_stats['feature_point']:5d}: "
                   f"最大误差 = {feat_stats['max_error']:.6f}, "
                   f"平均误差 = {feat_stats['mean_error']:.6f}\n")
    
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
