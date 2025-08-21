#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较两个pred_anchor文件的脚本
统计各锚点、各维度的最大误差
"""

import numpy as np
import os
import sys

def load_bin_file(file_path, shape):
    """
    加载二进制文件
    
    Args:
        file_path: 文件路径
        shape: 数据形状 (batch_size, num_anchors, anchor_dims)
    
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

def compare_anchors(data1, data2, tolerance=1e-6):
    """
    比较两个锚点数据并统计误差
    
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
    
    # 按锚点维度统计最大误差
    if len(data1.shape) == 3:  # (batch, anchors, dims)
        batch_size, num_anchors, anchor_dims = data1.shape
        
        print(f"\n=== 按锚点维度统计最大误差 ===")
        print(f"锚点数量: {num_anchors}")
        print(f"锚点维度: {anchor_dims}")
        
        # 锚点维度名称（根据常见的锚点格式）
        dim_names = ['x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'vx', 'vy', 'vz', 'conf']
        
        dim_max_errors = []
        for i in range(anchor_dims):
            dim_diff = abs_diff[:, :, i]
            max_error = np.max(dim_diff)
            mean_error = np.mean(dim_diff)
            dim_max_errors.append({
                'dim': i,
                'dim_name': dim_names[i] if i < len(dim_names) else f'dim_{i}',
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        dim_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        print(f"\n各锚点维度最大误差:")
        for i, dim_stats in enumerate(dim_max_errors):
            print(f"  {dim_stats['dim_name']:4s}: "
                  f"最大误差 = {dim_stats['max_error']:.6f}, "
                  f"平均误差 = {dim_stats['mean_error']:.6f}")
        
        # 按锚点统计最大误差
        print(f"\n=== 按锚点统计最大误差 ===")
        print(f"锚点数量: {num_anchors}")
        
        anchor_max_errors = []
        for i in range(num_anchors):
            anchor_diff = abs_diff[:, i, :]
            max_error = np.max(anchor_diff)
            mean_error = np.mean(anchor_diff)
            anchor_max_errors.append({
                'anchor': i,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        anchor_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        print(f"\n前20个最大误差的锚点:")
        for i, anchor_stats in enumerate(anchor_max_errors[:20]):
            print(f"  锚点 {anchor_stats['anchor']:3d}: "
                  f"最大误差 = {anchor_stats['max_error']:.6f}, "
                  f"平均误差 = {anchor_stats['mean_error']:.6f}")
        
        # 分析锚点内容（显示前几个锚点的具体值）
        print(f"\n=== 锚点内容分析 ===")
        print(f"前5个锚点的具体值:")
        for anchor_idx in range(min(5, num_anchors)):
            anchor1 = data1[0, anchor_idx]
            anchor2 = data2[0, anchor_idx]
            print(f"\n锚点 {anchor_idx}:")
            for dim_idx in range(anchor_dims):
                dim_name = dim_names[dim_idx] if dim_idx < len(dim_names) else f'dim_{dim_idx}'
                diff = abs(anchor1[dim_idx] - anchor2[dim_idx])
                print(f"  {dim_name:4s}: C++={anchor1[dim_idx]:8.4f}, Python={anchor2[dim_idx]:8.4f}, 差异={diff:8.6f}")
    
    return stats, dim_max_errors, anchor_max_errors

def analyze_anchor_distribution(data1, data2):
    """
    分析锚点分布特征
    
    Args:
        data1: 第一个数据
        data2: 第二个数据
    """
    print(f"\n=== 锚点分布分析 ===")
    
    # 分析每个维度的分布
    batch_size, num_anchors, anchor_dims = data1.shape
    
    for dim_idx in range(anchor_dims):
        dim_name = ['x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'vx', 'vy', 'vz', 'conf'][dim_idx] if dim_idx < 11 else f'dim_{dim_idx}'
        
        dim1 = data1[0, :, dim_idx]
        dim2 = data2[0, :, dim_idx]
        
        print(f"\n{dim_name} 维度:")
        print(f"  C++:   min={dim1.min():8.4f}, max={dim1.max():8.4f}, mean={dim1.mean():8.4f}, std={dim1.std():8.4f}")
        print(f"  Python: min={dim2.min():8.4f}, max={dim2.max():8.4f}, mean={dim2.mean():8.4f}, std={dim2.std():8.4f}")
        
        # 计算分布差异
        mean_diff = abs(dim1.mean() - dim2.mean())
        std_diff = abs(dim1.std() - dim2.std())
        print(f"  差异:  mean_diff={mean_diff:8.6f}, std_diff={std_diff:8.6f}")

def main():
    # 文件路径
    file1_path = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_0_pred_anchor_1*900*11_float32.bin"
    file2_path = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_pred_anchor_1*900*11_float32.bin"
    
    # 数据形状
    shape = (1, 900, 11)  # (batch_size, num_anchors, anchor_dims)
    
    print("=== Pred Anchor 比较脚本 ===")
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
    stats, dim_max_errors, anchor_max_errors = compare_anchors(data1, data2)
    
    if stats is None:
        return
    
    # 分析锚点分布
    analyze_anchor_distribution(data1, data2)
    
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
    tolerance = 1e-6
    if stats['max_abs_error'] < tolerance:
        print(f"\n✓ 数据完全一致 (最大误差 < {tolerance})")
    else:
        print(f"\n✗ 数据存在差异 (最大误差 >= {tolerance})")
    
    # 保存详细结果到文件
    output_file = "pred_anchor_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Pred Anchor 比较结果 ===\n")
        f.write(f"文件1: {file1_path}\n")
        f.write(f"文件2: {file2_path}\n")
        f.write(f"数据形状: {shape}\n\n")
        
        f.write("=== 总体统计 ===\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\n=== 各锚点维度最大误差 ===\n")
        for i, dim_stats in enumerate(dim_max_errors):
            f.write(f"{dim_stats['dim_name']:4s}: "
                   f"最大误差 = {dim_stats['max_error']:.6f}, "
                   f"平均误差 = {dim_stats['mean_error']:.6f}\n")
        
        f.write(f"\n=== 前50个最大误差的锚点 ===\n")
        for i, anchor_stats in enumerate(anchor_max_errors[:50]):
            f.write(f"锚点 {anchor_stats['anchor']:3d}: "
                   f"最大误差 = {anchor_stats['max_error']:.6f}, "
                   f"平均误差 = {anchor_stats['mean_error']:.6f}\n")
    
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 