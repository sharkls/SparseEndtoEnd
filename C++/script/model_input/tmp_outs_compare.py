#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较两个路径下的tmp_outs0~5文件的脚本
分析最大绝对误差和余弦距离
注意：此脚本只比较tmp_outs0~5数据，anchor和instance_feature的比较已移至other_input_compare.py
"""

import numpy as np
import os
import sys
from pathlib import Path

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

def cosine_distance(a, b):
    """
    计算两个向量的余弦距离
    
    Args:
        a, b: 输入向量
    
    Returns:
        余弦距离 (0表示完全相同，2表示完全相反)
    """
    # 计算余弦相似度
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # 避免除零
    if norm_a == 0 or norm_b == 0:
        return 2.0  # 最大距离
    
    cosine_similarity = dot_product / (norm_a * norm_b)
    # 将相似度转换为距离: distance = 1 - similarity
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance

def compare_tmp_outs(data1, data2, name):
    """
    比较两个tmp_outs数据并计算误差
    
    Args:
        data1: 第一个数据
        data2: 第二个数据
        name: 数据名称
    
    Returns:
        误差统计信息
    """
    if data1.shape != data2.shape:
        print(f"错误: {name} 数据形状不匹配. data1: {data1.shape}, data2: {data2.shape}")
        return None
    
    print(f"\n=== {name} 数据比较 ===")
    print(f"数据形状: {data1.shape}")
    
    # 计算绝对误差
    abs_diff = np.abs(data1 - data2)
    
    # 计算相对误差（避免除零）
    tolerance = 1e-8
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
    
    # 计算余弦距离
    print(f"\n=== {name} 余弦距离分析 ===")
    
    # 按实例计算余弦距离
    batch_size, num_instances, feature_dim = data1.shape
    instance_cosine_distances = []
    
    for i in range(num_instances):
        instance1 = data1[0, i, :]  # 第一个batch，第i个实例
        instance2 = data2[0, i, :]
        
        cos_dist = cosine_distance(instance1, instance2)
        instance_cosine_distances.append(cos_dist)
    
    instance_cosine_distances = np.array(instance_cosine_distances)
    
    # 余弦距离统计
    cos_stats = {
        'min_cosine_distance': np.min(instance_cosine_distances),
        'max_cosine_distance': np.max(instance_cosine_distances),
        'mean_cosine_distance': np.mean(instance_cosine_distances),
        'std_cosine_distance': np.std(instance_cosine_distances),
        'median_cosine_distance': np.median(instance_cosine_distances)
    }
    
    # 打印余弦距离统计
    print(f"余弦距离统计:")
    print(f"  最小余弦距离: {cos_stats['min_cosine_distance']:.6f}")
    print(f"  最大余弦距离: {cos_stats['max_cosine_distance']:.6f}")
    print(f"  平均余弦距离: {cos_stats['mean_cosine_distance']:.6f}")
    print(f"  余弦距离标准差: {cos_stats['std_cosine_distance']:.6f}")
    print(f"  余弦距离中位数: {cos_stats['median_cosine_distance']:.6f}")
    
    # 找出余弦距离最大和最小的实例
    max_cos_idx = np.argmax(instance_cosine_distances)
    min_cos_idx = np.argmin(instance_cosine_distances)
    
    print(f"\n余弦距离分析:")
    print(f"  余弦距离最大的实例 {max_cos_idx}: {instance_cosine_distances[max_cos_idx]:.6f}")
    print(f"  余弦距离最小的实例 {min_cos_idx}: {instance_cosine_distances[min_cos_idx]:.6f}")
    
    # 按特征维度统计最大误差
    print(f"\n=== {name} 按特征维度统计最大误差 ===")
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
    
    # 按实例统计最大误差
    print(f"\n=== {name} 按实例统计最大误差 ===")
    print(f"实例数量: {num_instances}")
    
    instance_max_errors = []
    for i in range(num_instances):
        instance_diff = abs_diff[:, i, :]
        max_error = np.max(instance_diff)
        mean_error = np.mean(instance_diff)
        instance_max_errors.append({
            'instance': i,
            'max_error': max_error,
            'mean_error': mean_error,
            'cosine_distance': instance_cosine_distances[i]
        })
    
    # 按最大误差排序
    instance_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
    
    print(f"\n前10个最大误差的实例:")
    for i, inst_stats in enumerate(instance_max_errors[:10]):
        print(f"  实例 {inst_stats['instance']:5d}: "
              f"最大误差 = {inst_stats['max_error']:.6f}, "
              f"平均误差 = {inst_stats['mean_error']:.6f}, "
              f"余弦距离 = {inst_stats['cosine_distance']:.6f}")
    
    return stats, cos_stats, column_max_errors, instance_max_errors

def main():
    # 文件路径
    base_path1 = "/share/Code/SparseEnd2End/script/tutorial/asset"
    base_path2 = "/share/Code/SparseEnd2End/C++/Output/val_bin"
    
    # 数据形状
    shape = (1, 900, 512)  # (batch_size, num_instances, feature_dim)
    
    print("=== Tmp_outs 比较脚本 ===")
    print(f"参考路径: {base_path1}")
    print(f"目标路径: {base_path2}")
    print(f"期望形状: {shape}")
    
    # 比较所有tmp_outs文件
    all_results = {}
    
    for i in range(6):
        tmp_name = f"tmp_outs{i}"
        
        # 构建文件路径
        file1_path = os.path.join(base_path1, f"sample_1_{tmp_name}_1*900*512_float32.bin")
        file2_path = os.path.join(base_path2, f"sample_1_{tmp_name}_1*900*512_float32.bin")
        
        print(f"\n{'='*60}")
        print(f"比较 {tmp_name}")
        print(f"文件1: {file1_path}")
        print(f"文件2: {file2_path}")
        
        # 加载数据
        data1 = load_bin_file(file1_path, shape)
        data2 = load_bin_file(file2_path, shape)
        
        if data1 is None or data2 is None:
            print(f"跳过 {tmp_name} 的比较")
            continue
        
        # 比较数据
        result = compare_tmp_outs(data1, data2, tmp_name)
        if result is not None:
            all_results[tmp_name] = result
    
    # 打印总体统计
    print(f"\n{'='*60}")
    print("=== 总体统计 ===")
    
    for tmp_name, (stats, cos_stats, col_errors, inst_errors) in all_results.items():
        print(f"\n{tmp_name}:")
        print(f"  最大绝对误差: {stats['max_abs_error']:.6f}")
        print(f"  平均绝对误差: {stats['mean_abs_error']:.6f}")
        print(f"  最大余弦距离: {cos_stats['max_cosine_distance']:.6f}")
        print(f"  平均余弦距离: {cos_stats['mean_cosine_distance']:.6f}")
        print(f"  不同元素百分比: {stats['percentage_different']:.2f}%")
    
    # 保存详细结果到文件
    output_file = "tmp_outs_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Tmp_outs 比较结果 ===\n")
        f.write(f"参考路径: {base_path1}\n")
        f.write(f"目标路径: {base_path2}\n")
        f.write(f"数据形状: {shape}\n\n")
        
        for tmp_name, (stats, cos_stats, col_errors, inst_errors) in all_results.items():
            f.write(f"=== {tmp_name} ===\n")
            f.write("绝对误差统计:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n余弦距离统计:\n")
            for key, value in cos_stats.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\n前20个最大误差的特征维度:\n")
            for i, col_stats in enumerate(col_errors[:20]):
                f.write(f"  特征维度 {col_stats['feature_dim']:3d}: "
                       f"最大误差 = {col_stats['max_error']:.6f}, "
                       f"平均误差 = {col_stats['mean_error']:.6f}\n")
            
            f.write(f"\n前20个最大误差的实例:\n")
            for i, inst_stats in enumerate(inst_errors[:20]):
                f.write(f"  实例 {inst_stats['instance']:5d}: "
                       f"最大误差 = {inst_stats['max_error']:.6f}, "
                       f"平均误差 = {inst_stats['mean_error']:.6f}, "
                       f"余弦距离 = {inst_stats['cosine_distance']:.6f}\n")
            
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 判断是否一致
    tolerance = 1e-6
    all_consistent = True
    
    for tmp_name, (stats, cos_stats, col_errors, inst_errors) in all_results.items():
        if stats['max_abs_error'] > tolerance:
            print(f"\n✗ {tmp_name} 存在差异 (最大误差 >= {tolerance})")
            all_consistent = False
        else:
            print(f"\n✓ {tmp_name} 数据完全一致 (最大误差 < {tolerance})")
    
    if all_consistent:
        print(f"\n🎉 所有数据完全一致!")
    else:
        print(f"\n⚠️  部分数据存在差异，请检查!")

if __name__ == "__main__":
    main()
