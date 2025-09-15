#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较GPU版本与CPU版本 第一帧缓存数据 的差异：
- cached_feature (1*600*256 float32)
- cached_anchor  (1*600*11  float32)
- cached_track_id (1*600     int32)
"""

import numpy as np
import os
import sys

def load_bin_file(file_path, shape, dtype=np.float32):
    """
    加载二进制文件
    
    Args:
        file_path: 文件路径
        shape: 数据形状
        dtype: 数据类型
    
    Returns:
        numpy数组
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return None
    
    try:
        # 读取二进制文件
        data = np.fromfile(file_path, dtype=dtype)
        
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

def compare_data(data1, data2, name, tolerance=1e-6):
    """
    比较两个数据并统计误差
    
    Args:
        data1: 第一个数据
        data2: 第二个数据
        name: 数据名称
        tolerance: 容差
    
    Returns:
        误差统计信息
    """
    if data1 is None or data2 is None:
        return None
        
    if data1.shape != data2.shape:
        print(f"错误: {name} 数据形状不匹配. data1: {data1.shape}, data2: {data2.shape}")
        return None
    
    print(f"\n=== {name} 数据比较 ===")
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
    
    print(f"最大绝对误差: {stats['max_abs_error']:.6f}")
    print(f"平均绝对误差: {stats['mean_abs_error']:.6f}")
    print(f"最大相对误差: {stats['max_relative_error']:.6f}")
    print(f"平均相对误差: {stats['mean_relative_error']:.6f}")
    print(f"不同元素数量: {stats['num_different_elements']}")
    print(f"总元素数量: {stats['total_elements']}")
    print(f"不同元素百分比: {stats['percentage_different']:.2f}%")
    
    # 判断是否一致
    if stats['max_abs_error'] < tolerance:
        print(f"✓ {name} 数据完全一致 (最大误差 < {tolerance})")
    else:
        print(f"✗ {name} 数据存在差异 (最大误差 >= {tolerance})")
    
    return stats

def compare_cached_feature():
    """比较缓存特征数据"""
    print("\n" + "="*50)
    print("缓存特征数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_0_output_cached_feature_1*600*256_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_0_output_cached_feature_1*600*256_float32.bin"
    
    # 数据形状
    shape = (1, 600, 256)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "缓存特征", tolerance=1e-1)
        
        # 按实例统计最大误差
        instance_max_errors = []
        for i in range(600):
            instance_diff = np.abs(data1[0, i] - data2[0, i])
            max_error = np.max(instance_diff)
            mean_error = np.mean(instance_diff)
            instance_max_errors.append({
                'instance': i,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        instance_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        print(f"\n前10个最大误差的实例:")
        for i, inst_stats in enumerate(instance_max_errors[:10]):
            print(f"  实例 {inst_stats['instance']:3d}: "
                  f"最大误差 = {inst_stats['max_error']:.6f}, "
                  f"平均误差 = {inst_stats['mean_error']:.6f}")
        
        # 按特征维度统计最大误差
        print(f"\n前10个最大误差的特征维度:")
        feature_max_errors = []
        for i in range(256):
            feature_diff = np.abs(data1[:, :, i] - data2[:, :, i])
            max_error = np.max(feature_diff)
            mean_error = np.mean(feature_diff)
            feature_max_errors.append({
                'feature_dim': i,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        feature_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        for i, feat_stats in enumerate(feature_max_errors[:10]):
            print(f"  特征维度 {feat_stats['feature_dim']:3d}: "
                  f"最大误差 = {feat_stats['max_error']:.6f}, "
                  f"平均误差 = {feat_stats['mean_error']:.6f}")
        
        return stats, instance_max_errors, feature_max_errors
    return None, None, None

def compare_cached_anchor():
    """比较缓存锚点数据"""
    print("\n" + "="*50)
    print("缓存锚点数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_0_output_cached_anchor_1*600*11_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_0_output_cached_anchor_1*600*11_float32.bin"
    
    # 数据形状
    shape = (1, 600, 11)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "缓存锚点", tolerance=1e-1)
        
        # 按锚点统计最大误差
        anchor_max_errors = []
        for i in range(600):
            anchor_diff = np.abs(data1[0, i] - data2[0, i])
            max_error = np.max(anchor_diff)
            mean_error = np.mean(anchor_diff)
            anchor_max_errors.append({
                'anchor': i,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        anchor_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        print(f"\n前10个最大误差的锚点:")
        for i, anchor_stats in enumerate(anchor_max_errors[:10]):
            print(f"  锚点 {anchor_stats['anchor']:3d}: "
                  f"最大误差 = {anchor_stats['max_error']:.6f}, "
                  f"平均误差 = {anchor_stats['mean_error']:.6f}")
        
        # 按锚点维度统计最大误差
        print(f"\n各锚点维度最大误差:")
        for i in range(11):
            dim_diff = np.abs(data1[:, :, i] - data2[:, :, i])
            max_error = np.max(dim_diff)
            mean_error = np.mean(dim_diff)
            print(f"  维度 {i:2d}: 最大误差 = {max_error:.6f}, 平均误差 = {mean_error:.6f}")
        
        return stats, anchor_max_errors
    return None, None

def compare_cached_track_id():
    """比较缓存跟踪ID数据"""
    print("\n" + "="*50)
    print("缓存跟踪ID数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_0_output_cached_track_id_1*600_int32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_0_output_cached_track_id_1*600_int32.bin"
    
    # 数据形状
    shape = (1, 600)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape, dtype=np.int32)
    data2 = load_bin_file(cpu_file, shape, dtype=np.int32)
    
    if data1 is not None and data2 is not None:
        # 对于整数类型，使用不同的比较方式
        print(f"\n=== 缓存跟踪ID 数据比较 ===")
        print(f"数据形状: {data1.shape}")
        
        # 统计不同的track_id
        diff_mask = (data1 != data2)
        num_different = np.sum(diff_mask)
        
        stats = {
            'num_different_elements': int(num_different),
            'total_elements': int(data1.size),
            'percentage_different': float(num_different / data1.size * 100)
        }
        
        print(f"不同元素数量: {stats['num_different_elements']}")
        print(f"总元素数量: {stats['total_elements']}")
        print(f"不同元素百分比: {stats['percentage_different']:.2f}%")
        
        if num_different > 0:
            print(f"✗ 缓存跟踪ID 数据存在差异")
            print(f"\n前10个不同的Track ID:")
            diff_indices = np.where(diff_mask)[0]
            for i, idx in enumerate(diff_indices[:10]):
                print(f"  索引 {idx:3d}: GPU={data1[0, idx]} vs CPU={data2[0, idx]}")
        else:
            print(f"✓ 缓存跟踪ID 数据完全一致")
        
        return stats
    return None

def print_all_track_ids():
    """打印所有600个track_id值进行对比"""
    print("=== 所有600个Track ID值详细对比 ===")
    
    # 文件路径
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_0_output_cached_track_id_1*600_int32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_0_output_cached_track_id_1*600_int32.bin"
    
    # 加载数据
    gpu_data = load_bin_file(gpu_file, (1, 600), dtype=np.int32)
    cpu_data = load_bin_file(cpu_file, (1, 600), dtype=np.int32)
    
    if gpu_data is None or cpu_data is None:
        print("无法加载数据")
        return
    
    print(f"GPU版本 track_id 形状: {gpu_data.shape}")
    print(f"CPU版本 track_id 形状: {cpu_data.shape}")
    print()
    
    # 打印所有600个track_id值对比（每行10个）
    print("所有600个 track_id 值对比:")
    print("索引 | GPU版本 | CPU版本 | 是否相同 | 索引 | GPU版本 | CPU版本 | 是否相同")
    print("-" * 80)
    
    for i in range(0, 600, 2):  # 每行显示2个对比
        # 第一个值
        gpu_val1 = gpu_data[0, i]
        cpu_val1 = cpu_data[0, i]
        same1 = "✓" if gpu_val1 == cpu_val1 else "✗"
        
        # 第二个值（如果存在）
        if i + 1 < 600:
            gpu_val2 = gpu_data[0, i + 1]
            cpu_val2 = cpu_data[0, i + 1]
            same2 = "✓" if gpu_val2 == cpu_val2 else "✗"
            print(f"{i:3d} | {gpu_val1:7d} | {cpu_val1:7d} | {same1:6s} | {i+1:3d} | {gpu_val2:7d} | {cpu_val2:7d} | {same2}")
        else:
            print(f"{i:3d} | {gpu_val1:7d} | {cpu_val1:7d} | {same1:6s} |")
    
    # 统计差异
    diff_mask = (gpu_data != cpu_data)
    num_different = np.sum(diff_mask)
    
    print(f"\n=== 差异统计 ===")
    print(f"总元素数量: {gpu_data.size}")
    print(f"不同元素数量: {num_different}")
    print(f"不同元素百分比: {num_different / gpu_data.size * 100:.2f}%")
    
    # if num_different > 0:
    #     print(f"\n=== 所有不同的 track_id 详细列表 ===")
    #     diff_indices = np.where(diff_mask)[1]  # 获取列索引
    #     print("索引 | GPU版本 | CPU版本 | 差异")
    #     print("-" * 40)
    #     for idx in diff_indices:
    #         gpu_val = gpu_data[0, idx]
    #         cpu_val = cpu_data[0, idx]
    #         diff = gpu_val - cpu_val
    #         print(f"{idx:3d} | {gpu_val:7d} | {cpu_val:7d} | {diff:+4d}")
    
    # 打印GPU版本的完整数组（每行20个值）
    print(f"\n=== GPU版本完整 track_id 数组 (每行20个) ===")
    gpu_flat = gpu_data.flatten()
    for i in range(0, len(gpu_flat), 20):
        end_idx = min(i + 20, len(gpu_flat))
        values = gpu_flat[i:end_idx]
        print(f"[{i:3d}:{end_idx:3d}] " + " ".join(f"{val:3d}" for val in values))
    
    # 打印CPU版本的完整数组（每行20个值）
    print(f"\n=== CPU版本完整 track_id 数组 (每行20个) ===")
    cpu_flat = cpu_data.flatten()
    for i in range(0, len(cpu_flat), 20):
        end_idx = min(i + 20, len(cpu_flat))
        values = cpu_flat[i:end_idx]
        print(f"[{i:3d}:{end_idx:3d}] " + " ".join(f"{val:3d}" for val in values))
    
    # 分析排序情况
    print(f"\n=== 排序分析 ===")
    gpu_sorted = np.sort(gpu_flat)
    cpu_sorted = np.sort(cpu_flat)
    
    print("GPU版本排序后前20个值:", gpu_sorted[:20])
    print("CPU版本排序后前20个值:", cpu_sorted[:20])
    print("GPU版本排序后后20个值:", gpu_sorted[-20:])
    print("CPU版本排序后后20个值:", cpu_sorted[-20:])
    
    # 检查是否包含相同的值
    gpu_unique = np.unique(gpu_flat)
    cpu_unique = np.unique(cpu_flat)
    print(f"GPU版本唯一值数量: {len(gpu_unique)}")
    print(f"CPU版本唯一值数量: {len(cpu_unique)}")
    
    # 检查值范围
    print(f"GPU版本值范围: [{gpu_flat.min()}, {gpu_flat.max()}]")
    print(f"CPU版本值范围: [{cpu_flat.min()}, {cpu_flat.max()}]")

def main():
    """主函数"""
    print("=== GPU版本 vs CPU版本 第一帧缓存数据比较脚本 ===")
    print("GPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin_gpu/")
    print("CPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin/")
    
    # 存储所有比较结果
    all_results = {}
    
    # 比较缓存特征
    feature_stats, feature_instance_max_errors, feature_max_errors = compare_cached_feature()
    if feature_stats:
        all_results['cached_feature'] = feature_stats
        all_results['feature_instance_max_errors'] = feature_instance_max_errors
        all_results['feature_max_errors'] = feature_max_errors
    
    # 比较缓存锚点
    anchor_stats, anchor_max_errors = compare_cached_anchor()
    if anchor_stats:
        all_results['cached_anchor'] = anchor_stats
        all_results['anchor_max_errors'] = anchor_max_errors
    
    # 比较缓存跟踪ID
    track_id_stats = compare_cached_track_id()
    if track_id_stats:
        all_results['cached_track_id'] = track_id_stats
    
    # 保存详细结果到文件
    output_file = "gpu_vs_cpu_first_frame_cached_data_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== GPU版本 vs CPU版本 第一帧缓存数据比较结果 ===\n\n")
        f.write("GPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin_gpu/\n")
        f.write("CPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin/\n\n")
        
        for data_type, stats in all_results.items():
            if data_type not in ['feature_instance_max_errors', 'feature_max_errors', 'anchor_max_errors']:
                f.write(f"=== {data_type} ===\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
        
        # 保存特征详细误差信息
        if 'feature_instance_max_errors' in all_results:
            f.write("=== 前50个最大误差的缓存特征实例 ===\n")
            for i, inst_stats in enumerate(all_results['feature_instance_max_errors'][:50]):
                f.write(f"实例 {inst_stats['instance']:3d}: "
                       f"最大误差 = {inst_stats['max_error']:.6f}, "
                       f"平均误差 = {inst_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        if 'feature_max_errors' in all_results:
            f.write("=== 前50个最大误差的特征维度 ===\n")
            for i, feat_stats in enumerate(all_results['feature_max_errors'][:50]):
                f.write(f"特征维度 {feat_stats['feature_dim']:3d}: "
                       f"最大误差 = {feat_stats['max_error']:.6f}, "
                       f"平均误差 = {feat_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        # 保存锚点详细误差信息
        if 'anchor_max_errors' in all_results:
            f.write("=== 前50个最大误差的缓存锚点 ===\n")
            for i, anchor_stats in enumerate(all_results['anchor_max_errors'][:50]):
                f.write(f"锚点 {anchor_stats['anchor']:3d}: "
                       f"最大误差 = {anchor_stats['max_error']:.6f}, "
                       f"平均误差 = {anchor_stats['mean_error']:.6f}\n")
            f.write("\n")
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 总结
    print(f"\n=== 比较总结 ===")
    basic_data_types = [k for k in all_results.keys() if k not in ['feature_instance_max_errors', 'feature_max_errors', 'anchor_max_errors']]
    print(f"比较的数据类型数量: {len(basic_data_types)}")
    
    tolerance = 1e-1
    consistent_count = 0
    for data_type, stats in all_results.items():
        if data_type in basic_data_types:
            if data_type == 'cached_track_id':
                # 对于整数类型，检查是否完全一致
                if stats['num_different_elements'] == 0:
                    consistent_count += 1
                    print(f"✓ {data_type}: 完全一致")
                else:
                    print(f"✗ {data_type}: 存在差异 ({stats['num_different_elements']} 个不同元素)")
            else:
                # 对于浮点类型，检查误差
                if stats['max_abs_error'] < tolerance:
                    consistent_count += 1
                    print(f"✓ {data_type}: 完全一致")
                else:
                    print(f"✗ {data_type}: 存在差异 (最大误差: {stats['max_abs_error']:.6f})")
    
    print(f"\n完全一致的数据类型: {consistent_count}/{len(basic_data_types)}")
    
    # 特别说明第一帧缓存数据的特点
    print(f"\n=== 第一帧缓存数据特点说明 ===")
    print("1. 包含缓存特征数据 (600个实例, 256维特征)")
    print("2. 包含缓存锚点数据 (600个锚点, 11维)")
    print("3. 包含缓存跟踪ID数据 (600个实例)")
    print("4. 这些数据是从900个实例中选择的Top-600个最高置信度的实例")
    print("5. 所有数据都经过GPU内存处理和16字节对齐")
    print_all_track_ids()

if __name__ == "__main__":
    main()