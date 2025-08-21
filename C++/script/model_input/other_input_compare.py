#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较SparseBEV.cpp保存的各种输入数据与Python脚本保存的对应数据
包括时间间隔、图像宽高、Lidar2img变换矩阵、锚点、空间形状、层级起始索引等
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

def compare_time_interval():
    """比较时间间隔数据"""
    print("\n" + "="*50)
    print("时间间隔数据比较")
    print("="*50)
    
    # 文件路径
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_0_input_time_interval_1_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_time_interval_1_float32.bin"
    
    # 数据形状
    shape = (1,)
    
    # 加载数据
    data1 = load_bin_file(cpp_file, shape)
    data2 = load_bin_file(python_file, shape)
    
    if data1 is not None and data2 is not None:
        return compare_data(data1, data2, "时间间隔")
    return None

def compare_image_wh():
    """比较图像宽高数据"""
    print("\n" + "="*50)
    print("图像宽高数据比较")
    print("="*50)
    
    # 文件路径
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_0_input_image_wh_1*6*2_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_image_wh_1*6*2_float32.bin"
    
    # 数据形状
    shape = (1, 6, 2)
    
    # 加载数据
    data1 = load_bin_file(cpp_file, shape)
    data2 = load_bin_file(python_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "图像宽高")
        
        # 打印每个摄像头的宽高信息
        print(f"\n各摄像头图像尺寸:")
        for cam_idx in range(6):
            cpp_wh = data1[0, cam_idx]
            python_wh = data2[0, cam_idx]
            print(f"  摄像头 {cam_idx}: C++({cpp_wh[0]:.0f}, {cpp_wh[1]:.0f}) vs Python({python_wh[0]:.0f}, {python_wh[1]:.0f})")
        
        return stats
    return None

def compare_lidar2img():
    """比较Lidar2img变换矩阵数据"""
    print("\n" + "="*50)
    print("Lidar2img变换矩阵数据比较")
    print("="*50)
    
    # 文件路径
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_0_input_lidar2img_1*6*4*4_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_lidar2img_1*6*4*4_float32.bin"
    
    # 数据形状
    shape = (1, 6, 4, 4)
    
    # 加载数据
    data1 = load_bin_file(cpp_file, shape)
    data2 = load_bin_file(python_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "Lidar2img变换矩阵")
        
        # 打印每个摄像头的变换矩阵信息
        print(f"\n各摄像头变换矩阵最大误差:")
        for cam_idx in range(6):
            cam_diff = np.abs(data1[0, cam_idx] - data2[0, cam_idx])
            max_error = np.max(cam_diff)
            mean_error = np.mean(cam_diff)
            print(f"  摄像头 {cam_idx}: 最大误差 = {max_error:.6f}, 平均误差 = {mean_error:.6f}")
        
        return stats
    return None

def compare_anchor():
    """比较锚点数据"""
    print("\n" + "="*50)
    print("锚点数据比较")
    print("="*50)
    
    # 文件路径
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_0_input_anchor_1*900*11_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_anchor_1*900*11_float32.bin"
    
    # 数据形状
    shape = (1, 900, 11)
    
    # 加载数据
    data1 = load_bin_file(cpp_file, shape)
    data2 = load_bin_file(python_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "锚点")
        
        # 按锚点统计最大误差
        anchor_max_errors = []
        for i in range(900):
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
        
        return stats, anchor_max_errors
    return None, None

def compare_spatial_shapes():
    """比较空间形状数据"""
    print("\n" + "="*50)
    print("空间形状数据比较")
    print("="*50)
    
    # 文件路径
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_0_input_spatial_shapes_6*4*2_int32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_spatial_shapes_6*4*2_int32.bin"
    
    # 数据形状
    shape = (6, 4, 2)
    
    # 加载数据
    data1 = load_bin_file(cpp_file, shape, dtype=np.int32)
    data2 = load_bin_file(python_file, shape, dtype=np.int32)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "空间形状")
        
        # 打印空间形状信息
        print(f"\n各层级空间形状:")
        for level in range(4):
            for cam_idx in range(6):
                cpp_shape = data1[cam_idx, level]
                python_shape = data2[cam_idx, level]
                print(f"  摄像头 {cam_idx}, 层级 {level}: C++{tuple(cpp_shape)} vs Python{tuple(python_shape)}")
        
        return stats
    return None

def compare_level_start_index():
    """比较层级起始索引数据"""
    print("\n" + "="*50)
    print("层级起始索引数据比较")
    print("="*50)
    
    # 文件路径
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_0_input_level_start_index_6*4_int32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_level_start_index_6*4_int32.bin"
    
    # 数据形状
    shape = (6, 4)
    
    # 加载数据
    data1 = load_bin_file(cpp_file, shape, dtype=np.int32)
    data2 = load_bin_file(python_file, shape, dtype=np.int32)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "层级起始索引")
        
        # 打印层级起始索引信息
        print(f"\n各摄像头层级起始索引:")
        for cam_idx in range(6):
            cpp_indices = data1[cam_idx]
            python_indices = data2[cam_idx]
            print(f"  摄像头 {cam_idx}: C++{tuple(cpp_indices)} vs Python{tuple(python_indices)}")
        
        return stats
    return None

def compare_instance_feature():
    """比较实例特征数据"""
    print("\n" + "="*50)
    print("实例特征数据比较")
    print("="*50)
    
    # 文件路径
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_0_input_instance_feature_1*900*256_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_instance_feature_1*900*256_float32.bin"
    
    # 数据形状
    shape = (1, 900, 256)
    
    # 加载数据
    data1 = load_bin_file(cpp_file, shape)
    data2 = load_bin_file(python_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "实例特征")
        
        # 按实例统计最大误差
        instance_max_errors = []
        for i in range(900):
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

def main():
    """主函数"""
    print("=== SparseBEV.cpp vs Python 输入数据比较脚本 ===")
    
    # 存储所有比较结果
    all_results = {}
    
    # 比较时间间隔
    time_stats = compare_time_interval()
    if time_stats:
        all_results['time_interval'] = time_stats
    
    # 比较图像宽高
    image_wh_stats = compare_image_wh()
    if image_wh_stats:
        all_results['image_wh'] = image_wh_stats
    
    # 比较Lidar2img变换矩阵
    lidar2img_stats = compare_lidar2img()
    if lidar2img_stats:
        all_results['lidar2img'] = lidar2img_stats
    
    # 比较锚点
    anchor_stats, anchor_max_errors = compare_anchor()
    if anchor_stats:
        all_results['anchor'] = anchor_stats
        all_results['anchor_max_errors'] = anchor_max_errors
    
    # 比较空间形状
    spatial_shapes_stats = compare_spatial_shapes()
    if spatial_shapes_stats:
        all_results['spatial_shapes'] = spatial_shapes_stats
    
    # 比较层级起始索引
    level_start_index_stats = compare_level_start_index()
    if level_start_index_stats:
        all_results['level_start_index'] = level_start_index_stats
    
    # 比较实例特征
    instance_feature_stats, instance_max_errors, feature_max_errors = compare_instance_feature()
    if instance_feature_stats:
        all_results['instance_feature'] = instance_feature_stats
        all_results['instance_max_errors'] = instance_max_errors
        all_results['feature_max_errors'] = feature_max_errors
    
    # 保存详细结果到文件
    output_file = "other_input_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== SparseBEV.cpp vs Python 输入数据比较结果 ===\n\n")
        
        for data_type, stats in all_results.items():
            if data_type not in ['anchor_max_errors', 'instance_max_errors', 'feature_max_errors']:
                f.write(f"=== {data_type} ===\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
        
        # 保存锚点详细误差信息
        if 'anchor_max_errors' in all_results:
            f.write("=== 前50个最大误差的锚点 ===\n")
            for i, anchor_stats in enumerate(all_results['anchor_max_errors'][:50]):
                f.write(f"锚点 {anchor_stats['anchor']:3d}: "
                       f"最大误差 = {anchor_stats['max_error']:.6f}, "
                       f"平均误差 = {anchor_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        # 保存实例特征详细误差信息
        if 'instance_max_errors' in all_results:
            f.write("=== 前50个最大误差的实例 ===\n")
            for i, inst_stats in enumerate(all_results['instance_max_errors'][:50]):
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
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 总结
    print(f"\n=== 比较总结 ===")
    print(f"比较的数据类型数量: {len([k for k in all_results.keys() if k not in ['anchor_max_errors', 'instance_max_errors', 'feature_max_errors']])}")
    
    tolerance = 1e-6
    consistent_count = 0
    for data_type, stats in all_results.items():
        if data_type not in ['anchor_max_errors', 'instance_max_errors', 'feature_max_errors']:
            if stats['max_abs_error'] < tolerance:
                consistent_count += 1
                print(f"✓ {data_type}: 完全一致")
            else:
                print(f"✗ {data_type}: 存在差异 (最大误差: {stats['max_abs_error']:.6f})")
    
    print(f"\n完全一致的数据类型: {consistent_count}/{len([k for k in all_results.keys() if k not in ['anchor_max_errors', 'instance_max_errors', 'feature_max_errors']])}")

if __name__ == "__main__":
    main() 