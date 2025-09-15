#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较GPU版本SparseBEV.cpp保存的第二帧推理输入数据与原始CPU版本数据的差异
包括时间间隔、图像宽高、Lidar2img变换矩阵、锚点、空间形状、层级起始索引、实例特征、临时数据等
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
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_time_interval_1_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_time_interval_1_float32.bin"
    
    # 数据形状
    shape = (1,)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
    if data1 is not None and data2 is not None:
        return compare_data(data1, data2, "时间间隔")
    return None

def compare_image_wh():
    """比较图像宽高数据"""
    print("\n" + "="*50)
    print("图像宽高数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_image_wh_1*6*2_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_image_wh_1*6*2_float32.bin"
    
    # 数据形状
    shape = (1, 6, 2)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape, dtype=np.float32)
    data2 = load_bin_file(cpu_file, shape, dtype=np.float32)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "图像宽高")
        
        # 打印每个摄像头的宽高信息
        print(f"\n各摄像头图像尺寸:")
        for cam_idx in range(6):
            gpu_wh = data1[0, cam_idx]
            cpu_wh = data2[0, cam_idx]
            print(f"  摄像头 {cam_idx}: GPU({gpu_wh[0]}, {gpu_wh[1]}) vs CPU({cpu_wh[0]:.0f}, {cpu_wh[1]:.0f})")
        
        return stats
    return None

def compare_lidar2img():
    """比较Lidar2img变换矩阵数据"""
    print("\n" + "="*50)
    print("Lidar2img变换矩阵数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_lidar2img_1*6*4*4_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_lidar2img_1*6*4*4_float32.bin"
    
    # 数据形状
    shape = (1, 6, 4, 4)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
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
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_anchor_1*900*11_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_anchor_1*900*11_float32.bin"
    
    # 数据形状
    shape = (1, 900, 11)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
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
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_spatial_shapes_6*4*2_int32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_spatial_shapes_6*4*2_int32.bin"
    
    # 数据形状
    shape = (6, 4, 2)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape, dtype=np.int32)
    data2 = load_bin_file(cpu_file, shape, dtype=np.int32)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "空间形状")
        
        # 打印空间形状信息
        print(f"\n各层级空间形状:")
        for level in range(4):
            for cam_idx in range(6):
                gpu_shape = data1[cam_idx, level]
                cpu_shape = data2[cam_idx, level]
                print(f"  摄像头 {cam_idx}, 层级 {level}: GPU{tuple(gpu_shape)} vs CPU{tuple(cpu_shape)}")
        
        return stats
    return None

def compare_level_start_index():
    """比较层级起始索引数据"""
    print("\n" + "="*50)
    print("层级起始索引数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_level_start_index_6*4_int32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_level_start_index_6*4_int32.bin"
    
    # 数据形状
    shape = (6, 4)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape, dtype=np.int32)
    data2 = load_bin_file(cpu_file, shape, dtype=np.int32)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "层级起始索引")
        
        # 打印层级起始索引信息
        print(f"\n各摄像头层级起始索引:")
        for cam_idx in range(6):
            gpu_indices = data1[cam_idx]
            cpu_indices = data2[cam_idx]
            print(f"  摄像头 {cam_idx}: GPU{tuple(gpu_indices)} vs CPU{tuple(cpu_indices)}")
        
        return stats
    return None

def compare_instance_feature():
    """比较实例特征数据"""
    print("\n" + "="*50)
    print("实例特征数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_instance_feature_1*900*256_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_instance_feature_1*900*256_float32.bin"
    
    # 数据形状
    shape = (1, 900, 256)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
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

def compare_temp_instance_feature():
    """比较临时实例特征数据（第二帧特有）"""
    print("\n" + "="*50)
    print("临时实例特征数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_temp_instance_feature_1*600*256_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_temp_instance_feature_1*600*256_float32.bin"
    
    # 数据形状
    shape = (1, 600, 256)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "临时实例特征")
        
        # 按实例统计最大误差
        temp_instance_max_errors = []
        for i in range(600):
            instance_diff = np.abs(data1[0, i] - data2[0, i])
            max_error = np.max(instance_diff)
            mean_error = np.mean(instance_diff)
            temp_instance_max_errors.append({
                'instance': i,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        temp_instance_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        print(f"\n前10个最大误差的临时实例:")
        for i, inst_stats in enumerate(temp_instance_max_errors[:10]):
            print(f"  临时实例 {inst_stats['instance']:3d}: "
                  f"最大误差 = {inst_stats['max_error']:.6f}, "
                  f"平均误差 = {inst_stats['mean_error']:.6f}")
        
        return stats, temp_instance_max_errors
    return None, None

def compare_temp_anchor():
    """比较临时锚点数据（第二帧特有）"""
    print("\n" + "="*50)
    print("临时锚点数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_temp_anchor_1*600*11_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_temp_anchor_1*600*11_float32.bin"
    
    # 数据形状
    shape = (1, 600, 11)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "临时锚点")
        
        # 按锚点统计最大误差
        temp_anchor_max_errors = []
        for i in range(600):
            anchor_diff = np.abs(data1[0, i] - data2[0, i])
            max_error = np.max(anchor_diff)
            mean_error = np.mean(anchor_diff)
            temp_anchor_max_errors.append({
                'anchor': i,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        temp_anchor_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        print(f"\n前10个最大误差的临时锚点:")
        for i, anchor_stats in enumerate(temp_anchor_max_errors[:10]):
            print(f"  临时锚点 {anchor_stats['anchor']:3d}: "
                  f"最大误差 = {anchor_stats['max_error']:.6f}, "
                  f"平均误差 = {anchor_stats['mean_error']:.6f}")
        
        return stats, temp_anchor_max_errors
    return None, None

def compare_mask():
    """比较mask数据（第二帧特有）"""
    print("\n" + "="*50)
    print("Mask数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_mask_1_int32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_mask_1_int32.bin"
    
    # 数据形状
    shape = (1,)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape, dtype=np.int32)
    data2 = load_bin_file(cpu_file, shape, dtype=np.int32)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "Mask")
        
        print(f"\nMask值:")
        print(f"  GPU: {data1[0]}")
        print(f"  CPU: {data2[0]}")
        
        return stats
    return None

def compare_pred_track_id():
    """比较track_id数据（第二帧特有）"""
    print("\n" + "="*50)
    print("Track ID数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_track_id_1*900_int32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_track_id_1*900_int32.bin"
    
    # 数据形状
    shape = (1, 900)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape, dtype=np.int32)
    data2 = load_bin_file(cpu_file, shape, dtype=np.int32)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "预测Track ID")
        
        # 统计不同的track_id
        diff_mask = (data1[0] != data2[0])
        num_different = np.sum(diff_mask)
        
        print(f"\nTrack ID差异统计:")
        print(f"  不同元素数量: {num_different}")
        print(f"  总元素数量: {data1.size}")
        print(f"  不同元素百分比: {num_different / data1.size * 100:.2f}%")
        
        if num_different > 0:
            print(f"\n前10个不同的Track ID:")
            diff_indices = np.where(diff_mask)[0]
            for i, idx in enumerate(diff_indices[:10]):
                print(f"  索引 {idx:3d}: GPU={data1[0, idx]} vs CPU={data2[0, idx]}")
        
        return stats
    return None

def compare_global_to_lidar_mat():
    """比较global_to_lidar变换矩阵数据"""
    print("\n" + "="*50)
    print("Global_to_lidar变换矩阵数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_input_global_to_lidar_mat_4*4_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_input_global_to_lidar_mat_4*4_float32.bin"
    
    # 数据形状
    shape = (4, 4)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "Global_to_lidar变换矩阵")
        
        # 打印变换矩阵的详细信息
        print(f"\nGPU版本变换矩阵:")
        for i in range(4):
            print(f"  [{data1[i,0]:8.4f} {data1[i,1]:8.4f} {data1[i,2]:8.4f} {data1[i,3]:8.4f}]")
        
        print(f"\nCPU版本变换矩阵:")
        for i in range(4):
            print(f"  [{data2[i,0]:8.4f} {data2[i,1]:8.4f} {data2[i,2]:8.4f} {data2[i,3]:8.4f}]")
        
        # 计算逐元素误差
        print(f"\n逐元素误差矩阵:")
        error_matrix = np.abs(data1 - data2)
        for i in range(4):
            print(f"  [{error_matrix[i,0]:8.6f} {error_matrix[i,1]:8.6f} {error_matrix[i,2]:8.6f} {error_matrix[i,3]:8.6f}]")
        
        # 分析变换矩阵的特殊性质
        print(f"\n变换矩阵性质分析:")
        
        # 检查是否为旋转矩阵（R^T * R = I）
        gpu_rotation = data1[:3, :3]
        cpu_rotation = data2[:3, :3]
        
        gpu_orthogonal = np.dot(gpu_rotation.T, gpu_rotation)
        cpu_orthogonal = np.dot(cpu_rotation.T, cpu_rotation)
        
        gpu_orthogonal_error = np.max(np.abs(gpu_orthogonal - np.eye(3)))
        cpu_orthogonal_error = np.max(np.abs(cpu_orthogonal - np.eye(3)))
        
        print(f"  GPU旋转矩阵正交性误差: {gpu_orthogonal_error:.8f}")
        print(f"  CPU旋转矩阵正交性误差: {cpu_orthogonal_error:.8f}")
        
        # 检查行列式（旋转矩阵的行列式应该为1）
        gpu_det = np.linalg.det(gpu_rotation)
        cpu_det = np.linalg.det(cpu_rotation)
        
        print(f"  GPU旋转矩阵行列式: {gpu_det:.8f}")
        print(f"  CPU旋转矩阵行列式: {cpu_det:.8f}")
        
        # 检查平移向量
        gpu_translation = data1[:3, 3]
        cpu_translation = data2[:3, 3]
        translation_error = np.max(np.abs(gpu_translation - cpu_translation))
        
        print(f"  平移向量误差: {translation_error:.6f}")
        print(f"  GPU平移向量: [{gpu_translation[0]:.4f}, {gpu_translation[1]:.4f}, {gpu_translation[2]:.4f}]")
        print(f"  CPU平移向量: [{cpu_translation[0]:.4f}, {cpu_translation[1]:.4f}, {cpu_translation[2]:.4f}]")
        
        # 检查齐次坐标行
        gpu_homogeneous = data1[3, :]
        cpu_homogeneous = data2[3, :]
        homogeneous_error = np.max(np.abs(gpu_homogeneous - cpu_homogeneous))
        
        print(f"  齐次坐标行误差: {homogeneous_error:.6f}")
        print(f"  GPU齐次坐标行: [{gpu_homogeneous[0]:.4f}, {gpu_homogeneous[1]:.4f}, {gpu_homogeneous[2]:.4f}, {gpu_homogeneous[3]:.4f}]")
        print(f"  CPU齐次坐标行: [{cpu_homogeneous[0]:.4f}, {cpu_homogeneous[1]:.4f}, {cpu_homogeneous[2]:.4f}, {cpu_homogeneous[3]:.4f}]")
        
        # 计算变换矩阵的逆矩阵误差
        try:
            gpu_inv = np.linalg.inv(data1)
            cpu_inv = np.linalg.inv(data2)
            inv_error = np.max(np.abs(gpu_inv - cpu_inv))
            print(f"  逆矩阵误差: {inv_error:.6f}")
        except np.linalg.LinAlgError:
            print(f"  逆矩阵计算失败（矩阵奇异）")
        
        # 按行统计最大误差
        row_max_errors = []
        for i in range(4):
            row_diff = np.abs(data1[i] - data2[i])
            max_error = np.max(row_diff)
            mean_error = np.mean(row_diff)
            row_max_errors.append({
                'row': i,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        print(f"\n各行最大误差:")
        for row_stats in row_max_errors:
            print(f"  第{row_stats['row']}行: 最大误差 = {row_stats['max_error']:.6f}, 平均误差 = {row_stats['mean_error']:.6f}")
        
        # 按列统计最大误差
        col_max_errors = []
        for j in range(4):
            col_diff = np.abs(data1[:, j] - data2[:, j])
            max_error = np.max(col_diff)
            mean_error = np.mean(col_diff)
            col_max_errors.append({
                'col': j,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        print(f"\n各列最大误差:")
        for col_stats in col_max_errors:
            print(f"  第{col_stats['col']}列: 最大误差 = {col_stats['max_error']:.6f}, 平均误差 = {col_stats['mean_error']:.6f}")
        
        return stats, row_max_errors, col_max_errors
    return None, None, None

def main():
    """主函数"""
    print("=== GPU版本 vs CPU版本 第二帧推理输入数据比较脚本 ===")
    print("GPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin_gpu/")
    print("CPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin/")
    
    # 存储所有比较结果
    all_results = {}
    
    # 比较基础输入数据
    time_stats = compare_time_interval()
    if time_stats:
        all_results['time_interval'] = time_stats
    
    # 添加global_to_lidar_mat比较
    global_to_lidar_stats, row_max_errors, col_max_errors = compare_global_to_lidar_mat()
    if global_to_lidar_stats:
        all_results['global_to_lidar_mat'] = global_to_lidar_stats
        all_results['global_to_lidar_row_errors'] = row_max_errors
        all_results['global_to_lidar_col_errors'] = col_max_errors
    
    image_wh_stats = compare_image_wh()
    if image_wh_stats:
        all_results['image_wh'] = image_wh_stats
    
    lidar2img_stats = compare_lidar2img()
    if lidar2img_stats:
        all_results['lidar2img'] = lidar2img_stats
    
    anchor_stats, anchor_max_errors = compare_anchor()
    if anchor_stats:
        all_results['anchor'] = anchor_stats
        all_results['anchor_max_errors'] = anchor_max_errors
    
    spatial_shapes_stats = compare_spatial_shapes()
    if spatial_shapes_stats:
        all_results['spatial_shapes'] = spatial_shapes_stats
    
    level_start_index_stats = compare_level_start_index()
    if level_start_index_stats:
        all_results['level_start_index'] = level_start_index_stats
    
    instance_feature_stats, instance_max_errors, feature_max_errors = compare_instance_feature()
    if instance_feature_stats:
        all_results['instance_feature'] = instance_feature_stats
        all_results['instance_max_errors'] = instance_max_errors
        all_results['feature_max_errors'] = feature_max_errors
    
    # 比较第二帧特有的临时数据
    temp_instance_feature_stats, temp_instance_max_errors = compare_temp_instance_feature()
    if temp_instance_feature_stats:
        all_results['temp_instance_feature'] = temp_instance_feature_stats
        all_results['temp_instance_max_errors'] = temp_instance_max_errors
    
    temp_anchor_stats, temp_anchor_max_errors = compare_temp_anchor()
    if temp_anchor_stats:
        all_results['temp_anchor'] = temp_anchor_stats
        all_results['temp_anchor_max_errors'] = temp_anchor_max_errors
    
    mask_stats = compare_mask()
    if mask_stats:
        all_results['mask'] = mask_stats
    
    pred_track_id_stats = compare_pred_track_id()
    if pred_track_id_stats:
        all_results['track_id'] = pred_track_id_stats
    
    # 保存详细结果到文件
    output_file = "gpu_vs_cpu_second_frame_input_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== GPU版本 vs CPU版本 第二帧推理输入数据比较结果 ===\n\n")
        f.write("GPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin_gpu/\n")
        f.write("CPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin/\n\n")
        
        for data_type, stats in all_results.items():
            # 修复：添加global_to_lidar_row_errors和global_to_lidar_col_errors到排除列表
            if data_type not in ['anchor_max_errors', 'instance_max_errors', 'feature_max_errors', 
                               'temp_instance_max_errors', 'temp_anchor_max_errors',
                               'global_to_lidar_row_errors', 'global_to_lidar_col_errors']:
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
        
        # 保存临时实例特征详细误差信息
        if 'temp_instance_max_errors' in all_results:
            f.write("=== 前50个最大误差的临时实例 ===\n")
            for i, inst_stats in enumerate(all_results['temp_instance_max_errors'][:50]):
                f.write(f"临时实例 {inst_stats['instance']:3d}: "
                       f"最大误差 = {inst_stats['max_error']:.6f}, "
                       f"平均误差 = {inst_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        # 保存临时锚点详细误差信息
        if 'temp_anchor_max_errors' in all_results:
            f.write("=== 前50个最大误差的临时锚点 ===\n")
            for i, anchor_stats in enumerate(all_results['temp_anchor_max_errors'][:50]):
                f.write(f"临时锚点 {anchor_stats['anchor']:3d}: "
                       f"最大误差 = {anchor_stats['max_error']:.6f}, "
                       f"平均误差 = {anchor_stats['mean_error']:.6f}\n")
            f.write("\n")

        # 在保存详细结果的部分添加：
        if 'global_to_lidar_row_errors' in all_results:
            f.write("=== Global_to_lidar变换矩阵各行最大误差 ===\n")
            for i, row_stats in enumerate(all_results['global_to_lidar_row_errors']):
                f.write(f"第{row_stats['row']}行: 最大误差 = {row_stats['max_error']:.6f}, "
                       f"平均误差 = {row_stats['mean_error']:.6f}\n")
            f.write("\n")

        if 'global_to_lidar_col_errors' in all_results:
            f.write("=== Global_to_lidar变换矩阵各列最大误差 ===\n")
            for i, col_stats in enumerate(all_results['global_to_lidar_col_errors']):
                f.write(f"第{col_stats['col']}列: 最大误差 = {col_stats['max_error']:.6f}, "
                       f"平均误差 = {col_stats['mean_error']:.6f}\n")
            f.write("\n")
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 总结
    print(f"\n=== 比较总结 ===")
    # 修复：添加global_to_lidar_row_errors和global_to_lidar_col_errors到排除列表
    basic_data_types = [k for k in all_results.keys() if k not in ['anchor_max_errors', 'instance_max_errors', 
                                                                  'feature_max_errors', 'temp_instance_max_errors', 
                                                                  'temp_anchor_max_errors', 'global_to_lidar_row_errors', 
                                                                  'global_to_lidar_col_errors']]
    print(f"比较的数据类型数量: {len(basic_data_types)}")
    
    tolerance = 1e-6
    consistent_count = 0
    for data_type, stats in all_results.items():
        if data_type in basic_data_types:
            if stats['max_abs_error'] < tolerance:
                consistent_count += 1
                print(f"✓ {data_type}: 完全一致")
            else:
                print(f"✗ {data_type}: 存在差异 (最大误差: {stats['max_abs_error']:.6f})")
    
    print(f"\n完全一致的数据类型: {consistent_count}/{len(basic_data_types)}")
    
    # 特别说明第二帧的特点
    print(f"\n=== 第二帧推理特点说明 ===")
    print("1. 包含第一帧的所有基础输入数据")
    print("2. 新增临时实例特征数据 (600个实例)")
    print("3. 新增临时锚点数据 (600个锚点)")
    print("4. 新增mask数据 (用于标识是否为第一帧)")
    print("5. 新增预测track_id数据 (900个track_id)")
    print("6. 所有数据都经过GPU内存处理和16字节对齐")

if __name__ == "__main__":
    main()