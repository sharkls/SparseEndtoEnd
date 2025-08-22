#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较SparseBEV.cpp第二帧推理保存的输入数据与Python脚本保存的对应数据
包括时间间隔、图像宽高、Lidar2img变换矩阵、锚点、空间形状、层级起始索引等
以及第二帧特有的时序相关数据：临时实例特征、临时锚点、掩码、跟踪ID等
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
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_1_input_time_interval_1_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_1_time_interval_1_float32.bin"
    
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
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_1_input_image_wh_1*6*2_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_1_image_wh_1*6*2_float32.bin"
    
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
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_1_input_lidar2img_1*6*4*4_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_1_lidar2img_1*6*4*4_float32.bin"
    
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
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_1_input_anchor_1*900*11_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_1_anchor_1*900*11_float32.bin"
    
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
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_1_input_spatial_shapes_6*4*2_int32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_1_spatial_shapes_6*4*2_int32.bin"
    
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
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_1_input_level_start_index_6*4_int32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_1_level_start_index_6*4_int32.bin"
    
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
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_1_input_instance_feature_1*900*256_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_1_instance_feature_1*900*256_float32.bin"
    
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

def compare_temp_instance_feature():
    """比较临时实例特征数据（第二帧特有）"""
    print("\n" + "="*50)
    print("临时实例特征数据比较（第二帧特有）")
    print("="*50)
    
    # 文件路径
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_1_temp_instance_feature_1*600*256_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_1_temp_instance_feature_1*600*256_float32.bin"
    
    # 数据形状
    shape = (1, 600, 256)
    
    # 加载数据
    data1 = load_bin_file(cpp_file, shape)
    data2 = load_bin_file(python_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "临时实例特征")
        
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
        
        print(f"\n前10个最大误差的临时实例:")
        for i, inst_stats in enumerate(instance_max_errors[:10]):
            print(f"  临时实例 {inst_stats['instance']:3d}: "
                  f"最大误差 = {inst_stats['max_error']:.6f}, "
                  f"平均误差 = {inst_stats['mean_error']:.6f}")
        
        return stats, instance_max_errors
    return None, None

def compare_temp_anchor():
    """比较临时锚点数据（第二帧特有）"""
    print("\n" + "="*50)
    print("临时锚点数据比较（第二帧特有）")
    print("="*50)
    
    # 文件路径
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_1_temp_anchor_1*600*11_float32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_1_temp_anchor_1*600*11_float32.bin"
    
    # 数据形状
    shape = (1, 600, 11)
    
    # 加载数据
    data1 = load_bin_file(cpp_file, shape)
    data2 = load_bin_file(python_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "临时锚点")
        
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
        
        print(f"\n前10个最大误差的临时锚点:")
        for i, anchor_stats in enumerate(anchor_max_errors[:10]):
            print(f"  临时锚点 {anchor_stats['anchor']:3d}: "
                  f"最大误差 = {anchor_stats['max_error']:.6f}, "
                  f"平均误差 = {anchor_stats['mean_error']:.6f}")
        
        return stats, anchor_max_errors
    return None, None

def compare_mask():
    """比较掩码数据（第二帧特有）"""
    print("\n" + "="*50)
    print("掩码数据比较（第二帧特有）")
    print("="*50)
    
    # 文件路径
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_1_mask_1_int32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_1_mask_1_int32.bin"
    
    # 数据形状
    shape = (1,)
    
    # 加载数据
    data1 = load_bin_file(cpp_file, shape, dtype=np.int32)
    data2 = load_bin_file(python_file, shape, dtype=np.int32)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "掩码")
        
        # 打印掩码值
        print(f"\n掩码值比较:")
        print(f"  C++掩码值: {data1[0]}")
        print(f"  Python掩码值: {data2[0]}")
        
        return stats
    return None

def compare_track_id():
    """比较跟踪ID数据（第二帧特有）"""
    print("\n" + "="*50)
    print("跟踪ID数据比较（第二帧特有）")
    print("="*50)
    
    # 文件路径
    cpp_file = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_1_track_id_1*900_int32.bin"
    python_file = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_1_track_id_1*900_int32.bin"
    
    # 数据形状
    shape = (1, 900)
    
    # 加载数据
    data1 = load_bin_file(cpp_file, shape, dtype=np.int32)
    data2 = load_bin_file(python_file, shape, dtype=np.int32)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "跟踪ID")
        
        # 直接打印完整的track_id数组
        print(f"\n=== 完整的track_id数组 (900个元素) ===")
        print("C++ track_id数组:")
        cpp_track_ids = data1[0]
        print("  " + " ".join([f"{val:4d}" for val in cpp_track_ids]))
        
        print(f"\nPython track_id数组:")
        python_track_ids = data2[0]
        print("  " + " ".join([f"{val:4d}" for val in python_track_ids]))
        
        # 统计非-1的跟踪ID数量
        cpp_valid_ids = np.sum(data1[0] != -1)
        python_valid_ids = np.sum(data2[0] != -1)
        print(f"\n有效跟踪ID统计:")
        print(f"  C++有效ID数量: {cpp_valid_ids}")
        print(f"  Python有效ID数量: {python_valid_ids}")
        
        # 找到所有非-1的跟踪ID进行比较
        cpp_non_neg1_indices = np.where(data1[0] != -1)[0]
        python_non_neg1_indices = np.where(data2[0] != -1)[0]
        
        print(f"\nC++中非-1的track_id值:")
        cpp_non_neg1_values = data1[0][data1[0] != -1]
        if len(cpp_non_neg1_values) > 0:
            print(f"  [{' '.join([f'{val}' for val in cpp_non_neg1_values])}]")
        else:
            print("  []")
        
        print(f"\nPython中非-1的track_id值:")
        python_non_neg1_values = data2[0][data2[0] != -1]
        if len(python_non_neg1_values) > 0:
            print(f"  [{' '.join([f'{val}' for val in python_non_neg1_values])}]")
        else:
            print("  []")
        
        # 找到前10个非-1的跟踪ID进行比较
        cpp_non_neg1 = data1[0][data1[0] != -1][:10]
        python_non_neg1 = data2[0][data2[0] != -1][:10]
        
        if len(cpp_non_neg1) > 0:
            print(f"\n前{min(len(cpp_non_neg1), 10)}个非-1的跟踪ID:")
            for i in range(min(len(cpp_non_neg1), 10)):
                print(f"  位置 {i}: C++({cpp_non_neg1[i]}) vs Python({python_non_neg1[i] if i < len(python_non_neg1) else 'N/A'})")
        
        return stats
    return None

def main():
    """主函数"""
    print("=== SparseBEV.cpp 第二帧推理 vs Python 输入数据比较脚本 ===")
    
    # 存储所有比较结果
    all_results = {}
    
    # 比较基础输入数据
    print("\n" + "="*60)
    print("基础输入数据比较")
    print("="*60)
    
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
    
    # 比较第二帧特有的时序相关数据
    print("\n" + "="*60)
    print("第二帧特有的时序相关数据比较")
    print("="*60)
    
    # 比较临时实例特征
    temp_instance_feature_stats, temp_instance_max_errors = compare_temp_instance_feature()
    if temp_instance_feature_stats:
        all_results['temp_instance_feature'] = temp_instance_feature_stats
        all_results['temp_instance_max_errors'] = temp_instance_max_errors
    
    # 比较临时锚点
    temp_anchor_stats, temp_anchor_max_errors = compare_temp_anchor()
    if temp_anchor_stats:
        all_results['temp_anchor'] = temp_anchor_stats
        all_results['temp_anchor_max_errors'] = temp_anchor_max_errors
    
    # 比较掩码
    mask_stats = compare_mask()
    if mask_stats:
        all_results['mask'] = mask_stats
    
    # 比较跟踪ID
    track_id_stats = compare_track_id()
    if track_id_stats:
        all_results['track_id'] = track_id_stats
    
    # 保存详细结果到文件
    output_file = "second_frame_infer_input_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== SparseBEV.cpp 第二帧推理 vs Python 输入数据比较结果 ===\n\n")
        
        # 保存基础输入数据结果
        f.write("=== 基础输入数据 ===\n")
        for data_type, stats in all_results.items():
            if data_type not in ['anchor_max_errors', 'instance_max_errors', 'feature_max_errors', 
                                'temp_instance_max_errors', 'temp_anchor_max_errors']:
                if data_type in ['time_interval', 'image_wh', 'lidar2img', 'anchor', 'spatial_shapes', 
                                'level_start_index', 'instance_feature']:
                    f.write(f"--- {data_type} ---\n")
                    for key, value in stats.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
        
        # 保存第二帧特有数据结果
        f.write("=== 第二帧特有的时序相关数据 ===\n")
        for data_type, stats in all_results.items():
            if data_type not in ['anchor_max_errors', 'instance_max_errors', 'feature_max_errors', 
                                'temp_instance_max_errors', 'temp_anchor_max_errors']:
                if data_type in ['temp_instance_feature', 'temp_anchor', 'mask', 'track_id']:
                    f.write(f"--- {data_type} ---\n")
                    for key, value in stats.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
        
        # 保存详细误差信息
        if 'anchor_max_errors' in all_results:
            f.write("=== 前50个最大误差的锚点 ===\n")
            for i, anchor_stats in enumerate(all_results['anchor_max_errors'][:50]):
                f.write(f"锚点 {anchor_stats['anchor']:3d}: "
                       f"最大误差 = {anchor_stats['max_error']:.6f}, "
                       f"平均误差 = {anchor_stats['mean_error']:.6f}\n")
            f.write("\n")
        
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
        
        if 'temp_instance_max_errors' in all_results:
            f.write("=== 前50个最大误差的临时实例 ===\n")
            for i, inst_stats in enumerate(all_results['temp_instance_max_errors'][:50]):
                f.write(f"临时实例 {inst_stats['instance']:3d}: "
                       f"最大误差 = {inst_stats['max_error']:.6f}, "
                       f"平均误差 = {inst_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        if 'temp_anchor_max_errors' in all_results:
            f.write("=== 前50个最大误差的临时锚点 ===\n")
            for i, anchor_stats in enumerate(all_results['temp_anchor_max_errors'][:50]):
                f.write(f"临时锚点 {anchor_stats['anchor']:3d}: "
                       f"最大误差 = {anchor_stats['max_error']:.6f}, "
                       f"平均误差 = {anchor_stats['mean_error']:.6f}\n")
            f.write("\n")
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 总结
    print(f"\n=== 比较总结 ===")
    
    # 基础输入数据类型
    basic_types = ['time_interval', 'image_wh', 'lidar2img', 'anchor', 'spatial_shapes', 
                   'level_start_index', 'instance_feature']
    # 第二帧特有数据类型
    second_frame_types = ['temp_instance_feature', 'temp_anchor', 'mask', 'track_id']
    
    basic_count = len([k for k in basic_types if k in all_results])
    second_frame_count = len([k for k in second_frame_types if k in all_results])
    
    print(f"基础输入数据类型数量: {basic_count}")
    print(f"第二帧特有数据类型数量: {second_frame_count}")
    print(f"总数据类型数量: {basic_count + second_frame_count}")
    
    tolerance = 1e-2
    basic_consistent_count = 0
    second_frame_consistent_count = 0
    
    # 检查基础输入数据一致性
    print(f"\n基础输入数据一致性:")
    for data_type in basic_types:
        if data_type in all_results:
            stats = all_results[data_type]
            if stats['max_abs_error'] < tolerance:
                basic_consistent_count += 1
                print(f"✓ {data_type}: 完全一致")
            else:
                print(f"✗ {data_type}: 存在差异 (最大误差: {stats['max_abs_error']:.6f})")
    
    # 检查第二帧特有数据一致性
    print(f"\n第二帧特有数据一致性:")
    for data_type in second_frame_types:
        if data_type in all_results:
            stats = all_results[data_type]
            if stats['max_abs_error'] < tolerance:
                second_frame_consistent_count += 1
                print(f"✓ {data_type}: 完全一致")
            else:
                print(f"✗ {data_type}: 存在差异 (最大误差: {stats['max_abs_error']:.6f})")
    
    print(f"\n基础输入数据完全一致: {basic_consistent_count}/{basic_count}")
    print(f"第二帧特有数据完全一致: {second_frame_consistent_count}/{second_frame_count}")
    print(f"总体完全一致: {basic_consistent_count + second_frame_consistent_count}/{basic_count + second_frame_count}")

if __name__ == "__main__":
    main()
