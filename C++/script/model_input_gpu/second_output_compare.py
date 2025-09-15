#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较GPU版本SparseBEV.cpp保存的第二帧推理输出数据与原始CPU版本数据的差异
包括pred_instance_feature、pred_anchor、pred_class_score、pred_quality_score、pred_track_id等
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

def compare_pred_instance_feature():
    """比较预测实例特征数据"""
    print("\n" + "="*50)
    print("预测实例特征数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_output_pred_instance_feature_1*900*256_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_output_pred_instance_feature_1*900*256_float32.bin"
    
    # 数据形状
    shape = (1, 900, 256)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "预测实例特征")
        
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

def compare_pred_anchor():
    """比较预测锚点数据"""
    print("\n" + "="*50)
    print("预测锚点数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_output_pred_anchor_1*900*11_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_output_pred_anchor_1*900*11_float32.bin"
    
    # 数据形状
    shape = (1, 900, 11)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "预测锚点")
        
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
        
        # 按锚点属性统计最大误差
        print(f"\n前10个最大误差的锚点属性:")
        attr_names = ['center_x', 'center_y', 'center_z', 'size_x', 'size_y', 'size_z', 
                     'yaw_x', 'yaw_y', 'vel_x', 'vel_y', 'vel_z']
        attr_max_errors = []
        for i in range(11):
            attr_diff = np.abs(data1[:, :, i] - data2[:, :, i])
            max_error = np.max(attr_diff)
            mean_error = np.mean(attr_diff)
            attr_max_errors.append({
                'attr': i,
                'attr_name': attr_names[i],
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        attr_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        for i, attr_stats in enumerate(attr_max_errors[:10]):
            print(f"  属性 {attr_stats['attr']:2d} ({attr_stats['attr_name']:8s}): "
                  f"最大误差 = {attr_stats['max_error']:.6f}, "
                  f"平均误差 = {attr_stats['mean_error']:.6f}")
        
        return stats, anchor_max_errors, attr_max_errors
    return None, None, None

def compare_pred_class_score():
    """比较预测类别分数数据"""
    print("\n" + "="*50)
    print("预测类别分数数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_output_pred_class_score_1*900*10_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_output_pred_class_score_1*900*10_float32.bin"
    
    # 数据形状
    shape = (1, 900, 10)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "预测类别分数")
        
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
        
        # 按类别统计最大误差
        print(f"\n各类别最大误差:")
        class_names = ['Car', 'Truck', 'Construction_vehicle', 'Bus', 'Trailer', 
                      'Barrier', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Traffic_cone']
        class_max_errors = []
        for i in range(10):
            class_diff = np.abs(data1[:, :, i] - data2[:, :, i])
            max_error = np.max(class_diff)
            mean_error = np.mean(class_diff)
            class_max_errors.append({
                'class': i,
                'class_name': class_names[i],
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        class_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        for i, class_stats in enumerate(class_max_errors):
            print(f"  类别 {class_stats['class']:2d} ({class_stats['class_name']:20s}): "
                  f"最大误差 = {class_stats['max_error']:.6f}, "
                  f"平均误差 = {class_stats['mean_error']:.6f}")
        
        return stats, instance_max_errors, class_max_errors
    return None, None, None

def compare_pred_quality_score():
    """比较预测质量分数数据"""
    print("\n" + "="*50)
    print("预测质量分数数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_output_pred_quality_score_1*900*2_float32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_output_pred_quality_score_1*900*2_float32.bin"
    
    # 数据形状
    shape = (1, 900, 2)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape)
    data2 = load_bin_file(cpu_file, shape)
    
    if data1 is not None and data2 is not None:
        stats = compare_data(data1, data2, "预测质量分数")
        
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
        
        # 按质量分数属性统计最大误差
        print(f"\n各质量分数属性最大误差:")
        quality_names = ['IoU', 'Center_Offset']
        quality_max_errors = []
        for i in range(2):
            quality_diff = np.abs(data1[:, :, i] - data2[:, :, i])
            max_error = np.max(quality_diff)
            mean_error = np.mean(quality_diff)
            quality_max_errors.append({
                'quality': i,
                'quality_name': quality_names[i],
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        for i, quality_stats in enumerate(quality_max_errors):
            print(f"  质量分数 {quality_stats['quality']:2d} ({quality_stats['quality_name']:15s}): "
                  f"最大误差 = {quality_stats['max_error']:.6f}, "
                  f"平均误差 = {quality_stats['mean_error']:.6f}")
        
        return stats, instance_max_errors, quality_max_errors
    return None, None, None

def compare_pred_track_id():
    """比较预测track_id数据"""
    print("\n" + "="*50)
    print("预测Track ID数据比较")
    print("="*50)
    
    # 文件路径 - GPU版本 vs 原始CPU版本
    gpu_file = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/sample_1_output_pred_track_id_1*900_int32.bin"
    cpu_file = "/share/Code/Sparse4d/C++/Output/val_bin/sample_1_output_pred_track_id_1*900_int32.bin"
    
    # 数据形状
    shape = (1, 900)
    
    # 加载数据
    data1 = load_bin_file(gpu_file, shape, dtype=np.int32)
    data2 = load_bin_file(cpu_file, shape, dtype=np.int32)
    
    if data1 is not None and data2 is not None:
        # 对于整数类型，使用不同的比较方式
        print(f"\n=== 预测Track ID 数据比较 ===")
        print(f"数据形状: {data1.shape}")
        
        # 计算差异
        diff_mask = (data1 != data2)
        num_different = np.sum(diff_mask)
        total_elements = data1.size
        
        stats = {
            'num_different_elements': int(num_different),
            'total_elements': int(total_elements),
            'percentage_different': float(num_different / total_elements * 100)
        }
        
        print(f"不同元素数量: {stats['num_different_elements']}")
        print(f"总元素数量: {stats['total_elements']}")
        print(f"不同元素百分比: {stats['percentage_different']:.2f}%")
        
        if num_different > 0:
            print(f"✗ 预测Track ID 数据存在差异")
            print(f"\n前10个不同的Track ID:")
            diff_indices = np.where(diff_mask)[0]
            for i, idx in enumerate(diff_indices[:10]):
                print(f"  索引 {idx:3d}: GPU={data1[0, idx]} vs CPU={data2[0, idx]}")
        else:
            print(f"✓ 预测Track ID 数据完全一致")
        
        return stats
    return None

def main():
    """主函数"""
    print("=== GPU版本 vs CPU版本 第二帧推理输出数据比较脚本 ===")
    print("GPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin_gpu/")
    print("CPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin/")
    
    # 存储所有比较结果
    all_results = {}
    
    # 比较预测输出数据
    pred_instance_feature_stats, instance_max_errors, feature_max_errors = compare_pred_instance_feature()
    if pred_instance_feature_stats:
        all_results['pred_instance_feature'] = pred_instance_feature_stats
        all_results['pred_instance_max_errors'] = instance_max_errors
        all_results['pred_feature_max_errors'] = feature_max_errors
    
    pred_anchor_stats, anchor_max_errors, attr_max_errors = compare_pred_anchor()
    if pred_anchor_stats:
        all_results['pred_anchor'] = pred_anchor_stats
        all_results['pred_anchor_max_errors'] = anchor_max_errors
        all_results['pred_attr_max_errors'] = attr_max_errors
    
    pred_class_score_stats, class_instance_max_errors, class_max_errors = compare_pred_class_score()
    if pred_class_score_stats:
        all_results['pred_class_score'] = pred_class_score_stats
        all_results['pred_class_instance_max_errors'] = class_instance_max_errors
        all_results['pred_class_max_errors'] = class_max_errors
    
    pred_quality_score_stats, quality_instance_max_errors, quality_max_errors = compare_pred_quality_score()
    if pred_quality_score_stats:
        all_results['pred_quality_score'] = pred_quality_score_stats
        all_results['pred_quality_instance_max_errors'] = quality_instance_max_errors
        all_results['pred_quality_max_errors'] = quality_max_errors
    
    pred_track_id_stats = compare_pred_track_id()
    if pred_track_id_stats:
        all_results['pred_track_id'] = pred_track_id_stats
    
    # 保存详细结果到文件
    output_file = "gpu_vs_cpu_second_frame_output_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== GPU版本 vs CPU版本 第二帧推理输出数据比较结果 ===\n\n")
        f.write("GPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin_gpu/\n")
        f.write("CPU版本路径: /share/Code/Sparse4d/C++/Output/val_bin/\n\n")
        
        for data_type, stats in all_results.items():
            # 排除列表类型的数据
            if data_type not in ['pred_instance_max_errors', 'pred_feature_max_errors', 
                               'pred_anchor_max_errors', 'pred_attr_max_errors',
                               'pred_class_instance_max_errors', 'pred_class_max_errors',
                               'pred_quality_instance_max_errors', 'pred_quality_max_errors']:
                f.write(f"=== {data_type} ===\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
        
        # 保存预测实例特征详细误差信息
        if 'pred_instance_max_errors' in all_results:
            f.write("=== 前50个最大误差的预测实例 ===\n")
            for i, inst_stats in enumerate(all_results['pred_instance_max_errors'][:50]):
                f.write(f"实例 {inst_stats['instance']:3d}: "
                       f"最大误差 = {inst_stats['max_error']:.6f}, "
                       f"平均误差 = {inst_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        if 'pred_feature_max_errors' in all_results:
            f.write("=== 前50个最大误差的预测特征维度 ===\n")
            for i, feat_stats in enumerate(all_results['pred_feature_max_errors'][:50]):
                f.write(f"特征维度 {feat_stats['feature_dim']:3d}: "
                       f"最大误差 = {feat_stats['max_error']:.6f}, "
                       f"平均误差 = {feat_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        # 保存预测锚点详细误差信息
        if 'pred_anchor_max_errors' in all_results:
            f.write("=== 前50个最大误差的预测锚点 ===\n")
            for i, anchor_stats in enumerate(all_results['pred_anchor_max_errors'][:50]):
                f.write(f"锚点 {anchor_stats['anchor']:3d}: "
                       f"最大误差 = {anchor_stats['max_error']:.6f}, "
                       f"平均误差 = {anchor_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        if 'pred_attr_max_errors' in all_results:
            f.write("=== 预测锚点各属性最大误差 ===\n")
            for i, attr_stats in enumerate(all_results['pred_attr_max_errors']):
                f.write(f"属性 {attr_stats['attr']:2d} ({attr_stats['attr_name']:8s}): "
                       f"最大误差 = {attr_stats['max_error']:.6f}, "
                       f"平均误差 = {attr_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        # 保存预测类别分数详细误差信息
        if 'pred_class_instance_max_errors' in all_results:
            f.write("=== 前50个最大误差的预测类别实例 ===\n")
            for i, inst_stats in enumerate(all_results['pred_class_instance_max_errors'][:50]):
                f.write(f"实例 {inst_stats['instance']:3d}: "
                       f"最大误差 = {inst_stats['max_error']:.6f}, "
                       f"平均误差 = {inst_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        if 'pred_class_max_errors' in all_results:
            f.write("=== 预测类别各类别最大误差 ===\n")
            for i, class_stats in enumerate(all_results['pred_class_max_errors']):
                f.write(f"类别 {class_stats['class']:2d} ({class_stats['class_name']:20s}): "
                       f"最大误差 = {class_stats['max_error']:.6f}, "
                       f"平均误差 = {class_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        # 保存预测质量分数详细误差信息
        if 'pred_quality_instance_max_errors' in all_results:
            f.write("=== 前50个最大误差的预测质量实例 ===\n")
            for i, inst_stats in enumerate(all_results['pred_quality_instance_max_errors'][:50]):
                f.write(f"实例 {inst_stats['instance']:3d}: "
                       f"最大误差 = {inst_stats['max_error']:.6f}, "
                       f"平均误差 = {inst_stats['mean_error']:.6f}\n")
            f.write("\n")
        
        if 'pred_quality_max_errors' in all_results:
            f.write("=== 预测质量各属性最大误差 ===\n")
            for i, quality_stats in enumerate(all_results['pred_quality_max_errors']):
                f.write(f"质量分数 {quality_stats['quality']:2d} ({quality_stats['quality_name']:15s}): "
                       f"最大误差 = {quality_stats['max_error']:.6f}, "
                       f"平均误差 = {quality_stats['mean_error']:.6f}\n")
            f.write("\n")
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 总结
    print(f"\n=== 比较总结 ===")
    basic_data_types = [k for k in all_results.keys() if k not in ['pred_instance_max_errors', 'pred_feature_max_errors',
                                                                  'pred_anchor_max_errors', 'pred_attr_max_errors',
                                                                  'pred_class_instance_max_errors', 'pred_class_max_errors',
                                                                  'pred_quality_instance_max_errors', 'pred_quality_max_errors']]
    print(f"比较的数据类型数量: {len(basic_data_types)}")
    
    tolerance = 1e-6
    consistent_count = 0
    for data_type, stats in all_results.items():
        if data_type in basic_data_types:
            if data_type == 'pred_track_id':
                # 对于track_id，检查是否有不同元素
                if stats['num_different_elements'] == 0:
                    consistent_count += 1
                    print(f"✓ {data_type}: 完全一致")
                else:
                    print(f"✗ {data_type}: 存在差异 ({stats['num_different_elements']} 个不同元素)")
            else:
                # 对于浮点数，检查最大绝对误差
                if stats['max_abs_error'] < tolerance:
                    consistent_count += 1
                    print(f"✓ {data_type}: 完全一致")
                else:
                    print(f"✗ {data_type}: 存在差异 (最大误差: {stats['max_abs_error']:.6f})")
    
    print(f"\n完全一致的数据类型: {consistent_count}/{len(basic_data_types)}")
    
    # 特别说明第二帧输出的特点
    print(f"\n=== 第二帧推理输出特点说明 ===")
    print("1. pred_instance_feature: 预测实例特征 (1*900*256)")
    print("2. pred_anchor: 预测锚点 (1*900*11)")
    print("3. pred_class_score: 预测类别分数 (1*900*10)")
    print("4. pred_quality_score: 预测质量分数 (1*900*2)")
    print("5. pred_track_id: 预测跟踪ID (1*900)")
    print("6. 所有数据都经过GPU推理和16字节对齐")

if __name__ == "__main__":
    main()
