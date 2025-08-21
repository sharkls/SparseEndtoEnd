#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较两个图像预处理文件的脚本
统计各通道、各摄像头的最大误差
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def load_bin_file(file_path, shape):
    """
    加载二进制文件
    
    Args:
        file_path: 文件路径
        shape: 数据形状 (batch_size, num_cameras, channels, height, width)
    
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

def compare_images(data1, data2, tolerance=1e-6):
    """
    比较两个图像数据并统计误差
    
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
    
    # 按摄像头统计最大误差
    if len(data1.shape) == 5:  # (batch, cameras, channels, height, width)
        batch_size, num_cameras, channels, height, width = data1.shape
        
        print(f"\n=== 按摄像头统计最大误差 ===")
        print(f"摄像头数量: {num_cameras}")
        print(f"通道数: {channels}")
        print(f"图像尺寸: {height}x{width}")
        
        camera_max_errors = []
        for i in range(num_cameras):
            camera_diff = abs_diff[:, i, :, :, :]
            max_error = np.max(camera_diff)
            mean_error = np.mean(camera_diff)
            camera_max_errors.append({
                'camera': i,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        camera_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        print(f"\n各摄像头最大误差:")
        for i, cam_stats in enumerate(camera_max_errors):
            print(f"  摄像头 {cam_stats['camera']}: "
                  f"最大误差 = {cam_stats['max_error']:.6f}, "
                  f"平均误差 = {cam_stats['mean_error']:.6f}")
        
        # 按通道统计最大误差
        print(f"\n=== 按通道统计最大误差 ===")
        channel_max_errors = []
        for i in range(channels):
            channel_diff = abs_diff[:, :, i, :, :]
            max_error = np.max(channel_diff)
            mean_error = np.mean(channel_diff)
            channel_max_errors.append({
                'channel': i,
                'max_error': max_error,
                'mean_error': mean_error
            })
        
        # 按最大误差排序
        channel_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        print(f"\n各通道最大误差:")
        channel_names = ['R', 'G', 'B']
        for i, ch_stats in enumerate(channel_max_errors):
            ch_name = channel_names[ch_stats['channel']] if ch_stats['channel'] < len(channel_names) else f'C{ch_stats["channel"]}'
            print(f"  通道 {ch_name}: "
                  f"最大误差 = {ch_stats['max_error']:.6f}, "
                  f"平均误差 = {ch_stats['mean_error']:.6f}")
        
        # 按像素位置统计最大误差
        print(f"\n=== 按像素位置统计最大误差 ===")
        pixel_max_errors = []
        for h in range(height):
            for w in range(width):
                pixel_diff = abs_diff[:, :, :, h, w]
                max_error = np.max(pixel_diff)
                mean_error = np.mean(pixel_diff)
                pixel_max_errors.append({
                    'height': h,
                    'width': w,
                    'max_error': max_error,
                    'mean_error': mean_error
                })
        
        # 按最大误差排序，只显示前10个
        pixel_max_errors.sort(key=lambda x: x['max_error'], reverse=True)
        
        print(f"\n前10个最大误差的像素位置:")
        for i, pix_stats in enumerate(pixel_max_errors[:10]):
            print(f"  像素 ({pix_stats['height']:3d}, {pix_stats['width']:3d}): "
                  f"最大误差 = {pix_stats['max_error']:.6f}, "
                  f"平均误差 = {pix_stats['mean_error']:.6f}")
    
    return stats, camera_max_errors, channel_max_errors, pixel_max_errors

def visualize_difference(data1, data2, output_dir="."):
    """
    可视化两个图像数据的差异
    
    Args:
        data1: 第一个数据
        data2: 第二个数据
        output_dir: 输出目录
    """
    if data1.shape != data2.shape:
        print("错误: 数据形状不匹配，无法可视化")
        return
    
    batch_size, num_cameras, channels, height, width = data1.shape
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个摄像头创建差异图
    for cam_idx in range(num_cameras):
        # 计算差异
        diff = np.abs(data1[0, cam_idx] - data2[0, cam_idx])
        
        # 创建图像
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'摄像头 {cam_idx} 图像差异对比', fontsize=16)
        
        # 原始图像1
        img1 = data1[0, cam_idx].transpose(1, 2, 0)  # (H, W, C)
        # 归一化到[0,1]用于显示
        img1_normalized = (img1 - img1.min()) / (img1.max() - img1.min())
        axes[0, 0].imshow(img1_normalized)
        axes[0, 0].set_title('原始图像1')
        axes[0, 0].axis('off')
        
        # 原始图像2
        img2 = data2[0, cam_idx].transpose(1, 2, 0)  # (H, W, C)
        img2_normalized = (img2 - img2.min()) / (img2.max() - img2.min())
        axes[0, 1].imshow(img2_normalized)
        axes[0, 1].set_title('原始图像2')
        axes[0, 1].axis('off')
        
        # 差异图
        diff_normalized = (diff.transpose(1, 2, 0) - diff.min()) / (diff.max() - diff.min())
        axes[0, 2].imshow(diff_normalized)
        axes[0, 2].set_title('差异图')
        axes[0, 2].axis('off')
        
        # 各通道差异
        channel_names = ['R', 'G', 'B']
        for ch_idx in range(channels):
            ch_diff = diff[ch_idx]
            ch_diff_normalized = (ch_diff - ch_diff.min()) / (ch_diff.max() - ch_diff.min())
            axes[1, ch_idx].imshow(ch_diff_normalized, cmap='hot')
            axes[1, ch_idx].set_title(f'{channel_names[ch_idx]}通道差异')
            axes[1, ch_idx].axis('off')
        
        # plt.tight_layout()
        plt.savefig(f'{output_dir}/camera_{cam_idx}_difference.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"摄像头 {cam_idx} 差异图已保存到: {output_dir}/camera_{cam_idx}_difference.png")

def main():
    # 文件路径
    file1_path = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_0_imgs_1*6*3*256*704_float32.bin"
    file2_path = "/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_imgs_1*6*3*256*704_float32.bin"
    
    # 数据形状
    shape = (1, 6, 3, 256, 704)  # (batch_size, num_cameras, channels, height, width)
    
    print("=== 图像预处理比较脚本 ===")
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
    stats, camera_max_errors, channel_max_errors, pixel_max_errors = compare_images(data1, data2)
    
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
    tolerance = 0.02
    if stats['max_abs_error'] < tolerance:
        print(f"\n✓ 数据完全一致 (最大误差 < {tolerance})")
    else:
        print(f"\n✗ 数据存在差异 (最大误差 >= {tolerance})")
    
    # 生成可视化
    print(f"\n=== 生成可视化 ===")
    visualize_difference(data1, data2, "img_comparison_plots")
    
    # 保存详细结果到文件
    output_file = "preprocess_img_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 图像预处理比较结果 ===\n")
        f.write(f"文件1: {file1_path}\n")
        f.write(f"文件2: {file2_path}\n")
        f.write(f"数据形状: {shape}\n\n")
        
        f.write("=== 总体统计 ===\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\n=== 各摄像头最大误差 ===\n")
        for i, cam_stats in enumerate(camera_max_errors):
            f.write(f"摄像头 {cam_stats['camera']}: "
                   f"最大误差 = {cam_stats['max_error']:.6f}, "
                   f"平均误差 = {cam_stats['mean_error']:.6f}\n")
        
        f.write(f"\n=== 各通道最大误差 ===\n")
        channel_names = ['R', 'G', 'B']
        for i, ch_stats in enumerate(channel_max_errors):
            ch_name = channel_names[ch_stats['channel']] if ch_stats['channel'] < len(channel_names) else f'C{ch_stats["channel"]}'
            f.write(f"通道 {ch_name}: "
                   f"最大误差 = {ch_stats['max_error']:.6f}, "
                   f"平均误差 = {ch_stats['mean_error']:.6f}\n")
        
        f.write(f"\n=== 前20个最大误差的像素位置 ===\n")
        for i, pix_stats in enumerate(pixel_max_errors[:20]):
            f.write(f"像素 ({pix_stats['height']:3d}, {pix_stats['width']:3d}): "
                   f"最大误差 = {pix_stats['max_error']:.6f}, "
                   f"平均误差 = {pix_stats['mean_error']:.6f}\n")
    
    print(f"\n详细结果已保存到: {output_file}")
    print(f"可视化图像已保存到: img_comparison_plots/ 目录")

if __name__ == "__main__":
    main() 