#!/usr/bin/env python3
"""
对比两个预处理后的bin文件
文件1：C++预处理输出
文件2：Python脚本生成的参考数据
同时对比时间戳是否一致
"""

import numpy as np
import os
import struct
import argparse
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

def load_bin_file(filepath: str, expected_shape: Tuple[int, ...], dtype: str = 'float32') -> np.ndarray:
    """
    加载二进制文件
    
    Args:
        filepath: 文件路径
        expected_shape: 期望的形状
        dtype: 数据类型
    
    Returns:
        加载的数据数组
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    # 读取文件
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # 计算元素数量
    element_size = np.dtype(dtype).itemsize
    num_elements = len(data) // element_size
    
    # 验证元素数量
    expected_elements = np.prod(expected_shape)
    if num_elements != expected_elements:
        raise ValueError(f"文件大小不匹配: 期望 {expected_elements} 个元素，实际 {num_elements} 个元素")
    
    # 转换为numpy数组
    array = np.frombuffer(data, dtype=dtype)
    array = array.reshape(expected_shape)
    
    return array

def load_timestamp_file(filepath: str) -> float:
    """
    加载时间戳文件（单个float32值）
    
    Args:
        filepath: 时间戳文件路径
    
    Returns:
        时间戳值
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"时间戳文件不存在: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    if len(data) != 4:
        raise ValueError(f"时间戳文件大小不正确: 期望4字节，实际{len(data)}字节")
    
    timestamp = struct.unpack('f', data)[0]
    return timestamp

def extract_timestamp_from_filename(filepath: str) -> int:
    """
    从文件名中提取时间戳
    
    Args:
        filepath: 文件路径
    
    Returns:
        时间戳（整数）
    """
    filename = os.path.basename(filepath)
    # 查找文件名中的时间戳数字
    import re
    timestamp_match = re.search(r'(\d{13,})', filename)
    if timestamp_match:
        return int(timestamp_match.group(1))
    else:
        raise ValueError(f"无法从文件名中提取时间戳: {filename}")

def compare_timestamps(cpp_timestamp: int, python_timestamp: float, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    对比两个时间戳
    
    Args:
        cpp_timestamp: C++输出的时间戳（整数）
        python_timestamp: Python参考数据的时间戳（float）
        tolerance: 容差
    
    Returns:
        对比结果字典
    """
    # 将C++时间戳转换为float进行比较
    cpp_timestamp_float = float(cpp_timestamp)
    
    diff = abs(cpp_timestamp_float - python_timestamp)
    relative_diff = diff / (abs(python_timestamp) + 1e-8)
    
    stats = {
        'cpp_timestamp': cpp_timestamp,
        'cpp_timestamp_float': cpp_timestamp_float,
        'python_timestamp': python_timestamp,
        'absolute_diff': diff,
        'relative_diff': relative_diff,
        'within_tolerance': diff <= tolerance,
        'tolerance': tolerance,
        'match': diff <= tolerance
    }
    
    return stats

def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    对比两个数组
    
    Args:
        arr1: 第一个数组
        arr2: 第二个数组
        tolerance: 容差
    
    Returns:
        对比结果字典
    """
    # 检查形状是否相同
    if arr1.shape != arr2.shape:
        return {
            'shape_match': False,
            'shape1': arr1.shape,
            'shape2': arr2.shape,
            'error': '形状不匹配'
        }
    
    # 计算差异
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    
    # 统计信息
    stats = {
        'shape_match': True,
        'shape': arr1.shape,
        'total_elements': arr1.size,
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff),
        'max_diff': np.max(abs_diff),
        'min_diff': np.min(abs_diff),
        'mean_abs_diff': np.mean(abs_diff),
        'max_relative_diff': np.max(abs_diff / (np.abs(arr2) + 1e-8)),
        'elements_within_tolerance': np.sum(abs_diff <= tolerance),
        'tolerance': tolerance,
        'match_percentage': np.sum(abs_diff <= tolerance) / arr1.size * 100,
        'arr1_stats': {
            'min': np.min(arr1),
            'max': np.max(arr1),
            'mean': np.mean(arr1),
            'std': np.std(arr1)
        },
        'arr2_stats': {
            'min': np.min(arr2),
            'max': np.max(arr2),
            'mean': np.mean(arr2),
            'std': np.std(arr2)
        }
    }
    
    return stats

def visualize_comparison(arr1: np.ndarray, arr2: np.ndarray, save_dir: str):
    """
    可视化对比结果
    
    Args:
        arr1: 第一个数组
        arr2: 第二个数组
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 差异分布直方图
    diff = arr1 - arr2
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.hist(diff.flatten(), bins=100, alpha=0.7, label='差异分布')
    plt.xlabel('差异值')
    plt.ylabel('频次')
    plt.title('差异分布直方图')
    plt.legend()
    
    # 2. 绝对值差异分布
    plt.subplot(2, 3, 2)
    abs_diff = np.abs(diff)
    plt.hist(abs_diff.flatten(), bins=100, alpha=0.7, label='绝对差异分布')
    plt.xlabel('绝对差异值')
    plt.ylabel('频次')
    plt.title('绝对差异分布')
    plt.legend()
    
    # 3. 第一个数组的分布
    plt.subplot(2, 3, 3)
    plt.hist(arr1.flatten(), bins=100, alpha=0.7, label='数组1分布')
    plt.xlabel('值')
    plt.ylabel('频次')
    plt.title('数组1值分布')
    plt.legend()
    
    # 4. 第二个数组的分布
    plt.subplot(2, 3, 4)
    plt.hist(arr2.flatten(), bins=100, alpha=0.7, label='数组2分布')
    plt.xlabel('值')
    plt.ylabel('频次')
    plt.title('数组2值分布')
    plt.legend()
    
    # 5. 散点图对比
    plt.subplot(2, 3, 5)
    # 随机采样以避免图像过于密集
    sample_size = min(10000, arr1.size)
    indices = np.random.choice(arr1.size, sample_size, replace=False)
    plt.scatter(arr1.flatten()[indices], arr2.flatten()[indices], alpha=0.5, s=1)
    plt.plot([arr1.min(), arr1.max()], [arr1.min(), arr1.max()], 'r--', label='y=x')
    plt.xlabel('数组1值')
    plt.ylabel('数组2值')
    plt.title('散点图对比')
    plt.legend()
    
    # 6. 差异热力图（取第一个相机的前几个通道）
    plt.subplot(2, 3, 6)
    if len(arr1.shape) >= 3:
        # 显示第一个相机的第一个通道
        cam_idx = 0
        channel_idx = 0
        slice_data = diff[cam_idx, channel_idx, :, :]
        plt.imshow(slice_data, cmap='RdBu_r', aspect='auto')
        plt.colorbar(label='差异值')
        plt.title(f'相机{cam_idx}通道{channel_idx}差异热力图')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, 'comparison_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果保存到: {os.path.join(save_dir, 'comparison_visualization.png')}")

def save_detailed_report(stats: Dict[str, Any], timestamp_stats: Dict[str, Any], save_dir: str, file1_name: str, file2_name: str):
    """
    保存详细报告
    
    Args:
        stats: 数据对比统计信息
        timestamp_stats: 时间戳对比统计信息
        save_dir: 保存目录
        file1_name: 文件1名称
        file2_name: 文件2名称
    """
    os.makedirs(save_dir, exist_ok=True)
    
    report_file = os.path.join(save_dir, 'comparison_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("预处理数据对比报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"文件1: {file1_name}\n")
        f.write(f"文件2: {file2_name}\n\n")
        
        # 时间戳对比结果
        f.write("时间戳对比:\n")
        f.write(f"  C++时间戳: {timestamp_stats['cpp_timestamp']}\n")
        f.write(f"  Python时间戳: {timestamp_stats['python_timestamp']:.6f}\n")
        f.write(f"  绝对差异: {timestamp_stats['absolute_diff']:.6e}\n")
        f.write(f"  相对差异: {timestamp_stats['relative_diff']:.6e}\n")
        f.write(f"  容差: {timestamp_stats['tolerance']:.6e}\n")
        f.write(f"  时间戳匹配: {'✅' if timestamp_stats['match'] else '❌'}\n\n")
        
        if not stats['shape_match']:
            f.write("❌ 形状不匹配\n")
            f.write(f"  文件1形状: {stats['shape1']}\n")
            f.write(f"  文件2形状: {stats['shape2']}\n")
            return
        
        f.write("✅ 形状匹配\n")
        f.write(f"  形状: {stats['shape']}\n")
        f.write(f"  总元素数: {stats['total_elements']:,}\n\n")
        
        f.write("差异统计:\n")
        f.write(f"  平均差异: {stats['mean_diff']:.6e}\n")
        f.write(f"  差异标准差: {stats['std_diff']:.6e}\n")
        f.write(f"  最大绝对差异: {stats['max_diff']:.6e}\n")
        f.write(f"  最小绝对差异: {stats['min_diff']:.6e}\n")
        f.write(f"  平均绝对差异: {stats['mean_abs_diff']:.6e}\n")
        f.write(f"  最大相对差异: {stats['max_relative_diff']:.6e}\n")
        f.write(f"  容差: {stats['tolerance']:.6e}\n")
        f.write(f"  在容差内的元素数: {stats['elements_within_tolerance']:,}\n")
        f.write(f"  匹配百分比: {stats['match_percentage']:.2f}%\n\n")
        
        f.write("数组1统计:\n")
        f.write(f"  最小值: {stats['arr1_stats']['min']:.6e}\n")
        f.write(f"  最大值: {stats['arr1_stats']['max']:.6e}\n")
        f.write(f"  平均值: {stats['arr1_stats']['mean']:.6e}\n")
        f.write(f"  标准差: {stats['arr1_stats']['std']:.6e}\n\n")
        
        f.write("数组2统计:\n")
        f.write(f"  最小值: {stats['arr2_stats']['min']:.6e}\n")
        f.write(f"  最大值: {stats['arr2_stats']['max']:.6e}\n")
        f.write(f"  平均值: {stats['arr2_stats']['mean']:.6e}\n")
        f.write(f"  标准差: {stats['arr2_stats']['std']:.6e}\n\n")
        
        # 综合判断
        f.write("综合评估:\n")
        timestamp_ok = timestamp_stats['match']
        data_ok = stats['match_percentage'] >= 95.0
        
        if timestamp_ok and data_ok:
            f.write("🎉 结果: 完全匹配 (时间戳和数据都匹配)\n")
        elif timestamp_ok and stats['match_percentage'] >= 80.0:
            f.write("✅ 结果: 时间戳匹配，数据部分匹配\n")
        elif not timestamp_ok and data_ok:
            f.write("⚠️  结果: 时间戳不匹配，但数据匹配\n")
        else:
            f.write("❌ 结果: 时间戳和数据都不匹配\n")
    
    print(f"详细报告保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='对比两个预处理后的bin文件')
    parser.add_argument('--file1', type=str, 
                       default='/share/Code/SparseEnd2End/C++/Output/val_bin/preprocessed_imgs_6*3*256*704_float32_1750732794366.bin',
                       help='第一个文件路径')
    parser.add_argument('--file2', type=str,
                       default='/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_imgs_1*6*3*256*704_float32.bin',
                       help='第二个文件路径')
    parser.add_argument('--timestamp_file', type=str,
                       default='/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_time_interval_1_float32.bin',
                       help='Python参考数据的时间戳文件路径')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='容差')
    parser.add_argument('--save_dir', type=str,
                       default='/share/Code/SparseEnd2End/C++/script/compare/results',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    print("开始对比bin文件...")
    print(f"文件1: {args.file1}")
    print(f"文件2: {args.file2}")
    print(f"时间戳文件: {args.timestamp_file}")
    print(f"容差: {args.tolerance}")
    print(f"保存目录: {args.save_dir}")
    print("-" * 50)
    
    try:
        # 加载时间戳
        print("加载时间戳文件...")
        python_timestamp = load_timestamp_file(args.timestamp_file)
        print(f"Python时间戳: {python_timestamp}")
        
        # 从C++文件名提取时间戳
        cpp_timestamp = extract_timestamp_from_filename(args.file1)
        print(f"C++时间戳: {cpp_timestamp}")
        
        # 对比时间戳
        print("对比时间戳...")
        timestamp_stats = compare_timestamps(cpp_timestamp, python_timestamp, args.tolerance)
        print(f"时间戳匹配: {'✅' if timestamp_stats['match'] else '❌'}")
        print(f"时间戳差异: {timestamp_stats['absolute_diff']:.6e}")
        
        # 加载数据
        print("加载文件1...")
        arr1 = load_bin_file(args.file1, (6, 3, 256, 704), 'float32')
        print(f"文件1加载成功，形状: {arr1.shape}")
        
        print("加载文件2...")
        arr2 = load_bin_file(args.file2, (1, 6, 3, 256, 704), 'float32')
        # 移除batch维度
        arr2 = arr2.squeeze(0)
        print(f"文件2加载成功，形状: {arr2.shape}")
        
        # 对比数据
        print("进行数据对比...")
        stats = compare_arrays(arr1, arr2, args.tolerance)
        
        # 打印简要结果
        print("\n对比结果:")
        print(f"时间戳匹配: {'✅' if timestamp_stats['match'] else '❌'}")
        print(f"形状匹配: {'✅' if stats['shape_match'] else '❌'}")
        if stats['shape_match']:
            print(f"数据匹配百分比: {stats['match_percentage']:.2f}%")
            print(f"最大绝对差异: {stats['max_diff']:.6e}")
            print(f"平均绝对差异: {stats['mean_abs_diff']:.6e}")
        
        # 保存详细报告
        print("\n保存详细报告...")
        save_detailed_report(stats, timestamp_stats, args.save_dir, 
                           os.path.basename(args.file1), 
                           os.path.basename(args.file2))
        
        # 生成可视化
        print("生成可视化...")
        visualize_comparison(arr1, arr2, args.save_dir)
        
        print("\n对比完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 