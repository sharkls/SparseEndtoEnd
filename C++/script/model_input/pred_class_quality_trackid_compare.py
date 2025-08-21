#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pred Class Score, Quality Score, Track ID 比较脚本
对比C++推理的分类得分、质量得分、跟踪ID数据与Python预期数据的差异
"""

import numpy as np
import os
import sys
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

def load_bin_file(file_path: str, expected_shape: Tuple[int, ...]) -> np.ndarray:
    """
    加载二进制文件并重塑为指定形状
    
    Args:
        file_path: 文件路径
        expected_shape: 期望的形状
        
    Returns:
        加载的数据数组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 读取二进制数据
    data = np.fromfile(file_path, dtype=np.float32)
    
    # 计算期望的元素数量
    expected_size = np.prod(expected_shape)
    
    if data.size != expected_size:
        raise ValueError(f"数据大小不匹配: 期望 {expected_size}, 实际 {data.size}")
    
    # 重塑数据
    data = data.reshape(expected_shape)
    
    print(f"成功加载文件: {file_path}")
    print(f"  形状: {data.shape}")
    print(f"  数据类型: {data.dtype}")
    print(f"  数值范围: [{data.min():.6f}, {data.max():.6f}]")
    
    return data

def compare_class_scores(cpp_data: np.ndarray, python_data: np.ndarray) -> Dict[str, Any]:
    """
    比较分类得分数据
    
    Args:
        cpp_data: C++分类得分数据
        python_data: Python分类得分数据
        
    Returns:
        比较结果字典
    """
    print("\n=== 分类得分数据比较 ===")
    
    # 计算差异
    diff = cpp_data - python_data
    abs_diff = np.abs(diff)
    rel_diff = np.abs(diff) / (np.abs(python_data) + 1e-8)
    
    # 按类别统计
    num_classes = cpp_data.shape[-1]
    class_stats = {}
    
    for i in range(num_classes):
        class_diff = abs_diff[..., i]
        class_rel_diff = rel_diff[..., i]
        
        class_stats[f"class_{i}"] = {
            "max_error": float(class_diff.max()),
            "mean_error": float(class_diff.mean()),
            "std_error": float(class_diff.std()),
            "max_rel_error": float(class_rel_diff.max()),
            "mean_rel_error": float(class_rel_diff.mean())
        }
    
    # 总体统计
    overall_stats = {
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "std_abs_error": float(abs_diff.std()),
        "max_rel_error": float(rel_diff.max()),
        "mean_rel_error": float(rel_diff.mean()),
        "std_rel_error": float(rel_diff.std()),
        "class_stats": class_stats
    }
    
    print(f"分类得分形状: {cpp_data.shape}")
    print(f"最大绝对误差: {overall_stats['max_abs_error']:.6f}")
    print(f"平均绝对误差: {overall_stats['mean_abs_error']:.6f}")
    print(f"最大相对误差: {overall_stats['max_rel_error']:.6f}")
    print(f"平均相对误差: {overall_stats['mean_rel_error']:.6f}")
    
    return overall_stats

def compare_quality_scores(cpp_data: np.ndarray, python_data: np.ndarray) -> Dict[str, Any]:
    """
    比较质量得分数据
    
    Args:
        cpp_data: C++质量得分数据
        python_data: Python质量得分数据
        
    Returns:
        比较结果字典
    """
    print("\n=== 质量得分数据比较 ===")
    
    # 计算差异
    diff = cpp_data - python_data
    abs_diff = np.abs(diff)
    rel_diff = np.abs(diff) / (np.abs(python_data) + 1e-8)
    
    # 按质量维度统计
    num_quality_dims = cpp_data.shape[-1]
    quality_stats = {}
    
    for i in range(num_quality_dims):
        dim_diff = abs_diff[..., i]
        dim_rel_diff = rel_diff[..., i]
        
        quality_stats[f"quality_dim_{i}"] = {
            "max_error": float(dim_diff.max()),
            "mean_error": float(dim_diff.mean()),
            "std_error": float(dim_diff.std()),
            "max_rel_error": float(dim_rel_diff.max()),
            "mean_rel_error": float(dim_rel_diff.mean())
        }
    
    # 总体统计
    overall_stats = {
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "std_abs_error": float(abs_diff.std()),
        "max_rel_error": float(rel_diff.max()),
        "mean_rel_error": float(rel_diff.mean()),
        "std_rel_error": float(rel_diff.std()),
        "quality_stats": quality_stats
    }
    
    print(f"质量得分形状: {cpp_data.shape}")
    print(f"最大绝对误差: {overall_stats['max_abs_error']:.6f}")
    print(f"平均绝对误差: {overall_stats['mean_abs_error']:.6f}")
    print(f"最大相对误差: {overall_stats['max_rel_error']:.6f}")
    print(f"平均相对误差: {overall_stats['mean_rel_error']:.6f}")
    
    return overall_stats

def compare_track_ids(cpp_data: np.ndarray, python_data: np.ndarray) -> Dict[str, Any]:
    """
    比较跟踪ID数据
    
    Args:
        cpp_data: C++跟踪ID数据
        python_data: Python跟踪ID数据
        
    Returns:
        比较结果字典
    """
    print("\n=== 跟踪ID数据比较 ===")
    
    # 计算差异
    diff = cpp_data - python_data
    abs_diff = np.abs(diff)
    
    # 统计信息
    stats = {
        "max_abs_error": int(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "std_abs_error": float(abs_diff.std()),
        "exact_match_count": int(np.sum(cpp_data == python_data)),
        "total_count": int(cpp_data.size),
        "match_percentage": float(np.sum(cpp_data == python_data) / cpp_data.size * 100),
        "cpp_unique_values": int(len(np.unique(cpp_data))),
        "python_unique_values": int(len(np.unique(python_data))),
        "cpp_value_range": [int(cpp_data.min()), int(cpp_data.max())],
        "python_value_range": [int(python_data.min()), int(python_data.max())]
    }
    
    print(f"跟踪ID形状: {cpp_data.shape}")
    print(f"最大绝对误差: {stats['max_abs_error']}")
    print(f"平均绝对误差: {stats['mean_abs_error']:.6f}")
    print(f"完全匹配数量: {stats['exact_match_count']} / {stats['total_count']}")
    print(f"匹配百分比: {stats['match_percentage']:.2f}%")
    print(f"C++唯一值数量: {stats['cpp_unique_values']}")
    print(f"Python唯一值数量: {stats['python_unique_values']}")
    print(f"C++值范围: [{stats['cpp_value_range'][0]}, {stats['cpp_value_range'][1]}]")
    print(f"Python值范围: [{stats['python_value_range'][0]}, {stats['python_value_range'][1]}]")
    
    return stats

def visualize_differences(cpp_data: np.ndarray, python_data: np.ndarray, 
                         data_type: str, save_path: str = None):
    """
    可视化数据差异
    
    Args:
        cpp_data: C++数据
        python_data: Python数据
        data_type: 数据类型名称
        save_path: 保存路径
    """
    # 计算差异
    diff = cpp_data - python_data
    abs_diff = np.abs(diff)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{data_type} 数据比较', fontsize=16)
    
    # 1. 差异分布直方图
    axes[0, 0].hist(abs_diff.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('绝对误差分布')
    axes[0, 0].set_xlabel('绝对误差')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. C++ vs Python 散点图（取前1000个点）
    flat_cpp = cpp_data.flatten()[:1000]
    flat_python = python_data.flatten()[:1000]
    axes[0, 1].scatter(flat_python, flat_cpp, alpha=0.6, s=10)
    axes[0, 1].plot([flat_python.min(), flat_python.max()], 
                     [flat_python.min(), flat_python.max()], 'r--', lw=2)
    axes[0, 1].set_title('C++ vs Python (前1000个点)')
    axes[0, 1].set_xlabel('Python值')
    axes[0, 1].set_ylabel('C++值')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 误差热力图（如果是2D数据）
    if len(cpp_data.shape) == 2:
        im = axes[1, 0].imshow(abs_diff, cmap='hot', aspect='auto')
        axes[1, 0].set_title('绝对误差热力图')
        plt.colorbar(im, ax=axes[1, 0])
    else:
        # 如果是3D数据，显示第一个batch
        im = axes[1, 0].imshow(abs_diff[0], cmap='hot', aspect='auto')
        axes[1, 0].set_title('绝对误差热力图 (第一个batch)')
        plt.colorbar(im, ax=axes[1, 0])
    
    # 4. 统计信息
    axes[1, 1].axis('off')
    stats_text = f"""
统计信息:
最大绝对误差: {abs_diff.max():.6f}
平均绝对误差: {abs_diff.mean():.6f}
标准差: {abs_diff.std():.6f}
C++数据范围: [{cpp_data.min():.6f}, {cpp_data.max():.6f}]
Python数据范围: [{python_data.min():.6f}, {python_data.max():.6f}]
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                     verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()

def save_comparison_results(results: Dict[str, Any], save_path: str):
    """
    保存比较结果到文件
    
    Args:
        results: 比较结果字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== Pred Class Score, Quality Score, Track ID 比较结果 ===\n\n")
        
        # 分类得分结果
        if 'class_scores' in results:
            f.write("分类得分比较结果:\n")
            f.write(f"  最大绝对误差: {results['class_scores']['max_abs_error']:.6f}\n")
            f.write(f"  平均绝对误差: {results['class_scores']['mean_abs_error']:.6f}\n")
            f.write(f"  最大相对误差: {results['class_scores']['max_rel_error']:.6f}\n")
            f.write(f"  平均相对误差: {results['class_scores']['mean_rel_error']:.6f}\n\n")
            
            f.write("各类别误差统计:\n")
            for class_name, stats in results['class_scores']['class_stats'].items():
                f.write(f"  {class_name}:\n")
                f.write(f"    最大误差: {stats['max_error']:.6f}\n")
                f.write(f"    平均误差: {stats['mean_error']:.6f}\n")
                f.write(f"    标准差: {stats['std_error']:.6f}\n")
            f.write("\n")
        
        # 质量得分结果
        if 'quality_scores' in results:
            f.write("质量得分比较结果:\n")
            f.write(f"  最大绝对误差: {results['quality_scores']['max_abs_error']:.6f}\n")
            f.write(f"  平均绝对误差: {results['quality_scores']['mean_abs_error']:.6f}\n")
            f.write(f"  最大相对误差: {results['quality_scores']['max_rel_error']:.6f}\n")
            f.write(f"  平均相对误差: {results['quality_scores']['mean_rel_error']:.6f}\n\n")
            
            f.write("各质量维度误差统计:\n")
            for dim_name, stats in results['quality_scores']['quality_stats'].items():
                f.write(f"  {dim_name}:\n")
                f.write(f"    最大误差: {stats['max_error']:.6f}\n")
                f.write(f"    平均误差: {stats['mean_error']:.6f}\n")
                f.write(f"    标准差: {stats['std_error']:.6f}\n")
            f.write("\n")
        
        # 跟踪ID结果
        if 'track_ids' in results:
            f.write("跟踪ID比较结果:\n")
            f.write(f"  最大绝对误差: {results['track_ids']['max_abs_error']}\n")
            f.write(f"  平均绝对误差: {results['track_ids']['mean_abs_error']:.6f}\n")
            f.write(f"  完全匹配数量: {results['track_ids']['exact_match_count']} / {results['track_ids']['total_count']}\n")
            f.write(f"  匹配百分比: {results['track_ids']['match_percentage']:.2f}%\n")
            f.write(f"  C++唯一值数量: {results['track_ids']['cpp_unique_values']}\n")
            f.write(f"  Python唯一值数量: {results['track_ids']['python_unique_values']}\n")
            f.write(f"  C++值范围: [{results['track_ids']['cpp_value_range'][0]}, {results['track_ids']['cpp_value_range'][1]}]\n")
            f.write(f"  Python值范围: [{results['track_ids']['python_value_range'][0]}, {results['track_ids']['python_value_range'][1]}]\n\n")
        
        # 总体评估
        f.write("=== 总体评估 ===\n")
        max_errors = []
        if 'class_scores' in results:
            max_errors.append(('分类得分', results['class_scores']['max_abs_error']))
        if 'quality_scores' in results:
            max_errors.append(('质量得分', results['quality_scores']['max_abs_error']))
        if 'track_ids' in results:
            max_errors.append(('跟踪ID', results['track_ids']['max_abs_error']))
        
        if max_errors:
            max_error_type, max_error_value = max(max_errors, key=lambda x: x[1])
            f.write(f"最大误差类型: {max_error_type} ({max_error_value:.6f})\n")
            
            if max_error_value < 1e-6:
                f.write("✓ 数据完全一致\n")
            elif max_error_value < 0.1:
                f.write("✓ 数据差异很小，可接受\n")
            elif max_error_value < 1.0:
                f.write("⚠ 数据存在一定差异，需要关注\n")
            else:
                f.write("✗ 数据差异较大，需要进一步调查\n")
    
    print(f"详细结果已保存到: {save_path}")

def main():
    """主函数"""
    print("=== Pred Class Score, Quality Score, Track ID 比较脚本 ===\n")
    
    # 文件路径
    cpp_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin"
    python_dir = "/share/Code/SparseEnd2End/script/tutorial/asset"
    sample_id = "sample_0"
    
    # 文件路径
    cpp_class_score = os.path.join(cpp_dir, f"{sample_id}_pred_class_score_1*900*10_float32.bin")
    python_class_score = os.path.join(python_dir, f"{sample_id}_pred_class_score_1*900*10_float32.bin")
    
    cpp_quality_score = os.path.join(cpp_dir, f"{sample_id}_pred_quality_score_1*900*2_float32.bin")
    python_quality_score = os.path.join(python_dir, f"{sample_id}_pred_quality_score_1*900*2_float32.bin")
    
    cpp_track_id = os.path.join(cpp_dir, f"{sample_id}_pred_track_id_1*900_int32.bin")
    python_track_id = os.path.join(python_dir, f"{sample_id}_pred_track_id_1*900_int32.bin")
    
    print(f"文件1 (C++): {cpp_dir}")
    print(f"文件2 (Python): {python_dir}")
    
    results = {}
    
    try:
        # 比较分类得分
        print(f"\n=== 加载分类得分数据 ===")
        cpp_class_data = load_bin_file(cpp_class_score, (1, 900, 10))
        python_class_data = load_bin_file(python_class_score, (1, 900, 10))
        results['class_scores'] = compare_class_scores(cpp_class_data, python_class_data)
        
        # 比较质量得分
        print(f"\n=== 加载质量得分数据 ===")
        cpp_quality_data = load_bin_file(cpp_quality_score, (1, 900, 2))
        python_quality_data = load_bin_file(python_quality_score, (1, 900, 2))
        results['quality_scores'] = compare_quality_scores(cpp_quality_data, python_quality_data)
        
        # 比较跟踪ID
        print(f"\n=== 加载跟踪ID数据 ===")
        cpp_track_data = load_bin_file(cpp_track_id, (1, 900)).astype(np.int32)
        python_track_data = load_bin_file(python_track_id, (1, 900)).astype(np.int32)
        results['track_ids'] = compare_track_ids(cpp_track_data, python_track_data)
        
        # 可视化差异
        print(f"\n=== 生成可视化图表 ===")
        visualize_differences(cpp_class_data, python_class_data, "分类得分", 
                            "class_score_comparison.png")
        visualize_differences(cpp_quality_data, python_quality_data, "质量得分", 
                            "quality_score_comparison.png")
        visualize_differences(cpp_track_data, python_track_data, "跟踪ID", 
                            "track_id_comparison.png")
        
        # 保存结果
        save_comparison_results(results, "pred_class_quality_trackid_comparison_results.txt")
        
        # 总体评估
        print(f"\n=== 总体评估 ===")
        max_errors = []
        if 'class_scores' in results:
            max_errors.append(('分类得分', results['class_scores']['max_abs_error']))
        if 'quality_scores' in results:
            max_errors.append(('质量得分', results['quality_scores']['max_abs_error']))
        if 'track_ids' in results:
            max_errors.append(('跟踪ID', results['track_ids']['max_abs_error']))
        
        if max_errors:
            max_error_type, max_error_value = max(max_errors, key=lambda x: x[1])
            print(f"最大误差类型: {max_error_type} ({max_error_value:.6f})")
            
            if max_error_value < 1e-6:
                print("✓ 数据完全一致")
            elif max_error_value < 0.1:
                print("✓ 数据差异很小，可接受")
            elif max_error_value < 1.0:
                print("⚠ 数据存在一定差异，需要关注")
            else:
                print("✗ 数据差异较大，需要进一步调查")
        
    except Exception as e:
        print(f"错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ 比较完成")
    else:
        print("\n✗ 比较失败")
        sys.exit(1) 