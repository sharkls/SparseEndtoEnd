#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载并打印track_id相关的bin文件内容
"""

import numpy as np
import os
import sys

def load_bin_file(file_path, dtype=np.int32):
    """
    加载bin文件
    
    Args:
        file_path: bin文件路径
        dtype: 数据类型，默认int32
    
    Returns:
        numpy数组
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] 文件不存在: {file_path}")
        return None
    
    try:
        # 读取bin文件
        data = np.fromfile(file_path, dtype=dtype)
        print(f"[INFO] 成功加载文件: {file_path}")
        print(f"[INFO] 文件大小: {os.path.getsize(file_path)} bytes")
        print(f"[INFO] 数据类型: {dtype}")
        print(f"[INFO] 数组形状: {data.shape}")
        print(f"[INFO] 数组大小: {data.size}")
        return data
    except Exception as e:
        print(f"[ERROR] 加载文件失败: {e}")
        return None

def print_array_content(data, title, max_display=None):
    """
    打印数组内容
    
    Args:
        data: numpy数组
        title: 标题
        max_display: 最大显示数量，None表示显示全部
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    if data is None:
        print("数据为空")
        return
    
    # 基本信息
    print(f"数组形状: {data.shape}")
    print(f"数组大小: {data.size}")
    print(f"数据类型: {data.dtype}")
    print(f"最小值: {data.min()}")
    print(f"最大值: {data.max()}")
    print(f"平均值: {data.mean():.4f}")
    
    # 统计-1的数量
    neg_one_count = np.sum(data == -1)
    print(f"-1的数量: {neg_one_count}")
    print(f"-1的比例: {neg_one_count/data.size*100:.2f}%")
    
    # 统计非-1的数量
    non_neg_one_count = np.sum(data != -1)
    print(f"非-1的数量: {non_neg_one_count}")
    print(f"非-1的比例: {non_neg_one_count/data.size*100:.2f}%")
    
    # 显示所有数据
    print(f"\n完整数据内容:")
    if max_display is not None and data.size > max_display:
        print(f"(由于数据量较大，只显示前{max_display}个值)")
        display_data = data[:max_display]
        print(f"前{max_display}个值:")
    else:
        display_data = data
        print(f"所有{data.size}个值:")
    
    # 按行打印，每行20个值
    values_per_line = 20
    for i in range(0, len(display_data), values_per_line):
        end_idx = min(i + values_per_line, len(display_data))
        line_values = display_data[i:end_idx]
        
        # 格式化索引范围
        if len(display_data) <= 100:  # 数据量小时显示索引
            indices = [f"[{j:3d}]" for j in range(i, end_idx)]
            values = [f"{val:4d}" for val in line_values]
            print(f"  {i:3d}-{end_idx-1:3d}: {' '.join(indices)} -> {' '.join(values)}")
        else:  # 数据量大时只显示值
            values = [f"{val:4d}" for val in line_values]
            print(f"  {i:3d}-{end_idx-1:3d}: {' '.join(values)}")
    
    if max_display is not None and data.size > max_display:
        print(f"\n... 还有 {data.size - max_display} 个值未显示")
        print(f"如需查看全部数据，请设置 max_display=None")

def main():
    """主函数"""
    print("Track ID Bin文件内容分析工具")
    print("="*80)
    
    # 文件路径
    base_dir = "/share/Code/SparseEnd2End/C++/Output/val_bin"
    pred_track_id_file = os.path.join(base_dir, "sample_1_pred_track_id_1*900_int32.bin")
    track_id_file = os.path.join(base_dir, "sample_1_track_id_1*900_int32.bin")
    
    # 检查文件是否存在
    if not os.path.exists(pred_track_id_file):
        print(f"[ERROR] 文件不存在: {pred_track_id_file}")
        print("请检查文件路径是否正确")
        return
    
    if not os.path.exists(track_id_file):
        print(f"[ERROR] 文件不存在: {track_id_file}")
        print("请检查文件路径是否正确")
        return
    
    # 加载文件
    print("[INFO] 开始加载文件...")
    
    # 加载pred_track_id文件
    pred_track_id_data = load_bin_file(pred_track_id_file, dtype=np.int32)
    
    # 加载track_id文件
    track_id_data = load_bin_file(track_id_file, dtype=np.int32)
    
    # 打印内容
    if pred_track_id_data is not None:
        # 对于pred_track_id，显示所有内容
        print_array_content(pred_track_id_data, "PRED_TRACK_ID 文件内容分析", max_display=None)
    
    if track_id_data is not None:
        # 对于track_id，显示所有内容
        print_array_content(track_id_data, "TRACK_ID 文件内容分析", max_display=None)
    
    # 如果两个文件都加载成功，进行对比分析
    if pred_track_id_data is not None and track_id_data is not None:
        print(f"\n{'='*80}")
        print("文件对比分析")
        print(f"{'='*80}")
        
        if pred_track_id_data.shape == track_id_data.shape:
            print(f"两个文件形状相同: {pred_track_id_data.shape}")
            
            # 计算差异
            diff_mask = pred_track_id_data != track_id_data
            diff_count = np.sum(diff_mask)
            print(f"差异元素数量: {diff_count}")
            print(f"差异比例: {diff_count/pred_track_id_data.size*100:.2f}%")
            
            if diff_count > 0:
                print(f"\n差异位置和值:")
                diff_indices = np.where(diff_mask)[0]
                for i, idx in enumerate(diff_indices[:20]):  # 只显示前20个差异
                    print(f"  索引[{idx:3d}]: pred_track_id={pred_track_id_data[idx]:4d}, track_id={track_id_data[idx]:4d}")
                
                if diff_count > 20:
                    print(f"  ... 还有 {diff_count - 20} 个差异未显示")
            else:
                print("两个文件内容完全相同")
        else:
            print(f"两个文件形状不同:")
            print(f"  pred_track_id: {pred_track_id_data.shape}")
            print(f"  track_id: {track_id_data.shape}")
    
    print(f"\n{'='*80}")
    print("分析完成")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()