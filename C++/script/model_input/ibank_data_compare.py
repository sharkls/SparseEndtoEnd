# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
"""
InstanceBank数据一致性验证脚本
比较InstanceBank保存的数据与PyTorch asset文件的数据
"""

import os
import sys
import numpy as np
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tool.utils.logger import logger_wrapper
from deploy.utils.utils import printArrayInformation

def read_bin_file(file_path, shape, dtype=np.float32):
    """读取二进制文件"""
    try:
        data = np.fromfile(file_path, dtype=dtype)
        if shape:
            data = data.reshape(shape)
        return data
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to read file {file_path}: {e}")
        return None

def calculate_metrics(x, y, name):
    """计算绝对误差和cosine distance"""
    if x is None or y is None:
        print(f"[ERROR] Cannot calculate metrics for {name}: data is None")
        return
    
    if x.shape != y.shape:
        print(f"[ERROR] Shape mismatch for {name}: {x.shape} vs {y.shape}")
        return
    
    # 绝对误差统计
    abs_diff = np.abs(x - y)
    max_abs_error = float(abs_diff.max())
    mean_abs_error = float(abs_diff.mean())
    std_abs_error = float(abs_diff.std())
    
    # 相对误差统计
    rel_diff = np.abs((x - y) / (np.abs(y) + 1e-8))  # 避免除零
    max_rel_error = float(rel_diff.max())
    mean_rel_error = float(rel_diff.mean())
    std_rel_error = float(rel_diff.std())
    
    # Cosine distance
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # 避免零向量
    x_norm = np.linalg.norm(x_flat)
    y_norm = np.linalg.norm(y_flat)
    
    if x_norm > 1e-8 and y_norm > 1e-8:
        cosine_similarity = np.dot(x_flat, y_flat) / (x_norm * y_norm)
        cosine_distance = 1 - cosine_similarity
    else:
        cosine_distance = 1.0  # 如果向量接近零，cosine distance为1
    
    print(f"\n{'='*80}")
    print(f"数据比较结果: {name}")
    print(f"{'='*80}")
    print(f"形状: {x.shape}")
    print(f"数据类型: {x.dtype}")
    
    print(f"\n绝对误差统计:")
    print(f"  最大绝对误差: {max_abs_error:.6f}")
    print(f"  平均绝对误差: {mean_abs_error:.6f}")
    print(f"  绝对误差标准差: {std_abs_error:.6f}")
    
    print(f"\n相对误差统计:")
    print(f"  最大相对误差: {max_rel_error:.6f}")
    print(f"  平均相对误差: {mean_rel_error:.6f}")
    print(f"  相对误差标准差: {std_rel_error:.6f}")
    
    print(f"\n相似性度量:")
    print(f"  Cosine Distance: {cosine_distance:.6f}")
    
    # 打印数据范围信息
    print(f"\n数据范围:")
    print(f"  PyTorch数据: Min={x.min():.6f}, Max={x.max():.6f}, Mean={x.mean():.6f}")
    print(f"  C++数据:     Min={y.min():.6f}, Max={y.max():.6f}, Mean={y.mean():.6f}")
    
    # 打印前几个元素进行对比
    print(f"\n前10个元素对比:")
    x_flat = x.flatten()
    y_flat = y.flatten()
    for i in range(min(10, len(x_flat))):
        print(f"  [{i:3d}]: PyTorch={x_flat[i]:10.6f}, C++={y_flat[i]:10.6f}, Diff={abs(x_flat[i]-y_flat[i]):10.6f}")

def main():
    """主函数"""
    logger, _, _ = logger_wrapper("", False)
    logger.setLevel(logging.INFO)
    
    # 文件路径配置
    pytorch_dir = "script/tutorial/asset"
    cpp_dir = "C++/Output/val_bin"
    
    # 要比较的数据文件
    data_files = [
        {
            "name": "ibank_cached_anchor",
            "pytorch_path": f"{pytorch_dir}/sample_0_ibank_cached_anchor_1*600*11_float32.bin",
            "cpp_path": f"{cpp_dir}/sample_0_ibank_cached_anchor_1*600*11_float32.bin",
            "shape": [1, 600, 11],
            "dtype": np.float32
        },
        {
            "name": "ibank_cached_feature", 
            "pytorch_path": f"{pytorch_dir}/sample_0_ibank_cached_feature_1*600*256_float32.bin",
            "cpp_path": f"{cpp_dir}/sample_0_ibank_cached_feature_1*600*256_float32.bin",
            "shape": [1, 600, 256],
            "dtype": np.float32
        },
        {
            "name": "ibank_confidence",
            "pytorch_path": f"{pytorch_dir}/sample_0_ibank_confidence_1*600_float32.bin", 
            "cpp_path": f"{cpp_dir}/sample_0_ibank_confidence_1*600_float32.bin",
            "shape": [1, 600],
            "dtype": np.float32
        },
        {
            "name": "ibank_updated_temp_track_id",
            "pytorch_path": f"{pytorch_dir}/sample_0_ibank_updated_temp_track_id_1*900_int32.bin",
            "cpp_path": f"{cpp_dir}/sample_0_ibank_updated_temp_track_id_1*900_int32.bin", 
            "shape": [1, 900],
            "dtype": np.int32
        }
    ]
    
    print("InstanceBank数据一致性验证")
    print("="*80)
    
    for data_file in data_files:
        name = data_file["name"]
        pytorch_path = data_file["pytorch_path"]
        cpp_path = data_file["cpp_path"]
        shape = data_file["shape"]
        dtype = data_file["dtype"]
        
        print(f"\n正在比较: {name}")
        print(f"PyTorch文件: {pytorch_path}")
        print(f"C++文件:     {cpp_path}")
        
        # 读取数据
        pytorch_data = read_bin_file(pytorch_path, shape, dtype)
        cpp_data = read_bin_file(cpp_path, shape, dtype)
        
        if pytorch_data is not None and cpp_data is not None:
            # 计算指标
            calculate_metrics(pytorch_data, cpp_data, name)
        else:
            print(f"[ERROR] Failed to read data for {name}")
    
    print(f"\n{'='*80}")
    print("InstanceBank数据一致性验证完成")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 