#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C++与Python输出结果对比脚本
对比SparseBEV8.6的C++实现与Python推理结果的差异
"""

import numpy as np
import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_bin_file(file_path, dtype=np.float32):
    """
    加载二进制文件
    
    Args:
        file_path: 文件路径
        dtype: 数据类型，默认为float32
    
    Returns:
        numpy数组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        data = np.fromfile(file_path, dtype=dtype)
        print(f"[INFO] 成功读取文件: {file_path}")
        print(f"[INFO] 数据形状: {data.shape}")
        print(f"[INFO] 数据类型: {data.dtype}")
        print(f"[INFO] 数据大小: {data.size} 元素")
        print(f"[INFO] 文件大小: {os.path.getsize(file_path)} 字节")
        return data
    except Exception as e:
        raise RuntimeError(f"读取文件失败: {e}")

def calculate_statistics(data, name):
    """计算数据统计信息"""
    print(f"\n[INFO] {name} 统计信息:")
    print(f"  最小值: {data.min():.6f}")
    print(f"  最大值: {data.max():.6f}")
    print(f"  平均值: {data.mean():.6f}")
    print(f"  标准差: {data.std():.6f}")
    print(f"  总和: {data.sum():.6f}")

def get_max_error(data1, data2):
    """计算最大误差"""
    if data1.shape != data2.shape:
        raise ValueError(f"数据形状不匹配: {data1.shape} vs {data2.shape}")
    
    max_error = np.max(np.abs(data1 - data2))
    return max_error

def get_error_percentage(data1, data2, threshold):
    """计算超过阈值的误差百分比"""
    if data1.shape != data2.shape:
        raise ValueError(f"数据形状不匹配: {data1.shape} vs {data2.shape}")
    
    abs_diff = np.abs(data1 - data2)
    error_count = np.sum(abs_diff > threshold)
    total_count = data1.size
    error_percentage = (error_count / total_count) * 100
    
    print(f"  误差 > {threshold}: {error_count}/{total_count} ({error_percentage:.2f}%)")
    return error_percentage

def reshape_data(data, shape_str):
    """根据形状字符串重塑数据"""
    if shape_str is None:
        return data
    
    # 解析形状字符串，如 "1*6*3*256*704"
    shape = [int(dim) for dim in shape_str.split('*')]
    
    if np.prod(shape) != data.size:
        raise ValueError(f"形状不匹配: 期望 {np.prod(shape)} 元素，实际 {data.size} 元素")
    
    return data.reshape(shape)

def compare_bin_files(cpp_file, python_file, shape_str=None, threshold=0.01, name="数据", dtype=np.float32):
    """
    对比两个bin文件
    
    Args:
        cpp_file: C++输出文件路径
        python_file: Python参考文件路径
        shape_str: 形状字符串，如"1*6*3*256*704"
        threshold: 误差阈值
        name: 数据名称
        dtype: 数据类型
    """
    print("=" * 80)
    print(f"{name} 对比分析")
    print("=" * 80)
    
    # 读取文件
    print(f"\n[INFO] 读取C++输出文件: {cpp_file}")
    cpp_data = load_bin_file(cpp_file, dtype)
    
    print(f"\n[INFO] 读取Python参考文件: {python_file}")
    python_data = load_bin_file(python_file, dtype)
    
    # 计算统计信息
    calculate_statistics(cpp_data, "C++输出")
    calculate_statistics(python_data, "Python参考")
    
    # 重塑数据（如果提供了形状字符串）
    if shape_str:
        cpp_data_reshaped = reshape_data(cpp_data, shape_str)
        python_data_reshaped = reshape_data(python_data, shape_str)
        print(f"\n[INFO] 重塑后形状: {cpp_data_reshaped.shape}")
    else:
        cpp_data_reshaped = cpp_data
        python_data_reshaped = python_data
    
    # 计算误差
    print(f"\n[INFO] 开始计算误差...")
    max_error = get_max_error(cpp_data_reshaped, python_data_reshaped)
    
    # 计算不同阈值的误差百分比
    thresholds = [0.001, 0.01, 0.1, 0.5, 1.0]
    for t in thresholds:
        get_error_percentage(cpp_data_reshaped, python_data_reshaped, t)
    
    # 判断是否通过测试
    print(f"\n[INFO] 测试结果:")
    print(f"  最大误差: {max_error:.6f}")
    print(f"  阈值: {threshold:.6f}")
    
    if max_error <= threshold:
        print(f"  ✅ 测试通过! 最大误差 ({max_error:.6f}) <= 阈值 ({threshold:.6f})")
        return True
    else:
        print(f"  ❌ 测试失败! 最大误差 ({max_error:.6f}) > 阈值 ({threshold:.6f})")
        return False

def main():
    parser = argparse.ArgumentParser(description='对比C++与Python的输出结果')
    parser.add_argument('--cpp_dir', type=str, 
                       default='/share/Code/SparseEnd2End/C++/Output/val_bin/',
                       help='C++输出目录')
    parser.add_argument('--python_dir', type=str,
                       default='/share/Code/SparseEnd2End/script/tutorial/asset/',
                       help='Python参考数据目录')
    parser.add_argument('--save_dir', type=str,
                       default='/share/Code/SparseEnd2End/C++/script/compare/results',
                       help='结果保存目录')
    parser.add_argument('--tolerance', type=float, default=0.02,
                       help='误差容忍度')
    
    args = parser.parse_args()
    
    print("开始对比C++与Python输出结果...")
    print(f"C++输出目录: {args.cpp_dir}")
    print(f"Python参考目录: {args.python_dir}")
    print(f"结果保存目录: {args.save_dir}")
    print(f"误差容忍度: {args.tolerance}")
    print("-" * 50)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 定义要对比的文件
    comparison_files = [
        # # 预处理阶段
        # {
        #     "name": "原始图像",
        #     "cpp_file": "sample_0_ori_imgs_1*6*3*900*1600_uint8.bin",
        #     "python_file": "sample_0_ori_imgs_1*6*3*900*1600_uint8.bin",
        #     "shape_str": "1*6*3*900*1600",
        #     "threshold": 0.0,
        #     "dtype": np.uint8
        # },
        {
            "name": "预处理图像",
            "cpp_file": "sample_0_imgs_1*6*3*256*704_float32.bin",
            "python_file": "sample_0_imgs_1*6*3*256*704_float32.bin",
            "shape_str": "1*6*3*256*704",
            "threshold": 0.02
        },
        {
            "name": "特征提取结果",
            "cpp_file": "sample_0_feature_1*89760*256_float32.bin",
            "python_file": "sample_0_feature_1*89760*256_float32.bin",
            "shape_str": "1*89760*256",
            "threshold": 0.01
        },
        
        # 输入变量
        {
            "name": "时间间隔",
            "cpp_file": "sample_0_time_interval_1_float32.bin",
            "python_file": "sample_0_time_interval_1_float32.bin",
            "shape_str": "1",
            "threshold": 0.001
        },
        {
            "name": "图像宽高",
            "cpp_file": "sample_0_image_wh_1*6*2_float32.bin",
            "python_file": "sample_0_image_wh_1*6*2_float32.bin",
            "shape_str": "1*6*2",
            "threshold": 0.001
        },
        {
            "name": "Lidar2img变换矩阵",
            "cpp_file": "sample_0_lidar2img_1*6*4*4_float32.bin",
            "python_file": "sample_0_lidar2img_1*6*4*4_float32.bin",
            "shape_str": "1*6*4*4",
            "threshold": 0.001
        },
        {
            "name": "锚点数据",
            "cpp_file": "sample_0_anchor_1*900*11_float32.bin",
            "python_file": "sample_0_anchor_1*900*11_float32.bin",
            "shape_str": "1*900*11",
            "threshold": 0.001
        },
        {
            "name": "空间形状",
            "cpp_file": "sample_0_spatial_shapes_6*4*2_int32.bin",
            "python_file": "sample_0_spatial_shapes_6*4*2_int32.bin",
            "shape_str": "6*4*2",
            "threshold": 0.0,
            "dtype": np.int32
        },
        {
            "name": "层级起始索引",
            "cpp_file": "sample_0_level_start_index_6*4_int32.bin",
            "python_file": "sample_0_level_start_index_6*4_int32.bin",
            "shape_str": "6*4",
            "threshold": 0.0,
            "dtype": np.int32
        },
        {
            "name": "实例特征",
            "cpp_file": "sample_0_instance_feature_1*900*256_float32.bin",
            "python_file": "sample_0_instance_feature_1*900*256_float32.bin",
            "shape_str": "1*900*256",
            "threshold": 0.001
        },
        
        # 推理结果
        {
            "name": "预测实例特征",
            "cpp_file": "sample_0_pred_instance_feature_1*900*256_float32.bin",
            "python_file": "sample_0_pred_instance_feature_1*900*256_float32.bin",
            "shape_str": "1*900*256",
            "threshold": 0.001
        },
        {
            "name": "预测锚点",
            "cpp_file": "sample_0_pred_anchor_1*900*11_float32.bin",
            "python_file": "sample_0_pred_anchor_1*900*11_float32.bin",
            "shape_str": "1*900*11",
            "threshold": 0.001
        },
        {
            "name": "预测分类得分",
            "cpp_file": "sample_0_pred_class_score_1*900*10_float32.bin",
            "python_file": "sample_0_pred_class_score_1*900*10_float32.bin",
            "shape_str": "1*900*10",
            "threshold": 0.001
        },
        {
            "name": "预测质量得分",
            "cpp_file": "sample_0_pred_quality_score_1*900*2_float32.bin",
            "python_file": "sample_0_pred_quality_score_1*900*2_float32.bin",
            "shape_str": "1*900*2",
            "threshold": 0.001
        },
        {
            "name": "预测跟踪ID",
            "cpp_file": "sample_0_pred_track_id_1*900_int32.bin",
            "python_file": "sample_0_pred_track_id_1*900_int32.bin",
            "shape_str": "1*900",
            "threshold": 0.0,
            "dtype": np.int32
        },
        
        # 时序数据（第二帧）
        {
            "name": "临时实例特征",
            "cpp_file": "sample_1_temp_instance_feature_1*600*256_float32.bin",
            "python_file": "sample_1_temp_instance_feature_1*600*256_float32.bin",
            "shape_str": "1*600*256",
            "threshold": 0.001
        },
        {
            "name": "临时锚点",
            "cpp_file": "sample_1_temp_anchor_1*600*11_float32.bin",
            "python_file": "sample_1_temp_anchor_1*600*11_float32.bin",
            "shape_str": "1*600*11",
            "threshold": 0.001
        },
        {
            "name": "掩码",
            "cpp_file": "sample_1_mask_1_int32.bin",
            "python_file": "sample_1_mask_1_int32.bin",
            "shape_str": "1",
            "threshold": 0.0,
            "dtype": np.int32
        },
        {
            "name": "跟踪ID",
            "cpp_file": "sample_1_track_id_1*900_int32.bin",
            "python_file": "sample_1_track_id_1*900_int32.bin",
            "shape_str": "1*900",
            "threshold": 0.0,
            "dtype": np.int32
        }
    ]
    
    # 执行对比
    results = []
    for comp in comparison_files:
        cpp_path = os.path.join(args.cpp_dir, comp["cpp_file"])
        python_path = os.path.join(args.python_dir, comp["python_file"])
        
        print(f"\n{'='*20} 对比 {comp['name']} {'='*20}")
        
        try:
            dtype = comp.get("dtype", np.float32)
            success = compare_bin_files(
                cpp_path, 
                python_path, 
                comp["shape_str"], 
                comp["threshold"], 
                comp["name"],
                dtype
            )
            results.append({
                "name": comp["name"],
                "success": success,
                "cpp_file": comp["cpp_file"],
                "python_file": comp["python_file"]
            })
        except Exception as e:
            print(f"  ❌ 对比失败: {e}")
            results.append({
                "name": comp["name"],
                "success": False,
                "error": str(e),
                "cpp_file": comp["cpp_file"],
                "python_file": comp["python_file"]
            })
    
    # 生成总结报告
    print(f"\n{'='*50}")
    print("对比总结报告")
    print(f"{'='*50}")
    
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    
    print(f"总测试项目: {total_count}")
    print(f"通过测试: {success_count}")
    print(f"失败测试: {total_count - success_count}")
    print(f"通过率: {(success_count/total_count)*100:.1f}%")
    
    print(f"\n详细结果:")
    for result in results:
        status = "✅ 通过" if result["success"] else "❌ 失败"
        print(f"  {result['name']}: {status}")
        if not result["success"] and "error" in result:
            print(f"    错误: {result['error']}")
    
    # 保存报告
    report_file = os.path.join(args.save_dir, "comparison_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("C++与Python输出结果对比报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"总测试项目: {total_count}\n")
        f.write(f"通过测试: {success_count}\n")
        f.write(f"失败测试: {total_count - success_count}\n")
        f.write(f"通过率: {(success_count/total_count)*100:.1f}%\n\n")
        
        f.write("详细结果:\n")
        for result in results:
            status = "通过" if result["success"] else "失败"
            f.write(f"  {result['name']}: {status}\n")
            if not result["success"] and "error" in result:
                f.write(f"    错误: {result['error']}\n")
    
    print(f"\n报告已保存到: {report_file}")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main()) 