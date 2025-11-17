#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 FP32 和 FP16 ONNX 模型的推理结果差异
使用方法: python compare_onnx_fp32_fp16.py [--input-path PATH] [--output-dir DIR]
"""

import os
import sys
import argparse
import numpy as np
import onnxruntime as ort
from typing import Dict, Tuple, List


def load_input_data(input_path: str = None) -> np.ndarray:
    """
    加载输入数据
    
    Args:
        input_path: 输入数据文件路径（可选）
    
    Returns:
        input_img: 图像数据 (1, 6, 3, 256, 704)
    """
    if input_path and os.path.exists(input_path):
        print(f"从文件加载输入数据: {input_path}")
        input_img = np.fromfile(input_path, dtype=np.float32).reshape(1, 6, 3, 256, 704)
    else:
        print("生成随机输入数据...")
        # 生成归一化后的随机数据（模拟预处理后的图像）
        np.random.seed(42)  # 固定随机种子，确保可重复
        input_img = np.random.randn(1, 6, 3, 256, 704).astype(np.float32)
        # 归一化到 [0, 1] 范围（模拟预处理）
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
    
    print(f"输入数据形状: {input_img.shape}")
    print(f"输入数据范围: [{input_img.min():.6f}, {input_img.max():.6f}]")
    print(f"输入数据类型: {input_img.dtype}")
    
    return input_img


def run_onnx_inference(onnx_path: str, input_data: np.ndarray, use_fp16: bool = False) -> Dict[str, np.ndarray]:
    """
    运行 ONNX 模型推理
    
    Args:
        onnx_path: ONNX 模型路径
        input_data: 输入数据
        use_fp16: 是否使用 FP16 精度（输入数据会转换为 FP16）
    
    Returns:
        outputs: 输出字典 {output_name: output_array}
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")
    
    print(f"\n{'='*80}")
    print(f"加载 ONNX 模型: {onnx_path}")
    print(f"{'='*80}")
    
    # 创建推理会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # 获取输入输出信息
    input_info = session.get_inputs()[0]
    output_infos = session.get_outputs()
    
    print(f"输入名称: {input_info.name}")
    print(f"输入形状: {input_info.shape}")
    print(f"输入类型: {input_info.type}")
    
    print(f"\n输出数量: {len(output_infos)}")
    for i, output_info in enumerate(output_infos):
        print(f"  输出 {i}: {output_info.name}, 形状: {output_info.shape}, 类型: {output_info.type}")
    
    # 准备输入数据
    if use_fp16:
        # 如果模型是 FP16，将输入转换为 FP16
        input_data_fp16 = input_data.astype(np.float16)
        input_feed = {input_info.name: input_data_fp16}
        print(f"\n使用 FP16 输入数据")
        print(f"  输入数据范围: [{input_data_fp16.min():.6f}, {input_data_fp16.max():.6f}]")
    else:
        input_feed = {input_info.name: input_data}
        print(f"\n使用 FP32 输入数据")
    
    # 执行推理
    print("\n执行推理...")
    output_names = [output.name for output in output_infos]
    outputs = session.run(output_names, input_feed)
    
    # 转换为字典
    output_dict = {name: output for name, output in zip(output_names, outputs)}
    
    # 打印输出信息
    print("\n推理结果:")
    for name, output in output_dict.items():
        print(f"  {name}:")
        print(f"    形状: {output.shape}")
        print(f"    类型: {output.dtype}")
        print(f"    范围: [{output.min():.6f}, {output.max():.6f}]")
        print(f"    均值: {output.mean():.6f}")
        print(f"    标准差: {output.std():.6f}")
    
    return output_dict


def compare_outputs(output_fp32: Dict[str, np.ndarray], 
                   output_fp16: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """
    对比 FP32 和 FP16 的输出结果
    
    Args:
        output_fp32: FP32 模型的输出
        output_fp16: FP16 模型的输出
    
    Returns:
        comparison: 对比结果字典
    """
    print(f"\n{'='*80}")
    print("对比 FP32 和 FP16 输出结果")
    print(f"{'='*80}")
    
    comparison = {}
    
    # 确保输出名称一致
    common_outputs = set(output_fp32.keys()) & set(output_fp16.keys())
    if not common_outputs:
        raise ValueError("FP32 和 FP16 模型的输出名称不一致！")
    
    for output_name in common_outputs:
        fp32_output = output_fp32[output_name]
        fp16_output = output_fp16[output_name]
        
        # 转换为 FP32 进行对比（FP16 输出需要转换）
        if fp16_output.dtype == np.float16:
            fp16_output_fp32 = fp16_output.astype(np.float32)
        else:
            fp16_output_fp32 = fp16_output
        
        # 确保形状一致
        if fp32_output.shape != fp16_output.shape:
            print(f"⚠️  警告: {output_name} 的形状不一致!")
            print(f"    FP32: {fp32_output.shape}")
            print(f"    FP16: {fp16_output.shape}")
            continue
        
        # 计算差异
        diff = fp32_output - fp16_output_fp32
        abs_diff = np.abs(diff)
        
        # 计算相对误差（避免除零）
        relative_diff = np.abs(diff) / (np.abs(fp32_output) + 1e-8)
        
        # 统计信息
        stats = {
            'max_abs_diff': float(np.max(abs_diff)),
            'mean_abs_diff': float(np.mean(abs_diff)),
            'std_abs_diff': float(np.std(abs_diff)),
            'max_relative_diff': float(np.max(relative_diff)),
            'mean_relative_diff': float(np.mean(relative_diff)),
            'cosine_similarity': float(np.dot(fp32_output.flatten(), fp16_output_fp32.flatten()) / 
                                      (np.linalg.norm(fp32_output.flatten()) * np.linalg.norm(fp16_output_fp32.flatten()) + 1e-8)),
            'fp32_range': [float(np.min(fp32_output)), float(np.max(fp32_output))],
            'fp16_range': [float(np.min(fp16_output_fp32)), float(np.max(fp16_output_fp32))],
        }
        
        comparison[output_name] = stats
        
        # 打印对比结果
        print(f"\n输出: {output_name}")
        print(f"  最大绝对差异: {stats['max_abs_diff']:.6e}")
        print(f"  平均绝对差异: {stats['mean_abs_diff']:.6e}")
        print(f"  标准差: {stats['std_abs_diff']:.6e}")
        print(f"  最大相对差异: {stats['max_relative_diff']:.6e}")
        print(f"  平均相对差异: {stats['mean_relative_diff']:.6e}")
        print(f"  余弦相似度: {stats['cosine_similarity']:.8f}")
        print(f"  FP32 范围: [{stats['fp32_range'][0]:.6f}, {stats['fp32_range'][1]:.6f}]")
        print(f"  FP16 范围: [{stats['fp16_range'][0]:.6f}, {stats['fp16_range'][1]:.6f}]")
        
        # 计算差异分布
        diff_percentiles = np.percentile(abs_diff, [50, 75, 90, 95, 99])
        print(f"  差异百分位数:")
        print(f"    50%: {diff_percentiles[0]:.6e}")
        print(f"    75%: {diff_percentiles[1]:.6e}")
        print(f"    90%: {diff_percentiles[2]:.6e}")
        print(f"    95%: {diff_percentiles[3]:.6e}")
        print(f"    99%: {diff_percentiles[4]:.6e}")
    
    return comparison


def save_comparison_results(comparison: Dict, output_dir: str = None):
    """
    保存对比结果到文件
    
    Args:
        comparison: 对比结果字典
        output_dir: 输出目录（可选）
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "comparison_results.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("FP32 vs FP16 ONNX 模型推理结果对比\n")
            f.write("="*80 + "\n\n")
            
            for output_name, stats in comparison.items():
                f.write(f"输出: {output_name}\n")
                f.write(f"  最大绝对差异: {stats['max_abs_diff']:.6e}\n")
                f.write(f"  平均绝对差异: {stats['mean_abs_diff']:.6e}\n")
                f.write(f"  标准差: {stats['std_abs_diff']:.6e}\n")
                f.write(f"  最大相对差异: {stats['max_relative_diff']:.6e}\n")
                f.write(f"  平均相对差异: {stats['mean_relative_diff']:.6e}\n")
                f.write(f"  余弦相似度: {stats['cosine_similarity']:.8f}\n")
                f.write(f"  FP32 范围: [{stats['fp32_range'][0]:.6f}, {stats['fp32_range'][1]:.6f}]\n")
                f.write(f"  FP16 范围: [{stats['fp16_range'][0]:.6f}, {stats['fp16_range'][1]:.6f}]\n")
                f.write("\n")
        
        print(f"\n对比结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="对比 FP32 和 FP16 ONNX 模型的推理结果")
    parser.add_argument(
        "--fp32-onnx",
        type=str,
        default="/share/Code/Sparse4dE2E/deploy/onnx/onnx_backup/sparse4dbackbone.onnx",
        help="FP32 ONNX 模型路径"
    )
    parser.add_argument(
        "--fp16-onnx",
        type=str,
        default="/share/Code/Sparse4dE2E/deploy/onnx/sparse4dbackbone.onnx",
        help="FP16 ONNX 模型路径"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="输入数据文件路径（可选，如果不提供则生成随机数据）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（可选，保存对比结果）"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("FP32 vs FP16 ONNX 模型推理结果对比工具")
    print("="*80)
    
    # 加载输入数据
    input_data = load_input_data(args.input_path)
    
    # 运行 FP32 模型推理
    print(f"\n{'='*80}")
    print("FP32 模型推理")
    print(f"{'='*80}")
    output_fp32 = run_onnx_inference(args.fp32_onnx, input_data, use_fp16=False)
    
    # 运行 FP16 模型推理
    print(f"\n{'='*80}")
    print("FP16 模型推理")
    print(f"{'='*80}")
    output_fp16 = run_onnx_inference(args.fp16_onnx, input_data, use_fp16=True)
    
    # 对比结果
    comparison = compare_outputs(output_fp32, output_fp16)
    
    # 保存结果
    if args.output_dir:
        save_comparison_results(comparison, args.output_dir)
    
    # 总结
    print(f"\n{'='*80}")
    print("对比总结")
    print(f"{'='*80}")
    
    for output_name, stats in comparison.items():
        print(f"\n{output_name}:")
        if stats['cosine_similarity'] > 0.99:
            print(f"  ✅ 余弦相似度很高 ({stats['cosine_similarity']:.8f})，结果基本一致")
        elif stats['cosine_similarity'] > 0.95:
            print(f"  ⚠️  余弦相似度较高 ({stats['cosine_similarity']:.8f})，存在一定差异")
        else:
            print(f"  ❌ 余弦相似度较低 ({stats['cosine_similarity']:.8f})，差异较大")
        
        if stats['max_relative_diff'] < 0.01:
            print(f"  ✅ 最大相对差异很小 ({stats['max_relative_diff']:.6e})")
        elif stats['max_relative_diff'] < 0.1:
            print(f"  ⚠️  最大相对差异中等 ({stats['max_relative_diff']:.6e})")
        else:
            print(f"  ❌ 最大相对差异较大 ({stats['max_relative_diff']:.6e})")
    
    print("\n对比完成！")


if __name__ == "__main__":
    main()