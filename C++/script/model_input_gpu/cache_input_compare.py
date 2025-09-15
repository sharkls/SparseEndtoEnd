#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较GPU版本与CPU版本 cache_input 数据的差异：
- instance_feature (1*900*256 float32)
- anchor           (1*900*11  float32)
- confidence_logits(1*900*10  float32)

参照 first_cached_data_compare.py 的误差分析形式：
- 输出整体误差指标（最大/平均绝对误差，最大/平均相对误差，不同元素比例等）
- 对实例/维度进行Top-N的误差排名
- 将详细结果保存到文本文件
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, Any


def load_bin_file(file_path: str, shape: Tuple[int, ...], dtype=np.float32) -> Optional[np.ndarray]:
    """加载二进制文件并按形状reshape。"""
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return None
    try:
        data = np.fromfile(file_path, dtype=dtype)
        if data.size != int(np.prod(shape)):
            print(f"错误: 文件大小不匹配. 期望: {int(np.prod(shape))}, 实际: {data.size}. 文件: {file_path}")
            return None
        data = data.reshape(shape)
        print(f"成功加载: {file_path}")
        print(f"  形状: {data.shape}, dtype: {data.dtype}, 数值范围: [{data.min():.6f}, {data.max():.6f}]")
        return data
    except Exception as e:
        print(f"错误: 加载失败: {file_path}, {e}")
        return None


def compare_data(data1: np.ndarray, data2: np.ndarray, name: str, tolerance: float = 1e-6) -> Optional[Dict[str, Any]]:
    """逐元素比较两个同形状数组，返回误差统计。"""
    if data1 is None or data2 is None:
        return None
    if data1.shape != data2.shape:
        print(f"错误: {name} 形状不一致: {data1.shape} vs {data2.shape}")
        return None

    print(f"\n=== {name} 数据比较 ===")
    abs_diff = np.abs(data1 - data2)

    relative_diff = np.zeros_like(abs_diff)
    denom = np.abs(data2)
    mask = denom > tolerance
    relative_diff[mask] = abs_diff[mask] / denom[mask]

    stats = {
        "max_abs_error": float(np.max(abs_diff)),
        "mean_abs_error": float(np.mean(abs_diff)),
        "std_abs_error": float(np.std(abs_diff)),
        "max_relative_error": float(np.max(relative_diff)),
        "mean_relative_error": float(np.mean(relative_diff)),
        "std_relative_error": float(np.std(relative_diff)),
        "num_different_elements": int(np.sum(abs_diff > tolerance)),
        "total_elements": int(abs_diff.size),
        "percentage_different": float(np.sum(abs_diff > tolerance) / abs_diff.size * 100.0),
    }

    print(f"最大绝对误差: {stats['max_abs_error']:.6f}")
    print(f"平均绝对误差: {stats['mean_abs_error']:.6f}")
    print(f"最大相对误差: {stats['max_relative_error']:.6f}")
    print(f"平均相对误差: {stats['mean_relative_error']:.6f}")
    print(f"不同元素数量/比例: {stats['num_different_elements']} / {stats['percentage_different']:.2f}%")

    if stats["max_abs_error"] < tolerance:
        print(f"✓ {name} 完全一致 (max_abs_error < {tolerance})")
    else:
        print(f"✗ {name} 存在差异 (max_abs_error >= {tolerance})")

    return stats


def analyze_instance_feature(gpu_root: str, cpu_root: str, sample_idx: int) -> Tuple[Optional[Dict[str, Any]], Optional[list], Optional[list]]:
    print("\n" + "=" * 50)
    print(f"cache_input instance_feature 比较 (sample_{sample_idx})")
    print("=" * 50)

    gpu_file = os.path.join(gpu_root, f"sample_{sample_idx}_cache_input_instance_feature_1*900*256_float32.bin")
    cpu_file = os.path.join(cpu_root, f"sample_{sample_idx}_cache_input_instance_feature_1*900*256_float32.bin")
    shape = (1, 900, 256)

    data_gpu = load_bin_file(gpu_file, shape, dtype=np.float32)
    data_cpu = load_bin_file(cpu_file, shape, dtype=np.float32)
    stats = compare_data(data_gpu, data_cpu, "cache_input.instance_feature", tolerance=2e-1) if (data_gpu is not None and data_cpu is not None) else None

    instance_errors = []
    feature_errors = []
    if stats is not None:
        for i in range(900):
            diff = np.abs(data_gpu[0, i] - data_cpu[0, i])
            instance_errors.append({
                "instance": i,
                "max_error": float(np.max(diff)),
                "mean_error": float(np.mean(diff)),
            })
        instance_errors.sort(key=lambda x: x["max_error"], reverse=True)

        for d in range(256):
            diff = np.abs(data_gpu[:, :, d] - data_cpu[:, :, d])
            feature_errors.append({
                "feature_dim": d,
                "max_error": float(np.max(diff)),
                "mean_error": float(np.mean(diff)),
            })
        feature_errors.sort(key=lambda x: x["max_error"], reverse=True)

        print("\n前10个最大误差的实例:")
        for item in instance_errors[:10]:
            print(f"  实例 {item['instance']:3d}: 最大误差={item['max_error']:.6f}, 平均误差={item['mean_error']:.6f}")

        print("\n前10个最大误差的特征维度:")
        for item in feature_errors[:10]:
            print(f"  维度 {item['feature_dim']:3d}: 最大误差={item['max_error']:.6f}, 平均误差={item['mean_error']:.6f}")

    return stats, instance_errors if stats else None, feature_errors if stats else None


def analyze_anchor(gpu_root: str, cpu_root: str, sample_idx: int) -> Tuple[Optional[Dict[str, Any]], Optional[list]]:
    print("\n" + "=" * 50)
    print(f"cache_input anchor 比较 (sample_{sample_idx})")
    print("=" * 50)

    gpu_file = os.path.join(gpu_root, f"sample_{sample_idx}_cache_input_anchor_1*900*11_float32.bin")
    cpu_file = os.path.join(cpu_root, f"sample_{sample_idx}_cache_input_anchor_1*900*11_float32.bin")
    shape = (1, 900, 11)

    data_gpu = load_bin_file(gpu_file, shape, dtype=np.float32)
    data_cpu = load_bin_file(cpu_file, shape, dtype=np.float32)
    stats = compare_data(data_gpu, data_cpu, "cache_input.anchor", tolerance=2e-1) if (data_gpu is not None and data_cpu is not None) else None

    anchor_errors = []
    if stats is not None:
        for i in range(900):
            diff = np.abs(data_gpu[0, i] - data_cpu[0, i])
            anchor_errors.append({
                "anchor": i,
                "max_error": float(np.max(diff)),
                "mean_error": float(np.mean(diff)),
            })
        anchor_errors.sort(key=lambda x: x["max_error"], reverse=True)

        print("\n前10个最大误差的锚点:")
        for item in anchor_errors[:10]:
            print(f"  锚点 {item['anchor']:3d}: 最大误差={item['max_error']:.6f}, 平均误差={item['mean_error']:.6f}")

        print("\n各锚点维度最大误差:")
        for d in range(11):
            diff = np.abs(data_gpu[:, :, d] - data_cpu[:, :, d])
            print(f"  维度 {d:2d}: 最大误差={np.max(diff):.6f}, 平均误差={np.mean(diff):.6f}")

    return stats, anchor_errors if stats else None


def analyze_confidence_logits(gpu_root: str, cpu_root: str, sample_idx: int) -> Optional[Dict[str, Any]]:
    print("\n" + "=" * 50)
    print(f"cache_input confidence_logits 比较 (sample_{sample_idx})")
    print("=" * 50)

    gpu_file = os.path.join(gpu_root, f"sample_{sample_idx}_cache_input_confidence_logits_1*900*10_float32.bin")
    cpu_file = os.path.join(cpu_root, f"sample_{sample_idx}_cache_input_confidence_logits_1*900*10_float32.bin")
    shape = (1, 900, 10)

    data_gpu = load_bin_file(gpu_file, shape, dtype=np.float32)
    data_cpu = load_bin_file(cpu_file, shape, dtype=np.float32)
    stats = compare_data(data_gpu, data_cpu, "cache_input.confidence_logits", tolerance=2e-1) if (data_gpu is not None and data_cpu is not None) else None

    if stats is not None:
        # 每个类维度的误差统计Top-10
        class_errors = []
        for c in range(10):
            diff = np.abs(data_gpu[:, :, c] - data_cpu[:, :, c])
            class_errors.append({
                "class_dim": c,
                "max_error": float(np.max(diff)),
                "mean_error": float(np.mean(diff)),
            })
        class_errors.sort(key=lambda x: x["max_error"], reverse=True)
        print("\n前10个最大误差的类别维度:")
        for item in class_errors[:10]:
            print(f"  类别维度 {item['class_dim']:2d}: 最大误差={item['max_error']:.6f}, 平均误差={item['mean_error']:.6f}")

    return stats


def run_for_sample(sample_idx: int) -> Dict[str, Any]:
    gpu_root = "/share/Code/Sparse4d/C++/Output/val_bin_gpu/"
    cpu_root = "/share/Code/Sparse4d/C++/Output/val_bin/"

    all_results: Dict[str, Any] = {}

    feat_stats, feat_inst_errs, feat_dim_errs = analyze_instance_feature(gpu_root, cpu_root, sample_idx)
    if feat_stats:
        all_results["instance_feature"] = feat_stats
        all_results["feature_instance_max_errors"] = feat_inst_errs
        all_results["feature_max_errors"] = feat_dim_errs

    anchor_stats, anchor_errs = analyze_anchor(gpu_root, cpu_root, sample_idx)
    if anchor_stats:
        all_results["anchor"] = anchor_stats
        all_results["anchor_max_errors"] = anchor_errs

    cls_stats = analyze_confidence_logits(gpu_root, cpu_root, sample_idx)
    if cls_stats:
        all_results["confidence_logits"] = cls_stats

    return all_results


def save_results(all_results: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== GPU vs CPU cache_input 数据比较结果 ===\n\n")
        f.write("GPU路径: /share/Code/Sparse4d/C++/Output/val_bin_gpu/\n")
        f.write("CPU路径: /share/Code/Sparse4d/C++/Output/val_bin/\n\n")

        basic_keys = [k for k in all_results.keys() if k not in [
            "feature_instance_max_errors", "feature_max_errors", "anchor_max_errors"
        ]]

        for key in basic_keys:
            f.write(f"=== {key} ===\n")
            for kk, vv in all_results[key].items():
                f.write(f"{kk}: {vv}\n")
            f.write("\n")

        if "feature_instance_max_errors" in all_results and all_results["feature_instance_max_errors"]:
            f.write("=== 前50个最大误差的实例 (instance_feature) ===\n")
            for item in all_results["feature_instance_max_errors"][:50]:
                f.write(f"实例 {item['instance']:3d}: 最大误差={item['max_error']:.6f}, 平均误差={item['mean_error']:.6f}\n")
            f.write("\n")

        if "feature_max_errors" in all_results and all_results["feature_max_errors"]:
            f.write("=== 前50个最大误差的特征维度 (instance_feature) ===\n")
            for item in all_results["feature_max_errors"][:50]:
                f.write(f"特征维度 {item['feature_dim']:3d}: 最大误差={item['max_error']:.6f}, 平均误差={item['mean_error']:.6f}\n")
            f.write("\n")

        if "anchor_max_errors" in all_results and all_results["anchor_max_errors"]:
            f.write("=== 前50个最大误差的锚点 (anchor) ===\n")
            for item in all_results["anchor_max_errors"][:50]:
                f.write(f"锚点 {item['anchor']:3d}: 最大误差={item['max_error']:.6f}, 平均误差={item['mean_error']:.6f}\n")
            f.write("\n")


def summarize(all_results: Dict[str, Any]) -> None:
    print("\n=== 比较总结 ===")
    basic_keys = [k for k in all_results.keys() if k not in [
        "feature_instance_max_errors", "feature_max_errors", "anchor_max_errors"
    ]]
    print(f"比较的数据类型数量: {len(basic_keys)}")

    tolerance = 2e-1
    consistent = 0
    for key in basic_keys:
        stats = all_results[key]
        if stats.get("max_abs_error", 1.0) < tolerance:
            consistent += 1
            print(f"✓ {key}: 完全一致")
        else:
            print(f"✗ {key}: 存在差异 (最大误差: {stats.get('max_abs_error', 0):.6f})")
    print(f"\n完全一致的数据类型: {consistent}/{len(basic_keys)}")


if __name__ == "__main__":
    print("=== GPU vs CPU cache_input 数据比较脚本 ===")
    print("GPU路径: /share/Code/Sparse4d/C++/Output/val_bin_gpu/")
    print("CPU路径: /share/Code/Sparse4d/C++/Output/val_bin/")

    results_0 = run_for_sample(0)
    output_0 = "gpu_vs_cpu_first_frame_cache_input_comparison_results.txt"
    if results_0:
        save_results(results_0, output_0)
        summarize(results_0)
        print(f"\n第一帧结果已保存: {output_0}")

    # # 如果第二帧也已导出，可一并比较
    # results_1 = run_for_sample(1)
    # if results_1:
    #     output_1 = "gpu_vs_cpu_second_frame_cache_input_comparison_results.txt"
    #     save_results(results_1, output_1)
    #     summarize(results_1)
    #     print(f"\n第二帧结果已保存: {output_1}")
