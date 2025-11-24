#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试 NaN 产生位置的工具

功能：
1. 分析失败案例的输入数据特征
2. 检查输入数据中是否存在 NaN/Inf
3. 统计失败案例的分布
4. 生成详细的调试报告
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# 导入验证脚本
sys.path.insert(0, str(Path(__file__).parent))
import validate_sparsebox_plugin

def analyze_input_data(anchor, instance, attrs, sample_idx, node_idx):
    """分析输入数据的特征"""
    stats = {
        "sample": sample_idx,
        "node": node_idx,
        "anchor_stats": {},
        "instance_stats": {},
        "has_nan": False,
        "has_inf": False,
    }
    
    # 分析 anchor 数据
    if anchor is not None:
        stats["anchor_stats"] = {
            "shape": anchor.shape,
            "min": float(np.min(anchor)),
            "max": float(np.max(anchor)),
            "mean": float(np.mean(anchor)),
            "std": float(np.std(anchor)),
            "nan_count": int(np.isnan(anchor).sum()),
            "inf_count": int(np.isinf(anchor).sum()),
            "finite_count": int(np.isfinite(anchor).sum()),
        }
        
        # 检查每个维度的统计
        if anchor.shape[-1] == 11:  # anchor 格式: [batch, num_anchor, 11]
            for dim in range(11):
                dim_data = anchor[..., dim]
                stats["anchor_stats"][f"dim_{dim}"] = {
                    "min": float(np.min(dim_data)),
                    "max": float(np.max(dim_data)),
                    "mean": float(np.mean(dim_data)),
                    "nan_count": int(np.isnan(dim_data).sum()),
                    "inf_count": int(np.isinf(dim_data).sum()),
                }
    
    # 分析 instance 数据
    if instance is not None:
        stats["instance_stats"] = {
            "shape": instance.shape,
            "min": float(np.min(instance)),
            "max": float(np.max(instance)),
            "mean": float(np.mean(instance)),
            "std": float(np.std(instance)),
            "nan_count": int(np.isnan(instance).sum()),
            "inf_count": int(np.isinf(instance).sum()),
            "finite_count": int(np.isfinite(instance).sum()),
        }
    
    # 检查是否有 NaN/Inf
    stats["has_nan"] = (
        stats["anchor_stats"].get("nan_count", 0) > 0 or
        stats["instance_stats"].get("nan_count", 0) > 0
    )
    stats["has_inf"] = (
        stats["anchor_stats"].get("inf_count", 0) > 0 or
        stats["instance_stats"].get("inf_count", 0) > 0
    )
    
    return stats

def analyze_failed_cases(onnx_path, plugin_so, asset_dir, fp16=True):
    """分析所有失败案例"""
    onnx_path = Path(onnx_path)
    plugin_so = Path(plugin_so)
    asset_dir = Path(asset_dir)
    
    # 获取所有节点
    nodes = validate_sparsebox_plugin.list_sparsebox_nodes(onnx_path)
    num_nodes = len(nodes)
    
    # 查找所有样本
    anchor_files = list(asset_dir.glob('sample_*_anchor_*.bin'))
    sample_indices = sorted(set([int(f.name.split('_')[1]) for f in anchor_files]))
    
    print(f"找到 {len(sample_indices)} 个样本")
    print(f"找到 {num_nodes} 个 SparseBox 节点\n")
    
    failed_cases = []
    all_stats = []
    
    # 测试所有样本和节点
    for sample_idx in sample_indices:
        for node_idx in range(num_nodes):
            try:
                node = nodes[node_idx]
                attrs = validate_sparsebox_plugin.parse_plugin_attrs(node)
                
                # 加载样本
                anchor, instance = validate_sparsebox_plugin.load_sample(asset_dir, sample_idx)
                
                # 构建 engine
                input_shapes = {"anchor": tuple(anchor.shape)}
                if int(attrs["num_learnable_pts"][0]) > 0:
                    input_shapes["instance_feature"] = tuple(instance.shape)
                
                engine = validate_sparsebox_plugin.build_trt_engine(
                    attrs, input_shapes, plugin_so, fp16
                )
                
                # 运行验证
                trt_inputs = {
                    "anchor": anchor.astype(np.float16 if fp16 else np.float32)
                }
                if int(attrs["num_learnable_pts"][0]) > 0:
                    trt_inputs["instance_feature"] = instance.astype(
                        np.float16 if fp16 else np.float32
                    )
                
                trt_output = validate_sparsebox_plugin.run_engine(engine, trt_inputs).astype(np.float32)
                ref_output = validate_sparsebox_plugin.run_pytorch_reference(
                    anchor, instance, attrs, "fp32"
                )
                
                metrics = validate_sparsebox_plugin.compare_outputs(ref_output, trt_output)
                
                # 检查是否失败
                max_diff = metrics["max_abs_diff"]
                is_valid = not (np.isnan(max_diff) or np.isinf(max_diff))
                is_success = is_valid and max_diff < 2.0
                
                # 分析输入数据
                stats = analyze_input_data(anchor, instance, attrs, sample_idx, node_idx)
                stats["metrics"] = metrics
                stats["success"] = is_success
                all_stats.append(stats)
                
                if not is_success:
                    failed_cases.append({
                        "sample": sample_idx,
                        "node": node_idx,
                        "stats": stats,
                        "metrics": metrics,
                    })
                    
            except Exception as e:
                print(f"  样本 {sample_idx}, Node {node_idx}: 错误 - {e}")
                failed_cases.append({
                    "sample": sample_idx,
                    "node": node_idx,
                    "error": str(e)
                })
    
    return failed_cases, all_stats

def print_analysis_report(failed_cases, all_stats):
    """打印分析报告"""
    print("\n" + "="*80)
    print("失败案例详细分析")
    print("="*80)
    
    if not failed_cases:
        print("✅ 没有失败案例！")
        return
    
    print(f"\n总失败案例数: {len(failed_cases)}")
    
    # 按节点统计
    node_failures = {}
    for case in failed_cases:
        if "error" not in case:
            node = case["node"]
            if node not in node_failures:
                node_failures[node] = []
            node_failures[node].append(case)
    
    print(f"\n按节点统计失败次数:")
    for node in sorted(node_failures.keys()):
        print(f"  Node {node}: {len(node_failures[node])} 次失败")
    
    # 分析输入数据特征
    print(f"\n失败案例输入数据特征:")
    print("-" * 80)
    
    for case in failed_cases[:10]:  # 只显示前10个
        if "error" in case:
            continue
            
        stats = case["stats"]
        print(f"\n样本 {case['sample']}, Node {case['node']}:")
        print(f"  输入数据检查:")
        print(f"    Anchor NaN: {stats['anchor_stats'].get('nan_count', 0)}")
        print(f"    Anchor Inf: {stats['anchor_stats'].get('inf_count', 0)}")
        print(f"    Instance NaN: {stats['instance_stats'].get('nan_count', 0)}")
        print(f"    Instance Inf: {stats['instance_stats'].get('inf_count', 0)}")
        
        if "anchor_stats" in stats and "dim_0" in stats["anchor_stats"]:
            print(f"  Anchor 维度统计:")
            for dim in range(min(11, 8)):  # 只显示前8个维度
                dim_key = f"dim_{dim}"
                if dim_key in stats["anchor_stats"]:
                    dim_stats = stats["anchor_stats"][dim_key]
                    print(f"    Dim {dim}: min={dim_stats['min']:.6f}, max={dim_stats['max']:.6f}, "
                          f"mean={dim_stats['mean']:.6f}, NaN={dim_stats['nan_count']}, Inf={dim_stats['inf_count']}")
        
        print(f"  输出误差: max={case['metrics']['max_abs_diff']:.6e}")
    
    # 统计共同特征
    print(f"\n共同特征分析:")
    print("-" * 80)
    
    # 检查是否有输入 NaN/Inf
    has_input_nan = sum(1 for case in failed_cases 
                        if "stats" in case and case["stats"].get("has_nan", False))
    has_input_inf = sum(1 for case in failed_cases 
                        if "stats" in case and case["stats"].get("has_inf", False))
    
    print(f"  输入包含 NaN 的案例: {has_input_nan}/{len(failed_cases)}")
    print(f"  输入包含 Inf 的案例: {has_input_inf}/{len(failed_cases)}")
    
    # 分析 anchor 范围
    if failed_cases:
        anchor_mins = []
        anchor_maxs = []
        for case in failed_cases:
            if "stats" in case and "anchor_stats" in case["stats"]:
                anchor_stats = case["stats"]["anchor_stats"]
                if "min" in anchor_stats:
                    anchor_mins.append(anchor_stats["min"])
                if "max" in anchor_stats:
                    anchor_maxs.append(anchor_stats["max"])
        
        if anchor_mins and anchor_maxs:
            print(f"  Anchor 范围: min={min(anchor_mins):.6f}, max={max(anchor_maxs):.6f}")

def main():
    parser = argparse.ArgumentParser(description="调试 NaN 产生位置")
    parser.add_argument("--onnx", required=True, help="Head ONNX path")
    parser.add_argument("--plugin-so", required=True, help="Plugin shared library")
    parser.add_argument("--asset-dir", default="script/tutorial/asset", help="Asset directory")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--max-samples", type=int, default=None, help="最大验证样本数")
    args = parser.parse_args()
    
    asset_dir = Path(args.asset_dir)
    onnx_path = Path(args.onnx)
    plugin_so = Path(args.plugin_so)
    
    print("开始分析失败案例...")
    failed_cases, all_stats = analyze_failed_cases(
        onnx_path, plugin_so, asset_dir, args.fp16
    )
    
    print_analysis_report(failed_cases, all_stats)
    
    return 0 if len(failed_cases) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

