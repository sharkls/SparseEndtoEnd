#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析失败案例的模式和特征

通过多次运行测试，找出失败案例的共同特征
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
import validate_sparsebox_plugin

def run_multiple_tests(onnx_path, plugin_so, asset_dir, num_runs=5, fp16=True, max_samples=10):
    """运行多次测试，收集失败案例"""
    all_failures = []
    
    for run_idx in range(num_runs):
        print(f"\n运行 {run_idx + 1}/{num_runs}...")
        
        # 获取所有节点
        nodes = validate_sparsebox_plugin.list_sparsebox_nodes(onnx_path)
        num_nodes = len(nodes)
        
        # 查找所有样本
        anchor_files = list(asset_dir.glob('sample_*_anchor_*.bin'))
        sample_indices = sorted(set([int(f.name.split('_')[1]) for f in anchor_files]))[:max_samples]
        
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
                    
                    if not is_success:
                        all_failures.append({
                            "run": run_idx,
                            "sample": sample_idx,
                            "node": node_idx,
                            "max_diff": max_diff,
                            "has_nan": np.isnan(trt_output).any(),
                            "nan_count": int(np.isnan(trt_output).sum()) if np.isnan(trt_output).any() else 0,
                        })
                        
                except Exception as e:
                    pass  # 忽略错误
    
    return all_failures

def analyze_patterns(failures):
    """分析失败模式"""
    print(f"\n{'='*80}")
    print("失败模式分析")
    print(f"{'='*80}")
    
    if not failures:
        print("✅ 没有失败案例！")
        return
    
    print(f"\n总失败次数: {len(failures)}")
    
    # 按样本和节点统计
    case_counts = defaultdict(int)
    nan_cases = []
    
    for failure in failures:
        key = (failure["sample"], failure["node"])
        case_counts[key] += 1
        if failure["has_nan"]:
            nan_cases.append(failure)
    
    print(f"\n失败案例分布 (样本, 节点):")
    for (sample, node), count in sorted(case_counts.items(), key=lambda x: -x[1]):
        print(f"  样本 {sample}, Node {node}: {count} 次失败")
    
    print(f"\n包含 NaN 的失败案例: {len(nan_cases)}/{len(failures)}")
    
    if nan_cases:
        print(f"\nNaN 案例详情:")
        for case in nan_cases[:10]:
            print(f"  运行 {case['run']}, 样本 {case['sample']}, Node {case['node']}: "
                  f"NaN数量={case['nan_count']}, max_diff={case['max_diff']}")
    
    # 分析是否是非确定性的
    print(f"\n非确定性分析:")
    case_stability = {}
    for (sample, node), count in case_counts.items():
        stability = count / len(set(f["run"] for f in failures if (f["sample"], f["node"]) == (sample, node)))
        case_stability[(sample, node)] = stability
    
    print(f"  总是失败的案例: {sum(1 for s in case_stability.values() if s >= 0.8)}")
    print(f"  偶尔失败的案例: {sum(1 for s in case_stability.values() if 0.2 <= s < 0.8)}")
    print(f"  很少失败的案例: {sum(1 for s in case_stability.values() if s < 0.2)}")

def main():
    parser = argparse.ArgumentParser(description="分析失败案例的模式")
    parser.add_argument("--onnx", required=True, help="Head ONNX path")
    parser.add_argument("--plugin-so", required=True, help="Plugin shared library")
    parser.add_argument("--asset-dir", default="script/tutorial/asset", help="Asset directory")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--num-runs", type=int, default=3, help="运行次数")
    parser.add_argument("--max-samples", type=int, default=10, help="最大样本数")
    args = parser.parse_args()
    
    failures = run_multiple_tests(
        Path(args.onnx), Path(args.plugin_so), Path(args.asset_dir),
        args.num_runs, args.fp16, args.max_samples
    )
    
    analyze_patterns(failures)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

