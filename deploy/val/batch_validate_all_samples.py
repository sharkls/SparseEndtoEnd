#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量验证所有样本的 SparseBox 插件精度
"""

import argparse
import sys
from pathlib import Path
import glob

# 导入验证脚本
sys.path.insert(0, str(Path(__file__).parent))
import validate_sparsebox_plugin

def find_available_samples(asset_dir: Path) -> list:
    """查找所有可用的样本索引"""
    anchor_files = list(asset_dir.glob('sample_*_anchor_*.bin'))
    sample_indices = sorted(set([int(f.name.split('_')[1]) for f in anchor_files]))
    return sample_indices

def main():
    parser = argparse.ArgumentParser(description="批量验证所有样本")
    parser.add_argument("--onnx", required=True, help="Head ONNX path")
    parser.add_argument("--plugin-so", required=True, help="Plugin shared library")
    parser.add_argument("--asset-dir", default="script/tutorial/asset", help="Asset directory")
    parser.add_argument("--max-samples", type=int, default=None, help="最大验证样本数")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    args = parser.parse_args()
    
    asset_dir = Path(args.asset_dir)
    onnx_path = Path(args.onnx)
    plugin_so = Path(args.plugin_so)
    
    # 查找所有样本
    sample_indices = find_available_samples(asset_dir)
    if args.max_samples:
        sample_indices = sample_indices[:args.max_samples]
    
    print(f"找到 {len(sample_indices)} 个样本: {sample_indices}")
    print(f"开始批量验证...\n")
    
    # 获取所有节点
    nodes = validate_sparsebox_plugin.list_sparsebox_nodes(onnx_path)
    num_nodes = len(nodes)
    print(f"找到 {num_nodes} 个 SparseBox 节点\n")
    
    # 验证结果统计
    results = []
    failed_cases = []
    
    for sample_idx in sample_indices:
        print(f"{'='*60}")
        print(f"验证样本 {sample_idx}")
        print(f"{'='*60}")
        
        for node_idx in range(num_nodes):
            try:
                # 运行验证
                node = nodes[node_idx]
                attrs = validate_sparsebox_plugin.parse_plugin_attrs(node)
                
                # 检查样本文件是否存在
                anchor_file = list(asset_dir.glob(f'sample_{sample_idx}_anchor_*.bin'))
                inst_file = list(asset_dir.glob(f'sample_{sample_idx}_instance_feature_*.bin'))
                
                if not anchor_file or not inst_file:
                    print(f"  Node {node_idx}: 样本文件不存在，跳过")
                    continue
                
                anchor, instance = validate_sparsebox_plugin.load_sample(asset_dir, sample_idx)
                
                # 构建 engine
                input_shapes = {"anchor": tuple(anchor.shape)}
                if int(attrs["num_learnable_pts"][0]) > 0:
                    input_shapes["instance_feature"] = tuple(instance.shape)
                
                engine = validate_sparsebox_plugin.build_trt_engine(
                    attrs, input_shapes, plugin_so, args.fp16
                )
                
                # 运行验证
                trt_inputs = {
                    "anchor": anchor.astype(validate_sparsebox_plugin.np.float16 if args.fp16 else validate_sparsebox_plugin.np.float32)
                }
                if int(attrs["num_learnable_pts"][0]) > 0:
                    trt_inputs["instance_feature"] = instance.astype(
                        validate_sparsebox_plugin.np.float16 if args.fp16 else validate_sparsebox_plugin.np.float32
                    )
                
                trt_output = validate_sparsebox_plugin.run_engine(engine, trt_inputs).astype(validate_sparsebox_plugin.np.float32)
                # 关键修复：由于插件在 FP16 模式下输出 FP32，应该使用 FP32 参考实现进行比较
                # 这样可以确保比较的是相同精度的输出
                ref_output = validate_sparsebox_plugin.run_pytorch_reference(
                    anchor, instance, attrs, "fp32"  # 总是使用 FP32 参考，因为插件输出 FP32
                )
                
                metrics = validate_sparsebox_plugin.compare_outputs(ref_output, trt_output)
                
                results.append({
                    "sample": sample_idx,
                    "node": node_idx,
                    "metrics": metrics
                })
                
                max_diff = metrics["max_abs_diff"]
                status = "✅" if max_diff < 2.0 else "⚠️"
                print(f"  Node {node_idx}: max={max_diff:.6e}, mean={metrics['mean_abs_diff']:.6e}, median={metrics['median_abs_diff']:.6f} {status}")
                
                if max_diff >= 2.0:
                    failed_cases.append({
                        "sample": sample_idx,
                        "node": node_idx,
                        "max_diff": max_diff,
                        "mean_diff": metrics["mean_abs_diff"]
                    })
                    
            except Exception as e:
                print(f"  Node {node_idx}: 错误 - {e}")
                failed_cases.append({
                    "sample": sample_idx,
                    "node": node_idx,
                    "error": str(e)
                })
    
    # 生成统计报告
    print(f"\n{'='*60}")
    print("验证统计报告")
    print(f"{'='*60}")
    
    total_cases = len(results)
    success_cases = sum(1 for r in results if r["metrics"]["max_abs_diff"] < 2.0)
    failed_count = len(failed_cases)
    
    print(f"\n总验证次数: {total_cases}")
    print(f"成功次数: {success_cases} ({100*success_cases/total_cases:.1f}%)")
    print(f"失败次数: {failed_count} ({100*failed_count/total_cases:.1f}%)")
    
    if failed_cases:
        print(f"\n失败案例详情:")
        for case in failed_cases:
            if "error" in case:
                print(f"  样本 {case['sample']}, Node {case['node']}: 错误 - {case['error']}")
            else:
                print(f"  样本 {case['sample']}, Node {case['node']}: max_diff={case['max_diff']:.6e}, mean_diff={case['mean_diff']:.6e}")
        
        # 按节点统计失败次数
        print(f"\n按节点统计失败次数:")
        node_failures = {}
        for case in failed_cases:
            if "error" not in case:
                node = case['node']
                node_failures[node] = node_failures.get(node, 0) + 1
        for node, count in sorted(node_failures.items()):
            print(f"  Node {node}: {count} 次失败")
    
    # 按样本统计
    print(f"\n按样本统计:")
    sample_stats = {}
    for r in results:
        sample = r["sample"]
        if sample not in sample_stats:
            sample_stats[sample] = {"total": 0, "success": 0, "failed": 0}
        sample_stats[sample]["total"] += 1
        if r["metrics"]["max_abs_diff"] < 2.0:
            sample_stats[sample]["success"] += 1
        else:
            sample_stats[sample]["failed"] += 1
    
    for sample in sorted(sample_stats.keys()):
        stats = sample_stats[sample]
        print(f"  样本 {sample}: {stats['success']}/{stats['total']} 成功 ({100*stats['success']/stats['total']:.1f}%)")
    
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

