#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
追踪 kernel 中 NaN 产生位置的工具

通过分析特定失败案例，找出 NaN 产生的具体位置
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import validate_sparsebox_plugin

def trace_single_case(onnx_path, plugin_so, asset_dir, sample_idx, node_idx, fp16=True):
    """追踪单个失败案例"""
    onnx_path = Path(onnx_path)
    plugin_so = Path(plugin_so)
    asset_dir = Path(asset_dir)
    
    # 获取节点
    nodes = validate_sparsebox_plugin.list_sparsebox_nodes(onnx_path)
    node = nodes[node_idx]
    attrs = validate_sparsebox_plugin.parse_plugin_attrs(node)
    
    # 加载样本
    anchor, instance = validate_sparsebox_plugin.load_sample(asset_dir, sample_idx)
    
    print(f"\n{'='*80}")
    print(f"追踪样本 {sample_idx}, Node {node_idx}")
    print(f"{'='*80}")
    
    # 分析输入数据
    print(f"\n输入数据统计:")
    print(f"  Anchor shape: {anchor.shape}")
    print(f"  Anchor range: [{np.min(anchor):.6f}, {np.max(anchor):.6f}]")
    print(f"  Anchor NaN: {np.isnan(anchor).sum()}, Inf: {np.isinf(anchor).sum()}")
    
    if instance is not None:
        print(f"  Instance shape: {instance.shape}")
        print(f"  Instance range: [{np.min(instance):.6f}, {np.max(instance):.6f}]")
        print(f"  Instance NaN: {np.isnan(instance).sum()}, Inf: {np.isinf(instance).sum()}")
    
    # 检查 anchor 的每个维度
    print(f"\nAnchor 各维度统计:")
    dim_names = ["centerX", "centerY", "centerZ", "log_size_x", "log_size_y", 
                 "log_size_z", "sinYaw", "cosYaw", "dim8", "dim9", "dim10"]
    for i in range(min(11, anchor.shape[-1])):
        dim_data = anchor[..., i]
        print(f"  {dim_names[i] if i < len(dim_names) else f'dim{i}'}: "
              f"min={np.min(dim_data):.6f}, max={np.max(dim_data):.6f}, "
              f"mean={np.mean(dim_data):.6f}, NaN={np.isnan(dim_data).sum()}, Inf={np.isinf(dim_data).sum()}")
    
    # 构建 engine 并运行
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
    
    # 分析输出
    print(f"\n输出分析:")
    print(f"  TensorRT output shape: {trt_output.shape}")
    print(f"  TensorRT output range: [{np.min(trt_output):.6f}, {np.max(trt_output):.6f}]")
    print(f"  TensorRT output NaN: {np.isnan(trt_output).sum()}, Inf: {np.isinf(trt_output).sum()}")
    print(f"  Reference output range: [{np.min(ref_output):.6f}, {np.max(ref_output):.6f}]")
    print(f"  Reference output NaN: {np.isnan(ref_output).sum()}, Inf: {np.isinf(ref_output).sum()}")
    
    # 找出 NaN 的位置
    nan_mask = np.isnan(trt_output)
    if nan_mask.any():
        nan_indices = np.where(nan_mask)
        print(f"\nNaN 位置分析 (前10个):")
        for i in range(min(10, len(nan_indices[0]))):
            b, anchor_idx, kp, xyz = nan_indices[0][i], nan_indices[1][i], nan_indices[2][i], nan_indices[3][i]
            print(f"  Position [{b}, {anchor_idx}, {kp}, {xyz}]:")
            print(f"    TensorRT output: {trt_output[b, anchor_idx, kp, xyz]}")
            print(f"    Reference output: {ref_output[b, anchor_idx, kp, xyz]}")
            
            # 检查对应的输入
            if anchor_idx < anchor.shape[1]:
                anchor_data = anchor[0, anchor_idx, :]
                print(f"    Anchor input: center=({anchor_data[0]:.6f}, {anchor_data[1]:.6f}, {anchor_data[2]:.6f}), "
                      f"log_size=({anchor_data[3]:.6f}, {anchor_data[4]:.6f}, {anchor_data[5]:.6f})")
                if instance is not None and anchor_idx < instance.shape[1]:
                    inst_data = instance[0, anchor_idx, :]
                    print(f"    Instance input range: [{np.min(inst_data):.6f}, {np.max(inst_data):.6f}]")
    
    # 计算误差
    diff = np.abs(trt_output - ref_output)
    print(f"\n误差统计:")
    print(f"  Max diff: {np.max(diff):.6e}")
    print(f"  Mean diff: {np.mean(diff):.6e}")
    print(f"  Median diff: {np.median(diff):.6e}")
    
    # 保存输出用于进一步分析
    output_dir = Path("deploy/val/debug_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"sample_{sample_idx}_node_{node_idx}_trt_output.npy", trt_output)
    np.save(output_dir / f"sample_{sample_idx}_node_{node_idx}_ref_output.npy", ref_output)
    np.save(output_dir / f"sample_{sample_idx}_node_{node_idx}_diff.npy", diff)
    print(f"\n输出已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="追踪 kernel 中 NaN 产生位置")
    parser.add_argument("--onnx", required=True, help="Head ONNX path")
    parser.add_argument("--plugin-so", required=True, help="Plugin shared library")
    parser.add_argument("--asset-dir", default="script/tutorial/asset", help="Asset directory")
    parser.add_argument("--sample", type=int, required=True, help="样本索引")
    parser.add_argument("--node", type=int, required=True, help="节点索引")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    args = parser.parse_args()
    
    trace_single_case(
        args.onnx, args.plugin_so, args.asset_dir,
        args.sample, args.node, args.fp16
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

