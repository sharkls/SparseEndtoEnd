#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试工具：追踪 FP16 模式下的溢出点
"""

import numpy as np
import torch
from pathlib import Path
import sys
import argparse

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from deploy.val.validate_sparsebox_plugin import (
    list_sparsebox_nodes,
    parse_plugin_attrs,
    load_sample,
    build_trt_engine,
    run_engine,
    run_pytorch_reference,
)

def debug_single_case(onnx_path, plugin_so, asset_dir, sample_idx, node_idx, use_fp16=True):
    """调试单个案例，找出溢出点"""
    print(f"\n{'='*60}")
    print(f"调试: 样本 {sample_idx}, Node {node_idx}, FP16={use_fp16}")
    print(f"{'='*60}\n")
    
    # 加载数据
    anchor, instance = load_sample(asset_dir, sample_idx)
    print(f"Anchor shape: {anchor.shape}")
    print(f"Instance shape: {instance.shape}")
    
    # 加载 ONNX 模型
    import onnx
    onnx_path_obj = Path(onnx_path) if isinstance(onnx_path, str) else onnx_path
    onnx_model = onnx.load(str(onnx_path_obj))
    nodes = list_sparsebox_nodes(onnx_path_obj)
    node = nodes[node_idx]
    attrs = parse_plugin_attrs(node)
    
    print(f"\n节点属性:")
    print(f"  embed_dims: {int(attrs['embed_dims'][0])}")
    print(f"  num_pts: {int(attrs['num_pts'][0])}")
    print(f"  num_learnable_pts: {int(attrs['num_learnable_pts'][0])}")
    
    # 检查输入数据范围
    print(f"\n输入数据范围检查:")
    print(f"  Anchor center: [{anchor[..., 0].min():.3f}, {anchor[..., 0].max():.3f}]")
    print(f"  Anchor size_log: [{anchor[..., 3].min():.3f}, {anchor[..., 3].max():.3f}]")
    print(f"  Instance feature: [{instance.min():.6f}, {instance.max():.6f}]")
    print(f"  Instance feature mean: {instance.mean():.6f}, std: {instance.std():.6f}")
    
    # 检查权重和偏置
    if "fc_weight" in attrs and "fc_bias" in attrs:
        weight = attrs["fc_weight"].reshape(int(attrs["num_learnable_pts"][0]) * 3, int(attrs["embed_dims"][0]))
        bias = attrs["fc_bias"].reshape(int(attrs["num_learnable_pts"][0]) * 3)
        print(f"\n权重和偏置范围:")
        print(f"  Weight: [{weight.min():.6f}, {weight.max():.6f}], mean={weight.mean():.6f}, std={weight.std():.6f}")
        print(f"  Bias: [{bias.min():.6f}, {bias.max():.6f}], mean={bias.mean():.6f}, std={bias.std():.6f}")
        
        # 模拟线性层计算，检查可能的溢出
        print(f"\n模拟线性层计算（检查溢出点）:")
        instance_fp16 = instance.astype(np.float16) if use_fp16 else instance
        weight_fp32 = weight.astype(np.float32)
        bias_fp32 = bias.astype(np.float32)
        
        # 转换为 torch tensor 进行计算
        inst_t = torch.from_numpy(instance_fp16).cuda() if use_fp16 and torch.cuda.is_available() else torch.from_numpy(instance_fp16)
        weight_t = torch.from_numpy(weight_fp32).cuda() if use_fp16 and torch.cuda.is_available() else torch.from_numpy(weight_fp32)
        bias_t = torch.from_numpy(bias_fp32).cuda() if use_fp16 and torch.cuda.is_available() else torch.from_numpy(bias_fp32)
        
        if use_fp16 and torch.cuda.is_available():
            inst_t = inst_t.half()
            weight_t = weight_t.float()  # 权重使用 FP32
            bias_t = bias_t.float()  # 偏置使用 FP32
        
        # 计算线性层输出
        with torch.no_grad():
            linear_out = torch.nn.functional.linear(inst_t, weight_t, bias_t)
            linear_out_fp32 = linear_out.float()
        
        print(f"  Linear 输出范围: [{linear_out_fp32.min():.6f}, {linear_out_fp32.max():.6f}]")
        print(f"  Linear 输出 mean: {linear_out_fp32.mean():.6f}, std: {linear_out_fp32.std():.6f}")
        print(f"  是否有 NaN: {torch.isnan(linear_out_fp32).any().item()}")
        print(f"  是否有 Inf: {torch.isinf(linear_out_fp32).any().item()}")
        print(f"  超出 FP16 范围 (>65504): {(linear_out_fp32.abs() > 65504).sum().item()}")
        print(f"  超出合理范围 (>100): {(linear_out_fp32.abs() > 100).sum().item()}")
        
        # 计算 sigmoid
        sigmoid_out = torch.sigmoid(linear_out_fp32) - 0.5
        print(f"\n  Sigmoid 输出范围: [{sigmoid_out.min():.6f}, {sigmoid_out.max():.6f}]")
        print(f"  是否有 NaN: {torch.isnan(sigmoid_out).any().item()}")
        print(f"  是否有 Inf: {torch.isinf(sigmoid_out).any().item()}")
    
    # 运行 TensorRT 和 PyTorch 参考实现
    print(f"\n运行 TensorRT 和 PyTorch 参考实现...")
    input_shapes = {"anchor": tuple(anchor.shape)}
    if int(attrs["num_learnable_pts"][0]) > 0:
        input_shapes["instance_feature"] = tuple(instance.shape)
    
    engine = build_trt_engine(attrs, input_shapes, plugin_so, use_fp16)
    
    trt_inputs = {
        "anchor": anchor.astype(np.float16 if use_fp16 else np.float32)
    }
    if int(attrs["num_learnable_pts"][0]) > 0:
        trt_inputs["instance_feature"] = instance.astype(np.float16 if use_fp16 else np.float32)
    
    trt_output = run_engine(engine, trt_inputs).astype(np.float32)
    ref_output = run_pytorch_reference(anchor, instance, attrs, "fp16" if use_fp16 else "fp32")
    
    # 计算差异
    diff = np.abs(ref_output - trt_output)
    print(f"\n输出差异分析:")
    print(f"  最大误差: {diff.max():.6e}")
    print(f"  平均误差: {diff.mean():.6e}")
    print(f"  中位数误差: {np.median(diff):.6e}")
    print(f"  是否有 NaN: {np.isnan(diff).any()}")
    print(f"  是否有 Inf: {np.isinf(diff).any()}")
    print(f"  误差 > 1e6 的点数: {(diff > 1e6).sum()}")
    print(f"  误差 > 1e10 的点数: {(diff > 1e10).sum()}")
    
    if diff.max() > 1e6:
        print(f"\n⚠️  发现巨大误差！")
        # 找出最大误差的位置
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  最大误差位置: batch={max_idx[0]}, anchor={max_idx[1]}, keypoint={max_idx[2]}, axis={max_idx[3]}")
        print(f"  TensorRT 输出: {trt_output[max_idx]:.6e}")
        print(f"  PyTorch 参考: {ref_output[max_idx]:.6e}")
        print(f"  差异: {diff[max_idx]:.6e}")
        
        # 分析该 anchor 的输入
        anchor_idx = max_idx[1]
        print(f"\n  问题 Anchor {anchor_idx} 的输入:")
        print(f"    center: ({anchor[0, anchor_idx, 0]:.3f}, {anchor[0, anchor_idx, 1]:.3f}, {anchor[0, anchor_idx, 2]:.3f})")
        print(f"    size_log: ({anchor[0, anchor_idx, 3]:.3f}, {anchor[0, anchor_idx, 4]:.3f}, {anchor[0, anchor_idx, 5]:.3f})")
        print(f"    size_exp: ({np.exp(anchor[0, anchor_idx, 3]):.3f}, {np.exp(anchor[0, anchor_idx, 4]):.3f}, {np.exp(anchor[0, anchor_idx, 5]):.3f})")
        if int(attrs["num_learnable_pts"][0]) > 0:
            print(f"    instance_feature range: [{instance[0, anchor_idx].min():.6f}, {instance[0, anchor_idx].max():.6f}]")
    
    return diff.max() < 2.0

def main():
    parser = argparse.ArgumentParser(description="调试 FP16 溢出点")
    parser.add_argument("--onnx", required=True, help="Head ONNX path")
    parser.add_argument("--plugin-so", required=True, help="Plugin shared library")
    parser.add_argument("--asset-dir", default="script/tutorial/asset", help="Asset directory")
    parser.add_argument("--sample-index", type=int, required=True, help="Sample index")
    parser.add_argument("--node-index", type=int, required=True, help="Node index")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    args = parser.parse_args()
    
    success = debug_single_case(
        args.onnx, args.plugin_so, Path(args.asset_dir),
        args.sample_index, args.node_index, args.fp16
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

