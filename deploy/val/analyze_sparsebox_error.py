#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度分析 SparseBox 插件误差来源

功能：
1. 加载差值文件和原始输入数据
2. 分析问题 anchor 的输入特征（anchor 值、instance_feature 值）
3. 模拟 CUDA 内核的计算过程，找出数值溢出/下溢点
4. 输出详细的分析报告
"""

import argparse
import numpy as np
from pathlib import Path
import sys

def load_bin_file(filepath: Path, dtype=np.float32):
    """加载二进制文件"""
    if not filepath.exists():
        return None
    return np.fromfile(filepath, dtype=dtype)

def analyze_anchor_values(anchor_data, anchor_idx, topk=5):
    """分析指定 anchor 的值"""
    # anchor 形状: [1, N, 11]
    # 索引: [batch, anchor_idx, :]
    anchor = anchor_data[0, anchor_idx, :]
    
    print(f"\n=== Anchor {anchor_idx} 详细分析 ===")
    print(f"  center: ({anchor[0]:.6f}, {anchor[1]:.6f}, {anchor[2]:.6f})")
    print(f"  size (log): ({anchor[3]:.6f}, {anchor[4]:.6f}, {anchor[5]:.6f})")
    print(f"  sin_yaw: {anchor[6]:.6f}, cos_yaw: {anchor[7]:.6f}")
    
    # 计算 exp(size) 检查溢出
    size_x = np.exp(anchor[3])
    size_y = np.exp(anchor[4])
    size_z = np.exp(anchor[5])
    
    print(f"  size (exp): ({size_x:.6e}, {size_y:.6e}, {size_z:.6e})")
    
    # FP16 范围检查
    fp16_max = 65504.0
    fp16_min = 6.103515625e-05
    
    if size_x > fp16_max or size_y > fp16_max or size_z > fp16_max:
        print(f"  ⚠️  WARNING: size 值超出 FP16 最大值 ({fp16_max})")
    if size_x < fp16_min or size_y < fp16_min or size_z < fp16_min:
        print(f"  ⚠️  WARNING: size 值低于 FP16 最小值 ({fp16_min})")
    
    return {
        'center': anchor[:3],
        'size_log': anchor[3:6],
        'size_exp': np.array([size_x, size_y, size_z]),
        'yaw': anchor[6:8]
    }

def analyze_instance_feature(inst_data, anchor_idx, embed_dims):
    """分析 instance_feature 的值"""
    if inst_data is None:
        return None
    
    # instance_feature 形状: [1, N, embed_dims]
    inst = inst_data[0, anchor_idx, :]
    
    stats = {
        'min': float(np.min(inst)),
        'max': float(np.max(inst)),
        'mean': float(np.mean(inst)),
        'std': float(np.std(inst)),
        'abs_max': float(np.max(np.abs(inst)))
    }
    
    print(f"\n=== Instance Feature (anchor {anchor_idx}) ===")
    print(f"  shape: {inst.shape}")
    print(f"  min: {stats['min']:.6f}, max: {stats['max']:.6f}")
    print(f"  mean: {stats['mean']:.6f}, std: {stats['std']:.6f}")
    print(f"  abs_max: {stats['abs_max']:.6f}")
    
    # 检查是否有异常大的值
    fp16_max = 65504.0
    if stats['abs_max'] > fp16_max:
        print(f"  ⚠️  WARNING: 有值超出 FP16 最大值")
    
    return stats

def simulate_sigmoid_centered(x):
    """模拟 CUDA 内核中的 sigmoid_centered 函数"""
    # CUDA 实现: 1.f / (1.f + expf(-x)) - 0.5f
    # 数值稳定性问题：
    # - 如果 x 很大（> 88），expf(-x) 会下溢为 0，结果 = 1.0 - 0.5 = 0.5
    # - 如果 x 很小（< -88），expf(-x) 会溢出为 inf，结果 = 1/(1+inf) - 0.5 = -0.5
    
    # 使用数值稳定的实现
    if x > 88.0:
        return 0.5  # expf(-x) ≈ 0
    elif x < -88.0:
        return -0.5  # expf(-x) ≈ inf, 1/(1+inf) ≈ 0
    else:
        exp_neg_x = np.exp(-x)
        return 1.0 / (1.0 + exp_neg_x) - 0.5

def simulate_learnable_point_computation(
    inst_feature, fc_weight, fc_bias, size, embed_dims
):
    """模拟可学习点的计算过程"""
    # 1. 线性变换: accum = inst_feature @ fc_weight.T + fc_bias
    # fc_weight 形状: [num_learnable_pts * 3, embed_dims]
    # 对于第一个可学习点，使用前 3 行
    
    accum = np.zeros(3)
    for k in range(embed_dims):
        val = inst_feature[k]
        accum[0] += val * fc_weight[0 * embed_dims + k]
        accum[1] += val * fc_weight[1 * embed_dims + k]
        accum[2] += val * fc_weight[2 * embed_dims + k]
    accum += fc_bias[:3]
    
    print(f"\n  Linear output (accum): ({accum[0]:.6f}, {accum[1]:.6f}, {accum[2]:.6f})")
    
    # 2. sigmoid_centered
    sig_x = simulate_sigmoid_centered(accum[0])
    sig_y = simulate_sigmoid_centered(accum[1])
    sig_z = simulate_sigmoid_centered(accum[2])
    
    print(f"  Sigmoid centered: ({sig_x:.6f}, {sig_y:.6f}, {sig_z:.6f})")
    
    # 3. 乘以 size
    local_x = sig_x * size[0]
    local_y = sig_y * size[1]
    local_z = sig_z * size[2]
    
    print(f"  Local (after * size): ({local_x:.6f}, {local_y:.6f}, {local_z:.6f})")
    
    # 检查溢出
    fp16_max = 65504.0
    if abs(local_x) > fp16_max or abs(local_y) > fp16_max or abs(local_z) > fp16_max:
        print(f"  ⚠️  WARNING: Local 值超出 FP16 范围")
    
    return np.array([local_x, local_y, local_z])

def main():
    parser = argparse.ArgumentParser(description="深度分析 SparseBox 插件误差")
    parser.add_argument("--diff-file", required=True, help="差值文件路径 (val/diff_node*_sample*.npy)")
    parser.add_argument("--asset-dir", default="script/tutorial/asset", help="资源目录")
    parser.add_argument("--sample-index", type=int, default=0, help="样本索引")
    parser.add_argument("--topk", type=int, default=10, help="分析前 topk 个误差最大的点")
    parser.add_argument("--onnx", help="ONNX 文件路径（用于获取插件属性）")
    args = parser.parse_args()
    
    # 加载差值文件
    diff_path = Path(args.diff_file)
    if not diff_path.exists():
        print(f"错误: 差值文件不存在: {diff_path}")
        sys.exit(1)
    
    diff = np.load(diff_path)
    print(f"差值文件形状: {diff.shape}")
    
    # 找出误差最大的点
    flat_diff = diff.reshape(-1)
    topk_indices = np.argsort(flat_diff)[::-1][:args.topk]
    
    # 加载原始输入数据
    asset_dir = Path(args.asset_dir)
    anchor_file = asset_dir / f"sample_{args.sample_index}_anchor_1*900*11_float32.bin"
    inst_file = asset_dir / f"sample_{args.sample_index}_instance_feature_1*900*256_float32.bin"
    
    if not anchor_file.exists():
        # 尝试 glob 匹配
        matches = list(asset_dir.glob(f"sample_{args.sample_index}_anchor_*.bin"))
        if matches:
            anchor_file = matches[0]
        else:
            print(f"错误: 找不到 anchor 文件: {anchor_file}")
            sys.exit(1)
    
    anchor_data = load_bin_file(anchor_file).reshape(1, -1, 11)
    print(f"Anchor 数据形状: {anchor_data.shape}")
    
    inst_data = None
    if inst_file.exists():
        inst_data = load_bin_file(inst_file)
        embed_dims = inst_data.size // (anchor_data.shape[1] * 1)
        inst_data = inst_data.reshape(1, anchor_data.shape[1], embed_dims)
        print(f"Instance feature 形状: {inst_data.shape}, embed_dims={embed_dims}")
    else:
        # 尝试 glob 匹配
        matches = list(asset_dir.glob(f"sample_{args.sample_index}_instance_feature_*.bin"))
        if matches:
            inst_file = matches[0]
            inst_data = load_bin_file(inst_file)
            embed_dims = inst_data.size // (anchor_data.shape[1] * 1)
            inst_data = inst_data.reshape(1, anchor_data.shape[1], embed_dims)
            print(f"Instance feature 形状: {inst_data.shape}, embed_dims={embed_dims}")
    
    # 分析每个误差最大的点
    print(f"\n{'='*60}")
    print(f"分析前 {args.topk} 个误差最大的点")
    print(f"{'='*60}")
    
    for rank, flat_idx in enumerate(topk_indices, 1):
        b, anchor_idx, kp_idx, xyz = np.unravel_index(flat_idx, diff.shape)
        error = flat_diff[flat_idx]
        
        print(f"\n{'#'*60}")
        print(f"# {rank}. 误差: {error:.6e}")
        print(f"# Anchor: {anchor_idx}, Keypoint: {kp_idx}, Axis: {xyz}")
        print(f"{'#'*60}")
        
        # 分析 anchor 值
        anchor_info = analyze_anchor_values(anchor_data, anchor_idx)
        
        # 分析 instance_feature（如果存在）
        if inst_data is not None:
            inst_stats = analyze_instance_feature(inst_data, anchor_idx, embed_dims)
        
        # 如果是可学习点，模拟计算过程
        # 注意：这里需要从 ONNX 获取 fix_scale, fc_weight, fc_bias
        # 暂时跳过，因为需要解析 ONNX
        
        print(f"\n  建议检查:")
        print(f"    1. anchor[{anchor_idx}][3/4/5] (size log) 是否过大导致 exp 溢出")
        print(f"    2. 如果是可学习点，检查 linear 输出是否导致 sigmoid 数值不稳定")
        print(f"    3. 检查最终输出是否超出 FP16 范围")

if __name__ == "__main__":
    main()

