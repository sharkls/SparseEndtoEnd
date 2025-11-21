#!/usr/bin/env python3
import numpy as np
import sys
from pathlib import Path

# 加载验证脚本保存的输出
onnx_path = Path("deploy/onnx/sparse4dhead1st.onnx")
asset_dir = Path("script/tutorial/asset")

# 重新运行一次验证，但这次保存 PyTorch 和 TensorRT 的原始输出
import onnx
import tensorrt as trt
import torch
from cuda import cudart
from modules.ops.sparse_box3d_keypoints import sparse_box3d_keypoints

# 简化版本：直接加载差值文件并检查
diff = np.load("val/debug_output.npy")
print(f"差值形状: {diff.shape}")
print(f"最大值: {diff.max():.6e}")
print(f"是否有 NaN: {np.isnan(diff).any()}")
print(f"是否有 Inf: {np.isinf(diff).any()}")

# 找出有问题的点
problem_mask = diff > 1e6  # 误差大于 1e6 的点
num_problems = problem_mask.sum()
print(f"\n误差 > 1e6 的点数: {num_problems} / {diff.size} ({100*num_problems/diff.size:.2f}%)")

if num_problems > 0:
    # 找出这些点的位置
    problem_indices = np.where(problem_mask)
    print(f"\n前 10 个问题点的位置:")
    for i in range(min(10, len(problem_indices[0]))):
        b, anchor, kp, xyz = problem_indices[0][i], problem_indices[1][i], problem_indices[2][i], problem_indices[3][i]
        print(f"  [{b}, {anchor}, {kp}, {xyz}]: diff = {diff[b, anchor, kp, xyz]:.6e}")

