#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TopK 稳定性测试脚本
- 加载 pred_class_score 的 bin 文件 (形状 [1,900,10])
- 计算 confidence = pred_class_score.max(dim=-1).values -> [1,900]
- 分别用 topk / topk_onnx_compatiblev2 / topk_onnx_compatiblev3 做排序取前K
- 打印每种方法的 topK 索引与对应confidence
- 额外打印：在 topK 中出现“confidence值完全相等”的分组及各自的索引
"""

import os
import sys
import argparse
import numpy as np
import torch

# 方便直接运行
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from modules.head.sparse4d_blocks.instance_bank import (
    topk as topk_plain,
    topk_onnx_compatiblev2,
)

# v3 可能不存在，容错
try:
    from modules.head.sparse4d_blocks.instance_bank import topk_onnx_compatiblev3
    HAS_V3 = True
except Exception:
    HAS_V3 = False


def read_pred_class_score(path: str) -> torch.Tensor:
    shape = (1, 900, 10)
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != np.prod(shape):
        raise ValueError(f"文件大小不匹配: 期望元素数={np.prod(shape)}, 实际={arr.size}")
    arr = arr.reshape(shape)
    return torch.from_numpy(arr)  # float32


def to_2d_confidence(scores_1x900x10: torch.Tensor) -> torch.Tensor:
    # shape [1,900,10] -> [1,900]
    conf = scores_1x900x10.max(dim=-1).values
    return conf


def run_topk_plain(conf_2d: torch.Tensor, k: int):
    # topk_plain 返回 indices 为展平的 (bs*k,) 需要还原为 [bs,k]
    bs, N = conf_2d.shape
    conf_k, _, flat_idx = topk_plain(conf_2d, k)
    idx = flat_idx.view(bs, k)
    # 与 conf_2d 对齐的 confidence
    conf_sel = torch.gather(conf_2d, 1, idx)
    return idx, conf_sel


def run_topk_v2(conf_2d: torch.Tensor, k: int):
    # v2 返回 indices 形状为 [bs,k]
    conf_k, _, idx = topk_onnx_compatiblev2(conf_2d, k)
    # 与 conf_2d 对齐的 confidence
    conf_sel = torch.gather(conf_2d, 1, idx)
    return idx, conf_sel


def run_topk_v3(conf_2d: torch.Tensor, k: int):
    conf_k, _, idx = topk_onnx_compatiblev3(conf_2d, k)
    conf_sel = torch.gather(conf_2d, 1, idx)
    return idx, conf_sel


def print_method_result(name: str, idx: torch.Tensor, conf_sel: torch.Tensor, topn_print: int = 20):
    idx_np = idx.cpu().numpy()
    conf_np = conf_sel.cpu().numpy()
    print(f"\n==== {name} 结果 ====")
    print(f"Top-{idx_np.shape[1]} 索引(前{topn_print}个): {idx_np[0, :topn_print].tolist()}")
    print(f"Top-{idx_np.shape[1]} 置信度(前{topn_print}个): {conf_np[0, :topn_print].round(6).tolist()}")

    # 打印topK中“置信度完全相等”的分组
    vals = conf_np[0]
    groups = {}
    for j, v in enumerate(vals):
        groups.setdefault(float(v), []).append(int(idx_np[0, j]))
    dup_groups = {v: inds for v, inds in groups.items() if len(inds) > 1}
    if dup_groups:
        print("等值分组(置信度完全相等):")
        for v, inds in dup_groups.items():
            print(f"  值={v:.6f} -> 索引={inds}")
    else:
        print("TopK中无完全相等的置信度值分组")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bin",
        type=str,
        default="C++/Output/val_bin/sample_0_pred_class_score_1*900*10_float32.bin",
        # default="script/tutorial/asset/sample_0_pred_class_score_1*900*10_float32.bin",
        help="pred_class_score bin 文件路径 (float32, 形状 1*900*10)",
    )
    parser.add_argument("--k", type=int, default=20, help="TopK 大小")
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_printoptions(precision=6, linewidth=200, sci_mode=False)

    print(f"加载文件: {args.bin}")
    scores = read_pred_class_score(args.bin)
    print(f"形状: {tuple(scores.shape)}, dtype: {scores.dtype}")

    conf = to_2d_confidence(scores)  # [1,900]
    print(f"max后confidence形状: {tuple(conf.shape)}")

    # 三种方法
    idx_plain, conf_plain = run_topk_plain(conf, args.k)
    idx_v2, conf_v2 = run_topk_v2(conf, args.k)

    print_method_result("topk(plain)", idx_plain, conf_plain)
    print_method_result("topk_onnx_compatiblev2", idx_v2, conf_v2)

    if HAS_V3:
        idx_v3, conf_v3 = run_topk_v3(conf, args.k)
        print_method_result("topk_onnx_compatiblev3", idx_v3, conf_v3)

        # 对比三者在“相同置信度”的排序差异（仅在出现等值时有意义）
        print("\n==== 三者在等值分组中的索引顺序对比(若存在等值) ====")
        vals = conf_plain[0].cpu().numpy()
        groups = {}
        for j, v in enumerate(vals):
            groups.setdefault(float(v), []).append(j)  # j是topK内位置
        dup_vals = [v for v, pos in groups.items() if len(pos) > 1]
        if dup_vals:
            plain_idx = idx_plain[0].cpu().numpy()
            v2_idx = idx_v2[0].cpu().numpy()
            v3_idx = idx_v3[0].cpu().numpy()
            for v in dup_vals:
                pos_list = groups[v]
                print(f"值={v:.6f}")
                print(f"  topk(plain) 索引序列: {[int(plain_idx[p]) for p in pos_list]}")
                print(f"  v2 索引序列:         {[int(v2_idx[p]) for p in pos_list]}")
                print(f"  v3 索引序列:         {[int(v3_idx[p]) for p in pos_list]}")
        else:
            print("TopK中未检测到等值分组，三者排序一致性对比略过")
    else:
        print("未检测到 topk_onnx_compatiblev3，可先在模块中添加后再测试。")


if __name__ == "__main__":
    main() 