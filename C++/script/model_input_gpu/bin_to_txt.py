#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将二进制bin文件转换为txt格式，便于直接对比差异。
- 每行格式: index val1 val2 ...
- 支持三类数据：
  1) cached_feature: 形状(1, 600, 256), dtype=float32
  2) cached_anchor:  形状(1, 600, 11),  dtype=float32
  3) cached_track_id:形状(1, 600),      dtype=int32
可直接运行，默认处理第一帧GPU/CPU的三类文件，并输出到同目录下的txt文件。
也可通过命令行参数自定义输入输出。
"""

import argparse
import os
import numpy as np
from typing import Tuple


def load_bin(file_path: str, shape: Tuple[int, ...], dtype=np.float32):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    data = np.fromfile(file_path, dtype=dtype)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"文件大小不匹配: 期望{np.prod(shape)}, 实际{data.size}, 文件: {file_path}")
    return data.reshape(shape)


def save_array_to_txt_index_first(arr: np.ndarray, out_path: str, float_fmt: str = "{:.6f}"):
    """
    将数组保存为txt。
    - 若arr形状为(1, N, C)，则写出N行: idx v1 v2 ... vC
    - 若arr形状为(1, N)或(N, )，则写出N行: idx v
    """
    # 规范到(1, N, C)或(1, N)
    if arr.ndim == 3:
        _, n, c = arr.shape
        lines = []
        for i in range(n):
            row_vals = arr[0, i]
            if np.issubdtype(row_vals.dtype, np.floating):
                vals_str = " ".join(float_fmt.format(float(v)) for v in row_vals.tolist())
            else:
                vals_str = " ".join(str(int(v)) for v in row_vals.tolist())
            lines.append(f"{i} {vals_str}")
    elif arr.ndim == 2:
        # 形如(1, N)
        _, n = arr.shape
        lines = [f"{i} {int(arr[0, i])}" for i in range(n)] if np.issubdtype(arr.dtype, np.integer) \
                 else [f"{i} {float_fmt.format(float(arr[0, i]))}" for i in range(n)]
    elif arr.ndim == 1:
        n = arr.shape[0]
        lines = [f"{i} {int(arr[i])}" for i in range(n)] if np.issubdtype(arr.dtype, np.integer) \
                 else [f"{i} {float_fmt.format(float(arr[i]))}" for i in range(n)]
    else:
        raise ValueError(f"不支持的数组维度: {arr.shape}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"已保存: {out_path} (共{len(lines)}行)")


def export_default_first_frame():
    """导出默认第一帧三类数据(GPU/CPU)。"""
    base_gpu = "/share/Code/Sparse4d/C++/Output/val_bin_gpu"
    base_cpu = "/share/Code/Sparse4d/C++/Output/val_bin"

    files = [
        # (描述, 路径, 形状, dtype, 输出txt路径)
        ("gpu_feature", f"{base_gpu}/sample_0_output_cached_feature_1*600*256_float32.bin", (1, 600, 256), np.float32,
         f"{base_gpu}/sample_0_output_cached_feature_1*600*256_float32.txt"),
        ("cpu_feature", f"{base_cpu}/sample_0_output_cached_feature_1*600*256_float32.bin", (1, 600, 256), np.float32,
         f"{base_cpu}/sample_0_output_cached_feature_1*600*256_float32.txt"),
        ("gpu_anchor", f"{base_gpu}/sample_0_output_cached_anchor_1*600*11_float32.bin", (1, 600, 11), np.float32,
         f"{base_gpu}/sample_0_output_cached_anchor_1*600*11_float32.txt"),
        ("cpu_anchor", f"{base_cpu}/sample_0_output_cached_anchor_1*600*11_float32.bin", (1, 600, 11), np.float32,
         f"{base_cpu}/sample_0_output_cached_anchor_1*600*11_float32.txt"),
        ("gpu_track_id", f"{base_gpu}/sample_0_output_cached_track_id_1*600_int32.bin", (1, 600), np.int32,
         f"{base_gpu}/sample_0_output_cached_track_id_1*600_int32.txt"),
        ("cpu_track_id", f"{base_cpu}/sample_0_output_cached_track_id_1*600_int32.bin", (1, 600), np.int32,
         f"{base_cpu}/sample_0_output_cached_track_id_1*600_int32.txt"),

         ("temp_instance_feature", f"{base_gpu}/sample_1_input_temp_instance_feature_1*600*256_float32.bin", (1, 600, 256), np.float32,
         f"{base_gpu}/sample_1_input_temp_instance_feature_1*600*256_float32.txt"),
         ("temp_instance_feature", f"{base_cpu}/sample_1_input_temp_instance_feature_1*600*256_float32.bin", (1, 600, 256), np.float32,
         f"{base_cpu}/sample_1_input_temp_instance_feature_1*600*256_float32.txt"),
         ("temp_anchor", f"{base_gpu}/sample_1_input_temp_anchor_1*600*11_float32.bin", (1, 600, 11), np.float32,
         f"{base_gpu}/sample_1_input_temp_anchor_1*600*11_float32.txt"),
         ("temp_anchor", f"{base_cpu}/sample_1_input_temp_anchor_1*600*11_float32.bin", (1, 600, 11), np.float32,
         f"{base_cpu}/sample_1_input_temp_anchor_1*600*11_float32.txt"),
    ]

    for name, bin_path, shape, dtype, out_txt in files:
        try:
            arr = load_bin(bin_path, shape, dtype)
            save_array_to_txt_index_first(arr, out_txt, float_fmt="{:.6f}")
        except Exception as e:
            print(f"[{name}] 失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="将bin文件导出为txt(每行: index 后接各列数值)")
    parser.add_argument("--bin", dest="bin_path", default=None, help="输入bin文件路径")
    parser.add_argument("--shape", dest="shape", default=None,
                        help="形状，例如: 1,600,256 或 1,600 或 600")
    parser.add_argument("--dtype", dest="dtype", default="float32", choices=["float32", "int32"],
                        help="数据类型")
    parser.add_argument("--out", dest="out_path", default=None, help="输出txt路径")
    parser.add_argument("--floatfmt", dest="float_fmt", default="{:.6f}", help="浮点数格式，如{:.6f}")
    parser.add_argument("--run-default", dest="run_default", action="store_true",
                        help="不带任何参数，直接导出默认GPU/CPU第一帧的三类txt对比文件")

    args = parser.parse_args()

    if args.run_default or (args.bin_path is None and args.out_path is None):
        export_default_first_frame()
        return

    if not args.bin_path or not args.out_path or not args.shape:
        parser.error("--bin, --shape, --out 必须同时提供；或使用 --run-default")

    # 解析shape
    try:
        shape = tuple(int(x) for x in args.shape.split(","))
    except Exception:
        raise ValueError("形状解析失败，请使用逗号分隔的整数，如 1,600,256")

    dtype = np.float32 if args.dtype == "float32" else np.int32

    arr = load_bin(args.bin_path, shape, dtype)
    save_array_to_txt_index_first(arr, args.out_path, float_fmt=args.float_fmt)


if __name__ == "__main__":
    main() 