#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Head Engine Validation Script
=============================
验证 Head1 和 Head2 TensorRT Engine 与 PyTorch 中间结果的一致性

Usage:
    # 验证 Head1 FP32
    python validate_head_engine.py --head head1 --precision fp32 --data_dir val_data_e2e_fp32
    
    # 验证 Head2 FP32
    python validate_head_engine.py --head head2 --precision fp32 --data_dir val_data_e2e_fp32
    
    # 验证 Head1 FP16
    python validate_head_engine.py --head head1 --precision fp16 --data_dir val_data_e2e_fp32
    
    # 验证所有
    python validate_head_engine.py --head all --precision all --data_dir val_data_e2e_fp32
"""

import argparse
import os
import sys
import glob
import ctypes
import time
import numpy as np
import tensorrt as trt
import torch

# 简单的表格打印函数（避免额外依赖）
def tabulate_simple(rows, headers, tablefmt="grid"):
    """简单的表格打印，不依赖 tabulate 库"""
    if not rows:
        return ""
    
    # 计算每列宽度
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # 生成分隔线
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    
    # 格式化行
    def format_row(row):
        return "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
    
    lines = [sep, format_row(headers), sep]
    for row in rows:
        lines.append(format_row(row))
    lines.append(sep)
    
    return "\n".join(lines)

# 尝试导入 tabulate，如果失败则使用简单版本
try:
    from tabulate import tabulate
except ImportError:
    tabulate = tabulate_simple

# ============================================================================
# TensorRT Wrapper
# ============================================================================
class TRTWrapper:
    def __init__(self, engine_path, verbose=False):
        log_level = trt.Logger.WARNING if not verbose else trt.Logger.INFO
        self.trt_logger = trt.Logger(log_level)
        
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.trt_logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
    
    def get_binding_dtype(self, name):
        dtype = self.engine.get_tensor_dtype(name)
        if dtype == trt.DataType.HALF:
            return torch.float16, np.float16
        elif dtype == trt.DataType.INT32:
            return torch.int32, np.int32
        elif hasattr(trt.DataType, 'INT64') and dtype == trt.DataType.INT64:
            return torch.int64, np.int64
        elif dtype == trt.DataType.BOOL:
            return torch.bool, np.bool_
        else:
            return torch.float32, np.float32

    def infer(self, input_map):
        inputs = []
        
        for name in self.input_names:
            tensor = input_map[name]
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor).cuda()
            elif not tensor.is_cuda:
                tensor = tensor.cuda()
            
            # 转换为 engine 期望的精度
            torch_dtype, _ = self.get_binding_dtype(name)
            if tensor.dtype != torch_dtype:
                if torch_dtype == torch.float16:
                    tensor = tensor.half()
                elif torch_dtype == torch.float32:
                    tensor = tensor.float()
                elif torch_dtype == torch.int32:
                    tensor = tensor.int()
            
            self.context.set_input_shape(name, tuple(tensor.shape))
            inputs.append(tensor)
        
        # 分配输出
        output_map = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dims = [max(1, s) for s in shape]
            torch_dtype, _ = self.get_binding_dtype(name)
            out_tensor = torch.empty(tuple(dims), dtype=torch_dtype, device='cuda')
            output_map[name] = out_tensor
        
        # 设置地址
        for i, name in enumerate(self.input_names):
            self.context.set_tensor_address(name, inputs[i].data_ptr())
        for name in self.output_names:
            self.context.set_tensor_address(name, output_map[name].data_ptr())
        
        # 执行
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        
        return output_map

# ============================================================================
# 数据加载工具
# ============================================================================
def parse_shape_from_filename(filename):
    """从文件名解析 shape，如 sample_0_feature_1*89760*256_float32.bin"""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    for part in parts:
        if '*' in part:
            try:
                return tuple(map(int, part.split('*')))
            except:
                pass
    return None

def parse_dtype_from_filename(filename):
    """从文件名解析 dtype"""
    basename = os.path.basename(filename).lower()
    if 'float16' in basename or 'fp16' in basename:
        return np.float16
    elif 'float64' in basename:
        return np.float64
    elif 'int32' in basename:
        return np.int32
    elif 'int64' in basename:
        return np.int64
    elif 'uint8' in basename:
        return np.uint8
    else:
        return np.float32

def find_file(data_dir, name_pattern, sample_prefix="sample_0"):
    """查找匹配的 bin 文件"""
    pattern = os.path.join(data_dir, f"{sample_prefix}_{name_pattern}_*.bin")
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(data_dir, f"{name_pattern}.bin")
        files = glob.glob(pattern)
    if not files:
        return None
    return files[0]

def load_bin(path, dtype=None, shape=None):
    """加载 bin 文件"""
    if not path or not os.path.exists(path):
        return None
    
    if dtype is None:
        dtype = parse_dtype_from_filename(path)
    if shape is None:
        shape = parse_shape_from_filename(path)
    
    data = np.fromfile(path, dtype=dtype)
    if shape is not None:
        try:
            data = data.reshape(shape)
        except:
            pass
    return data

# ============================================================================
# 误差分析工具
# ============================================================================
def compute_error_metrics(pred, expected, name=""):
    """计算详细的误差指标"""
    pred_flat = pred.flatten().astype(np.float64)
    exp_flat = expected.flatten().astype(np.float64)
    
    # 确保形状匹配
    min_len = min(len(pred_flat), len(exp_flat))
    pred_flat = pred_flat[:min_len]
    exp_flat = exp_flat[:min_len]
    
    diff = np.abs(pred_flat - exp_flat)
    
    metrics = {
        "name": name,
        "shape": str(pred.shape),
        "max_diff": float(np.max(diff)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "median_diff": float(np.median(diff)),
        "p99_diff": float(np.percentile(diff, 99)),
    }
    
    # 余弦相似度
    norm_pred = np.linalg.norm(pred_flat)
    norm_exp = np.linalg.norm(exp_flat)
    if norm_pred > 0 and norm_exp > 0:
        metrics["cos_sim"] = float(np.dot(pred_flat, exp_flat) / (norm_pred * norm_exp))
    else:
        metrics["cos_sim"] = 0.0
    
    # 相对误差
    non_zero_mask = np.abs(exp_flat) > 1e-8
    if np.any(non_zero_mask):
        rel_diff = diff[non_zero_mask] / np.abs(exp_flat[non_zero_mask])
        metrics["max_rel_diff"] = float(np.max(rel_diff))
        metrics["mean_rel_diff"] = float(np.mean(rel_diff))
    else:
        metrics["max_rel_diff"] = 0.0
        metrics["mean_rel_diff"] = 0.0
    
    return metrics

def print_error_table(metrics_list, title=""):
    """打印误差表格"""
    if title:
        print(f"\n{'='*80}")
        print(f" {title}")
        print(f"{'='*80}")
    
    headers = ["Output", "Shape", "Max Diff", "Mean Diff", "P99 Diff", "Cos Sim", "Status"]
    rows = []
    
    for m in metrics_list:
        # 判断状态
        if m["max_diff"] < 1e-5:
            status = "✓ EXACT"
        elif m["max_diff"] < 1e-3 and m["cos_sim"] > 0.999:
            status = "✓ GOOD"
        elif m["max_diff"] < 0.1 and m["cos_sim"] > 0.99:
            status = "⚠ ACCEPTABLE"
        else:
            status = "✗ MISMATCH"
        
        rows.append([
            m["name"][:25],
            m["shape"][:15],
            f"{m['max_diff']:.6e}",
            f"{m['mean_diff']:.6e}",
            f"{m['p99_diff']:.6e}",
            f"{m['cos_sim']:.6f}",
            status
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))

# ============================================================================
# Head1 验证
# ============================================================================
def validate_head1(engine_path, data_dir, sample_idx=0, verbose=False):
    """验证 Head1 Engine"""
    print(f"\n{'#'*80}")
    print(f"# Validating Head1 Engine")
    print(f"# Engine: {engine_path}")
    print(f"# Data: {data_dir} (sample_{sample_idx})")
    print(f"{'#'*80}")
    
    wrapper = TRTWrapper(engine_path, verbose=verbose)
    
    print(f"\nEngine Inputs: {wrapper.input_names}")
    print(f"Engine Outputs: {wrapper.output_names}")
    
    # Head1 输入映射
    head1_input_mapping = {
        "feature": "feature",
        "spatial_shapes": "spatial_shapes",
        "level_start_index": "level_start_index",
        "instance_feature": "instance_feature",
        "anchor": "anchor",
        "time_interval": "time_interval",
        "image_wh": "image_wh",
        "lidar2img": "lidar2img",
    }
    
    # 加载输入
    inputs = {}
    sample_prefix = f"sample_{sample_idx}"
    print(f"\nLoading inputs from {sample_prefix}...")
    
    for engine_name in wrapper.input_names:
        # 查找对应的 bin 文件
        data_name = head1_input_mapping.get(engine_name, engine_name)
        path = find_file(data_dir, data_name, sample_prefix)
        
        if not path:
            print(f"  [ERROR] Input '{engine_name}' not found!")
            return None
        
        data = load_bin(path)
        if data is None:
            print(f"  [ERROR] Failed to load '{path}'!")
            return None
        
        inputs[engine_name] = data
        print(f"  ✓ {engine_name}: {data.shape} ({data.dtype})")
    
    # 执行推理
    print("\nRunning inference...")
    start_time = time.time()
    outputs = wrapper.infer(inputs)
    inference_time = (time.time() - start_time) * 1000
    print(f"Inference time: {inference_time:.2f} ms")
    
    # 加载 PyTorch 参考输出并比较
    metrics_list = []
    print("\nComparing outputs...")
    
    for name in wrapper.output_names:
        engine_out = outputs[name].cpu().numpy()
        
        # 查找对应的 PyTorch 输出
        pt_path = find_file(data_dir, name, sample_prefix)
        if not pt_path:
            print(f"  [WARN] PyTorch output '{name}' not found, skipping...")
            continue
        
        pt_out = load_bin(pt_path)
        if pt_out is None:
            continue
        
        # 尝试 reshape
        try:
            pt_out = pt_out.reshape(engine_out.shape)
        except:
            print(f"  [WARN] Shape mismatch for '{name}': engine={engine_out.shape}, pt={pt_out.shape}")
            continue
        
        # 计算误差
        if np.issubdtype(engine_out.dtype, np.integer):
            # 整数类型：计算匹配率
            match_rate = np.mean(engine_out.flatten() == pt_out.flatten())
            metrics_list.append({
                "name": name,
                "shape": str(engine_out.shape),
                "max_diff": 1 - match_rate,
                "mean_diff": 1 - match_rate,
                "p99_diff": 1 - match_rate,
                "cos_sim": match_rate,
            })
        else:
            metrics = compute_error_metrics(engine_out, pt_out.astype(np.float32), name)
            metrics_list.append(metrics)
    
    print_error_table(metrics_list, "Head1 Output Comparison")
    
    return {
        "inference_time": inference_time,
        "metrics": metrics_list,
        "outputs": outputs
    }

# ============================================================================
# Head2 验证
# ============================================================================
def validate_head2(engine_path, data_dir, sample_idx=1, verbose=False):
    """验证 Head2 Engine"""
    print(f"\n{'#'*80}")
    print(f"# Validating Head2 Engine")
    print(f"# Engine: {engine_path}")
    print(f"# Data: {data_dir} (sample_{sample_idx})")
    print(f"{'#'*80}")
    
    wrapper = TRTWrapper(engine_path, verbose=verbose)
    
    print(f"\nEngine Inputs: {wrapper.input_names}")
    print(f"Engine Outputs: {wrapper.output_names}")
    
    # Head2 输入映射
    head2_input_mapping = {
        "feature": "feature",
        "spatial_shapes": "spatial_shapes",
        "level_start_index": "level_start_index",
        "instance_feature": "instance_feature",
        "anchor": "anchor",
        "time_interval": "time_interval",
        "image_wh": "image_wh",
        "lidar2img": "lidar2img",
        "temp_instance_feature": "temp_instance_feature",
        "temp_anchor": "temp_anchor",
        "mask": "mask",
        "track_id": "track_id",
    }
    
    # 加载输入
    inputs = {}
    sample_prefix = f"sample_{sample_idx}"
    print(f"\nLoading inputs from {sample_prefix}...")
    
    for engine_name in wrapper.input_names:
        data_name = head2_input_mapping.get(engine_name, engine_name)
        path = find_file(data_dir, data_name, sample_prefix)
        
        if not path:
            print(f"  [ERROR] Input '{engine_name}' not found!")
            return None
        
        data = load_bin(path)
        if data is None:
            print(f"  [ERROR] Failed to load '{path}'!")
            return None
        
        inputs[engine_name] = data
        print(f"  ✓ {engine_name}: {data.shape} ({data.dtype})")
    
    # 执行推理
    print("\nRunning inference...")
    start_time = time.time()
    outputs = wrapper.infer(inputs)
    inference_time = (time.time() - start_time) * 1000
    print(f"Inference time: {inference_time:.2f} ms")
    
    # 加载 PyTorch 参考输出并比较
    metrics_list = []
    print("\nComparing outputs...")
    
    for name in wrapper.output_names:
        engine_out = outputs[name].cpu().numpy()
        
        pt_path = find_file(data_dir, name, sample_prefix)
        if not pt_path:
            print(f"  [WARN] PyTorch output '{name}' not found, skipping...")
            continue
        
        pt_out = load_bin(pt_path)
        if pt_out is None:
            continue
        
        try:
            pt_out = pt_out.reshape(engine_out.shape)
        except:
            print(f"  [WARN] Shape mismatch for '{name}': engine={engine_out.shape}, pt={pt_out.shape}")
            continue
        
        if np.issubdtype(engine_out.dtype, np.integer):
            match_rate = np.mean(engine_out.flatten() == pt_out.flatten())
            metrics_list.append({
                "name": name,
                "shape": str(engine_out.shape),
                "max_diff": 1 - match_rate,
                "mean_diff": 1 - match_rate,
                "p99_diff": 1 - match_rate,
                "cos_sim": match_rate,
            })
        else:
            metrics = compute_error_metrics(engine_out, pt_out.astype(np.float32), name)
            metrics_list.append(metrics)
    
    print_error_table(metrics_list, "Head2 Output Comparison")
    
    return {
        "inference_time": inference_time,
        "metrics": metrics_list,
        "outputs": outputs
    }

# ============================================================================
# 主函数
# ============================================================================
def load_plugins(plugin_dir):
    """加载 TensorRT 插件"""
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")
    
    plugins = [
        os.path.join(plugin_dir, "dfa_plugin/lib/deformableAttentionAggr.so"),
        os.path.join(plugin_dir, "ln_plugin/lib/customLayerNorm.so"),
        os.path.join(plugin_dir, "sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so"),
    ]
    
    for p in plugins:
        if os.path.exists(p):
            ctypes.CDLL(p)
            print(f"✓ Loaded plugin: {os.path.basename(p)}")
        else:
            print(f"✗ Plugin not found: {p}")

def main():
    parser = argparse.ArgumentParser(description="Validate Head1/Head2 TensorRT Engine")
    parser.add_argument("--head", type=str, default="all", choices=["head1", "head2", "all"],
                        help="Which head to validate: head1, head2, or all")
    parser.add_argument("--precision", type=str, default="all", choices=["fp32", "fp16", "all"],
                        help="Engine precision: fp32, fp16, or all")
    parser.add_argument("--data_dir", type=str, default="val_data_e2e_fp32",
                        help="Directory containing PyTorch intermediate data")
    parser.add_argument("--engine_dir", type=str, default="engine",
                        help="Directory containing TensorRT engines")
    parser.add_argument("--plugin_dir", type=str, default=".",
                        help="Directory containing plugin .so files")
    parser.add_argument("--sample_head1", type=int, default=0,
                        help="Sample index for Head1 validation (default: 0)")
    parser.add_argument("--sample_head2", type=int, default=1,
                        help="Sample index for Head2 validation (default: 1)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()
    
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理相对路径
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(script_dir, args.data_dir)
    if not os.path.isabs(args.engine_dir):
        args.engine_dir = os.path.join(script_dir, args.engine_dir)
    if not os.path.isabs(args.plugin_dir):
        args.plugin_dir = os.path.join(script_dir, args.plugin_dir)
    
    print("="*80)
    print(" Head Engine Validation Tool")
    print("="*80)
    print(f"Data Directory:   {args.data_dir}")
    print(f"Engine Directory: {args.engine_dir}")
    print(f"Plugin Directory: {args.plugin_dir}")
    print("="*80)
    
    # 加载插件
    print("\nLoading TensorRT plugins...")
    load_plugins(args.plugin_dir)
    
    # 确定要验证的配置
    heads = ["head1", "head2"] if args.head == "all" else [args.head]
    precisions = ["fp32", "fp16"] if args.precision == "all" else [args.precision]
    
    results = {}
    
    for head in heads:
        for precision in precisions:
            # 构造 engine 路径
            if precision == "fp32":
                if head == "head1":
                    engine_name = "sparse4dhead1st-fp32.engine"
                else:
                    engine_name = "sparse4dhead2nd-fp32.engine"
            else:  # fp16
                if head == "head1":
                    engine_name = "sparse4dhead1st.engine"
                else:
                    engine_name = "sparse4dhead2nd.engine"
            
            engine_path = os.path.join(args.engine_dir, engine_name)
            
            if not os.path.exists(engine_path):
                print(f"\n[SKIP] Engine not found: {engine_path}")
                continue
            
            key = f"{head}_{precision}"
            
            if head == "head1":
                results[key] = validate_head1(
                    engine_path, args.data_dir, 
                    sample_idx=args.sample_head1,
                    verbose=args.verbose
                )
            else:
                results[key] = validate_head2(
                    engine_path, args.data_dir,
                    sample_idx=args.sample_head2,
                    verbose=args.verbose
                )
    
    # 打印总结
    print(f"\n{'='*80}")
    print(" SUMMARY")
    print(f"{'='*80}")
    
    summary_rows = []
    for key, result in results.items():
        if result is None:
            summary_rows.append([key, "FAILED", "-", "-"])
            continue
        
        max_diff = max(m["max_diff"] for m in result["metrics"]) if result["metrics"] else 0
        mean_cos = np.mean([m["cos_sim"] for m in result["metrics"]]) if result["metrics"] else 0
        
        if max_diff < 1e-3:
            status = "✓ PASS"
        elif max_diff < 0.1:
            status = "⚠ ACCEPTABLE"
        else:
            status = "✗ FAIL"
        
        summary_rows.append([
            key,
            status,
            f"{max_diff:.6e}",
            f"{mean_cos:.6f}",
            f"{result['inference_time']:.2f} ms"
        ])
    
    print(tabulate(
        summary_rows,
        headers=["Config", "Status", "Max Error", "Avg Cos Sim", "Inference Time"],
        tablefmt="grid"
    ))

if __name__ == "__main__":
    main()

