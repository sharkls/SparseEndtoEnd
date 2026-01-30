"""
    # 测试所有校准数据（默认）
    python3 script/tutorial/053.benchmark_quantization_error_backbone.py

    # 只测试前 10 个样本（快速验证）
    python3 script/tutorial/053.benchmark_quantization_error_backbone.py --max-samples 10

    # 指定自定义路径
    python3 script/tutorial/053.benchmark_quantization_error_backbone.py \
        --engine deploy/engine/sparse4dbackbone_int8.engine \
        --calib-dir deploy/calibration_data \
        --max-samples 50
"""

import torch
import numpy as np
import tensorrt as trt
import os
import argparse
import sys
import glob
from typing import Optional, Dict, Any

# 引用项目路径
sys.path.append(os.getcwd())
from tool.utils.config import read_cfg
from tool.runner.checkpoint import load_checkpoint
from deploy.export_backbone_onnx_quant import Sparse4DBackbone

def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    try:
        from modules.sparse4d_detector import Sparse4D
    except ImportError:
        pass
    return eval(type)(**cfg2)

class TRTWrapperTorch:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        try:
            with open(engine_path, "rb") as f:
                self.runtime = trt.Runtime(self.logger)
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"[ERROR] Failed to load engine: {e}")
            sys.exit(1)
            
        self.context = self.engine.create_execution_context()
        self.io_tensors = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dims = []
            for dim in shape:
                if dim == -1: dims.append(1)
                else: dims.append(dim)
            dtype_trt = self.engine.get_tensor_dtype(name)
            if dtype_trt == trt.DataType.FLOAT: dtype_torch = torch.float32
            elif dtype_trt == trt.DataType.HALF: dtype_torch = torch.float16
            elif dtype_trt == trt.DataType.INT32: dtype_torch = torch.int32
            elif dtype_trt == trt.DataType.INT8: dtype_torch = torch.int8
            else: dtype_torch = torch.float32
            tensor = torch.zeros(tuple(dims), dtype=dtype_torch, device='cuda')
            self.io_tensors[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

    def infer(self, input_map):
        for name, tensor in self.io_tensors.items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name in input_map:
                    src = input_map[name]
                    if isinstance(src, np.ndarray): src = torch.from_numpy(src).cuda()
                    if src.dtype != tensor.dtype: src = src.to(tensor.dtype)
                    if src.numel() == tensor.numel(): src = src.view_as(tensor)
                    tensor.copy_(src)
                    self.context.set_input_shape(name, tensor.shape)
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle=stream)
        torch.cuda.current_stream().synchronize()
        results = {}
        for name, tensor in self.io_tensors.items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                results[name] = tensor
        return results

def compare_tensors(name, pt_out, trt_out):
    pt_np = pt_out.detach().cpu().float().numpy().flatten()
    trt_np = trt_out.detach().cpu().float().numpy().flatten()
    min_len = min(len(pt_np), len(trt_np))
    pt_np = pt_np[:min_len]
    trt_np = trt_np[:min_len]
    norm_pt = np.linalg.norm(pt_np)
    norm_trt = np.linalg.norm(trt_np)
    cos_sim = np.dot(pt_np, trt_np) / (norm_pt * norm_trt + 1e-6)
    abs_diff = np.abs(pt_np - trt_np)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    rel_err = mean_diff / (np.mean(np.abs(pt_np)) + 1e-6)
    
    return {
        "cos_sim": cos_sim,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_err": rel_err
    }

def print_statistics(name, stats_list):
    """Print aggregated statistics across all samples"""
    cos_sims = [s["cos_sim"] for s in stats_list]
    max_diffs = [s["max_diff"] for s in stats_list]
    mean_diffs = [s["mean_diff"] for s in stats_list]
    rel_errs = [s["rel_err"] for s in stats_list]
    
    print(f"\n{'='*60}")
    print(f"Layer: {name} (Aggregated over {len(stats_list)} samples)")
    print(f"{'='*60}")
    print(f"Cosine Similarity:")
    print(f"  -> Mean:  {np.mean(cos_sims):.5f}")
    print(f"  -> Min:   {np.min(cos_sims):.5f}")
    print(f"  -> Max:   {np.max(cos_sims):.5f}")
    print(f"  -> Std:   {np.std(cos_sims):.5f}")
    print(f"Max Absolute Difference:")
    print(f"  -> Mean:  {np.mean(max_diffs):.5f}")
    print(f"  -> Min:   {np.min(max_diffs):.5f}")
    print(f"  -> Max:   {np.max(max_diffs):.5f}")
    print(f"Mean Absolute Difference:")
    print(f"  -> Mean:  {np.mean(mean_diffs):.5f}")
    print(f"  -> Min:   {np.min(mean_diffs):.5f}")
    print(f"  -> Max:   {np.max(mean_diffs):.5f}")
    print(f"Relative Error:")
    print(f"  -> Mean:  {np.mean(rel_errs):.5f}")
    print(f"  -> Min:   {np.min(rel_errs):.5f}")
    print(f"  -> Max:   {np.max(rel_errs):.5f}")
    
    mean_cos_sim = np.mean(cos_sims)
    if mean_cos_sim < 0.99:
        print(f"\n[WARNING] Average cosine similarity ({mean_cos_sim:.5f}) < 0.99!")
        print("          Check quantization scale (amax) or calibration data quality.")
    else:
        print(f"\n[PASS] Average cosine similarity ({mean_cos_sim:.5f}) >= 0.99.")
    print(f"{'='*60}\n")

def load_all_calibration_data(data_dir, max_samples=None):
    """Load all calibration data files"""
    pattern = os.path.join(data_dir, "backbone", "*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No calibration data found in {pattern}")
        return []
    
    if max_samples is not None:
        files = files[:max_samples]
    
    print(f"[INFO] Found {len(files)} calibration data files. Loading...")
    inputs = []
    for i, fpath in enumerate(files):
        try:
            data = np.load(fpath)
            if "img" in data:
                img = torch.from_numpy(data["img"]).cuda()
                # NaN check
                img = torch.nan_to_num(img, nan=0.0)
                inputs.append(img)
            else:
                print(f"[WARN] File {fpath} does not contain 'img' key. Skipping.")
        except Exception as e:
            print(f"[WARN] Failed to load {fpath}: {e}. Skipping.")
    
    print(f"[INFO] Successfully loaded {len(inputs)} samples.")
    return inputs

def main():
    parser = argparse.ArgumentParser(description="Benchmark Backbone INT8 Quantization Error")
    parser.add_argument("--max-samples", type=int, default=None, 
                        help="Maximum number of calibration samples to test (default: all)")
    parser.add_argument("--engine", type=str, default="deploy/engine/sparse4dbackbone_int8.engine",
                        help="Path to TensorRT engine file")
    parser.add_argument("--calib-dir", type=str, default="deploy/calibration_data",
                        help="Directory containing calibration data")
    args = parser.parse_args()
    
    CFG_PATH = "dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py"
    CKPT_PATH = "ckpt/sparse4dv3_r50.pth"
    BACKBONE_ENGINE = args.engine
    CALIB_DATA_DIR = args.calib_dir
    
    if not os.path.exists(BACKBONE_ENGINE):
        print(f"[ERROR] Engine not found: {BACKBONE_ENGINE}")
        return

    print("[1/4] Loading PyTorch model...")
    cfg = read_cfg(CFG_PATH)
    if "img_backbone" in cfg["model"]:
        cfg["model"]["img_backbone"]["with_cp"] = False
    full_model = build_module(cfg["model"])
    load_checkpoint(full_model, CKPT_PATH, map_location="cpu")
    full_model.cuda().eval()
    pt_backbone = Sparse4DBackbone(full_model).cuda().eval()
    
    print(f"[2/4] Loading TensorRT Engine: {BACKBONE_ENGINE}...")
    trt_backbone = TRTWrapperTorch(BACKBONE_ENGINE)
    
    print("[3/4] Loading all calibration data...")
    all_inputs = load_all_calibration_data(CALIB_DATA_DIR, max_samples=args.max_samples)
    if not all_inputs:
        print("[WARN] No calibration data found. Using random input as fallback (Expect bad results!).")
        all_inputs = [torch.randn(1, 6, 3, 256, 704).cuda()]
    
    print(f"[4/4] Running inference & Comparing over {len(all_inputs)} samples...")
    stats_list = []
    
    for idx, real_input in enumerate(all_inputs):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Processing sample {idx + 1}/{len(all_inputs)}...")
        
        with torch.no_grad():
            pt_out = pt_backbone(real_input)
        
        trt_inputs = {"img": real_input}
        trt_outs = trt_backbone.infer(trt_inputs)
        
        if "feature" in trt_outs:
            trt_out_feat = trt_outs['feature']
            stats = compare_tensors("Backbone Output (Feature)", pt_out, trt_out_feat)
            stats_list.append(stats)
        else:
            print(f"[ERROR] Output 'feature' not found in Engine outputs: {list(trt_outs.keys())}")
            break
    
    if stats_list:
        print_statistics("Backbone Output (Feature)", stats_list)
    else:
        print("[ERROR] No valid statistics collected!")

if __name__ == "__main__":
    main()
