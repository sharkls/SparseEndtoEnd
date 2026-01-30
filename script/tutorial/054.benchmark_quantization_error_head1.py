
import torch
import numpy as np
import tensorrt as trt
import os
import argparse
import sys
import glob
import copy
import ctypes
from typing import Optional, Dict, Any

# 引用项目路径
sys.path.append(os.getcwd())
from tool.utils.config import read_cfg
from tool.runner.checkpoint import load_checkpoint
from deploy.export_head_onnx_quant import Sparse4DHead1st

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
        self._load_plugins()
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

    def _load_plugins(self):
        plugin_paths = [
            "deploy/dfa_plugin/lib/deformableAttentionAggr.so",
            "deploy/ln_plugin/lib/customLayerNorm.so",
            "deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so"
        ]
        trt.init_libnvinfer_plugins(self.logger, "")
        for path in plugin_paths:
            if os.path.exists(path): ctypes.CDLL(path)
            else: print(f"[WARN] Plugin not found: {path}")

    def infer(self, input_map):
        for name, tensor in self.io_tensors.items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name in input_map:
                    src = input_map[name]
                    if isinstance(src, np.ndarray): src = torch.from_numpy(src).cuda()
                    if src.dtype != tensor.dtype: src = src.to(tensor.dtype)
                    if src.numel() == tensor.numel(): src = src.view_as(tensor)
                    tensor.copy_(src)
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
    if mean_cos_sim < 0.98:
        print(f"\n[WARNING] Average cosine similarity ({mean_cos_sim:.5f}) < 0.98!")
        print("          Check quantization scale (amax) or calibration data quality.")
    else:
        print(f"\n[PASS] Average cosine similarity ({mean_cos_sim:.5f}) >= 0.98.")
    print(f"{'='*60}\n")

def load_all_calibration_data(data_dir, max_samples=None):
    """Load all Head1 calibration data files"""
    pattern = os.path.join(data_dir, "head1", "*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No calibration data found in {pattern}")
        return []
    
    if max_samples is not None:
        files = files[:max_samples]
    
    print(f"[INFO] Found {len(files)} Head1 calibration data files. Loading...")
    all_inputs = []
    for i, fpath in enumerate(files):
        try:
            data = np.load(fpath)
            inputs = {}
            for k in data.files:
                tensor = torch.from_numpy(data[k]).cuda()
                # NaN check
                tensor = torch.nan_to_num(tensor, nan=0.0)
                inputs[k] = tensor
            all_inputs.append(inputs)
        except Exception as e:
            print(f"[WARN] Failed to load {fpath}: {e}. Skipping.")
    
    print(f"[INFO] Successfully loaded {len(all_inputs)} samples.")
    return all_inputs

def main():
    parser = argparse.ArgumentParser(description="Benchmark Head1 INT8 Quantization Error")
    parser.add_argument("--max-samples", type=int, default=None, 
                        help="Maximum number of calibration samples to test (default: all)")
    parser.add_argument("--engine", type=str, default="deploy/engine/sparse4dhead1st_int8.engine",
                        help="Path to TensorRT engine file")
    parser.add_argument("--calib-dir", type=str, default="deploy/calibration_data",
                        help="Directory containing calibration data")
    args = parser.parse_args()
    
    CFG_PATH = "dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py"
    CKPT_PATH = "ckpt/sparse4dv3_r50.pth"
    HEAD1_ENGINE = args.engine
    CALIB_DATA_DIR = args.calib_dir
    
    if not os.path.exists(HEAD1_ENGINE):
        print(f"[ERROR] Engine not found: {HEAD1_ENGINE}")
        return

    print("[1/4] Loading PyTorch model...")
    cfg = read_cfg(CFG_PATH)
    model = build_module(cfg["model"])
    load_checkpoint(model, CKPT_PATH, map_location="cpu")
    model.cuda().eval()
    pt_head1 = Sparse4DHead1st(copy.deepcopy(model))
    pt_head1.eval().cuda()
    
    print(f"[2/4] Loading TensorRT Engine: {HEAD1_ENGINE}...")
    trt_head1 = TRTWrapperTorch(HEAD1_ENGINE)
    
    print("[3/4] Loading all Head1 calibration data...")
    all_inputs = load_all_calibration_data(CALIB_DATA_DIR, max_samples=args.max_samples)
    if not all_inputs:
        print("[ERROR] No calibration data loaded!")
        return
    
    print(f"[4/4] Running inference & Comparing over {len(all_inputs)} samples...")
    
    # Statistics for each output
    stats_feature = []
    stats_anchor = []
    stats_class = []
    stats_quality = []
    
    for idx, inputs in enumerate(all_inputs):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Processing sample {idx + 1}/{len(all_inputs)}...")
        
        pt_inputs = {k: v.float() for k, v in inputs.items()}
        trt_inputs = {k: v.float() for k, v in inputs.items()}
        
        with torch.no_grad():
            pt_outs = pt_head1.head_forward(
                pt_head1.model.head,
                pt_inputs["feature"], pt_inputs["spatial_shapes"], pt_inputs["level_start_index"],
                pt_inputs["instance_feature"], pt_inputs["anchor"], pt_inputs["time_interval"],
                pt_inputs["image_wh"], pt_inputs["lidar2img"]
            )
        
        trt_outs = trt_head1.infer(trt_inputs)
        
        # Collect statistics for each output
        if "pred_instance_feature" in trt_outs:
            stats_feature.append(compare_tensors("pred_instance_feature", pt_outs[0], trt_outs["pred_instance_feature"]))
        if "pred_anchor" in trt_outs:
            stats_anchor.append(compare_tensors("pred_anchor", pt_outs[1], trt_outs["pred_anchor"]))
        if "pred_class_score" in trt_outs:
            stats_class.append(compare_tensors("pred_class_score", pt_outs[2], trt_outs["pred_class_score"]))
        if "pred_quality_score" in trt_outs:
            stats_quality.append(compare_tensors("pred_quality_score", pt_outs[3], trt_outs["pred_quality_score"]))
    
    # Print aggregated statistics
    if stats_feature:
        print_statistics("pred_instance_feature", stats_feature)
    if stats_anchor:
        print_statistics("pred_anchor", stats_anchor)
    if stats_class:
        print_statistics("pred_class_score", stats_class)
    if stats_quality:
        print_statistics("pred_quality_score", stats_quality)

if __name__ == "__main__":
    main()
