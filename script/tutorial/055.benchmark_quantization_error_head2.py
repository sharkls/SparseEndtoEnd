
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

sys.path.append(os.getcwd())
from tool.utils.config import read_cfg
from tool.runner.checkpoint import load_checkpoint
from deploy.export_head_onnx_quant import Sparse4DHead1st, Sparse4DHead2nd

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
    
    print(f"Layer: {name}")
    print(f"  -> Cosine Similarity: {cos_sim:.5f}")
    if cos_sim < 0.98:
        print("  [WARNING] Similarity < 0.98")
    else:
        print("  [PASS] High similarity.")

def load_npz(path):
    print(f"[INFO] Loading {path}")
    data = np.load(path)
    inputs = {}
    for k in data.files:
        inputs[k] = torch.from_numpy(data[k]).cuda()
    return inputs

def main():
    CFG_PATH = "dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py"
    CKPT_PATH = "ckpt/sparse4dv3_r50.pth"
    HEAD2_ENGINE = "deploy/engine/sparse4dhead2nd_int8.engine"
    
    # Locate data files for Frame 0 and Frame 1
    # Assuming standard structure: deploy/calibration_data/head1/000000.npz and head2/000001.npz
    f0_path = "deploy/calibration_data/head1/000000.npz"
    f1_path = "deploy/calibration_data/head2/000001.npz"
    
    if not os.path.exists(f0_path) or not os.path.exists(f1_path):
        print("[ERROR] Calibration data missing. Please run 051.generate_calibration_data.py first.")
        return

    print("[1/5] Loading PyTorch model...")
    cfg = read_cfg(CFG_PATH)
    model = build_module(cfg["model"])
    load_checkpoint(model, CKPT_PATH, map_location="cpu")
    model.cuda().eval()
    
    # Wrappers for PyTorch
    pt_head1 = Sparse4DHead1st(model) # Use same model instance to share state
    pt_head2 = Sparse4DHead2nd(model)
    
    print(f"[2/5] Loading TensorRT Engine: {HEAD2_ENGINE}...")
    trt_head2 = TRTWrapperTorch(HEAD2_ENGINE)
    
    # ---------------------------------------------------------
    # STEP 1: Run PyTorch Head 1 to warm up InstanceBank state
    # ---------------------------------------------------------
    print("[3/5] Running PyTorch Head 1 (Warmup State)...")
    inputs0 = load_npz(f0_path)
    inputs0 = {k: v.float() for k, v in inputs0.items()}
    
    with torch.no_grad():
        # This updates model.head.instance_bank internally
        _ = pt_head1.head_forward(
            model.head,
            inputs0["feature"], inputs0["spatial_shapes"], inputs0["level_start_index"],
            inputs0["instance_feature"], inputs0["anchor"], inputs0["time_interval"],
            inputs0["image_wh"], inputs0["lidar2img"]
        )

    # ---------------------------------------------------------
    # STEP 2: Run PyTorch Head 2
    # ---------------------------------------------------------
    print("[4/5] Running PyTorch Head 2...")
    inputs1 = load_npz(f1_path)
    inputs1 = {k: v.float() for k, v in inputs1.items()}
    
    with torch.no_grad():
        # NOTE: We use inputs from file, BUT we rely on InstanceBank's internal state for "next frame" logic 
        # if head_forward uses it.
        # Actually, head_forward inputs like temp_instance_feature are explicitly passed.
        # BUT, the ground truth temp_instance_feature in '000001.npz' matches what PyTorch produced in Frame 0.
        # So passing them explicitly is correct, AS LONG AS they match the InstanceBank state if InstanceBank is used.
        # Let's check Sparse4DHead2nd.head_forward again.
        # It uses: instance_feature, anchor, etc. from arguments.
        # It DOES NOT use self.instance_bank.cached_feature directly in computation, 
        # it only uses it to UPDATE the bank at the end.
        # 
        # WAIT: The key issue in the previous run was likely `mask` or `track_id` misalignment 
        # or the `temp_anchor` projection logic.
        
        pt_outs = pt_head2.head_forward(
            model.head,
            inputs1["feature"], inputs1["spatial_shapes"], inputs1["level_start_index"],
            inputs1["instance_feature"], inputs1["anchor"], inputs1["time_interval"],
            inputs1["temp_instance_feature"], inputs1["temp_anchor"], inputs1["mask"],
            inputs1["track_id"], inputs1["image_wh"], inputs1["lidar2img"]
        )

    # ---------------------------------------------------------
    # STEP 3: Run TensorRT Head 2
    # ---------------------------------------------------------
    print("[5/5] Running TensorRT Head 2...")
    trt_inputs = {k: v.float() for k, v in inputs1.items()}
    
    # Important: Mask and TrackID are usually INT32 in ONNX, but loaded as Float from .npz
    # We must ensure they are cast correctly for TRT input binding if TRT expects INT32.
    # Our TRTWrapper handles casting to the engine's expected dtype automatically.
    
    trt_outs = trt_head2.infer(trt_inputs)
    
    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------
    compare_tensors("pred_instance_feature", pt_outs[0], trt_outs["pred_instance_feature"])
    compare_tensors("pred_anchor", pt_outs[1], trt_outs["pred_anchor"])
    compare_tensors("pred_class_score", pt_outs[2], trt_outs["pred_class_score"])
    compare_tensors("pred_quality_score", pt_outs[3], trt_outs["pred_quality_score"])
    compare_tensors("pred_track_id", pt_outs[4], trt_outs["pred_track_id"])

if __name__ == "__main__":
    main()
