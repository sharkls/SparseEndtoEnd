import argparse
import os
import sys
import numpy as np
import tensorrt as trt
import torch
import ctypes
import time
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
/bin/python3 visualize_tmp.py --data_dir val_data_e2e_fp32 --plugin_dir .
"""

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from modules.head.sparse4d_blocks.sparse3d_embedding import SparseBox3DKeyPointsGenerator
    from dataset.config.nusc_std_bbox3d import *
except ImportError:
    print("[WARN] Could not import project modules. Visualization might fail.")
    # Mock constants if import fails
    X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))

def find_file(data_dir, name_pattern, sample_prefix="sample_0"):
    pattern = os.path.join(data_dir, f"{sample_prefix}_{name_pattern}_*.bin")
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(data_dir, f"{name_pattern}.bin")
        files = glob.glob(pattern)
    if not files and sample_prefix != "sample_0":
        pattern = os.path.join(data_dir, f"sample_0_{name_pattern}_*.bin")
        files = glob.glob(pattern)
    if not files: return None
    return files[0]

def load_bin(path, dtype=np.float32, shape=None):
    if not os.path.exists(path): return None
    data = np.fromfile(path, dtype=dtype)
    if shape is not None:
        try: data = data.reshape(shape)
        except: pass
    return data

class TRTWrapper:
    def __init__(self, engine_path, verbose=False):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(logger)
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

    def infer(self, input_map, output_shapes=None):
        inputs = []
        allocations = []
        
        for name in self.input_names:
            tensor = input_map[name]
            if isinstance(tensor, np.ndarray): tensor = torch.from_numpy(tensor).cuda()
            
            dtype_trt = self.engine.get_tensor_dtype(name)
            if dtype_trt == trt.DataType.HALF and tensor.dtype != torch.float16:
                tensor = tensor.half()
            elif dtype_trt == trt.DataType.FLOAT and tensor.dtype != torch.float32:
                tensor = tensor.float()
            
            self.context.set_input_shape(name, tensor.shape)
            inputs.append(tensor)
            allocations.append(tensor.data_ptr())

        output_map = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dims = []
            if output_shapes and name in output_shapes:
                dims = list(output_shapes[name])
            else:
                for s in shape: dims.append(1 if s < 0 else s)
            
            dtype_trt = self.engine.get_tensor_dtype(name)
            dtype_torch = torch.float32
            if dtype_trt == trt.DataType.HALF: dtype_torch = torch.float16
            elif dtype_trt == trt.DataType.INT32: dtype_torch = torch.int32
            
            out_tensor = torch.empty(tuple(dims), dtype=dtype_torch, device='cuda')
            allocations.append(out_tensor.data_ptr())
            output_map[name] = out_tensor

        for i, name in enumerate(self.input_names):
            self.context.set_tensor_address(name, inputs[i].data_ptr())
        for name in self.output_names:
            self.context.set_tensor_address(name, output_map[name].data_ptr())

        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        return output_map

# Simple Decoder based on modules/head/decoder/decoder.py
class SimpleDecoder:
    def __init__(self, num_output=300, score_threshold=0.3):
        self.num_output = num_output
        self.score_threshold = score_threshold

    def decode_box(self, box):
        # box: [N, 11] (X, Y, Z, W, L, H, SIN, COS, VX, VY, VZ)
        # Output: [N, 9] (X, Y, Z, W, L, H, YAW, VX, VY) (ignoring VZ for visualization)
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        decoded = torch.cat([
            box[:, [X, Y, Z]],
            box[:, [W, L, H]].exp(),
            yaw[:, None],
            box[:, VX:VY+1] # VX, VY
        ], dim=-1)
        return decoded

    def decode(self, cls_scores, box_preds, track_id=None):
        # cls_scores: [1, 900, 10]
        # box_preds: [1, 900, 11]
        # track_id: [1, 900]
        
        # 1. Score Processing
        scores = cls_scores[0].sigmoid() # [900, 10]
        max_scores, labels = scores.max(dim=-1) # [900], [900]
        
        # 2. TopK
        topk_scores, indices = max_scores.topk(self.num_output)
        
        # 3. Gather
        topk_labels = labels[indices]
        topk_boxes = box_preds[0][indices]
        
        if track_id is not None:
            topk_ids = track_id[0][indices]
        else:
            topk_ids = None
            
        # 4. Filter by threshold
        if self.score_threshold:
            mask = topk_scores >= self.score_threshold
            topk_scores = topk_scores[mask]
            topk_labels = topk_labels[mask]
            topk_boxes = topk_boxes[mask]
            if topk_ids is not None:
                topk_ids = topk_ids[mask]
                
        # 5. Decode Boxes
        if topk_boxes.shape[0] > 0:
            decoded_boxes = self.decode_box(topk_boxes)
        else:
            decoded_boxes = torch.zeros((0, 9)).to(topk_boxes)
        
        return {
            "boxes": decoded_boxes.cpu().numpy(),
            "scores": topk_scores.cpu().numpy(),
            "labels": topk_labels.cpu().numpy(),
            "ids": topk_ids.cpu().numpy() if topk_ids is not None else None
        }

def visualize_bev_comparison(gt_data, pytorch_res, engine_res, save_path="vis_comparison.png"):
    """
    Visualize GT, Pytorch Prediction, and Engine Prediction side-by-side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    
    titles = ["Ground Truth (Green)", "Pytorch Pred (Blue)", "Engine Pred (Red)"]
    colors = ['green', 'blue', 'red']
    
    for ax, title in zip(axes, titles):
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title(title)

    # 1. Ground Truth
    ax_gt = axes[0]
    if gt_data and len(gt_data["boxes"]) > 0:
        for i, box in enumerate(gt_data["boxes"]):
            # GT Box: [x, y, z, w, l, h, yaw, ...]
            if len(box) >= 7:
                x, y, z, w, l, h, yaw = box[:7]
                c, s = np.cos(yaw), np.sin(yaw)
                R = np.array([[c, -s], [s, c]])
                corners = np.array([
                    [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
                ])
                corners_global = (R @ corners.T).T + np.array([x, y])
                poly = patches.Polygon(corners_global, closed=True, fill=False, edgecolor=colors[0], linewidth=1.5)
                ax_gt.add_patch(poly)
                
                label = f"{int(gt_data['labels'][i])}" if gt_data['labels'] is not None else ""
                ax_gt.text(x, y, label, color=colors[0], fontsize=8, clip_on=True)

    # 2. Pytorch Prediction
    ax_pt = axes[1]
    if pytorch_res and len(pytorch_res["boxes"]) > 0:
        for i, (box, score) in enumerate(zip(pytorch_res["boxes"], pytorch_res["scores"])):
            x, y, z, w, l, h, yaw = box[:7]
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            corners = np.array([
                [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
            ])
            corners_global = (R @ corners.T).T + np.array([x, y])
            poly = patches.Polygon(corners_global, closed=True, fill=False, edgecolor=colors[1], linewidth=1.5)
            ax_pt.add_patch(poly)
            ax_pt.text(x, y+1, f"{score:.2f}", color=colors[1], fontsize=8, clip_on=True)

    # 3. Engine Prediction
    ax_eng = axes[2]
    if engine_res and len(engine_res["boxes"]) > 0:
        for i, (box, score) in enumerate(zip(engine_res["boxes"], engine_res["scores"])):
            x, y, z, w, l, h, yaw = box[:7]
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            corners = np.array([
                [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
            ])
            corners_global = (R @ corners.T).T + np.array([x, y])
            poly = patches.Polygon(corners_global, closed=True, fill=False, edgecolor=colors[2], linewidth=1.5)
            ax_eng.add_patch(poly)
            ax_eng.text(x, y+1, f"{score:.2f}", color=colors[2], fontsize=8, clip_on=True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Comparison saved to {save_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--plugin_dir", default="deploy")
    args = parser.parse_args()

    # Load Plugins
    plugins = [
        os.path.join(args.plugin_dir, "dfa_plugin/lib/deformableAttentionAggr.so"),
        os.path.join(args.plugin_dir, "ln_plugin/lib/customLayerNorm.so"),
        os.path.join(args.plugin_dir, "sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so"),
    ]
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")
    for p in plugins:
        if os.path.exists(p): ctypes.CDLL(p)
    
    print("Loading Engines...")
    engine_dir = os.path.join(os.path.dirname(__file__), "engine")
    backbone = TRTWrapper(os.path.join(engine_dir, "sparse4dbackbone.engine"))
    head1 = TRTWrapper(os.path.join(engine_dir, "sparse4dhead1st.engine"))
    head2 = TRTWrapper(os.path.join(engine_dir, "sparse4dhead2nd.engine"))
    
    decoder = SimpleDecoder(num_output=300, score_threshold=0.3)

    # --- Frame 1 Comparison & Validation ---
    print("\n" + "="*50)
    print("Running Frame 1: Full Offline Input Validation")
    print("="*50)
    
    head2_inputs = {}
    
    # 1. Load ALL inputs from offline files (PyTorch dumped)
    print(f"Loading Head2 Inputs from {args.data_dir} (sample_1)...")
    for name in head2.input_names:
        path = find_file(args.data_dir, name, "sample_1")
        if not path:
            print(f"[ERR] Input '{name}' not found for sample_1!")
            continue
            
        # Determine Dtype
        dtype = np.float32
        if "mask" in name or "id" in name or "spatial" in name or "level" in name:
            dtype = np.int32
            
        data = load_bin(path, dtype=dtype)
        
        # Determine Shape from filename if possible
        if data is not None:
            basename = os.path.basename(path)
            # Try to parse shape from filename like: sample_1_feature_1*89760*256_float32.bin
            try:
                parts = basename.split('_')
                shape_part = [p for p in parts if '*' in p]
                if shape_part:
                    dims = list(map(int, shape_part[0].split('*')))
                    data = data.reshape(dims)
            except:
                pass # Keep flat if parsing fails, TRT wrapper might handle or fail
            
            head2_inputs[name] = data
            # print(f"  - Loaded {name}: shape={data.shape}, dtype={data.dtype}")

    # 2. Run Engine Inference
    print("\nRunning Head2 Engine Inference...")
    head2_out = head2.infer(head2_inputs)
    
    # 3. Validate Outputs against Offline PyTorch Outputs
    print("\nValidating Outputs (Engine vs PyTorch Bin):")
    print("-" * 60)
    print(f"{'Output Name':<30} | {'Shape':<15} | {'Max Diff':<12} | {'Mean Diff':<12}")
    print("-" * 60)
    
    for name in head2.output_names:
        path = find_file(args.data_dir, name, "sample_1")
        if not path:
            print(f"{name:<30} | {'[Not Found]':<15} | {'-':<12} | {'-':<12}")
            continue
            
        dtype = np.int32 if "id" in name or "label" in name else np.float32
        pt_out = load_bin(path, dtype=dtype)
        
        eng_out = head2_out[name].cpu().numpy()
        
        # Try to reshape PT output to match Engine output
        try:
            pt_out = pt_out.reshape(eng_out.shape)
            if dtype == np.float32:
                diff = np.abs(eng_out - pt_out)
                max_diff = diff.max()
                mean_diff = diff.mean()
                print(f"{name:<30} | {str(eng_out.shape):<15} | {max_diff:<12.6f} | {mean_diff:<12.6f}")
            else:
                # For Ints/IDs, check exact match count or similar
                mismatch = (eng_out != pt_out).sum()
                print(f"{name:<30} | {str(eng_out.shape):<15} | Mismatch: {mismatch}")
        except Exception as e:
            print(f"{name:<30} | {str(eng_out.shape):<15} | Shape Error")

    # 4. Decode & Visualize
    print("\nDecoding & Visualizing...")
    
    # A. Decode Engine Result
    engine_res = decoder.decode(head2_out["pred_class_score"], head2_out["pred_anchor"], head2_out["pred_track_id"])
    print(f"Engine Detection: {len(engine_res['boxes'])} boxes")
    
    # B. Decode PyTorch Result (from bins)
    # Using the offline outputs we just verified against
    pt_cls = load_bin(find_file(args.data_dir, "pred_class_score", "sample_1"), shape=(1, 900, 10))
    pt_anchor = load_bin(find_file(args.data_dir, "pred_anchor", "sample_1"), shape=(1, 900, 11))
    pt_id = load_bin(find_file(args.data_dir, "pred_track_id", "sample_1"), shape=(1, 900), dtype=np.int32)
    
    if pt_cls is not None and pt_anchor is not None:
        pytorch_res = decoder.decode(
            torch.from_numpy(pt_cls).cuda(), 
            torch.from_numpy(pt_anchor).cuda(), 
            torch.from_numpy(pt_id).cuda() if pt_id is not None else None
        )
        print(f"PyTorch Detection: {len(pytorch_res['boxes'])} boxes")
    else:
        pytorch_res = {"boxes": [], "scores": []}
        print("PyTorch Detection: Failed to load raw head outputs")

    # C. Load GT (if available)
    gt_data = None
    path_gt_box = find_file(args.data_dir, "gt_boxes", "sample_1")
    path_gt_lbl = find_file(args.data_dir, "gt_labels", "sample_1")
    if path_gt_box:
        gt_boxes = load_bin(path_gt_box, shape=(-1, 9))
        gt_labels = load_bin(path_gt_lbl, dtype=np.int32)
        gt_data = {"boxes": gt_boxes, "labels": gt_labels}
        print(f"GT: {len(gt_boxes)} boxes")
    else:
        print("GT: Not found")

    visualize_bev_comparison(gt_data, pytorch_res, engine_res, "vis_comparison_frame1.png")



if __name__ == "__main__":
    main()

