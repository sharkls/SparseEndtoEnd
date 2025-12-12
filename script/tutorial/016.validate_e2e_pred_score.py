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

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from modules.head.sparse4d_blocks.sparse3d_embedding import SparseBox3DKeyPointsGenerator
    from dataset.config.nusc_std_bbox3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ
except ImportError:
    print("[WARN] Could not import SparseBox3DKeyPointsGenerator. Ego motion compensation might fail.")
    X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))

# ============================================================================
# Simple Decoder for Detection Results
# ============================================================================
class SimpleDecoder:
    def __init__(self, num_output=300, score_threshold=0.3):
        self.num_output = num_output
        self.score_threshold = score_threshold

    def decode_box(self, box):
        """Decode anchor to box: [N, 11] -> [N, 9] (X, Y, Z, W, L, H, YAW, VX, VY)"""
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        decoded = torch.cat([
            box[:, [X, Y, Z]],
            box[:, [W, L, H]].exp(),
            yaw[:, None],
            box[:, VX:VY+1]
        ], dim=-1)
        return decoded

    def decode(self, cls_scores, box_preds, track_id=None):
        """
        Decode head outputs to detection results.
        cls_scores: [1, 900, 10], box_preds: [1, 900, 11], track_id: [1, 900]
        """
        scores = cls_scores[0].sigmoid()
        max_scores, labels = scores.max(dim=-1)
        
        topk_scores, indices = max_scores.topk(self.num_output)
        topk_labels = labels[indices]
        topk_boxes = box_preds[0][indices]
        
        if track_id is not None:
            topk_ids = track_id[0][indices]
        else:
            topk_ids = None
        
        if self.score_threshold:
            mask = topk_scores >= self.score_threshold
            topk_scores = topk_scores[mask]
            topk_labels = topk_labels[mask]
            topk_boxes = topk_boxes[mask]
            if topk_ids is not None:
                topk_ids = topk_ids[mask]
        
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

# ============================================================================
# Visualization Functions
# ============================================================================
def draw_boxes_on_ax(ax, boxes, scores=None, labels=None, color='red', linewidth=1.5, show_score=True):
    """Draw 3D boxes on BEV axis"""
    if boxes is None or len(boxes) == 0:
        return
    
    for i, box in enumerate(boxes):
        if len(box) >= 7:
            x, y, z, w, l, h, yaw = box[:7]
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            corners = np.array([
                [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
            ])
            corners_global = (R @ corners.T).T + np.array([x, y])
            poly = patches.Polygon(corners_global, closed=True, fill=False, 
                                   edgecolor=color, linewidth=linewidth)
            ax.add_patch(poly)
            
            # Add score label
            if show_score and scores is not None and i < len(scores):
                ax.text(x, y+1, f"{scores[i]:.2f}", color=color, fontsize=7, clip_on=True)

def visualize_bev_comparison(pytorch_res, engine_res, gt_data=None, save_path="vis_e2e_comparison.png", title_suffix=""):
    """
    Visualize PyTorch prediction and Engine prediction side-by-side in BEV.
    """
    n_cols = 3 if gt_data else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(10*n_cols, 10))
    
    # Common setup
    for ax in axes:
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    idx = 0
    
    # Ground Truth (if available)
    if gt_data:
        ax_gt = axes[idx]
        ax_gt.set_title(f"Ground Truth (Green) {title_suffix}", fontsize=12)
        if gt_data and "boxes" in gt_data and len(gt_data["boxes"]) > 0:
            draw_boxes_on_ax(ax_gt, gt_data["boxes"], color='green', show_score=False)
            ax_gt.text(-55, 55, f"Count: {len(gt_data['boxes'])}", fontsize=10, color='green')
        idx += 1
    
    # PyTorch Prediction
    ax_pt = axes[idx]
    ax_pt.set_title(f"PyTorch Prediction (Blue) {title_suffix}", fontsize=12)
    if pytorch_res and "boxes" in pytorch_res and len(pytorch_res["boxes"]) > 0:
        draw_boxes_on_ax(ax_pt, pytorch_res["boxes"], pytorch_res.get("scores"), color='blue')
        ax_pt.text(-55, 55, f"Count: {len(pytorch_res['boxes'])}", fontsize=10, color='blue')
    idx += 1
    
    # Engine Prediction
    ax_eng = axes[idx]
    ax_eng.set_title(f"Engine Prediction (Red) {title_suffix}", fontsize=12)
    if engine_res and "boxes" in engine_res and len(engine_res["boxes"]) > 0:
        draw_boxes_on_ax(ax_eng, engine_res["boxes"], engine_res.get("scores"), color='red')
        ax_eng.text(-55, 55, f"Count: {len(engine_res['boxes'])}", fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Visualization] Saved to: {save_path}")
    plt.close(fig)

def visualize_bev_overlay(pytorch_res, engine_res, gt_data=None, save_path="vis_e2e_overlay.png", title_suffix=""):
    """
    Overlay PyTorch and Engine predictions on the same plot for direct comparison.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"Overlay: GT(Green) / PyTorch(Blue) / Engine(Red) {title_suffix}", fontsize=12)
    
    # Ground Truth
    if gt_data and "boxes" in gt_data:
        draw_boxes_on_ax(ax, gt_data["boxes"], color='green', linewidth=2, show_score=False)
    
    # PyTorch Prediction
    if pytorch_res and "boxes" in pytorch_res:
        draw_boxes_on_ax(ax, pytorch_res["boxes"], pytorch_res.get("scores"), color='blue', linewidth=1.5)
    
    # Engine Prediction
    if engine_res and "boxes" in engine_res:
        draw_boxes_on_ax(ax, engine_res["boxes"], engine_res.get("scores"), color='red', linewidth=1)
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor='green', label=f'GT ({len(gt_data["boxes"]) if gt_data else 0})'),
        patches.Patch(facecolor='none', edgecolor='blue', label=f'PyTorch ({len(pytorch_res["boxes"]) if pytorch_res else 0})'),
        patches.Patch(facecolor='none', edgecolor='red', label=f'Engine ({len(engine_res["boxes"]) if engine_res else 0})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Visualization] Saved to: {save_path}")
    plt.close(fig)

# ============================================================================
# Data Loading Utilities
# ============================================================================
def find_file(data_dir, name_pattern, sample_prefix="sample_0"):
    pattern = os.path.join(data_dir, f"{sample_prefix}_{name_pattern}_*.bin")
    files = glob.glob(pattern)
    if not files:
        # Try direct match
        pattern = os.path.join(data_dir, f"{name_pattern}.bin")
        files = glob.glob(pattern)
    
    if not files and sample_prefix != "sample_0":
        # Fallback
        pattern = os.path.join(data_dir, f"sample_0_{name_pattern}_*.bin")
        files = glob.glob(pattern)

    if not files:
        return None
    return files[0]

def load_bin(path, dtype=np.float32, shape=None):
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return None
    data = np.fromfile(path, dtype=dtype)
    if shape is not None:
        try:
            data = data.reshape(shape)
        except:
            print(f"[WARN] Reshape failed for {path} to {shape}")
    return data

class TRTWrapper:
    def __init__(self, engine_path, verbose=False):
        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.input_names = []
        self.output_names = []
        
        print(f"\n[Info] inspecting engine: {engine_path}")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            
            dtype_str = str(dtype)
            if dtype == trt.DataType.FLOAT: dtype_str = "FP32"
            elif dtype == trt.DataType.HALF: dtype_str = "FP16"
            elif dtype == trt.DataType.INT32: dtype_str = "INT32"
            elif dtype == trt.DataType.INT8: dtype_str = "INT8"
            
            print(f"  Binding {i}: Name='{name}', Mode={mode}, Dtype={dtype_str}, Shape={shape}")
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, input_map, output_shapes=None):
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        # Prepare Inputs
        for name in self.input_names:
            if name not in input_map:
                print(f"[Error] Missing input: {name}. Available: {list(input_map.keys())}")
                raise ValueError(f"Missing input: {name}")
            
            tensor = input_map[name]
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor).cuda()
            
            # Check input dtype expectation
            dtype_trt = self.engine.get_tensor_dtype(name)
            if dtype_trt == trt.DataType.HALF and tensor.dtype != torch.float16:
                # print(f"  [Auto-Cast] Converting input '{name}' from {tensor.dtype} to FP16")
                tensor = tensor.half()
            elif dtype_trt == trt.DataType.FLOAT and tensor.dtype != torch.float32:
                # print(f"  [Auto-Cast] Converting input '{name}' from {tensor.dtype} to FP32")
                tensor = tensor.float()
            elif dtype_trt == trt.DataType.INT32 and tensor.dtype != torch.int32:
                # print(f"  [Auto-Cast] Converting input '{name}' from {tensor.dtype} to INT32")
                tensor = tensor.int()

            # Auto-Reshape based on Engine expectation
            shape_trt = self.engine.get_tensor_shape(name)
            if len(shape_trt) > 1 and len(tensor.shape) == 1:
                # Check if elements match
                vol_trt = 1
                for s in shape_trt:
                    if s > 0: vol_trt *= s
                
                # Handle dynamic batch size (usually dim 0 is -1)
                if shape_trt[0] == -1:
                    vol_trt = -1 
                
                if tensor.numel() == vol_trt or vol_trt == -1: # Simple check
                     # Construct target shape, assuming batch=1 for dynamic dim 0
                    target_shape = list(shape_trt)
                    if target_shape[0] == -1: target_shape[0] = 1
                    try:
                        tensor = tensor.reshape(target_shape)
                        # print(f"  [Auto-Reshape] Reshaped '{name}' to {target_shape}")
                    except:
                        pass

            # Set dynamic shape
            self.context.set_input_shape(name, tensor.shape)
            self.inputs.append(tensor)
            self.allocations.append(tensor.data_ptr())

        # Prepare Outputs
        output_map = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dims = []
            
            # Check if we have an override for this output
            if output_shapes and name in output_shapes:
                dims = list(output_shapes[name])
            else:
                vol = 1
                for s in shape:
                    if s < 0: 
                        dims.append(1) # fallback
                    else:
                        dims.append(s)
                        vol *= s
            
            dtype_trt = self.engine.get_tensor_dtype(name)
            dtype_torch = torch.float32
            if dtype_trt == trt.DataType.HALF: dtype_torch = torch.float16
            elif dtype_trt == trt.DataType.INT32: dtype_torch = torch.int32
            elif dtype_trt == trt.DataType.BOOL: dtype_torch = torch.bool
            
            out_tensor = torch.empty(tuple(dims), dtype=dtype_torch, device='cuda')
            self.allocations.append(out_tensor.data_ptr())
            output_map[name] = out_tensor

        # Set Addresses
        for i, name in enumerate(self.input_names):
            self.context.set_tensor_address(name, self.inputs[i].data_ptr())
        for name in self.output_names:
            self.context.set_tensor_address(name, output_map[name].data_ptr())

        # Execute
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        
        return output_map

def validate_tensor(name, pred, expected_path, dtype=np.float32, threshold=0.1):
    if expected_path is None:
        return
        
    expected = load_bin(expected_path, dtype=dtype)
    if expected is None:
        return
        
    expected = expected.reshape(pred.shape)
    
    if np.issubdtype(dtype, np.integer):
        acc = np.mean(pred == expected)
        print(f"  [{name}] Accuracy: {acc*100:.2f}%")
    else:
        diff = np.abs(pred - expected)
        max_err = np.max(diff)
        mean_err = np.mean(diff)
        
        pred_flat = pred.flatten().astype(np.float32)
        exp_flat = expected.flatten().astype(np.float32)
        cos_sim = 0
        if np.linalg.norm(pred_flat) > 0 and np.linalg.norm(exp_flat) > 0:
            cos_sim = np.dot(pred_flat, exp_flat) / (np.linalg.norm(pred_flat) * np.linalg.norm(exp_flat))
            
        status = "MATCH" if max_err < threshold else "MISMATCH"
        print(f"  [{name}] Max Error: {max_err:.6f}, Mean Error: {mean_err:.6f}, Cos Sim: {cos_sim:.6f} -> {status}")

        # Set-based Matching Validation for Anchors (Handling permutation issues)
        if "anchor" in name and max_err > 10.0: # Only check if error is large
            print(f"    -> Large error detected. Checking Set-based Consistency (ignoring order)...")
            # pred: (1, 900, 11), expected: (1, 900, 11)
            # Take only xyz coordinates
            p_xyz = pred[0, :, :3] # (900, 3)
            e_xyz = expected[0, :, :3] # (900, 3)
            
            # Simple Nearest Neighbor for validation
            # For each pred, find closest expected
            # Compute pairwise distance matrix
            # Note: This is O(N^2), 900^2 is small enough
            p_xyz_t = torch.from_numpy(p_xyz).cuda().float()
            e_xyz_t = torch.from_numpy(e_xyz).cuda().float()
            
            dist = torch.cdist(p_xyz_t.unsqueeze(0), e_xyz_t.unsqueeze(0))[0] # (900, 900)
            min_dist, _ = torch.min(dist, dim=1) # (900,)
            
            mean_match_err = torch.mean(min_dist).item()
            max_match_err = torch.max(min_dist).item()
            
            print(f"    [Set-Matching] Mean Min Distance: {mean_match_err:.4f}m, Max: {max_match_err:.4f}m")
            if mean_match_err < 0.5:
                print(f"    => CONCLUSION: VALID. The large error is purely due to ordering mismatch.")
            else:
                print(f"    => CONCLUSION: INVALID. The predictions are actually far from GT.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to validation data")
    parser.add_argument("--plugin_dir", default="deploy", help="Path to deploy dir")
    parser.add_argument("--visualize", action="store_true", default=True, help="Enable visualization")
    parser.add_argument("--output_dir", default="./visualize/e2e", help="Directory to save visualization images")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Score threshold for visualization")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create decoder for visualization
    decoder = SimpleDecoder(num_output=300, score_threshold=args.score_threshold)

    # 1. Load Plugins
    plugins = [
        os.path.join(args.plugin_dir, "dfa_plugin/lib/deformableAttentionAggr.so"),
        os.path.join(args.plugin_dir, "ln_plugin/lib/customLayerNorm.so"),
        os.path.join(args.plugin_dir, "sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so"),
    ]
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")
    for p in plugins:
        if os.path.exists(p):
            ctypes.CDLL(p)
    
    # 2. Load Engines
    print("Loading Engines...")
    backbone_path = "deploy/engine/sparse4dbackbone.engine"
    head1_path = "deploy/engine/sparse4dhead1st.engine"
    head2_path = "deploy/engine/sparse4dhead2nd.engine"
    
    backbone = TRTWrapper(backbone_path)
    head1 = TRTWrapper(head1_path)
    head2 = TRTWrapper(head2_path)
    
    # 3. Frame 0 Validation (Backbone + Head1)
    print("\n>>> Validating Frame 0 (Backbone + Head1)")
    # 3.1 Run Backbone
    print("Running Backbone (Frame 0)...")
    backbone_inputs = {}
    img_path = find_file(args.data_dir, "imgs", "sample_0")
    backbone_inputs["img"] = load_bin(img_path, shape=(1, 6, 3, 256, 704))
    
    backbone_output_shapes = {
        "feature": (1, 89760, 256),
        "spatial_shapes": (6, 4, 2),
        "level_start_index": (6, 4)
    }
    
    backbone_out = backbone.infer(backbone_inputs, output_shapes=backbone_output_shapes)
    
    # Verify Backbone Output
    feature_gt_path = find_file(args.data_dir, "feature", "sample_0")
    if "feature" in backbone_out:
        validate_tensor("Backbone Feature", backbone_out["feature"].cpu().numpy(), feature_gt_path, threshold=0.1)
    
    # 3.2 Run Head 1
    print("Running Head 1 (Frame 0)...")
    head1_inputs = {}
    
    for name, tensor in backbone_out.items():
        if name in head1.input_names:
            head1_inputs[name] = tensor
            
    shapes = {
        "instance_feature": (1, 900, 256),
        "anchor": (1, 900, 11),
        "time_interval": (1,),
        "image_wh": (1, 6, 2),
        "lidar2img": (1, 6, 4, 4),
        "spatial_shapes": (6, 4, 2),
        "level_start_index": (6, 4)
    }
    
    for name in head1.input_names:
        if name in head1_inputs: continue
        path = find_file(args.data_dir, name, "sample_0")
        if path:
            # Auto-detect dtype from engine to ensure correct reading of INT32 files
            dtype_trt = head1.engine.get_tensor_dtype(name)
            dtype = np.float32
            if dtype_trt == trt.DataType.INT32: dtype = np.int32
            
            head1_inputs[name] = load_bin(path, shape=shapes.get(name), dtype=dtype)
        else:
            print(f"Warning: Missing input {name} for Head1")

    head1_out = head1.infer(head1_inputs)

    # ---------------------------------------------------------
    # Implement InstanceBank Logic (Ego-Motion Compensation + TopK)
    # ---------------------------------------------------------
    print("\n[InstanceBank] Calculating Ego-Motion Compensation...")
    
    # 0. TopK Selection (Using Predicted Scores)
    k = 600
    
    print("  [InstanceBank] Using Predicted Class Scores for TopK Selection (Engine behavior)")
    pred_cls = head1_out["pred_class_score"] 
    # Sigmoid -> Max
    confidence = pred_cls.sigmoid().max(dim=-1).values # (1, 900)
    
    # InstanceBank.py: confidence = torch.maximum(self.confidence * self.confidence_decay, confidence)
    # For Frame 0, self.confidence is None, so we just use predicted confidence.
    # Note: C++ uses applyConfidenceDecay logic even for first frame if not careful, 
    # but logically for first frame fusion is skipped or fused with 0.
    
    # Get Top-K
    topk_conf, topk_indices = torch.topk(confidence, k, dim=1) # (1, 600)
    
    pred_feature = head1_out["pred_instance_feature"] # (1, 900, 256)
    pred_anchor = head1_out["pred_anchor"] # (1, 900, 11)
    
    batch_size = 1
    def gather_topk(tensor, indices):
        dim = tensor.shape[-1]
        indices_expanded = indices.unsqueeze(-1).expand(batch_size, k, dim)
        return torch.gather(tensor, 1, indices_expanded)

    temp_instance_feature = gather_topk(pred_feature, topk_indices)
    temp_anchor = gather_topk(pred_anchor, topk_indices)
    
    print(f"  Selected Top-{k} instances from 900 queries.")

    # 1. Load Pose Matrices
    l2g_0 = None
    path_l2g_0 = find_file(args.data_dir, "lidar2global", "sample_0")
    if path_l2g_0:
        l2g_0 = load_bin(path_l2g_0, shape=(1, 4, 4), dtype=np.float64)
    else:
        path_g2l_0 = find_file(args.data_dir, "ibank_global2lidar", "sample_0")
        if not path_g2l_0:
            path_g2l_0 = find_file(args.data_dir, "global2lidar", "sample_0")
            
        if path_g2l_0:
             g2l_0 = load_bin(path_g2l_0, shape=(1, 4, 4), dtype=np.float64)
             l2g_0 = np.linalg.inv(g2l_0)

    g2l_1 = None
    path_g2l_1 = find_file(args.data_dir, "ibank_global2lidar", "sample_1")
    if not path_g2l_1:
        path_g2l_1 = find_file(args.data_dir, "global2lidar", "sample_1")
        
    if path_g2l_1:
        g2l_1 = load_bin(path_g2l_1, shape=(1, 4, 4), dtype=np.float64)
    else:
        path_l2g_1 = find_file(args.data_dir, "lidar2global", "sample_1")
        if path_l2g_1:
            l2g_1 = load_bin(path_l2g_1, shape=(1, 4, 4), dtype=np.float64)
            g2l_1 = np.linalg.inv(l2g_1)

    # 2. Load Timestamps to compute dt
    dt = 0.5
    path_ts_0 = find_file(args.data_dir, "ibank_timestamp", "sample_0")
    if not path_ts_0: path_ts_0 = find_file(args.data_dir, "timestamp", "sample_0")
    
    path_ts_1 = find_file(args.data_dir, "ibank_timestamp", "sample_1")
    if not path_ts_1: path_ts_1 = find_file(args.data_dir, "timestamp", "sample_1")
    
    if path_ts_0 and path_ts_1:
        ts0_data = np.fromfile(path_ts_0, dtype=np.float64)
        ts1_data = np.fromfile(path_ts_1, dtype=np.float64)
        
        if ts0_data.size > 0 and ts1_data.size > 0:
             dt = ts1_data[0] - ts0_data[0]
             if dt > 1e5: # us to s
                 dt /= 1e6
    
    T_temp2cur = None
    if l2g_0 is not None and g2l_1 is not None:
        l2g_0_t = torch.from_numpy(l2g_0).cuda().float()
        g2l_1_t = torch.from_numpy(g2l_1).cuda().float()
        
        # T_temp2cur = Global2Lidar(t) @ Lidar2Global(t-1)
        T_temp2cur = torch.matmul(g2l_1_t, l2g_0_t) # (1, 4, 4)
        
        print(f"  dt: {dt:.4f}s")
        print("  Applying Anchor Projection...")
        
        temp_anchor_before = temp_anchor.clone()
        projected_anchors = SparseBox3DKeyPointsGenerator.anchor_projection(
            temp_anchor,
            [T_temp2cur],
            time_intervals=[-dt]
        )
        temp_anchor = projected_anchors[0]
        
        # DEBUG: Check movement
        diff = torch.norm(temp_anchor[..., :3] - temp_anchor_before[..., :3], dim=-1)
        mean_mov = diff.mean().item()
        max_mov = diff.max().item()
        print(f"  [DEBUG] Mean anchor movement: {mean_mov:.4f}m, Max: {max_mov:.4f}m")
        
    else:
        print("  [WARN] Missing lidar2global/global2lidar matrices. Skipping compensation!")

    # Compare calculated temp_anchor with file temp_anchor (Ground Truth for Frame 1)
    path_temp_anchor_gt = find_file(args.data_dir, "temp_anchor", "sample_1")
    if path_temp_anchor_gt:
        print("\n  [DEBUG] Verifying Calculated Temp Anchor vs File...")
        # Check if GT file differs from Uncompensated Anchor
        gt_temp_anchor = load_bin(path_temp_anchor_gt, shape=(1, 600, 11))
        gt_temp_anchor_t = torch.from_numpy(gt_temp_anchor).cuda()
        
        # Compare GT with Before Compensation
        if 'temp_anchor_before' in locals():
            diff_uncomp = torch.norm(gt_temp_anchor_t[..., :3] - temp_anchor_before[..., :3], dim=-1).mean().item()
            print(f"  [DEBUG] GT vs Uncompensated Anchor Mean Diff: {diff_uncomp:.4f}m")
        
        validate_tensor("Temp Anchor (Calculated vs File)", temp_anchor.cpu().numpy(), path_temp_anchor_gt, threshold=0.5)

    prev_frame_state = {
        "temp_instance_feature": temp_instance_feature,
        "temp_anchor": temp_anchor
    }
    
    # Verify Head 1 Outputs
    expected_outputs = ["pred_instance_feature", "pred_anchor", "pred_quality_score"]
    for name in expected_outputs:
        if name in head1_out:
            path = find_file(args.data_dir, name, "sample_0")
            validate_tensor(f"Head1 {name}", head1_out[name].cpu().numpy(), path)

    # =====================================================================
    # Frame 0 Visualization
    # =====================================================================
    if args.visualize:
        print("\n[Visualization] Generating Frame 0 visualization...")
        
        # Decode Engine results
        engine_res_f0 = decoder.decode(
            head1_out["pred_class_score"],
            head1_out["pred_anchor"],
            None  # No track_id for head1
        )
        print(f"  Engine Detection (Frame 0): {len(engine_res_f0['boxes'])} boxes")
        
        # Decode PyTorch results (from saved bins)
        pt_cls_f0 = load_bin(find_file(args.data_dir, "pred_class_score", "sample_0"), shape=(1, 900, 10))
        pt_anchor_f0 = load_bin(find_file(args.data_dir, "pred_anchor", "sample_0"), shape=(1, 900, 11))
        
        if pt_cls_f0 is not None and pt_anchor_f0 is not None:
            pytorch_res_f0 = decoder.decode(
                torch.from_numpy(pt_cls_f0).cuda(),
                torch.from_numpy(pt_anchor_f0).cuda(),
                None
            )
            print(f"  PyTorch Detection (Frame 0): {len(pytorch_res_f0['boxes'])} boxes")
        else:
            pytorch_res_f0 = {"boxes": [], "scores": []}
            print("  PyTorch Detection (Frame 0): Failed to load")
        
        # Load GT (if available)
        gt_data_f0 = None
        path_gt_box_f0 = find_file(args.data_dir, "gt_boxes", "sample_0")
        path_gt_lbl_f0 = find_file(args.data_dir, "gt_labels", "sample_0")
        if path_gt_box_f0:
            gt_boxes_f0 = load_bin(path_gt_box_f0, shape=(-1, 9))
            gt_labels_f0 = load_bin(path_gt_lbl_f0, dtype=np.int32) if path_gt_lbl_f0 else None
            gt_data_f0 = {"boxes": gt_boxes_f0, "labels": gt_labels_f0}
            print(f"  GT (Frame 0): {len(gt_boxes_f0)} boxes")
        
        # Generate visualizations
        save_path_f0 = os.path.join(args.output_dir, "vis_e2e_frame0_comparison.png")
        visualize_bev_comparison(pytorch_res_f0, engine_res_f0, gt_data_f0, save_path_f0, "(Frame 0 - Head1)")
        
        save_path_f0_overlay = os.path.join(args.output_dir, "vis_e2e_frame0_overlay.png")
        visualize_bev_overlay(pytorch_res_f0, engine_res_f0, gt_data_f0, save_path_f0_overlay, "(Frame 0 - Head1)")

    # 4. Frame 1 Validation (Backbone + Head2)
    print("\n>>> Validating Frame 1 (Backbone + Head2) [Closed-Loop]")
    # 4.1 Run Backbone
    print("Running Backbone (Frame 1)...")
    backbone_inputs = {}
    img_path = find_file(args.data_dir, "imgs", "sample_1")
    backbone_inputs["img"] = load_bin(img_path, shape=(1, 6, 3, 256, 704))
    
    backbone_output_shapes = {
        "feature": (1, 89760, 256),
        "spatial_shapes": (6, 4, 2),
        "level_start_index": (6, 4)
    }
    
    backbone_out = backbone.infer(backbone_inputs, output_shapes=backbone_output_shapes)
    
    # Verify Backbone Output
    feature_gt_path = find_file(args.data_dir, "feature", "sample_1")
    if "feature" in backbone_out:
        validate_tensor("Backbone Feature", backbone_out["feature"].cpu().numpy(), feature_gt_path, threshold=0.1)

    # 4.2 Run Head 2
    print("Running Head 2 (Frame 1)...")
    head2_inputs = {}
    
    for name, tensor in backbone_out.items():
        if name in head2.input_names:
            head2_inputs[name] = tensor
            
    # Load other inputs from file BUT override temp_* with prev_frame_state
    shapes.update({
        "temp_instance_feature": (1, 600, 256),
        "temp_anchor": (1, 600, 11),
        "mask": (1,),
        "track_id": (1, 900),
    })
    
    for name in head2.input_names:
        if name in head2_inputs: continue
        
        # Override with previous frame state if available
        if name in prev_frame_state:
            tensor = prev_frame_state[name]
            if tensor.shape[1] != 600:
                 print(f"  [WARN] {name} shape mismatch: {tensor.shape}. Expected 600.")
            head2_inputs[name] = tensor
            print(f"  Using {name} from Frame 0 Output (Compensated & TopK)")
            continue

        path = find_file(args.data_dir, name, "sample_1")
        if path:
            # Auto-detect dtype from engine
            dtype_trt = head2.engine.get_tensor_dtype(name)
            dtype = np.float32
            if dtype_trt == trt.DataType.INT32: dtype = np.int32
            
            head2_inputs[name] = load_bin(path, shape=shapes.get(name), dtype=dtype)
        else:
             print(f"Warning: Missing input {name} for Head2")

    head2_out = head2.infer(head2_inputs)
    
    # Verify Head 2 Outputs
    expected_outputs.append("pred_track_id")
    for name in expected_outputs:
        if name in head2_out:
            path = find_file(args.data_dir, name, "sample_1")
            dtype = np.float32
            if "id" in name: dtype = np.int32
            validate_tensor(f"Head2 {name}", head2_out[name].cpu().numpy(), path, dtype=dtype)

    # =====================================================================
    # Frame 1 Visualization
    # =====================================================================
    if args.visualize:
        print("\n[Visualization] Generating Frame 1 visualization...")
        
        # Decode Engine results
        engine_res_f1 = decoder.decode(
            head2_out["pred_class_score"],
            head2_out["pred_anchor"],
            head2_out.get("pred_track_id")
        )
        print(f"  Engine Detection (Frame 1): {len(engine_res_f1['boxes'])} boxes")
        
        # Decode PyTorch results (from saved bins)
        pt_cls_f1 = load_bin(find_file(args.data_dir, "pred_class_score", "sample_1"), shape=(1, 900, 10))
        pt_anchor_f1 = load_bin(find_file(args.data_dir, "pred_anchor", "sample_1"), shape=(1, 900, 11))
        pt_id_f1 = load_bin(find_file(args.data_dir, "pred_track_id", "sample_1"), shape=(1, 900), dtype=np.int32)
        
        if pt_cls_f1 is not None and pt_anchor_f1 is not None:
            pytorch_res_f1 = decoder.decode(
                torch.from_numpy(pt_cls_f1).cuda(),
                torch.from_numpy(pt_anchor_f1).cuda(),
                torch.from_numpy(pt_id_f1).cuda() if pt_id_f1 is not None else None
            )
            print(f"  PyTorch Detection (Frame 1): {len(pytorch_res_f1['boxes'])} boxes")
        else:
            pytorch_res_f1 = {"boxes": [], "scores": []}
            print("  PyTorch Detection (Frame 1): Failed to load")
        
        # Load GT (if available)
        gt_data_f1 = None
        path_gt_box_f1 = find_file(args.data_dir, "gt_boxes", "sample_1")
        path_gt_lbl_f1 = find_file(args.data_dir, "gt_labels", "sample_1")
        if path_gt_box_f1:
            gt_boxes_f1 = load_bin(path_gt_box_f1, shape=(-1, 9))
            gt_labels_f1 = load_bin(path_gt_lbl_f1, dtype=np.int32) if path_gt_lbl_f1 else None
            gt_data_f1 = {"boxes": gt_boxes_f1, "labels": gt_labels_f1}
            print(f"  GT (Frame 1): {len(gt_boxes_f1)} boxes")
        
        # Generate visualizations
        save_path_f1 = os.path.join(args.output_dir, "vis_e2e_frame1_comparison.png")
        visualize_bev_comparison(pytorch_res_f1, engine_res_f1, gt_data_f1, save_path_f1, "(Frame 1 - Head2)")
        
        save_path_f1_overlay = os.path.join(args.output_dir, "vis_e2e_frame1_overlay.png")
        visualize_bev_overlay(pytorch_res_f1, engine_res_f1, gt_data_f1, save_path_f1_overlay, "(Frame 1 - Head2)")

    # 5. Frame 2 Validation (Backbone + Head2)
    print("\n>>> Validating Frame 2 (Backbone + Head2) [Closed-Loop Continue]")
    # 5.1 Run Backbone
    print("Running Backbone (Frame 2)...")
    backbone_inputs_f2 = {}
    img_path_f2 = find_file(args.data_dir, "imgs", "sample_2")
    if img_path_f2:
        backbone_inputs_f2["img"] = load_bin(img_path_f2, shape=(1, 6, 3, 256, 704))
        backbone_out_f2 = backbone.infer(backbone_inputs_f2, output_shapes=backbone_output_shapes)
        
        # Verify Backbone Output
        feature_gt_path_f2 = find_file(args.data_dir, "feature", "sample_2")
        if "feature" in backbone_out_f2:
            validate_tensor("Backbone Feature (Frame 2)", backbone_out_f2["feature"].cpu().numpy(), feature_gt_path_f2, threshold=0.1)
    else:
        print("  [WARN] Frame 2 image not found. Skipping Frame 2 Validation.")
        backbone_out_f2 = None
    
    # =====================================================================
    # PURE INFERENCE SIMULATION (Head1 -> Head2 -> Head2)
    # =====================================================================
    print("\n>>> Running Pure Inference Simulation (Head1 -> Head2 -> Head2)")
    print("    (Simulating deployment flow without using GT for alignment)")
    
    # 1. Use Head1 Outputs
    sim_pred_cls = head1_out["pred_class_score"]      # (1, 900, 10)
    sim_pred_feat = head1_out["pred_instance_feature"] # (1, 900, 256)
    sim_pred_anchor = head1_out["pred_anchor"]         # (1, 900, 11)
    
    # 2. Score & ID Generation (Simulate get_track_id for Frame 0)
    # Threshold from decoder config (default 0.3 here)
    sim_scores = sim_pred_cls.sigmoid().max(dim=-1).values # (1, 900)
    
    sim_track_id = torch.full((1, 900), -1, dtype=torch.int32).cuda()
    sim_mask = sim_scores >= args.score_threshold
    num_new = sim_mask.sum()
    if num_new > 0:
        new_ids = torch.arange(num_new, dtype=torch.int32).cuda() # start from 0
        sim_track_id[sim_mask] = new_ids
        
    print(f"  [Sim] Generated {num_new} track IDs for Frame 0 (Thresh={args.score_threshold})")

    # 3. TopK Selection (Simulate InstanceBank.cache)
    k_sim = 600
    
    # [Consistency Update] Add logic for Confidence Decay & Fusion (Same as InstanceBank.cache)
    confidence_decay = 0.6
    
    # Simulate previous frame confidence (For Frame 0, it's None)
    # In a real loop, this would come from the previous iteration's 'sim_topk_scores'
    prev_cached_confidence = None 
    
    if prev_cached_confidence is not None:
        print("  [Sim] Applying Confidence Decay & Fusion...")
        # Only fuse the recurrent queries (first k_sim)
        # Note: 'sim_scores' here is the full 900 queries from current frame
        sim_scores[:, :k_sim] = torch.maximum(
            prev_cached_confidence * confidence_decay,
            sim_scores[:, :k_sim]
        )
    else:
        print("  [Sim] First frame detected (or no history). Skipping Confidence Fusion.")

    sim_topk_scores, sim_topk_indices = torch.topk(sim_scores, k_sim, dim=1) # (1, 600)
    
    def gather_sim(tensor, indices):
        dim = tensor.shape[-1]
        indices_expanded = indices.unsqueeze(-1).expand(1, k_sim, dim)
        return torch.gather(tensor, 1, indices_expanded)

    sim_temp_feature = gather_sim(sim_pred_feat, sim_topk_indices)
    sim_temp_anchor = gather_sim(sim_pred_anchor, sim_topk_indices)
    
    # Track ID for next frame input: Gathered IDs padded to 900
    sim_temp_track_id = torch.gather(sim_track_id, 1, sim_topk_indices) # (1, 600)
    sim_input_track_id = torch.full((1, 900), -1, dtype=torch.int32).cuda()
    sim_input_track_id[:, :k_sim] = sim_temp_track_id
    
    # 4. Projection (Simulate InstanceBank.get)
    # Using same T and dt as calculated before
    if l2g_0 is not None and g2l_1 is not None and T_temp2cur is not None:
        print(f"  [Sim] Projecting anchors (Frame 0->1) dt={dt:.4f}s")
        sim_temp_anchor = SparseBox3DKeyPointsGenerator.anchor_projection(
            sim_temp_anchor,
            [T_temp2cur],
            time_intervals=[-dt]
        )[0]
    else:
        print("  [Sim] WARN: Skipping projection (missing transforms)")

    # 5. Run Head2 with Simulated Inputs
    sim_head2_inputs = {}
    # Base inputs from Frame 1
    for name, tensor in backbone_out.items():
        if name in head2.input_names:
            sim_head2_inputs[name] = tensor
            
    # Load static inputs for Frame 1
    for name in head2.input_names:
        if name in sim_head2_inputs: continue
        if name in ["temp_instance_feature", "temp_anchor", "track_id", "mask"]: continue # Skip IB inputs for now
        
        path = find_file(args.data_dir, name, "sample_1")
        if path:
            dtype_trt = head2.engine.get_tensor_dtype(name)
            dtype = np.float32
            if dtype_trt == trt.DataType.INT32: dtype = np.int32
            sim_head2_inputs[name] = load_bin(path, shape=shapes.get(name), dtype=dtype)
    
    # Override with Simulated InstanceBank inputs
    sim_head2_inputs["temp_instance_feature"] = sim_temp_feature
    sim_head2_inputs["temp_anchor"] = sim_temp_anchor
    sim_head2_inputs["track_id"] = sim_input_track_id
    sim_head2_inputs["mask"] = torch.tensor([1], dtype=torch.int32).cuda() # mask=1 means history valid
    
    # Also verify time_interval matches dt (if it's an input)
    if "time_interval" in sim_head2_inputs:
         # Overwrite loaded time_interval with calculated dt to be consistent
         sim_head2_inputs["time_interval"] = torch.tensor([dt], dtype=torch.float32).cuda()

    print("  [Sim] Running Head2 Inference (Frame 1)...")
    sim_head2_out = head2.infer(sim_head2_inputs)
    
    # 6. Visualize Simulated Results
    if args.visualize:
        sim_res_f1 = decoder.decode(
            sim_head2_out["pred_class_score"],
            sim_head2_out["pred_anchor"],
            sim_head2_out.get("pred_track_id")
        )
        print(f"  [Sim] Engine Detection (Frame 1): {len(sim_res_f1['boxes'])} boxes")
        
        save_path_sim = os.path.join(args.output_dir, "vis_e2e_frame1_sim_comparison.png")
        visualize_bev_comparison(pytorch_res_f1, sim_res_f1, gt_data_f1, save_path_sim, "(Frame 1 - Pure Sim)")
        
        save_path_sim_overlay = os.path.join(args.output_dir, "vis_e2e_frame1_sim_overlay.png")
        visualize_bev_overlay(pytorch_res_f1, sim_res_f1, gt_data_f1, save_path_sim_overlay, "(Frame 1 - Pure Sim)")

    # =====================================================================
    # Frame 1 -> Frame 2 Transition (Logic Verification)
    # =====================================================================
    if backbone_out_f2:
        print("\n[Sim] Transitioning Frame 1 -> Frame 2...")
        
        # 1. Get Frame 1 Outputs
        sim_pred_cls_f1 = sim_head2_out["pred_class_score"]       # (1, 900, 10)
        sim_pred_feat_f1 = sim_head2_out["pred_instance_feature"] # (1, 900, 256)
        sim_pred_anchor_f1 = sim_head2_out["pred_anchor"]         # (1, 900, 11)
        
        # 2. Score & ID Generation (Frame 1)
        sim_scores_f1 = sim_pred_cls_f1.sigmoid().max(dim=-1).values # (1, 900)
        
        # Track ID from Frame 1 Output (Engine generated IDs)
        sim_track_id_f1 = sim_head2_out["pred_track_id"]
        
        # 3. Confidence Fusion (Crucial Step!)
        print("  [Sim] Applying Confidence Decay & Fusion (Frame 1)...")
        
        # Fusion Logic:
        # recurrent_scores = max(prev_topk_scores * decay, current_scores[:600])
        sim_scores_f1_fused = sim_scores_f1.clone()
        
        fused_part = torch.maximum(
            sim_topk_scores * confidence_decay, # sim_topk_scores is from Frame 0 selection
            sim_scores_f1[:, :k_sim]
        )
        sim_scores_f1_fused[:, :k_sim] = fused_part
        
        # Verify if Fusion changed anything
        diff_fusion = (sim_scores_f1_fused - sim_scores_f1).abs().sum().item()
        print(f"  [Sim] Fusion Update Sum Diff: {diff_fusion:.4f}")
        if diff_fusion > 0:
            print("  [Sim] => Fusion Logic Active! History influenced current scores.")
        else:
            print("  [Sim] => Fusion Logic Inactive (Current scores higher or decay too strong).")
        
        # 4. TopK Selection (Frame 1)
        sim_topk_scores_f2, sim_topk_indices_f2 = torch.topk(sim_scores_f1_fused, k_sim, dim=1)
        
        sim_temp_feature_f2 = gather_sim(sim_pred_feat_f1, sim_topk_indices_f2)
        sim_temp_anchor_f2 = gather_sim(sim_pred_anchor_f1, sim_topk_indices_f2)
        
        # Track ID for Frame 2 input
        sim_temp_track_id_f2 = torch.gather(sim_track_id_f1, 1, sim_topk_indices_f2)
        sim_input_track_id_f2 = torch.full((1, 900), -1, dtype=torch.int32).cuda()
        sim_input_track_id_f2[:, :k_sim] = sim_temp_track_id_f2
        
        # 5. Projection (Frame 1 -> Frame 2)
        path_g2l_2 = find_file(args.data_dir, "ibank_global2lidar", "sample_2")
        if not path_g2l_2: path_g2l_2 = find_file(args.data_dir, "global2lidar", "sample_2")
        
        if g2l_1 is not None and path_g2l_2:
            g2l_2 = load_bin(path_g2l_2, shape=(1, 4, 4), dtype=np.float64)
            # Re-calculate inverse if needed or assume g2l_1 valid
            l2g_1_t = torch.inverse(torch.from_numpy(g2l_1)).cuda().float() 
            g2l_2_t = torch.from_numpy(g2l_2).cuda().float()
            
            T_1to2 = torch.matmul(g2l_2_t, l2g_1_t)
            
            # Calculate dt for Frame 2
            dt_f2 = 0.5 
            path_ts_2 = find_file(args.data_dir, "timestamp", "sample_2")
            if path_ts_1 and path_ts_2:
                 ts2 = np.fromfile(path_ts_2, dtype=np.float64)[0]
                 ts1 = np.fromfile(path_ts_1, dtype=np.float64)[0]
                 dt_f2 = (ts2 - ts1) / 1e6
            
            print(f"  [Sim] Projecting anchors (Frame 1->2) dt={dt_f2:.4f}s")
            sim_temp_anchor_f2 = SparseBox3DKeyPointsGenerator.anchor_projection(
                sim_temp_anchor_f2,
                [T_1to2],
                time_intervals=[-dt_f2]
            )[0]
        else:
            print("  [Sim] Missing transforms for Frame 2. Skipping projection.")
            T_1to2 = None

        # 6. Run Head 2 (Frame 2)
        print("  [Sim] Running Head2 Inference (Frame 2)...")
        sim_head2_inputs_f2 = {}
        
        for name, tensor in backbone_out_f2.items():
            if name in head2.input_names:
                sim_head2_inputs_f2[name] = tensor
        
        for name in head2.input_names:
            if name in sim_head2_inputs_f2: continue
            if name in ["temp_instance_feature", "temp_anchor", "track_id", "mask"]: continue
            
            path = find_file(args.data_dir, name, "sample_2")
            if path:
                dtype_trt = head2.engine.get_tensor_dtype(name)
                dtype = np.float32
                if dtype_trt == trt.DataType.INT32: dtype = np.int32
                sim_head2_inputs_f2[name] = load_bin(path, shape=shapes.get(name), dtype=dtype)

        sim_head2_inputs_f2["temp_instance_feature"] = sim_temp_feature_f2
        sim_head2_inputs_f2["temp_anchor"] = sim_temp_anchor_f2
        sim_head2_inputs_f2["track_id"] = sim_input_track_id_f2
        sim_head2_inputs_f2["mask"] = torch.tensor([1], dtype=torch.int32).cuda()
        
        if "time_interval" in sim_head2_inputs_f2:
             sim_head2_inputs_f2["time_interval"] = torch.tensor([dt_f2], dtype=torch.float32).cuda()

        sim_head2_out_f2 = head2.infer(sim_head2_inputs_f2)
        
        # Visualize Frame 2
        if args.visualize:
             sim_res_f2 = decoder.decode(
                sim_head2_out_f2["pred_class_score"],
                sim_head2_out_f2["pred_anchor"],
                sim_head2_out_f2.get("pred_track_id")
             )
             print(f"  [Sim] Engine Detection (Frame 2): {len(sim_res_f2['boxes'])} boxes")
             
             pt_cls_f2 = load_bin(find_file(args.data_dir, "pred_class_score", "sample_2"), shape=(1, 900, 10))
             pt_anchor_f2 = load_bin(find_file(args.data_dir, "pred_anchor", "sample_2"), shape=(1, 900, 11))
             pt_id_f2 = load_bin(find_file(args.data_dir, "pred_track_id", "sample_2"), shape=(1, 900), dtype=np.int32)
             
             pytorch_res_f2 = None
             if pt_cls_f2 is not None:
                 pytorch_res_f2 = decoder.decode(
                    torch.from_numpy(pt_cls_f2).cuda(),
                    torch.from_numpy(pt_anchor_f2).cuda(),
                    torch.from_numpy(pt_id_f2).cuda() if pt_id_f2 is not None else None
                 )
             else:
                 pytorch_res_f2 = {"boxes": [], "scores": []}
                 
             # Try to load GT for Frame 2
             gt_data_f2 = None
             path_gt_box_f2 = find_file(args.data_dir, "gt_boxes", "sample_2")
             if path_gt_box_f2:
                 gt_boxes_f2 = load_bin(path_gt_box_f2, shape=(-1, 9))
                 gt_data_f2 = {"boxes": gt_boxes_f2}
             
             save_path_sim_f2 = os.path.join(args.output_dir, "vis_e2e_frame2_sim_comparison.png")
             visualize_bev_comparison(pytorch_res_f2, sim_res_f2, gt_data_f2, save_path_sim_f2, "(Frame 2 - Pure Sim)")
             
             save_path_sim_overlay_f2 = os.path.join(args.output_dir, "vis_e2e_frame2_sim_overlay.png")
             visualize_bev_overlay(pytorch_res_f2, sim_res_f2, gt_data_f2, save_path_sim_overlay_f2, "(Frame 2 - Pure Sim)")

    # =====================================================================
    # Summary Statistics
    # =====================================================================
    print("\n" + "="*60)
    print(" VISUALIZATION SUMMARY")
    print("="*60)
    print(f"  Frame 0:")
    print(f"    - PyTorch: {len(pytorch_res_f0['boxes'])} detections")
    print(f"    - Engine:  {len(engine_res_f0['boxes'])} detections")
    if gt_data_f0:
        print(f"    - GT:      {len(gt_data_f0['boxes'])} objects")
    print(f"  Frame 1:")
    print(f"    - PyTorch: {len(pytorch_res_f1['boxes'])} detections")
    print(f"    - Engine:  {len(engine_res_f1['boxes'])} detections")
    if gt_data_f1:
        print(f"    - GT:      {len(gt_data_f1['boxes'])} objects")
    
    if backbone_out_f2 and 'sim_res_f2' in locals():
        print(f"  Frame 2:")
        print(f"    - PyTorch: {len(pytorch_res_f2['boxes']) if pytorch_res_f2 else 0} detections")
        print(f"    - Engine:  {len(sim_res_f2['boxes'])} detections")
        if gt_data_f2:
            print(f"    - GT:      {len(gt_data_f2['boxes'])} objects")
            
    print("="*60)
    print(f"  Output files:")
    print(f"    - {save_path_f0}")
    print(f"    - {save_path_f0_overlay}")
    print(f"    - {save_path_f1}")
    print(f"    - {save_path_f1_overlay}")
    if backbone_out_f2 and 'sim_res_f2' in locals():
        print(f"    - {save_path_sim_f2}")
        print(f"    - {save_path_sim_overlay_f2}")
    print("="*60)

if __name__ == "__main__":
    main()
