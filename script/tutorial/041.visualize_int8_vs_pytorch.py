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
        try:
            with open(engine_path, "rb") as f:
                self.runtime = trt.Runtime(logger)
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"[ERROR] Failed to load engine {engine_path}: {e}")
            sys.exit(1)
            
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        
        # Map input/output names to indices
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            # print(f"[ENGINE-DEBUG] Tensor: {name}, Mode: {mode}, Dtype: {dtype}, Shape: {shape}")
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, input_map, output_shapes=None):
        inputs = []
        allocations = []
        
        # print(f"[INFER-DEBUG] Starting inference. Input keys: {list(input_map.keys())}")

        for name in self.input_names:
            if name not in input_map:
                print(f"[ERROR] Missing input: {name}")
                continue
                
            tensor = input_map[name]
            if isinstance(tensor, np.ndarray): tensor = torch.from_numpy(tensor).cuda()
            
            # Type Conversion Logic
            dtype_trt = self.engine.get_tensor_dtype(name)
            if dtype_trt == trt.DataType.HALF and tensor.dtype != torch.float16:
                tensor = tensor.half()
            elif dtype_trt == trt.DataType.FLOAT and tensor.dtype != torch.float32:
                tensor = tensor.float()
            elif dtype_trt == trt.DataType.INT32 and tensor.dtype != torch.int32:
                tensor = tensor.int()
            
            # Dynamic Shape Handling
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                try:
                    self.context.set_input_shape(name, tensor.shape)
                except Exception as e:
                    print(f"[ERROR] Failed to set shape for {name}: {e}")
            inputs.append(tensor)
            allocations.append(tensor.data_ptr())

        output_map = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dims = []
            if output_shapes and name in output_shapes:
                dims = list(output_shapes[name])
            else:
                # Handle unknown dims (-1) with safe defaults or infer
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

class SimpleDecoder:
    def __init__(self, num_output=300, score_threshold=0.1):
        self.num_output = num_output
        self.score_threshold = score_threshold

    def decode_box(self, box):
        # box: [N, 11] (X, Y, Z, W, L, H, SIN, COS, VX, VY, VZ)
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        decoded = torch.cat([
            box[:, [X, Y, Z]],
            box[:, [W, L, H]].exp(),
            yaw[:, None],
            box[:, VX:VY+1]
        ], dim=-1)
        return decoded

    def decode(self, cls_scores, box_preds, track_id=None):
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
                
        decoded_boxes = self.decode_box(topk_boxes)
        
        return {
            "boxes": decoded_boxes.cpu().numpy(),
            "scores": topk_scores.cpu().numpy(),
            "labels": topk_labels.cpu().numpy(),
            "ids": topk_ids.cpu().numpy() if topk_ids is not None else None
        }

def visualize_bev(gt_res, pred_res, save_path="vis_bev.png", title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    def draw_on_ax(ax, res, color, label_prefix="", show_text=False, linestyle='-'):
        if not res or len(res["boxes"]) == 0: return
        for i, (box, score) in enumerate(zip(res["boxes"], res["scores"])):
            x, y, z, w, l, h, yaw = box[:7]
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            corners = np.array([
                [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
            ])
            corners_global = (R @ corners.T).T + np.array([x, y])
            
            poly = patches.Polygon(corners_global, closed=True, fill=False, edgecolor=color, linewidth=1.5, linestyle=linestyle)
            ax.add_patch(poly)
            
            if show_text:
                label = f"{score:.2f}"
                if res["ids"] is not None:
                    label += f" ID:{res['ids'][i]}"
                ax.text(x, y, label, color=color, fontsize=8, clip_on=True)

    # Plot GT (PyTorch FP32 Output)
    draw_on_ax(axes[0], gt_res, 'green', linestyle='-', show_text=True)
    axes[0].set_title(f"PyTorch FP32 Output (Green) {title_suffix}")
    
    # Plot Pred (TensorRT INT8 Output)
    draw_on_ax(axes[1], pred_res, 'red', linestyle='-', show_text=True)
    axes[1].set_title(f"TensorRT INT8 Output (Red) {title_suffix}")
    
    # Also overlay GT on Pred for direct comparison
    draw_on_ax(axes[1], gt_res, 'green', linestyle='--', show_text=False)

    for ax in axes:
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to val_data_e2e_fp32")
    parser.add_argument("--plugin_dir", default="deploy")
    parser.add_argument("--output_dir", default="./visualize/int8_vs_pytorch", help="Directory to save visualization images")
    parser.add_argument("--backbone", default="deploy/engine/sparse4dbackbone_int8.engine")
    parser.add_argument("--head1", default="deploy/engine/sparse4dhead1st_v3.engine") 
    parser.add_argument("--head2", default="deploy/engine/sparse4dhead2nd_v3.engine")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Plugins
    plugins = [
        os.path.join(args.plugin_dir, "dfa_plugin/lib/deformableAttentionAggr.so"),
        os.path.join(args.plugin_dir, "ln_plugin/lib/customLayerNorm.so"),
        os.path.join(args.plugin_dir, "sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so"),
    ]
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")
    for p in plugins:
        if os.path.exists(p): ctypes.CDLL(p)
        else: print(f"[WARN] Plugin not found: {p}")
    
    print("Loading INT8 Engines...")
    print(f"Backbone: {args.backbone}")
    backbone = TRTWrapper(args.backbone)
    print(f"Head1: {args.head1}")
    head1 = TRTWrapper(args.head1)
    print(f"Head2: {args.head2}")
    head2 = TRTWrapper(args.head2)
    
    decoder = SimpleDecoder(num_output=300, score_threshold=0.3)

    # ==========================
    # Frame 0 Pipeline
    # ==========================
    print("\nRunning Frame 0 Pipeline...")
    
    # 1. Backbone Inference
    img_path = find_file(args.data_dir, "imgs", "sample_0")
    if not img_path: raise FileNotFoundError("Sample 0 images not found")
    img = load_bin(img_path, shape=(1, 6, 3, 256, 704))
    
    # Explicit output shape for backbone feature
    backbone_out = backbone.infer({"img": img}, output_shapes={"feature": (1, 89760, 256)})
    
    # 2. Head1 Inference
    head1_inputs = {"feature": backbone_out["feature"]}
    
    # Load other inputs from file (spatial_shapes, etc.)
    for name in head1.input_names:
        if name not in head1_inputs:
            path = find_file(args.data_dir, name, "sample_0")
            
            dtype_trt = head1.engine.get_tensor_dtype(name)
            dtype = np.float32
            if dtype_trt == trt.DataType.INT32: dtype = np.int32
            
            data = load_bin(path, dtype=dtype)
            if data is not None:
                # Reshape logic
                shape_trt = head1.engine.get_tensor_shape(name)
                target_shape = list(shape_trt)
                if target_shape[0] == -1: target_shape[0] = 1 
                if data.size == np.prod(target_shape):
                    data = data.reshape(target_shape)
                head1_inputs[name] = data
            else:
                 print(f"[WARN] Input {name} not found for Head1")

    head1_out = head1.infer(head1_inputs)
    
    # 3. Visualize Frame 0
    # Decode INT8 Output
    pred_res0 = decoder.decode(head1_out["pred_class_score"], head1_out["pred_anchor"])
    
    # Load PyTorch Output (Ground Truth for comparison)
    gt_cls0 = load_bin(find_file(args.data_dir, "pred_class_score", "sample_0"), shape=(1, 900, 10))
    gt_anchor0 = load_bin(find_file(args.data_dir, "pred_anchor", "sample_0"), shape=(1, 900, 11))
    gt_res0 = decoder.decode(torch.from_numpy(gt_cls0).cuda(), torch.from_numpy(gt_anchor0).cuda())
    
    visualize_bev(gt_res0, pred_res0, os.path.join(args.output_dir, "vis_frame0_int8_vs_pytorch.png"), title_suffix="(Frame 0)")

    # ==========================
    # Frame 1 Pipeline
    # ==========================
    print("\nRunning Frame 1 Pipeline...")

    # 1. Instance Bank Update & Compensation
    # Using TopK from PyTorch output for alignment (simulating perfect association)
    # or use INT8 output? Let's use INT8 output to see accumulated error.
    # Actually, for visualization, using INT8 output is better to see real behavior.
    
    k = 600
    # Use INT8 Head1 output for selection
    scores = head1_out["pred_class_score"].sigmoid().max(dim=-1).values
    _, topk_indices = torch.topk(scores, k, dim=1)
    
    pred_feature = head1_out["pred_instance_feature"]
    pred_anchor = head1_out["pred_anchor"]
    
    def gather_topk(tensor, indices):
        indices_expanded = indices.unsqueeze(-1).expand(1, k, tensor.shape[-1])
        return torch.gather(tensor, 1, indices_expanded)

    temp_feature = gather_topk(pred_feature, topk_indices)
    temp_anchor = gather_topk(pred_anchor, topk_indices)
    
    # Ego Motion Compensation
    # Load transforms
    path_l2g_0 = find_file(args.data_dir, "lidar2global", "sample_0")
    if path_l2g_0:
        l2g_0 = load_bin(path_l2g_0, shape=(1, 4, 4), dtype=np.float64)
    else:
        # Try inferring
        path_g2l_0 = find_file(args.data_dir, "global2lidar", "sample_0") or find_file(args.data_dir, "ibank_global2lidar", "sample_0")
        if path_g2l_0:
             g2l_0_val = load_bin(path_g2l_0, shape=(4, 4), dtype=np.float64)
             l2g_0 = np.linalg.inv(g2l_0_val).reshape(1, 4, 4)
        else:
             l2g_0 = None

    g2l_1 = load_bin(find_file(args.data_dir, "ibank_global2lidar", "sample_1"), shape=(1, 4, 4), dtype=np.float64)
    if g2l_1 is None:
        g2l_1 = load_bin(find_file(args.data_dir, "global2lidar", "sample_1"), shape=(1, 4, 4), dtype=np.float64)

    if l2g_0 is not None and g2l_1 is not None:
        T = torch.matmul(torch.from_numpy(g2l_1).cuda().float(), torch.from_numpy(l2g_0).cuda().float())
        
        path_t0 = find_file(args.data_dir, "timestamp", "sample_0") or find_file(args.data_dir, "ibank_timestamp", "sample_0")
        path_t1 = find_file(args.data_dir, "timestamp", "sample_1") or find_file(args.data_dir, "ibank_timestamp", "sample_1")
        
        t0 = load_bin(path_t0, dtype=np.float64)
        t1 = load_bin(path_t1, dtype=np.float64)
        
        if t0 is not None and t1 is not None:
            dt = t1[0] - t0[0]
            if dt > 1e5: dt /= 1e6
            temp_anchor = SparseBox3DKeyPointsGenerator.anchor_projection(temp_anchor, [T], time_intervals=[-dt])[0]
            print(f"Motion compensation applied (dt={dt:.3f}s)")

    # 2. Backbone Frame 1
    img1 = load_bin(find_file(args.data_dir, "imgs", "sample_1"), shape=(1, 6, 3, 256, 704))
    bb_out1 = backbone.infer({"img": img1})
    
    # 3. Head2 Inference
    head2_inputs = {
        "feature": bb_out1["feature"],
        "temp_instance_feature": temp_feature,
        "temp_anchor": temp_anchor
    }
    
    for name in head2.input_names:
        if name not in head2_inputs:
            path = find_file(args.data_dir, name, "sample_1")
            
            dtype_trt = head2.engine.get_tensor_dtype(name)
            dtype = np.float32
            if dtype_trt == trt.DataType.INT32: dtype = np.int32
            
            data = load_bin(path, dtype=dtype)
            if data is not None:
                target_shape = list(head2.engine.get_tensor_shape(name))
                if target_shape[0] == -1: target_shape[0] = 1 
                if data.size == np.prod(target_shape):
                    data = data.reshape(target_shape)
                head2_inputs[name] = data
            else:
                 print(f"[WARN] Input {name} not found for Head2")

    head2_out = head2.infer(head2_inputs)

    # 4. Visualize Frame 1
    # Decode INT8 Output
    pred_res1 = decoder.decode(head2_out["pred_class_score"], head2_out["pred_anchor"], head2_out["pred_track_id"])
    
    # Load PyTorch Output
    gt_cls1 = load_bin(find_file(args.data_dir, "pred_class_score", "sample_1"), shape=(1, 900, 10))
    gt_anchor1 = load_bin(find_file(args.data_dir, "pred_anchor", "sample_1"), shape=(1, 900, 11))
    gt_id1 = load_bin(find_file(args.data_dir, "pred_track_id", "sample_1"), shape=(1, 900), dtype=np.int32)
    
    gt_res1 = decoder.decode(torch.from_numpy(gt_cls1).cuda(), torch.from_numpy(gt_anchor1).cuda(), torch.from_numpy(gt_id1).cuda())
    
    visualize_bev(gt_res1, pred_res1, os.path.join(args.output_dir, "vis_frame1_int8_vs_pytorch.png"), title_suffix="(Frame 1)")

if __name__ == "__main__":
    main()
