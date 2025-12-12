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
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            print(f"[ENGINE-DEBUG] Tensor: {name}, Mode: {mode}, Dtype: {dtype}, Shape: {shape}")
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, input_map, output_shapes=None):
        inputs = []
        allocations = []
        
        print(f"[INFER-DEBUG] Starting inference. Input keys: {list(input_map.keys())}")

        for name in self.input_names:
            tensor = input_map[name]
            if isinstance(tensor, np.ndarray): tensor = torch.from_numpy(tensor).cuda()
            
            # Print tensor details before processing
            print(f"[INFER-DEBUG] Processing input '{name}': Shape={tensor.shape}, Dtype={tensor.dtype}, Device={tensor.device}")

            dtype_trt = self.engine.get_tensor_dtype(name)
            if dtype_trt == trt.DataType.HALF and tensor.dtype != torch.float16:
                print(f"[WARN] Auto-converting {name} to HALF")
                tensor = tensor.half()
            elif dtype_trt == trt.DataType.FLOAT and tensor.dtype != torch.float32:
                print(f"[WARN] Auto-converting {name} to FLOAT")
                tensor = tensor.float()
            elif dtype_trt == trt.DataType.INT32 and tensor.dtype != torch.int32:
                print(f"[WARN] Auto-converting {name} to INT32")
                tensor = tensor.int()
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                print(f"[DEBUG] Setting input '{name}' shape to {tensor.shape}")
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
    def __init__(self, num_output=300, score_threshold=0.1):
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
        decoded_boxes = self.decode_box(topk_boxes)
        
        return {
            "boxes": decoded_boxes.cpu().numpy(),
            "scores": topk_scores.cpu().numpy(),
            "labels": topk_labels.cpu().numpy(),
            "ids": topk_ids.cpu().numpy() if topk_ids is not None else None
        }

def visualize_bev(gt_res, pred_res, save_path="vis_bev.png"):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    def draw_on_ax(ax, res, color, label_prefix="", show_text=False, linestyle='-'):
        if not res: return
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

    # Plot GT
    draw_on_ax(axes[0], gt_res, 'green', linestyle='--')
    axes[0].set_title("Ground Truth (Green)")
    
    # Plot Pred
    draw_on_ax(axes[1], pred_res, 'red', show_text=True)
    axes[1].set_title("Prediction (Red)")
    
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
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--plugin_dir", default="deploy")
    parser.add_argument("--output_dir", default="./visualize/inference_vs_gt", help="Directory to save visualization images")
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
    
    print("Loading Engines...")
    backbone = TRTWrapper("deploy/engine/sparse4dbackbone.engine")
    head1 = TRTWrapper("deploy/engine/sparse4dhead1st.engine")
    head2 = TRTWrapper("deploy/engine/sparse4dhead2nd.engine")
    
    decoder = SimpleDecoder(num_output=300, score_threshold=0.3) # Threshold 0.3 to filter noise

    # --- Frame 0 ---
    print("Running Frame 0...")
    # Inputs
    img_path = find_file(args.data_dir, "imgs", "sample_0")
    img = load_bin(img_path, shape=(1, 6, 3, 256, 704))
    backbone_out = backbone.infer({"img": img}, output_shapes={"feature": (1, 89760, 256)})
    
    head1_inputs = {"feature": backbone_out["feature"]}
    # Load static inputs
    for name in head1.input_names:
        if name not in head1_inputs:
            path = find_file(args.data_dir, name, "sample_0")
            
            # Auto-detect shape/dtype from engine
            dtype_trt = head1.engine.get_tensor_dtype(name)
            shape_trt = head1.engine.get_tensor_shape(name)
            
            # Map TRT dtype to Numpy dtype
            dtype = np.float32
            if dtype_trt == trt.DataType.INT32: dtype = np.int32
            
            # Load and Reshape
            data = load_bin(path, dtype=dtype)
            if data is not None:
                # Handle dynamic batch dimension if present (-1)
                target_shape = list(shape_trt)
                if target_shape[0] == -1: target_shape[0] = 1 # Assume batch=1
                
                # Check size match
                if data.size == np.prod(target_shape):
                    data = data.reshape(target_shape)
                else:
                    print(f"[WARN] Size mismatch for {name}: file={data.size}, engine={target_shape}")
            
            head1_inputs[name] = data
            
    head1_out = head1.infer(head1_inputs)
    
    # --- Instance Bank Logic (Compensation) ---
    print("Calculating Ego-Motion...")
    
    # TopK from GT (Alignment)
    k = 600
    path_cls_gt = find_file(args.data_dir, "pred_class_score", "sample_0")
    gt_cls = load_bin(path_cls_gt, shape=(1, 900, 10))
    confidence = torch.from_numpy(gt_cls).cuda().sigmoid().max(dim=-1).values
    _, topk_indices = torch.topk(confidence, k, dim=1)
    
    pred_feature = head1_out["pred_instance_feature"]
    pred_anchor = head1_out["pred_anchor"]
    
    def gather_topk(tensor, indices):
        dim = tensor.shape[-1]
        indices_expanded = indices.unsqueeze(-1).expand(1, k, dim)
        return torch.gather(tensor, 1, indices_expanded)

    temp_feature = gather_topk(pred_feature, topk_indices)
    temp_anchor = gather_topk(pred_anchor, topk_indices)
    
    # Projection
    # Try multiple patterns for lidar2global
    path_l2g_0 = find_file(args.data_dir, "lidar2global", "sample_0")
    if path_l2g_0:
        l2g_0 = load_bin(path_l2g_0, shape=(1, 4, 4), dtype=np.float64)
    else:
        # Fallback: Try Inverting global2lidar (or ibank_global2lidar)
        path_g2l_0 = find_file(args.data_dir, "global2lidar", "sample_0")
        if not path_g2l_0: path_g2l_0 = find_file(args.data_dir, "ibank_global2lidar", "sample_0")
        
        if path_g2l_0:
            print(f"[INFO] Computing lidar2global from {os.path.basename(path_g2l_0)}")
            g2l_0 = load_bin(path_g2l_0, shape=(4, 4), dtype=np.float64) # Load as 4x4
            l2g_0 = np.linalg.inv(g2l_0).reshape(1, 4, 4)
        else:
            l2g_0 = None
            print("[WARN] Could not find lidar2global or global2lidar for sample_0")

    g2l_1 = load_bin(find_file(args.data_dir, "ibank_global2lidar", "sample_1"), shape=(1, 4, 4), dtype=np.float64)
    
    # If ibank_global2lidar not found, try global2lidar
    if g2l_1 is None:
        g2l_1 = load_bin(find_file(args.data_dir, "global2lidar", "sample_1"), shape=(1, 4, 4), dtype=np.float64)

    if l2g_0 is not None and g2l_1 is not None:
        T = torch.matmul(torch.from_numpy(g2l_1).cuda().float(), torch.from_numpy(l2g_0).cuda().float())
        
        # Time
        def load_timestamp(prefix):
            path = find_file(args.data_dir, "timestamp", prefix)
            if not path: path = find_file(args.data_dir, "ibank_timestamp", prefix)
            return load_bin(path, dtype=np.float64)

        t0_data = load_timestamp("sample_0")
        t1_data = load_timestamp("sample_1")
        
        if t0_data is not None and t1_data is not None:
            t0 = t0_data[0]
            t1 = t1_data[0]
            dt = t1 - t0
            if dt > 1e5: dt /= 1e6 # us to s
            
            print(f"Applying projection dt={dt:.3f}s")
            temp_anchor = SparseBox3DKeyPointsGenerator.anchor_projection(temp_anchor, [T], time_intervals=[-dt])[0]
        else:
            print("[WARN] Timestamp missing, skipping projection time interval")
    else:
        print("[WARN] Transform missing, skipping anchor projection")
    
    # --- Frame 1 ---
    print("Running Frame 1...")
    img1 = load_bin(find_file(args.data_dir, "imgs", "sample_1"), shape=(1, 6, 3, 256, 704))
    bb_out1 = backbone.infer({"img": img1})
    
    head2_inputs = {
        "feature": bb_out1["feature"],
        "temp_instance_feature": temp_feature,
        "temp_anchor": temp_anchor
    }
    
    # Load other inputs
    for name in head2.input_names:
        if name not in head2_inputs:
            path = find_file(args.data_dir, name, "sample_1")
            
            # Auto-detect shape/dtype from engine
            dtype_trt = head2.engine.get_tensor_dtype(name)
            shape_trt = head2.engine.get_tensor_shape(name)
            
            # Map TRT dtype to Numpy dtype
            dtype = np.float32
            if dtype_trt == trt.DataType.INT32: dtype = np.int32
            
            # Load and Reshape
            data = load_bin(path, dtype=dtype)
            if data is not None:
                # Handle dynamic batch dimension if present (-1)
                target_shape = list(shape_trt)
                if target_shape[0] == -1: target_shape[0] = 1 # Assume batch=1
                
                # Check size match
                if data.size == np.prod(target_shape):
                    data = data.reshape(target_shape)
                else:
                     print(f"[WARN] Size mismatch for {name}: file={data.size}, engine={target_shape}")

            head2_inputs[name] = data
            
    head2_out = head2.infer(head2_inputs)
    
    # --- Decode & Visualize ---
    print("Decoding Results...")
    
    # 1. Decode Prediction (Head 2 Output)
    pred_res = decoder.decode(
        head2_out["pred_class_score"], 
        head2_out["pred_anchor"], 
        head2_out["pred_track_id"]
    )
    print(f"Detected {len(pred_res['boxes'])} objects in Pred.")
    
    # 2. Decode Ground Truth (from file)
    # We load the Raw Network Output GT, not the final annotated GT
    gt_cls = load_bin(find_file(args.data_dir, "pred_class_score", "sample_1"), shape=(1, 900, 10))
    gt_anchor = load_bin(find_file(args.data_dir, "pred_anchor", "sample_1"), shape=(1, 900, 11))
    gt_id = load_bin(find_file(args.data_dir, "pred_track_id", "sample_1"), shape=(1, 900), dtype=np.int32)
    
    gt_res = decoder.decode(
        torch.from_numpy(gt_cls).cuda(), 
        torch.from_numpy(gt_anchor).cuda(), 
        torch.from_numpy(gt_id).cuda()
    )
    print(f"Detected {len(gt_res['boxes'])} objects in GT (Raw Output).")
    
    # 3. Visualize
    visualize_bev(gt_res, pred_res, os.path.join(args.output_dir, "vis_frame1_bev.png"))
    
    # Also visualize Frame 0 for sanity check
    print("Visualizing Frame 0...")
    pred_res0 = decoder.decode(head1_out["pred_class_score"], head1_out["pred_anchor"])
    
    gt_cls0 = load_bin(find_file(args.data_dir, "pred_class_score", "sample_0"), shape=(1, 900, 10))
    gt_anchor0 = load_bin(find_file(args.data_dir, "pred_anchor", "sample_0"), shape=(1, 900, 11))
    gt_res0 = decoder.decode(torch.from_numpy(gt_cls0).cuda(), torch.from_numpy(gt_anchor0).cuda())
    
    visualize_bev(gt_res0, pred_res0, os.path.join(args.output_dir, "vis_frame0_bev.png"))

if __name__ == "__main__":
    main()

