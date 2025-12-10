import argparse
import os
import numpy as np
import tensorrt as trt
import torch
import ctypes
import time

import glob

def find_file(data_dir, name_pattern, sample_prefix="sample_1"):
    # Search for file matching pattern
    # Pattern example: {sample_prefix}_{name}_*.bin
    pattern = os.path.join(data_dir, f"{sample_prefix}_{name_pattern}_*.bin")
    files = glob.glob(pattern)
    if not files:
        # Try direct match
        pattern = os.path.join(data_dir, f"{name_pattern}.bin")
        files = glob.glob(pattern)
        
    if not files:
        # Try fallback to sample_0 if we were looking for something else and failed
        # This is useful if some static data (like maps?) is only saved with sample_0 prefix
        if sample_prefix != "sample_0":
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
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, input_map):
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        # Prepare Inputs
        for name in self.input_names:
            if name not in input_map:
                # Try mapping names if mismatch (e.g., feature_0 vs input_feature_0)
                print(f"[Error] Missing input: {name}. Available: {list(input_map.keys())}")
                raise ValueError(f"Missing input: {name}")
            
            tensor = input_map[name]
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor).cuda()
            
            # Set dynamic shape
            self.context.set_input_shape(name, tensor.shape)
            self.inputs.append(tensor)
            self.allocations.append(tensor.data_ptr())

        # Prepare Outputs
        output_map = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dims = []
            vol = 1
            for s in shape:
                if s < 0: 
                    dims.append(1) # fallback for unknown dim, usually should be inferred
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

def validate_engine(engine_path, data_dir, sample_prefix, engine_name="Head"):
    if not os.path.exists(engine_path):
        print(f"[{engine_name}] Engine not found: {engine_path}")
        return

    print(f"\n[{engine_name}] Loading Engine: {engine_path}")
    engine = TRTWrapper(engine_path)
    
    print(f"[{engine_name}] Loading inputs from {data_dir} ({sample_prefix})...")
    inputs = {}
    
    # Shapes definition
    shapes = {
        "feature": (1, 89760, 256),
        "instance_feature": (1, 900, 256),
        "anchor": (1, 900, 11),
        "time_interval": (1,),
        "temp_instance_feature": (1, 600, 256),
        "temp_anchor": (1, 600, 11),
        "image_wh": (1, 6, 2),
        "lidar2img": (1, 6, 4, 4),
        "spatial_shapes": (6, 4, 2),
        "level_start_index": (6, 4)
    }
    
    for name in engine.input_names:
        path = find_file(data_dir, name, sample_prefix=sample_prefix)
        shape = shapes.get(name)
        
        if path is None:
             print(f"[Error] [{engine_name}] Could not find file for input: {name}")
             return

        data = load_bin(path, shape=shape)
        if data is None:
            print(f"[Error] [{engine_name}] Could not load {name}")
            return
        inputs[name] = data

    # Inference
    print(f"[{engine_name}] >>> Running Inference...")
    start = time.time()
    outputs = engine.infer(inputs)
    end = time.time()
    print(f"[{engine_name}] Inference Time: {(end-start)*1000:.2f} ms")

    # Validation
    print(f"[{engine_name}] >>> Validation Results")
    expected_files = {
        "pred_instance_feature": "pred_instance_feature",
        "pred_anchor": "pred_anchor",
        "pred_class_score": "pred_class_score",
        "pred_quality_score": "pred_quality_score",
        "pred_track_id": "pred_track_id"
    }
    
    for name, out_tensor in outputs.items():
        if name not in expected_files:
            continue
            
        expected_path = find_file(data_dir, expected_files[name], sample_prefix=sample_prefix)
        if expected_path is None:
            continue
            
        pred = out_tensor.cpu().numpy()
        expected = load_bin(expected_path, dtype=pred.dtype)
        expected = expected.reshape(pred.shape)
        
        if np.issubdtype(pred.dtype, np.integer):
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
                
            status = "MATCH" if max_err < 0.1 else "MISMATCH"
            print(f"  [{name}] Max Error: {max_err:.6f}, Mean Error: {mean_err:.6f}, Cos Sim: {cos_sim:.6f} -> {status}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to validation data (e.g., deploy/val_data_plugin/head2)")
    parser.add_argument("--plugin_dir", default="deploy", help="Path to deploy dir containing plugins")
    args = parser.parse_args()

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
            print(f"Loaded {p}")
        else:
            print(f"Plugin not found: {p}")

    # 2. Validate Head 1 (1st frame)
    # Using sample_0 data
    validate_engine(
        engine_path="deploy/engine/sparse4dhead1st.engine",
        data_dir=args.data_dir,
        sample_prefix="sample_0",
        engine_name="Head1"
    )

    # 3. Validate Head 2 (Temporal frame)
    # Using sample_1 data
    validate_engine(
        engine_path="deploy/engine/sparse4dhead2nd.engine",
        data_dir=args.data_dir,
        sample_prefix="sample_1",
        engine_name="Head2"
    )

if __name__ == "__main__":
    main()
