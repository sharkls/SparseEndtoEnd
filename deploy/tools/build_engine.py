import tensorrt as trt
import os
import sys
import numpy as np
import argparse
from cuda import cudart
import glob

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def check_cuda_err(err):
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))

def get_input_shapes(head_type="head1"):
    # Define input shapes based on export script
    shapes = {}
    
    # Common inputs
    shapes['feature'] = (1, 89760, 256) 
    shapes['spatial_shapes'] = (6, 4, 2)
    shapes['level_start_index'] = (6, 4)
    shapes['instance_feature'] = (1, 900, 256)
    shapes['anchor'] = (1, 900, 11)
    shapes['time_interval'] = (1,)
    shapes['image_wh'] = (1, 6, 2)
    shapes['lidar2img'] = (1, 6, 4, 4)
    
    if head_type == "head2":
        shapes['temp_instance_feature'] = (1, 600, 256)
        shapes['temp_anchor'] = (1, 600, 11)
        shapes['mask'] = (1,)
        shapes['track_id'] = (1, 900)
    
    if head_type == "backbone":
        # Reset shapes for backbone
        shapes = {}
        # ONNX input is [-1, 6, 3, 256, 704], calibration data is [1, 6, 3, 256, 704]
        # We must include the batch dimension for explicit batch mode
        shapes['img'] = (1, 6, 3, 256, 704)

    return shapes

class FileCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir, input_shapes, cache_file, n_batches=0):
        super().__init__()
        self.data_dir = data_dir
        self.input_shapes = input_shapes
        self.cache_file = cache_file
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Calibration data dir not found: {data_dir}")
            
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if n_batches > 0:
            self.files = self.files[:n_batches]
        self.n_batches = len(self.files)
        self.current_batch = 0
        self.d_inputs = {}
        
        print(f"Found {self.n_batches} calibration files in {data_dir}")
        
        # Allocate device memory
        for name, shape in self.input_shapes.items():
            # Calculate size in bytes
            # Heuristic: shapes/index/mask/id -> int32, others -> float32
            # NOTE: "track_id" is int32, "instance_feature" is float32
            is_int = any(x in name for x in ["shapes", "index", "mask", "id"]) and "lidar" not in name
            dtype = np.int32 if is_int else np.float32
            
            # Check for specific names that might violate heuristic
            if name == "time_interval": dtype = np.float32
            
            size = int(np.prod(shape) * np.dtype(dtype).itemsize)
            
            err, ptr = cudart.cudaMalloc(size)
            check_cuda_err(err)
            # DEBUG info
            print(f"[DEBUG] Allocated {name} ({size} bytes): {ptr} type={type(ptr)}")
            self.d_inputs[name] = ptr

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        print(f"[DEBUG] get_batch called. Batch: {self.current_batch}")
        if self.current_batch >= self.n_batches:
            return None
            
        filepath = self.files[self.current_batch]
        data = np.load(filepath)
        
        # Copy data to device
        for name in names:
            if name in self.d_inputs:
                if name not in data:
                    print(f"ERROR: Input {name} not found in {filepath}")
                    continue
                
                host_data = data[name]
                
                # Reshape/Type check if necessary
                expected_shape = self.input_shapes[name]
                
                # Special handling: scalars in npz might be 0-d
                if host_data.ndim == 0 and len(expected_shape) == 1:
                    host_data = host_data.reshape(1)
                
                # Check flatten match
                if np.prod(host_data.shape) != np.prod(expected_shape):
                     print(f"WARNING: Shape mismatch for {name}. Expected {expected_shape}, got {host_data.shape}")
                     host_data = host_data.reshape(expected_shape) # Force reshape if total elements match
                
                # Enforce contiguous and type
                is_int = any(x in name for x in ["shapes", "index", "mask", "id"]) and "lidar" not in name
                if name == "time_interval": is_int = False
                
                target_dtype = np.int32 if is_int else np.float32
                
                if host_data.dtype != target_dtype:
                    host_data = host_data.astype(target_dtype)
                
                host_data = np.ascontiguousarray(host_data)
                
                # Copy
                err, = cudart.cudaMemcpy(self.d_inputs[name], host_data.ctypes.data, host_data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
                check_cuda_err(err)
            else:
                print(f"WARNING: Unknown input name {name} requested by TensorRT!")

        self.current_batch += 1
        if self.current_batch % 10 == 0:
            print(f"Calibration batch {self.current_batch}/{self.n_batches}")
            
        return [int(self.d_inputs[name]) for name in names]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            
    def __del__(self):
        if hasattr(self, 'd_inputs'):
            for ptr in self.d_inputs.values():
                cudart.cudaFree(ptr)

def build_engine(onnx_path, engine_path, head_type, mode="fp16", plugins=[], calib_dir=None, explicit_precision=False):
    print(f"Building {mode} engine for {head_type}...")
    
    # Initialize Builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Load Plugins
    for plugin_path in plugins:
        if os.path.exists(plugin_path):
            print(f"Loading plugin: {plugin_path}")
            trt.init_libnvinfer_plugins(TRT_LOGGER, "")
            import ctypes
            ctypes.CDLL(plugin_path)
        else:
            print(f"Warning: Plugin not found: {plugin_path}")

    # Parse ONNX
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        return False
        
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # Apply Explicit Precision logic if requested
    if mode == "int8" and explicit_precision:
        print("[INFO] Enabling Explicit Precision for INT8...")
        # config.set_flag(trt.BuilderFlag.STRICT_TYPES) # Avoid STRICT_TYPES globally as it might break plugins or other layers. 
        # Instead, we rely on layer.precision + PREFER_PRECISION_CONSTRAINTS or OBEY (TensorRT 8.2+)
        # For TensorRT 8.5, OBEY_PRECISION_CONSTRAINTS is preferred.
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        
        count_int8 = 0
        count_skip = 0
        
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = layer.name
            ltype = layer.type
            
            # Default to FP16/FP32 for safety
            # layer.precision = trt.float16 # Default
            
            # Skip Constant and Shape layers for precision forcing to avoid Int32 errors
            if ltype == trt.LayerType.CONSTANT or ltype == trt.LayerType.SHAPE:
                continue

            # CRITICAL OPTIMIZATION: Force small encoders to FP16 (Aggressive)
            # Apply to ALL layers in these scopes, not just MatMul, to avoid internal reformatting
            if "camera_encoder" in name or "anchor_encoder" in name:
                # Check if layer has inputs/outputs that support FP16? 
                # Simplest fix: catch the specific error for Constant (already handled above)
                # Also be careful with Gather indices.
                
                # Try-except block is not possible here as it fails at build time.
                # We can check number of outputs and their types if possible, but easier to just set precision.
                layer.precision = trt.float16
                layer.set_output_type(0, trt.float16)
                # print(f"[INFO] Forcing FP16 for Encoder Layer: {name}") # Too verbose
                count_skip += 1
                continue

            # Logic to identify layers to force to INT8
            # We target MatMul and FullyConnected layers that are NOT part of Attention mechanisms
            # (Attention usually contains 'attn' or 'Attention' in name)
            # We also target FFN layers specifically.
            
            is_matmul = (ltype == trt.LayerType.MATRIX_MULTIPLY or ltype == trt.LayerType.FULLY_CONNECTED)
            
            if is_matmul:
                # Exclude Attention layers if they are sensitive
                # 'fc' is common in names like 'pos_fc', 'weights_fc'
                # 'layers' usually implies the transformer block layers
                if "layers" in name or "fc" in name or "MatMul" in name:
                    layer.precision = trt.int8
                    layer.set_output_type(0, trt.int8)
                    # layer.set_input_type(0, trt.int8) # Removed as it causes AttributeError
                    print(f"[INFO] Forcing INT8 for layer: {name}")
                    count_int8 += 1
                else:
                    print(f"[INFO] Skipping MatMul (unknown context): {name}")
                    count_skip += 1
            
        print(f"[INFO] Explicit Precision: Forced {count_int8} layers to INT8. Skipped {count_skip} MatMuls.")

    # Set Optimization Profile (CRITICAL for Dynamic Shapes + INT8 Calibration)
    profile = builder.create_optimization_profile()
    input_shapes = get_input_shapes(head_type)
    
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        if name in input_shapes:
            shape = input_shapes[name]
            # Set Min, Opt, Max to the same shape for calibration
            profile.set_shape(name, shape, shape, shape)
            print(f"Set profile for {name}: {shape}")
        else:
            print(f"Warning: No shape defined for input {name}, using dynamic defaults which might fail.")
    
    config.add_optimization_profile(profile)

    # Config flags
    if mode == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
    elif mode == "int8":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            
            cache_file = engine_path.replace(".engine", ".cache")
            input_shapes = get_input_shapes(head_type)
            
            if calib_dir and os.path.exists(calib_dir):
                print(f"Using calibration data from {calib_dir}")
                calibrator = FileCalibrator(calib_dir, input_shapes, cache_file)
                config.int8_calibrator = calibrator
            else:
                print("Error: Calibration data directory not provided or does not exist for INT8 mode!")
                return False
        else:
            print("Warning: Platform does not support INT8!")

    # Memory pool
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30) # 4GB

    # Build
    print("Building engine (this may take a while)...")
    try:
        plan = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"Build exception: {e}")
        return False
    
    if plan:
        with open(engine_path, "wb") as f:
            f.write(plan)
        print(f"Engine saved to {engine_path}")
        return True
    else:
        print("Build failed!")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--head", choices=["head1", "head2", "backbone"], required=True)
    parser.add_argument("--mode", choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--plugins", nargs="+", default=[])
    parser.add_argument("--calib_dir", type=str, help="Directory containing calibration .npz files")
    parser.add_argument("--explicit_precision", action="store_true", help="Force INT8 precision for FFN/MatMul layers")
    args = parser.parse_args()
    
    success = build_engine(args.onnx, args.engine, args.head, args.mode, args.plugins, args.calib_dir, args.explicit_precision)
    if not success:
        sys.exit(1)
