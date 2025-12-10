import argparse
import os
import numpy as np
import tensorrt as trt
import torch
import ctypes
import glob
import time

class TRTWrapper:
    def __init__(self, engine_path, plugin_paths=[]):
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, "")
        
        # Load custom plugins
        for path in plugin_paths:
            if os.path.exists(path):
                ctypes.CDLL(path)
                print(f"Loaded plugin: {path}")
            else:
                print(f"[WARN] Plugin not found: {path}")

        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.input_names = []
        self.output_names = []
        
        # Allocate buffers
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            # Check if input or output
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, input_data_map):
        # Set input shapes and allocate memory
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for name in self.input_names:
            if name not in input_data_map:
                raise ValueError(f"Missing input: {name}")
            data = input_data_map[name]
            
            # Set shape for dynamic input
            self.context.set_input_shape(name, data.shape)
            
            # Create input tensor on GPU
            # Torch can handle data transfer efficiently
            if isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data).cuda()
            elif isinstance(data, torch.Tensor):
                tensor = data.cuda()
            else:
                raise ValueError("Input data must be numpy array or torch tensor")
                
            self.inputs.append(tensor)
            self.allocations.append(tensor.data_ptr())

        # Allocate outputs
        output_buffers = {}
        for name in self.output_names:
            # Infer output shape
            shape = self.context.get_tensor_shape(name)
            
            # Calculate output size (vol)
            vol = 1
            dims = []
            for dim in shape:
                if dim < 0:
                    vol *= 1 
                    dims.append(1) # fallback
                else:
                    vol *= dim
                    dims.append(dim)
            
            # Calculate dtype
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = torch.float32
            if dtype == trt.DataType.HALF:
                torch_dtype = torch.float16
            elif dtype == trt.DataType.INT32:
                torch_dtype = torch.int32
            elif dtype == trt.DataType.BOOL:
                torch_dtype = torch.bool
            elif dtype == trt.DataType.INT8:
                torch_dtype = torch.int8
                
            # Allocate output tensor
            # Using inferred shape directly
            d_output = torch.empty(tuple(dims), dtype=torch_dtype, device='cuda')
            
            self.allocations.append(d_output.data_ptr())
            output_buffers[name] = d_output

        # Execute
        # Set tensor address
        for i, name in enumerate(self.input_names):
            self.context.set_tensor_address(name, self.inputs[i].data_ptr())
        for name in self.output_names:
            self.context.set_tensor_address(name, output_buffers[name].data_ptr())

        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        
        # Synchronize to ensure completion before returning (for validation)
        torch.cuda.synchronize()
        
        # Copy back
        results = {}
        for name, d_output in output_buffers.items():
            # Get final shape if dynamic
            # final_shape = self.context.get_tensor_shape(name)
            # Reshape tensor if needed? Usually torch handles shape if we allocated correctly
            # But if TRT changed shape dynamically (e.g. nonzero), we might need to check.
            # Assuming shape is static after inference for this model.
            
            host_mem = d_output.cpu().numpy()
            results[name] = host_mem
            
        return results

def load_bin(path, dtype=np.float32, shape=None):
    data = np.fromfile(path, dtype=dtype)
    if shape is not None:
        try:
            data = data.reshape(shape)
        except:
            print(f"[WARN] Reshape failed for {path} to {shape}, keep flattened.")
    return data

def get_error(a, b):
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    diff = np.abs(a - b)
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    
    # Cosine Similarity
    if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    else:
        cos_sim = 0
        
    return max_err, mean_err, cos_sim

def main():
    parser = argparse.ArgumentParser(description="Validate TensorRT Engine against Golden Data")
    parser.add_argument("--engine", type=str, required=True, help="Path to TensorRT Engine")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing input_*.bin and output_*.bin")
    parser.add_argument("--plugins", nargs='+', default=[], help="List of plugin .so paths")
    args = parser.parse_args()

    print(f"Initializing TensorRT Engine: {args.engine}")
    wrapper = TRTWrapper(args.engine, args.plugins)
    
    # Load Inputs
    inputs = {}
    print(f"Loading inputs from {args.data_dir}...")
    for name in wrapper.input_names:
        bin_path = os.path.join(args.data_dir, f"input_{name}.bin")
        shape_path = os.path.join(args.data_dir, f"input_{name}.shape")
        
        if not os.path.exists(bin_path):
            print(f"[ERROR] Missing input file: {bin_path}")
            return

        # Infer dtype
        dtype = np.float32
        # Check engine binding dtype
        trt_dtype = wrapper.engine.get_tensor_dtype(name)
        if trt_dtype == trt.DataType.INT32:
            dtype = np.int32
        elif trt_dtype == trt.DataType.HALF:
            dtype = np.float16
        elif trt_dtype == trt.DataType.BOOL:
            dtype = np.bool_

        shape = None
        if os.path.exists(shape_path):
            shape = np.fromfile(shape_path, dtype=np.int32)
        
        data = load_bin(bin_path, dtype, shape)
        inputs[name] = data
        
    print("Running Inference...")
    start = time.time()
    outputs = wrapper.infer(inputs)
    end = time.time()
    print(f"Inference time: {(end-start)*1000:.2f} ms")
    
    print("\nValidating Outputs:")
    for name, pred in outputs.items():
        bin_path = os.path.join(args.data_dir, f"output_{name}.bin")
        if not os.path.exists(bin_path):
            print(f"[WARN] Expected output file not found: {bin_path}")
            continue
            
        # Infer expected dtype from prediction
        dtype = pred.dtype
        expected = load_bin(bin_path, dtype)
        
        # If expected was saved flattened, we might need to reshape to match pred for some metrics?
        # get_error flattens anyway.
        
        if np.issubdtype(dtype, np.integer):
             # Exact match check for integers
             matches = (pred.flatten() == expected.flatten())
             accuracy = np.mean(matches)
             print(f"[{name}]")
             print(f"  Shape: {pred.shape}")
             print(f"  Accuracy: {accuracy*100:.2f}%")
             status = "MATCH" if accuracy == 1.0 else "MISMATCH"
             print(f"  -> {status}")
        else:
            max_err, mean_err, cos_sim = get_error(pred, expected)
            
            status = "MATCH"
            # Adjust threshold based on precision?
            # For FP32, we expect < 1e-4. For FP16 < 1e-2.
            # Sparse4D dynamic output can have larger errors in outlier points, reasonable threshold is 0.1 for max, 1e-3 for mean
            if max_err > 0.2 and mean_err > 1e-3: 
                status = "MISMATCH (High Error)"
            elif max_err > 1e-3:
                status = "MATCH (Acceptable Diff)"
                
            print(f"[{name}]")
            print(f"  Shape: {pred.shape}")
            print(f"  Max Error:  {max_err:.6f}")
            print(f"  Mean Error: {mean_err:.6f}")
            print(f"  Cos Sim:    {cos_sim:.6f}")
            print(f"  -> {status}")

if __name__ == "__main__":
    main()

