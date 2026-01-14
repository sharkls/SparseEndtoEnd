import argparse
import os
import numpy as np
import onnxruntime as ort
import glob

def load_bin(path, dtype=np.float32):
    return np.fromfile(path, dtype=dtype)

def get_error(a, b):
    # Flatten
    a = a.flatten()
    b = b.flatten()
    diff = np.abs(a - b)
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    return max_err, mean_err

def main():
    parser = argparse.ArgumentParser(description="Validate ONNX model against saved PyTorch outputs")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing input_*.bin and output_*.bin")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for inputs (if saved as FP32 but model needs FP16)")
    args = parser.parse_args()

    print(f"Loading ONNX model: {args.onnx}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(args.onnx, providers=providers)

    inputs = {}
    print(f"Loading inputs from {args.data_dir}...")
    
    for node in session.get_inputs():
        name = node.name
        # Try to find corresponding bin file
        # Pattern: input_{name}.bin
        bin_path = os.path.join(args.data_dir, f"input_{name}.bin")
        if not os.path.exists(bin_path):
            print(f"[WARN] Input file not found for {name}: {bin_path}")
            continue
            
        # Determine dtype from ONNX expectation or file size? 
        # Usually we save as float32 or int32 based on tensor type.
        # Let's try to infer from ONNX type
        onnx_type = node.type
        dtype = np.float32
        if 'int32' in onnx_type:
            dtype = np.int32
        elif 'float16' in onnx_type:
            dtype = np.float16
        elif 'bool' in onnx_type:
            dtype = np.bool_
            
        data = load_bin(bin_path, dtype=dtype)
        
        # Reshape
        # ONNX shape might have dynamic dims (None or strings). 
        # We need to rely on the saved data being correct size, 
        # but reshaping is tricky if we don't know exact shape.
        # Ideally, we should save shape info too. 
        # For now, let's assume the user/export script saved shape info or we assume flatten matching.
        # Wait, ort expects shaped inputs.
        
        # Strategy: Use reshape based on node shape if fully defined, 
        # otherwise try to guess or use 1D if model accepts it (unlikely).
        # Better: export script should save shape. 
        # Or: Use the shape from the ONNX model if static.
        
        shape = node.shape
        if shape and all(isinstance(s, int) for s in shape):
            try:
                data = data.reshape(shape)
            except:
                print(f"[WARN] Failed to reshape {name} to {shape}. Data size: {data.size}")
        else:
             # Try to find a shape file? Or just try to match expected size.
             # For Sparse4D, inputs like spatial_shapes [6,4,2] are fixed.
             # Feature [1, N, 256] might vary.
             # Let's try to load a corresponding shape file if exists.
             shape_path = os.path.join(args.data_dir, f"input_{name}.shape")
             if os.path.exists(shape_path):
                 shape = np.fromfile(shape_path, dtype=np.int32)
                 data = data.reshape(shape)
             else:
                 print(f"[INFO] No shape file for {name}, trying 1D/flattened or inferred.")
                 # Fallback: if dynamic, we might be in trouble without shape file.
                 pass

        inputs[name] = data

    print("Running Inference...")
    outputs = session.run(None, inputs)
    
    print("\nValidating Outputs:")
    output_names = [n.name for n in session.get_outputs()]
    for i, name in enumerate(output_names):
        bin_path = os.path.join(args.data_dir, f"output_{name}.bin")
        if not os.path.exists(bin_path):
            print(f"[WARN] Expected output file not found for {name}: {bin_path}")
            continue
            
        pred = outputs[i]
        
        # Load expected
        # Again, dtype inference
        dtype = np.float32
        # Assuming output is mostly float
        
        expected = load_bin(bin_path, dtype=dtype)
        
        # Error calc
        try:
            max_err, mean_err = get_error(pred, expected)
            print(f"[{name}]")
            print(f"  Max Error:  {max_err:.6f}")
            print(f"  Mean Error: {mean_err:.6f}")
            
            if max_err > 1e-3:
                print(f"  -> POSSIBLE MISMATCH (Threshold 1e-3)")
            else:
                print(f"  -> MATCH")
        except Exception as e:
            print(f"  Error comparing {name}: {e}")
            print(f"  Pred shape: {pred.shape}, Expected size: {expected.size}")

if __name__ == "__main__":
    main()

