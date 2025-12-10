# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

import os
import numpy as np
import onnxruntime as ort
import argparse
import logging

def set_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def read_bin_file(file_path, dtype, shape):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = np.fromfile(file_path, dtype=dtype)
    
    expected_size = np.prod(shape)
    if data.size != expected_size:
        # Try to see if it matches ignoring batch dimension 1
        if data.size == expected_size // shape[0]:
             # Sometimes data is saved without batch dim, but ONNX expects it
             pass
        else:
            print(f"[WARN] Size mismatch for {os.path.basename(file_path)}: Expected {expected_size}, got {data.size}")
    
    return data.reshape(shape)

def get_error_percentage(pred, expected, threshold=0.1):
    diff = np.abs(pred - expected)
    max_error = np.max(diff)
    
    # Sort errors descending
    sorted_diff = np.sort(diff.flatten())[::-1]
    
    # Errors > threshold
    error_mask = diff > threshold
    error_count = np.sum(error_mask)
    percentage = error_count / diff.size
    
    return percentage, max_error

def main():
    parser = argparse.ArgumentParser(description="Verify Sparse4D Head ONNX Inference")
    parser.add_argument("--onnx_path", type=str, default="/share/Code/Sparse4dE2E/deploy/onnx/sparse4dhead1st.onnx", help="Path to ONNX model")
    parser.add_argument("--data_root", type=str, default="/share/Code/Sparse4dE2E/script/tutorial/asset/", help="Path to test data assets")
    args = parser.parse_args()

    logger = set_logger(None)
    logger.info(f"Loading ONNX model: {args.onnx_path}")

    # Initialize ONNX Runtime
    # Prefer CUDA if available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(args.onnx_path, providers=providers)
    except Exception as e:
        logger.error(f"Failed to create ONNX session: {e}")
        return

    logger.info("ONNX session created successfully.")

    # Define inputs and shapes (Based on C++ unit test filenames)
    # Names must match ONNX input names
    input_configs = {
        "feature": {"file": "sample_0_feature_1*89760*256_float32.bin", "dtype": np.float32, "shape": (1, 89760, 256)},
        "spatial_shapes": {"file": "sample_0_spatial_shapes_6*4*2_int32.bin", "dtype": np.int32, "shape": (6, 4, 2)}, 
        "level_start_index": {"file": "sample_0_level_start_index_6*4_int32.bin", "dtype": np.int32, "shape": (6, 4)},
        "instance_feature": {"file": "sample_0_instance_feature_1*900*256_float32.bin", "dtype": np.float32, "shape": (1, 900, 256)},
        "anchor": {"file": "sample_0_anchor_1*900*11_float32.bin", "dtype": np.float32, "shape": (1, 900, 11)},
        "time_interval": {"file": "sample_0_time_interval_1_float32.bin", "dtype": np.float32, "shape": (1,)},
        "image_wh": {"file": "sample_0_image_wh_1*6*2_float32.bin", "dtype": np.float32, "shape": (1, 6, 2)},
        "lidar2img": {"file": "sample_0_lidar2img_1*6*4*4_float32.bin", "dtype": np.float32, "shape": (1, 6, 4, 4)},
    }

    # Load inputs
    onnx_inputs = {}
    logger.info("Loading inputs...")
    for name, config in input_configs.items():
        file_path = os.path.join(args.data_root, config["file"])
        try:
            data = read_bin_file(file_path, config["dtype"], config["shape"])
            
            # Special handling for spatial_shapes and level_start_index to match typical ONNX batch requirements if needed
            # But based on C++ code they are passed as is. 
            # However, TensorRT bindings often flatten things. ONNX is strict about shapes.
            # Let's check session input shapes
            for input_meta in session.get_inputs():
                if input_meta.name == name:
                    # Check if we need to add batch dim or adjust
                    # This is a simple heuristic, might need adjustment based on actual ONNX export
                    if len(input_meta.shape) > len(data.shape) and input_meta.shape[0] == 1:
                         data = np.expand_dims(data, axis=0)
                    break
            
            onnx_inputs[name] = data
        except Exception as e:
            logger.error(f"Error loading input {name}: {e}")
            return

    # Run inference
    logger.info("Running inference...")
    try:
        outputs = session.run(None, onnx_inputs)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return

    # Define expected outputs and shapes
    output_names = [output.name for output in session.get_outputs()]
    logger.info(f"Model outputs: {output_names}")

    expected_configs = {
        "pred_instance_feature": {"file": "sample_0_pred_instance_feature_1*900*256_float32.bin", "dtype": np.float32, "shape": (1, 900, 256)},
        "pred_anchor": {"file": "sample_0_pred_anchor_1*900*11_float32.bin", "dtype": np.float32, "shape": (1, 900, 11)},
        "pred_class_score": {"file": "sample_0_pred_class_score_1*900*10_float32.bin", "dtype": np.float32, "shape": (1, 900, 10)},
        "pred_quality_score": {"file": "sample_0_pred_quality_score_1*900*2_float32.bin", "dtype": np.float32, "shape": (1, 900, 2)},
    }

    # Thresholds from C++ code
    thresholds = {
        "pred_instance_feature": 0.02, # Expected LE 0.02
        "pred_anchor": 0.02,           # Expected LE 0.02
        "pred_class_score": 0.01,      # Expected LE 0.01
        "pred_quality_score": 0.01     # Expected LE 0.01
    }

    # Verify results
    logger.info("\nVerifying Results:")
    for i, output_name in enumerate(output_names):
        if output_name not in expected_configs:
            logger.warning(f"Skipping verification for {output_name} (no expected data config)")
            continue
            
        config = expected_configs[output_name]
        file_path = os.path.join(args.data_root, config["file"])
        
        try:
            expected_data = read_bin_file(file_path, config["dtype"], config["shape"])
            pred_data = outputs[i]
            
            # Calculate Error
            # Using 0.1 as the error counting threshold (same as C++ GetErrorPercentage default)
            error_pct, max_err = get_error_percentage(pred_data, expected_data, threshold=0.1)
            
            limit = thresholds.get(output_name, 0.05)
            status = "PASS" if error_pct <= limit else "FAIL"
            
            print(f"Checking {output_name}:")
            print(f"  Error > 0.1 percentage = {error_pct:.5f} (Expected <= {limit})")
            print(f"  MaxError = {max_err:.5f}")
            print(f"  Pred range: [{np.min(pred_data):.5f}, {np.max(pred_data):.5f}]")
            print(f"  Expd range: [{np.min(expected_data):.5f}, {np.max(expected_data):.5f}]")
            print(f"  Status: {status}\n")
            
        except Exception as e:
            logger.error(f"Error verifying {output_name}: {e}")

if __name__ == "__main__":
    main()

