import os
import argparse
import numpy as np

def convert_npz_to_bin(npz_path, output_dir):
    if not os.path.exists(npz_path):
        print(f"[Error] NPZ file not found: {npz_path}")
        return False
        
    print(f"[Info] Converting {npz_path} to bins in {output_dir}...")
    data = np.load(npz_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for key in data.files:
        tensor = data[key]
        # Ensure float32 for most inputs unless specified otherwise by trtexec types
        # trtexec expects raw binary bytes.
        
        # Flatten and save
        bin_path = os.path.join(output_dir, f"input_{key}.bin")
        tensor.tofile(bin_path)
        print(f"  - Saved {key}: {tensor.shape} -> {bin_path}")
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head1-npz", type=str, default="deploy/calibration_data/head1/000000.npz")
    parser.add_argument("--head2-npz", type=str, default="deploy/calibration_data/head2/000000.npz")
    parser.add_argument("--output-dir", type=str, default="deploy/calibration_data_bin")
    args = parser.parse_args()
    
    # Convert Head 1
    if os.path.exists(args.head1_npz):
        convert_npz_to_bin(args.head1_npz, os.path.join(args.output_dir, "head1"))
    else:
        print(f"[Warning] Head1 NPZ not found at {args.head1_npz}")

    # Convert Head 2
    if os.path.exists(args.head2_npz):
        convert_npz_to_bin(args.head2_npz, os.path.join(args.output_dir, "head2"))
    else:
        print(f"[Warning] Head2 NPZ not found at {args.head2_npz}")
