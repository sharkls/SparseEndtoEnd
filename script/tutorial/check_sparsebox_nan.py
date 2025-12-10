import os
import numpy as np
import glob

def check_nan_inf(path, dtype=np.float16):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    data = np.fromfile(path, dtype=dtype)
    has_nan = np.isnan(data).any()
    has_inf = np.isinf(data).any()
    
    stats = f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}"
    print(f"Checking {os.path.basename(path)}:")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Stats: {stats}")
    
    if has_nan:
        print(f"  NaN indices: {np.where(np.isnan(data))[0][:5]}")

def main():
    base_dir = "deploy/val_data_plugin/real_data/sample_0"
    
    print("--- Checking SparseBox 0 ---")
    sb_dir = os.path.join(base_dir, "sparsebox_0")
    check_nan_inf(glob.glob(os.path.join(sb_dir, "*anchor*float16.bin"))[0])
    check_nan_inf(glob.glob(os.path.join(sb_dir, "*feature*float16.bin"))[0])
    check_nan_inf(glob.glob(os.path.join(sb_dir, "*keypoints*float16.bin"))[0])

    print("\n--- Checking SparseBox 1 ---")
    sb_dir = os.path.join(base_dir, "sparsebox_1")
    check_nan_inf(glob.glob(os.path.join(sb_dir, "*anchor*float16.bin"))[0])
    check_nan_inf(glob.glob(os.path.join(sb_dir, "*feature*float16.bin"))[0])
    check_nan_inf(glob.glob(os.path.join(sb_dir, "*keypoints*float16.bin"))[0])

if __name__ == "__main__":
    main()

