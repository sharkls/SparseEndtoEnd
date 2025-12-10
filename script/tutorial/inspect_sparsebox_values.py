import os
import glob
import numpy as np
import torch

def load_bin(path, dtype=np.float32):
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=dtype)
    return data

def inspect_dir(dir_path):
    print(f"\n--- Inspecting {os.path.basename(dir_path)} ---")
    
    # Feature
    feat_paths = glob.glob(os.path.join(dir_path, "*feature_*.bin"))
    if not feat_paths:
        print("Feature file not found")
        return
    feat = load_bin(feat_paths[0])
    print(f"Feature: shape={feat.shape}, min={feat.min()}, max={feat.max()}, mean={feat.mean()}, std={feat.std()}")
    
    # Anchor
    anchor_paths = glob.glob(os.path.join(dir_path, "*anchor_*.bin"))
    if not anchor_paths:
        print("Anchor file not found")
        return
    anchor = load_bin(anchor_paths[0])
    anchor = anchor.reshape(-1, 11)
    print(f"Anchor: shape={anchor.shape}")
    
    # Analyze specific anchor fields
    # [x, y, z, w, l, h, sin, cos, vx, vy, vz]
    xyz = anchor[:, 0:3]
    wlh = anchor[:, 3:6]
    sincos = anchor[:, 6:8]
    
    print(f"XYZ: min={xyz.min()}, max={xyz.max()}")
    print(f"WLH (log sizes): min={wlh.min()}, max={wlh.max()}")
    print(f"SinCos: min={sincos.min()}, max={sincos.max()}")
    
    # Check for potential issues
    if wlh.max() > 10:
        print(f"WARNING: Large log size detected! Max: {wlh.max()} -> exp({wlh.max()}) = {np.exp(wlh.max())}")
    
    if np.any(np.abs(sincos) > 1.01):
        print(f"WARNING: Sin/Cos > 1 detected! Max abs: {np.max(np.abs(sincos))}")

def main():
    base_dir = "deploy/val_data_plugin/real_data/sample_0"
    inspect_dir(os.path.join(base_dir, "sparsebox_0")) # Bad
    inspect_dir(os.path.join(base_dir, "sparsebox_2")) # Good

if __name__ == "__main__":
    main()
