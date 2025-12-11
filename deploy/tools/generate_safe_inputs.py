import numpy as np
import os
import argparse

def generate_safe_inputs(output_dir="deploy/val_data_trtexec"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认配置 (与 export_head_onnxv2.py 保持一致)
    NUMS_CAM = 6
    INPUT_H = 256
    INPUT_W = 704
    
    h_4x, w_4x = INPUT_H // 4, INPUT_W // 4
    h_8x, w_8x = INPUT_H // 8, INPUT_W // 8
    h_16x, w_16x = INPUT_H // 16, INPUT_W // 16
    h_32x, w_32x = INPUT_H // 32, INPUT_W // 32
    
    # 1. 生成 spatial_shapes [6, 4, 2]
    # 形状: [nums_cam, num_levels, 2] -> [H, W]
    shapes_per_cam = np.array([
        [h_4x, w_4x], 
        [h_8x, w_8x], 
        [h_16x, w_16x], 
        [h_32x, w_32x]
    ], dtype=np.int32)
    
    # 复制到所有相机
    spatial_shapes = np.tile(shapes_per_cam[None, ...], (NUMS_CAM, 1, 1))
    
    # 2. 生成 level_start_index [6, 4]
    # 计算每一层的特征点数量 (面积)
    areas = shapes_per_cam[:, 0] * shapes_per_cam[:, 1]
    # 计算起始索引 (累加)
    starts = np.cumsum(areas)
    # 也就是 [0, area0, area0+area1, area0+area1+area2]
    starts = np.concatenate(([0], starts[:-1]))
    
    level_start_index = np.tile(starts[None, ...], (NUMS_CAM, 1)).astype(np.int32)
    
    print(f"[Info] Generating safe inputs for trtexec...")
    print(f"  spatial_shapes: {spatial_shapes.shape}, dtype={spatial_shapes.dtype}")
    print(f"  level_start_index: {level_start_index.shape}, dtype={level_start_index.dtype}")
    
    sp_path = os.path.join(output_dir, "spatial_shapes.bin")
    ls_path = os.path.join(output_dir, "level_start_index.bin")
    
    spatial_shapes.tofile(sp_path)
    level_start_index.tofile(ls_path)
    
    print(f"[Success] Saved safe inputs to:\n  {sp_path}\n  {ls_path}")

if __name__ == "__main__":
    generate_safe_inputs()

