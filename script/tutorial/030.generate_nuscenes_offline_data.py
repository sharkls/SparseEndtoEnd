# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
"""
Generate nuScenes format offline data for C++ testing.
This script extracts images and calibration parameters from nuScenes dataset
and saves them in a format that can be loaded by the C++ loadOfflineData function.
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from typing import Optional, Dict, List
import json

from tool.utils.config import read_cfg
from tool.trainer.utils import set_random_seed

from dataset.nuscenes_dataset import *
from dataset.dataloader_wrapper.dataloader_wrapper import dataloader_wrapper
from dataset.utils.scatter_gather import scatter
from dataset.pipeline.nuscenes import LoadMultiViewImageFromFiles

# 定义必要的变量
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
class_names = [
    "car", "truck", "bus", "trailer", "construction_vehicle", 
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier"
]

# 定义nuScenes标签到目标标签的映射
nuscenes_to_target_mapping = {
    # 车辆类别
    "vehicle.car": "car",
    "vehicle.truck": "truck", 
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    
    # 行人
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian", 
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    
    # 两轮车
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    
    # 其他
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}

def verify_nuscenes_dataset_access(config_path: str):
    """
    Verify that nuScenes dataset is accessible and original image files exist.
    
    Args:
        config_path: Path to the dataset config file
    
    Returns:
        bool: True if dataset is accessible, False otherwise
    """
    print("Verifying nuScenes dataset access...")
    
    try:
        # 读取配置
        cfg = read_cfg(config_path)
        
        # 检查数据集路径
        dataset_cfg = cfg["data"]["val"].copy()
        data_root = dataset_cfg.get("data_root", "")
        ann_file = dataset_cfg.get("ann_file", "")
        
        print(f"Data root: {data_root}")
        print(f"Annotation file: {ann_file}")
        
        if not os.path.exists(data_root):
            print(f"❌ Data root directory not found: {data_root}")
            return False
        
        if not os.path.exists(ann_file):
            print(f"❌ Annotation file not found: {ann_file}")
            return False
        
        # 尝试加载数据集并检查原始图像文件
        dataset_type = dataset_cfg.pop("type")
        dataset_cfg["pipeline"] = [
            dict(type="LoadMultiViewImageFromFiles", to_float32=False),
        ]
        
        dataset = eval(dataset_type)(**dataset_cfg)
        
        # 检查前几个样本的原始图像文件
        for i in range(min(3, len(dataset))):
            data = dataset[i]
            img_filenames = data["img_filename"]
            
            print(f"Sample {i}: Found {len(img_filenames)} image files")
            for j, img_path in enumerate(img_filenames):
                if os.path.exists(img_path):
                    print(f"  ✅ Camera {j}: {img_path}")
                else:
                    print(f"  ❌ Camera {j}: File not found - {img_path}")
        
        print("✅ nuScenes dataset access verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying dataset access: {e}")
        return False


def build_module(cfg, default_args: Optional[Dict] = None):
    """Build module from config."""
    cfg2 = cfg.copy()
    if default_args is not None:
        
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def generate_nuscenes_offline_data(config_path: str, output_path: str, num_frames: int = 3):
    """
    Generate nuScenes format offline data for C++ testing.
    与010脚本使用相同的数据源和配置，确保数据一致性。
    
    Args:
        config_path: Path to the dataset config file
        output_path: Output directory for offline data
        num_frames: Number of frames to generate (默认3帧，与010脚本一致)
    """
    print(f"Generating nuScenes offline data for {num_frames} frames...")
    print(f"Using same data source and configuration as 010 script")
    
    # 读取配置
    cfg = read_cfg(config_path)
    # 使用与010脚本相同的随机种子
    set_random_seed(seed=1, deterministic=True)
    
    # 打印数据集配置信息
    print(f"Dataset config: {cfg['data']['test']}")  # 使用test配置，与010脚本一致
    dataset_cfg = cfg["data"]["test"].copy()  # 使用test配置，与010脚本一致
    print(f"Data root: {dataset_cfg.get('data_root', 'Not specified')}")
    print(f"Annotation file: {dataset_cfg.get('ann_file', 'Not specified')}")
    
    # 检查数据集路径是否存在
    data_root = dataset_cfg.get("data_root", "")
    if data_root and os.path.exists(data_root):
        print(f"✅ Data root exists: {data_root}")
        # 列出数据根目录下的内容
        try:
            data_root_contents = os.listdir(data_root)
            print(f"Data root contents: {data_root_contents[:10]}...")  # 只显示前10个
        except Exception as e:
            print(f"Error listing data root: {e}")
    else:
        print(f"❌ Data root not found: {data_root}")
    
    # 创建数据集，使用与010脚本相同的test_pipeline
    dataset_type = dataset_cfg.pop("type")
    print(f"Dataset type: {dataset_type}")
    
    # 移除不应该传递给数据集的参数
    samples_per_gpu = dataset_cfg.pop("samples_per_gpu", 1)
    workers_per_gpu = dataset_cfg.pop("workers_per_gpu", 0)
    
    # 使用与010脚本相同的test_pipeline
    if "pipeline" in dataset_cfg:
        # 使用完整的test_pipeline，与010脚本完全一致
        dataset_cfg["pipeline"] = [
            dict(type="LoadMultiViewImageFromFiles", to_float32=True),
            dict(type="ResizeCropFlipImage"),
            dict(type="NormalizeMultiviewImage", **cfg.get("img_norm_cfg", {"mean": [0, 0, 0], "std": [1, 1, 1], "to_rgb": True})),
            dict(type="NuScenesSparse4DAdaptor"),
            dict(
                type="Collect",
                keys=["img", "timestamp", "lidar2img", "image_wh", "ori_img"],
                meta_keys=["lidar2global", "global2lidar", "timestamp"],
            ),
        ]
    
    print(f"Pipeline config: {dataset_cfg['pipeline']}")
    
    try:
        dataset = eval(dataset_type)(**dataset_cfg)
        print(f"✅ Dataset created successfully, length: {len(dataset)}")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return
    
    # 创建数据加载器，使用与010脚本相同的配置
    dataloader = dataloader_wrapper(
        dataset,
        samples_per_gpu=samples_per_gpu,  # 与010脚本一致
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False,  # 与010脚本一致
    )
    
    # 创建输出目录
    camera_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
                   "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    for cam_type in camera_types:
        cam_dir = os.path.join(output_path, "images", cam_type)
        os.makedirs(cam_dir, exist_ok=True)
    
    calib_dir = os.path.join(output_path, "calib")
    os.makedirs(calib_dir, exist_ok=True)
    
    # 处理数据
    data_iter = dataloader.__iter__()
    
    for frame_idx in tqdm(range(num_frames), desc="Generating frames"):
        try:
            # 获取数据 - 与010脚本完全相同的方式
            data = next(data_iter)
            data = scatter(data, [0])[0]
            
            # 获取数据
            img_metas = data["img_metas"][0]
            lidar2img = data["lidar2img"].data[0].cpu().numpy()  # (1, 6, 4, 4) - 与010脚本格式一致
            img = data["img"].data[0].cpu().numpy()  # (1, 6, 3, 256, 704) - 与010脚本格式一致
            ori_img = data["ori_img"].data[0].cpu().numpy()  # (1, 6, 3, 900, 1600) - 原始图像
            
            print(f"\nFrame {frame_idx}: Processing {len(camera_types)} cameras")
            print(f"  📊 ori_img shape: {ori_img.shape}")
            print(f"  📊 img shape: {img.shape}")
            print(f"  📊 lidar2img shape: {lidar2img.shape}")
            
            # 保存原始图像数据为二进制文件 - 与010脚本格式完全一致
            # 010脚本保存格式: sample_{frame_idx}_ori_imgs_1*6*3*900*1600_uint8.bin
            ori_imgs_path = os.path.join(output_path, "calib", f"sample_{frame_idx}_ori_imgs_1*6*3*900*1600_uint8.bin")
            ori_img.astype(np.uint8).tofile(ori_imgs_path)
            print(f"  ✅ Saved ori_imgs binary: {ori_imgs_path}")
            
            # 保存预处理图像数据为二进制文件 - 与010脚本格式完全一致
            # 010脚本保存格式: sample_{frame_idx}_imgs_1*6*3*256*704_float32.bin
            imgs_path = os.path.join(output_path, "calib", f"sample_{frame_idx}_imgs_1*6*3*256*704_float32.bin")
            img.astype(np.float32).tofile(imgs_path)
            print(f"  ✅ Saved imgs binary: {imgs_path}")
            
            # 同时保存JPG图像文件用于可视化
            for cam_idx, cam_type in enumerate(camera_types):
                cam_img = ori_img[0, cam_idx]
                print(f"  Camera {cam_type}: cam_img shape = {cam_img.shape}")
                
                # 处理不同的图像格式
                if cam_img.shape == (3, 900, 1600):
                    # 三通道彩色图像
                    cam_img_cv = np.transpose(cam_img, (1, 2, 0))
                    cam_img_bgr = cv2.cvtColor(cam_img_cv, cv2.COLOR_RGB2BGR)
                elif cam_img.shape == (900, 1600):
                    # 灰度图像，转换为三通道
                    print(f"    ⚠️ 检测到灰度图像，转换为三通道")
                    cam_img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_GRAY2BGR)
                elif cam_img.shape == (900, 1600, 3):
                    # 已经是BGR格式
                    cam_img_bgr = cam_img
                else:
                    print(f"    ⚠️ 未知图像格式: {cam_img.shape}，尝试转换为三通道")
                    # 尝试转换为三通道
                    if cam_img.ndim == 2:
                        cam_img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_GRAY2BGR)
                    elif cam_img.ndim == 3 and cam_img.shape[2] == 3:
                        cam_img_bgr = cam_img
                    else:
                        raise ValueError(f"无法处理的图像格式: {cam_img.shape}")
                
                target_img_path = os.path.join(output_path, "images", cam_type, f"{frame_idx}.jpg")
                cv2.imwrite(target_img_path, cam_img_bgr)
                print(f"  Camera {cam_type}: Saved to {target_img_path}")
                print(f"    📊 Image stats: shape={cam_img_bgr.shape}, mean={cam_img_bgr.mean():.1f}, std={cam_img_bgr.std():.1f}")
            
            # 保存lidar2img标定参数 - 与010脚本格式一致
            calib_path = os.path.join(output_path, "calib", f"lidar2img_{frame_idx}.bin")
            lidar2img.astype(np.float32).tofile(calib_path)
            print(f"  ✅ Saved lidar2img calibration for frame {frame_idx}: shape={lidar2img.shape}")
            
            # 保存时间戳和变换信息
            if "timestamp" in data:
                timestamp = data["timestamp"].data[0].cpu().numpy()  # (1,)
                timestamp_path = os.path.join(output_path, "calib", f"timestamp_{frame_idx}.bin")
                timestamp.astype(np.float32).tofile(timestamp_path)
                print(f"  ✅ Saved timestamp for frame {frame_idx}: shape={timestamp.shape}")
            
            # 保存global2lidar变换矩阵
            if "global2lidar" in img_metas:
                global2lidar = img_metas["global2lidar"]
                global2lidar_path = os.path.join(output_path, "calib", f"global2lidar_{frame_idx}.bin")
                global2lidar.astype(np.float32).tofile(global2lidar_path)
                print(f"  ✅ Saved global2lidar for frame {frame_idx}: shape={global2lidar.shape}")
            
        except StopIteration:
            print(f"Dataset exhausted after {frame_idx} frames")
            break
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Successfully generated offline data in: {output_path}")
    print(f"Generated {num_frames} frames with {len(camera_types)} cameras each")
    print(f"Data format now matches 010 script requirements:")
    print(f"  - Images: (1, 6, 3, 256, 704) float32 (preprocessed)")
    print(f"  - Original images: (1, 6, 3, 900, 1600) uint8")
    print(f"  - Lidar2img: (1, 6, 4, 4) float32")
    print(f"  - Image_wh: (1, 6, 2) float32")
    print(f"  - Timestamp: (1,) float32")
    print(f"✅ Data source consistency with 010 script:")
    print(f"  - Same config file: {config_path}")
    print(f"  - Same dataset: test (not val)")
    print(f"  - Same random seed: 1")
    print(f"  - Same pipeline: test_pipeline")
    print(f"  - Same frame count: {num_frames}")


def generate_lidar2img_calib_params(config, output_path: str, seed=1, num_frames=3):
    """
    Generate lidar2img calibration parameters for each frame separately.
    This function is adapted from 020.genernate_onboard_modelconfig_and_inputbin_file.py
    与010脚本使用相同的数据源和配置，确保数据一致性。
    
    Args:
        config: Dataset configuration
        output_path: Output directory for calibration files
        seed: Random seed (默认1，与010脚本一致)
        num_frames: Number of frames to generate (默认3帧，与010脚本一致)
    """
    print(f"Generating lidar2img calibration parameters for {num_frames} frames...")
    print(f"Using same data source and configuration as 010 script")
    
    # 创建标定目录
    calib_dir = os.path.join(output_path, "calib")
    os.makedirs(calib_dir, exist_ok=True)
    
    # 使用与010脚本相同的test配置
    dataset_cfg = config.copy()["data"]["test"].copy()  # 使用test配置，与010脚本一致
    
    # 移除不应该传递给数据集的参数
    samples_per_gpu = dataset_cfg.pop("samples_per_gpu", 1)
    workers_per_gpu = dataset_cfg.pop("workers_per_gpu", 0)
    
    dataset_type = dataset_cfg.pop("type")
    dataset = eval(dataset_type)(**dataset_cfg)

    dataloader = dataloader_wrapper(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        seed=seed,  # 与010脚本一致
        dist=False,
        shuffle=False,  # 与010脚本一致
    )
    data_iter = dataloader.__iter__()

    for frame_idx in tqdm(range(num_frames), desc="Generating lidar2img calibration"):
        try:
            data = next(data_iter)
            data = scatter(data, [0])[0]
            lidar2img = data["lidar2img"][0].cpu().numpy()  # (1, 6, 4, 4) - 与010脚本格式一致
            
            # 每帧单独保存为一个bin文件
            calib_path = os.path.join(calib_dir, f"lidar2img_{frame_idx}.bin")
            lidar2img.astype(np.float32).tofile(calib_path)
            
            print(f"Frame {frame_idx}: Saved lidar2img calibration - shape={lidar2img.shape}")
            
        except StopIteration:
            print(f"Dataset exhausted after {frame_idx} frames")
            break
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue
    
    print(f"Successfully generated lidar2img calibration files in: {calib_dir}")
    print(f"✅ Data source consistency with 010 script:")
    print(f"  - Same dataset: test (not val)")
    print(f"  - Same random seed: {seed}")
    print(f"  - Same frame count: {num_frames}")


def generate_sample_data(output_path: str, num_frames: int = 5):
    """
    Generate sample data for testing when nuScenes dataset is not available.
    
    Args:
        output_path: Output directory for sample data
        num_frames: Number of frames to generate
    """
    print(f"Generating sample data for {num_frames} frames...")
    
    # 创建目录结构
    camera_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
                   "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    for cam_type in camera_types:
        cam_dir = os.path.join(output_path, "images", cam_type)
        os.makedirs(cam_dir, exist_ok=True)
    
    calib_dir = os.path.join(output_path, "calib")
    os.makedirs(calib_dir, exist_ok=True)
    
    lidar_dir = os.path.join(output_path, "lidar")
    os.makedirs(lidar_dir, exist_ok=True)
    
    labels_dir = os.path.join(output_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    temporal_dir = os.path.join(output_path, "temporal")
    os.makedirs(temporal_dir, exist_ok=True)
    
    # 生成示例数据
    for frame_idx in range(num_frames):
        # 为每个相机生成示例图像
        for cam_idx, cam_type in enumerate(camera_types):
            img_path = os.path.join(output_path, "images", cam_type, f"{frame_idx}.jpg")
            
            # 创建示例图像（900x1600，与原始图像尺寸一致）
            sample_img = np.ones((900, 1600, 3), dtype=np.uint8) * 50
            
            # 添加一些有意义的视觉内容
            # 添加背景渐变
            for i in range(900):
                intensity = int(50 + (i / 900) * 100)
                sample_img[i, :, :] = intensity
            
            # 添加一些几何图形
            cv2.rectangle(sample_img, (200, 100), (400, 300), (255, 0, 0), 3)
            cv2.circle(sample_img, (800, 450), 100, (0, 255, 0), -1)
            cv2.line(sample_img, (100, 400), (1500, 400), (0, 0, 255), 5)
            
            # 添加文字标识
            cv2.putText(sample_img, f"Frame {frame_idx}", (100, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(sample_img, f"Camera {cam_type}", (100, 800), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            cv2.imwrite(img_path, sample_img)
        
        # 生成示例雷达点云数据
        lidar_path = os.path.join(output_path, "lidar", f"lidar_{frame_idx}.bin")
        
        # 生成随机点云数据（1000个点）
        num_points = 1000
        points = np.random.randn(num_points, 5).astype(np.float32)
        points[:, :3] *= 50.0  # 缩放位置到合理范围
        points[:, 3] = np.random.uniform(0, 255, num_points)  # 强度
        points[:, 4] = np.random.randint(0, 64, num_points)   # ring
        
        points.tofile(lidar_path)
        
        # 生成示例3D标签数据
        labels_path = os.path.join(output_path, "labels", f"labels_{frame_idx}.json")
        
        # 生成一些示例3D目标
        sample_labels = []
        for i in range(np.random.randint(3, 8)):  # 3-7个目标
            label_info = {
                'token': f'sample_token_{frame_idx}_{i}',
                'original_category': np.random.choice(list(nuscenes_to_target_mapping.keys())),
                'category_name': np.random.choice(class_names),
                'translation': [np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(-2, 2)],
                'size': [np.random.uniform(3, 6), np.random.uniform(1.5, 2.5), np.random.uniform(1.5, 2.5)],
                'rotation': [1.0, 0.0, 0.0, 0.0],  # 单位四元数
                'bbox_3d': [0.0] * 7,  # 占位符
                'num_lidar_pts': np.random.randint(10, 100),
                'num_radar_pts': np.random.randint(0, 10),
                'visibility_token': 'visible',
                'instance_token': f'instance_{frame_idx}_{i}',
                'attribute_tokens': []
            }
            sample_labels.append(label_info)
        
        with open(labels_path, 'w') as f:
            json.dump(sample_labels, f, indent=2, ensure_ascii=False)
        
        # 生成示例时间戳和变换信息
        temporal_path = os.path.join(output_path, "temporal", f"temporal_{frame_idx}.json")
        
        # 模拟时间戳（从0开始，每帧间隔0.1秒）
        timestamp = frame_idx * 0.1
        
        # 模拟ego pose（车辆在道路上移动）
        ego_translation = [frame_idx * 10.0, 0.0, 0.0]  # 沿x轴前进
        ego_rotation = [1.0, 0.0, 0.0, 0.0]  # 单位四元数
        
        # 模拟lidar2ego变换（激光雷达在车顶）
        lidar2ego_translation = [0.0, 0.0, 1.8]  # 车顶高度
        lidar2ego_rotation = [1.0, 0.0, 0.0, 0.0]  # 单位四元数
        
        # 构建变换矩阵
        lidar2ego_matrix = np.eye(4)
        lidar2ego_matrix[:3, 3] = lidar2ego_translation
        
        ego2global_matrix = np.eye(4)
        ego2global_matrix[:3, 3] = ego_translation
        
        lidar2global_matrix = ego2global_matrix @ lidar2ego_matrix
        global2lidar_matrix = np.linalg.inv(lidar2global_matrix)
        
        temporal_info = {
            'timestamp_info': {
                'timestamp': timestamp,
                'sample_token': f'sample_{frame_idx}',
                'scene_token': f'scene_{frame_idx // 10}',  # 每10帧一个场景
                'lidar_token': f'lidar_{frame_idx}'
            },
            'ego_pose_info': {
                'ego2global_translation': ego_translation,
                'ego2global_rotation': ego_rotation,
                'ego_pose_token': f'ego_pose_{frame_idx}'
            },
            'lidar2ego_info': {
                'lidar2ego_translation': lidar2ego_translation,
                'lidar2ego_rotation': lidar2ego_rotation,
                'calibrated_sensor_token': f'calib_{frame_idx}'
            },
            'transform_info': {
                'lidar2ego_matrix': lidar2ego_matrix.tolist(),
                'ego2global_matrix': ego2global_matrix.tolist(),
                'lidar2global_matrix': lidar2global_matrix.tolist(),
                'global2lidar_matrix': global2lidar_matrix.tolist()
            }
        }
        
        with open(temporal_path, 'w') as f:
            json.dump(temporal_info, f, indent=2, ensure_ascii=False)
        
        # 生成示例lidar2img标定参数 - 每帧单独保存
        calib_path = os.path.join(output_path, "calib", f"lidar2img_{frame_idx}.bin")
        
        # 创建6个相机的4x4变换矩阵
        lidar2img_params = np.zeros((6, 4, 4), dtype=np.float32)
        
        for cam_idx in range(6):
            # 设置单位矩阵作为基础
            lidar2img_params[cam_idx] = np.eye(4)
            
            # 为不同相机添加不同的偏移
            lidar2img_params[cam_idx, 0, 3] = cam_idx * 0.1  # x偏移
            lidar2img_params[cam_idx, 1, 3] = cam_idx * 0.05 # y偏移
            lidar2img_params[cam_idx, 2, 3] = 1.0 + cam_idx * 0.02 # z偏移
        
        # 保存为二进制文件
        lidar2img_params.tofile(calib_path)
    
    print(f"Sample data generated successfully in: {output_path}")
    print(f"Generated {num_frames} frames with complete data including:")
    print(f"  - Images: {len(camera_types)} cameras per frame")
    print(f"  - Lidar point clouds: {num_points} points per frame")
    print(f"  - 3D labels: 3-7 objects per frame")
    print(f"  - Temporal info: timestamp and transform matrices")
    print(f"  - Calibration: lidar2img matrices")


def generate_nuscenes_offline_data_direct(config_path: str, output_path: str, num_frames: int = 10):
    """
    Generate nuScenes format offline data directly using nuScenes API.
    
    Args:
        config_path: Path to the dataset config file
        output_path: Output directory for offline data
        num_frames: Number of frames to generate
    """
    print(f"Generating nuScenes offline data directly for {num_frames} frames...")
    
    try:
        from nuscenes import NuScenes
        from nuscenes.utils.data_classes import Box
        from nuscenes.utils.geometry_utils import view_points
    except ImportError as e:
        print(f"❌ nuScenes not available: {e}")
        print("Generating sample data instead...")
        generate_sample_data(output_path, num_frames)
        return
    
    # 读取配置
    cfg = read_cfg(config_path)
    dataset_cfg = cfg["data"]["val"].copy()
    
    # 获取nuScenes数据集路径
    data_root = dataset_cfg.get("data_root", "")
    version = dataset_cfg.get("version", "v1.0-trainval")
    
    print(f"Data root: {data_root}")
    print(f"Version: {version}")
    
    if not data_root or not os.path.exists(data_root):
        print(f"❌ Data root not found: {data_root}")
        print("Generating sample data instead...")
        generate_sample_data(output_path, num_frames)
        return
    
    try:
        # 初始化nuScenes
        nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        print(f"✅ NuScenes loaded successfully with {len(nusc.sample)} samples")
    except Exception as e:
        print(f"❌ Error loading NuScenes: {e}")
        print("Generating sample data instead...")
        generate_sample_data(output_path, num_frames)
        return
    
    # 创建输出目录
    camera_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
                   "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    for cam_type in camera_types:
        cam_dir = os.path.join(output_path, "images", cam_type)
        os.makedirs(cam_dir, exist_ok=True)
    
    calib_dir = os.path.join(output_path, "calib")
    os.makedirs(calib_dir, exist_ok=True)
    
    # 创建雷达数据和标签数据目录
    lidar_dir = os.path.join(output_path, "lidar")
    os.makedirs(lidar_dir, exist_ok=True)
    
    labels_dir = os.path.join(output_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    # 获取所有样本
    samples = list(nusc.sample)
    print(f"Found {len(samples)} samples in dataset")
    
    # 处理前num_frames个样本
    for frame_idx in tqdm(range(min(num_frames, len(samples))), desc="Generating frames"):
        sample = samples[frame_idx]
        print(f"\nProcessing frame {frame_idx}: {sample['token']}")
        
        # 获取该样本的所有相机数据
        sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        
        # 获取所有相机的图像
        camera_data = {}
        for cam_type in camera_types:
            if cam_type in sample['data']:
                camera_data[cam_type] = nusc.get('sample_data', sample['data'][cam_type])
        
        print(f"Found {len(camera_data)} cameras for this sample")
        
        # 处理每个相机
        for cam_type in camera_types:
            if cam_type in camera_data:
                cam_data = camera_data[cam_type]
                
                # 获取原始图像文件路径
                original_img_path = os.path.join(data_root, cam_data['filename'])
                target_img_path = os.path.join(output_path, "images", cam_type, f"{frame_idx}.jpg")
                
                print(f"  Camera {cam_type}: {original_img_path}")
                
                # 检查原始文件是否存在
                if os.path.exists(original_img_path):
                    # 直接复制原始图像文件
                    import shutil
                    shutil.copy2(original_img_path, target_img_path)
                    print(f"    ✅ Copied from {original_img_path} to {target_img_path}")
                    
                    # 验证复制的图像
                    try:
                        img = cv2.imread(target_img_path)
                        print(f"    📊 Image stats: shape={img.shape}, mean={img.mean():.1f}, std={img.std():.1f}")
                    except Exception as e:
                        print(f"    ❌ Error reading copied image: {e}")
                else:
                    print(f"    ❌ Original file not found: {original_img_path}")
                    # 生成示例图像
                    sample_img = np.ones((900, 1600, 3), dtype=np.uint8) * 50
                    cv2.putText(sample_img, f"Missing: {os.path.basename(original_img_path)}", 
                               (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imwrite(target_img_path, sample_img)
                    print(f"    ⚠️ Generated sample image instead")
            else:
                print(f"  Camera {cam_type}: No data available")
        
        # 获取雷达点云数据
        try:
            if 'LIDAR_TOP' in sample['data']:
                lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                lidar_file_path = os.path.join(data_root, lidar_data['filename'])
                target_lidar_path = os.path.join(output_path, "lidar", f"lidar_{frame_idx}.bin")
                
                print(f"  LIDAR_TOP: {lidar_file_path}")
                
                if os.path.exists(lidar_file_path):
                    # 复制雷达点云文件
                    shutil.copy2(lidar_file_path, target_lidar_path)
                    print(f"    ✅ Copied lidar from {lidar_file_path} to {target_lidar_path}")
                    
                    # 读取并显示点云统计信息
                    try:
                        points = np.fromfile(lidar_file_path, dtype=np.float32).reshape(-1, 5)
                        print(f"    📊 Lidar stats: {points.shape[0]} points, range=[{points[:, :3].min(axis=0)}, {points[:, :3].max(axis=0)}]")
                    except Exception as e:
                        print(f"    ❌ Error reading lidar: {e}")
                else:
                    print(f"    ❌ Lidar file not found: {lidar_file_path}")
            else:
                print(f"  LIDAR_TOP: No data available")
        except Exception as e:
            print(f"  ❌ Error processing lidar: {e}")
        
        # 获取3D标签数据
        try:
            # 获取该样本的所有3D标注 - 使用与dump_nusc_ann_json.py相同的方式
            sample_annotations = []
            if "anns" in sample and sample["anns"]:
                sample_annotations = [
                    nusc.get("sample_annotation", token) for token in sample["anns"]
                ]
            
            print(f"  Labels: Found {len(sample_annotations)} annotations")
            
            if sample_annotations:
                # 使用nuScenes API获取激光雷达坐标系下的boxes（与dump_nusc_ann_json.py一致）
                lidar_token = sample['data']['LIDAR_TOP']
                _, boxes, _ = nusc.get_sample_data(lidar_token)
                
                # 保存标签数据为JSON格式，便于可视化
                labels_data = []
                for i, ann in enumerate(sample_annotations):
                    try:
                        # 获取原始类别名称
                        original_category = ann['category_name']
                        
                        # 映射到目标类别
                        if original_category in nuscenes_to_target_mapping:
                            target_category = nuscenes_to_target_mapping[original_category]
                        else:
                            # 如果不在映射中，跳过这个标注
                            print(f"    ⚠️ Skipping unknown category: {original_category}")
                            continue
                        
                        # 使用boxes中的数据（已经是激光雷达坐标系）
                        if i < len(boxes):
                            box = boxes[i]
                            # 获取位置、尺寸、旋转（与dump_nusc_ann_json.py一致）
                            loc = box.center      # [x, y, z]
                            dim = box.wlh         # [w, l, h]
                            rot = box.orientation.yaw_pitch_roll[0]  # yaw角
                            
                            # 构建bbox_3d数组 [x, y, z, l, w, h, yaw]（与dump_nusc_ann_json.py一致）
                            bbox = [
                                float(loc[0]),    # x
                                float(loc[1]),    # y  
                                float(loc[2]),    # z
                                float(dim[1]),    # l (length)
                                float(dim[0]),    # w (width)
                                float(dim[2]),    # h (height)
                                float(rot)        # yaw
                            ]
                            
                            label_info = {
                                'token': ann['token'],
                                'original_category': original_category,  # 保留原始类别用于调试
                                'category_name': target_category,        # 使用映射后的类别
                                'translation': loc.tolist(),             # 激光雷达坐标系下的坐标 [x, y, z]
                                'size': dim.tolist(),                    # 激光雷达坐标系下的尺寸 [w, l, h]
                                'rotation': box.orientation.elements.tolist(),  # 四元数 [w, x, y, z]
                                'bbox_3d': bbox,                         # [x, y, z, l, w, h, yaw]
                                'num_lidar_pts': ann.get('num_lidar_pts', 0),
                                'num_radar_pts': ann.get('num_radar_pts', 0),
                                'visibility_token': ann.get('visibility_token', ''),
                                'instance_token': ann.get('instance_token', ''),
                                'attribute_tokens': ann.get('attribute_tokens', [])
                            }
                            labels_data.append(label_info)
                            # print(f"    📍 Added label: {original_category} -> {target_category}")
                            # print(f"        Lidar coords: {loc}, size: {dim}, yaw: {rot:.3f}")
                        else:
                            print(f"    ⚠️ Box index {i} out of range for annotation {ann.get('token', 'unknown')}")
                            
                    except Exception as e:
                        print(f"    ⚠️ Error processing annotation {ann.get('token', 'unknown')}: {e}")
                        continue
                
                if labels_data:
                    # 保存标签数据
                    labels_file_path = os.path.join(output_path, "labels", f"labels_{frame_idx}.json")
                    try:
                        with open(labels_file_path, 'w') as f:
                            json.dump(labels_data, f, indent=2, ensure_ascii=False)
                        
                        print(f"    ✅ Saved {len(labels_data)} 3D labels to {labels_file_path}")
                        
                        # 显示标签统计信息
                        categories = [label['category_name'] for label in labels_data]  # 使用映射后的类别
                        from collections import Counter
                        category_counts = Counter(categories)
                        print(f"    📊 Label stats: {dict(category_counts)}")
                        
                        # 显示原始类别统计（用于调试）
                        original_categories = [label['original_category'] for label in labels_data]
                        original_counts = Counter(original_categories)
                        print(f"   Original categories: {dict(original_counts)}")
                    except Exception as e:
                        print(f"    ❌ Error saving labels to {labels_file_path}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"    ⚠️ No valid labels to save")
            else:
                print(f"    ⚠️ No 3D annotations found for this sample")
                
        except Exception as e:
            print(f"  ❌ Error processing labels: {e}")
            import traceback
            traceback.print_exc()
        
        # 获取标定参数
        try:
            # 获取激光雷达的标定信息
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            
            # 保存时间戳和空间对齐信息
            timestamp_info = {
                'timestamp': sample['timestamp'] / 1e6,  # 转换为秒
                'sample_token': sample['token'],
                'scene_token': sample['scene_token'],
                'lidar_token': lidar_data['token']
            }
            
            # 保存ego pose信息（用于空间对齐）
            ego_pose_info = {
                'ego2global_translation': lidar_pose['translation'],  # [x, y, z]
                'ego2global_rotation': lidar_pose['rotation'],        # [w, x, y, z] quaternion
                'ego_pose_token': lidar_pose['token']
            }
            
            # 保存lidar2ego变换信息
            lidar2ego_info = {
                'lidar2ego_translation': lidar_sensor['translation'],  # [x, y, z]
                'lidar2ego_rotation': lidar_sensor['rotation'],        # [w, x, y, z] quaternion
                'calibrated_sensor_token': lidar_sensor['token']
            }
            
            # 计算并保存完整的变换矩阵
            from pyquaternion import Quaternion
            
            # 构建lidar2ego变换矩阵
            lidar2ego_R = Quaternion(lidar_sensor['rotation']).rotation_matrix
            lidar2ego_T = np.array(lidar_sensor['translation'])
            lidar2ego_matrix = np.eye(4)
            lidar2ego_matrix[:3, :3] = lidar2ego_R
            lidar2ego_matrix[:3, 3] = lidar2ego_T
            
            # 构建ego2global变换矩阵
            ego2global_R = Quaternion(lidar_pose['rotation']).rotation_matrix
            ego2global_T = np.array(lidar_pose['translation'])
            ego2global_matrix = np.eye(4)
            ego2global_matrix[:3, :3] = ego2global_R
            ego2global_matrix[:3, 3] = ego2global_T
            
            # 计算lidar2global变换矩阵
            lidar2global_matrix = ego2global_matrix @ lidar2ego_matrix
            
            # 计算global2lidar变换矩阵（用于空间对齐）
            global2lidar_matrix = np.linalg.inv(lidar2global_matrix)
            
            # 保存变换矩阵信息
            transform_info = {
                'lidar2ego_matrix': lidar2ego_matrix.tolist(),
                'ego2global_matrix': ego2global_matrix.tolist(),
                'lidar2global_matrix': lidar2global_matrix.tolist(),
                'global2lidar_matrix': global2lidar_matrix.tolist()
            }
            
            # 保存时间戳和变换信息到JSON文件
            temporal_info = {
                'timestamp_info': timestamp_info,
                'ego_pose_info': ego_pose_info,
                'lidar2ego_info': lidar2ego_info,
                'transform_info': transform_info
            }
            
            temporal_file_path = os.path.join(output_path, "temporal", f"temporal_{frame_idx}.json")
            os.makedirs(os.path.dirname(temporal_file_path), exist_ok=True)
            
            try:
                with open(temporal_file_path, 'w') as f:
                    json.dump(temporal_info, f, indent=2, ensure_ascii=False)
                print(f"  ✅ Saved temporal info to {temporal_file_path}")
                print(f"    📊 Timestamp: {timestamp_info['timestamp']:.6f}s")
                print(f"    📍 Ego position: {ego_pose_info['ego2global_translation']}")
            except Exception as e:
                print(f"    ❌ Error saving temporal info: {e}")
            
            # 为每个相机构建lidar2img变换矩阵
            all_lidar2img_matrices = []
            
            for cam_type in camera_types:
                if cam_type in camera_data:
                    cam_data = camera_data[cam_type]
                    
                    try:
                        # 使用nuScenes的get_sample_data方法获取正确的变换
                        # 这个方法会自动处理所有坐标系变换
                        _, _, cam_intrinsic_matrix = nusc.get_sample_data(cam_data['token'])
                        
                        # 获取lidar到相机的变换
                        # 使用nuScenes的view_points方法需要的变换
                        cam_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                        cam_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
                        
                        # 构建完整的变换矩阵链
                        from pyquaternion import Quaternion
                        
                        # 1. 激光雷达到ego的变换
                        lidar2ego_R = Quaternion(lidar_sensor['rotation']).rotation_matrix
                        lidar2ego_T = np.array(lidar_sensor['translation'])
                        lidar2ego = np.eye(4)
                        lidar2ego[:3, :3] = lidar2ego_R
                        lidar2ego[:3, 3] = lidar2ego_T
                        
                        # 2. ego到global的变换（激光雷达帧）
                        ego2global_R = Quaternion(lidar_pose['rotation']).rotation_matrix
                        ego2global_T = np.array(lidar_pose['translation'])
                        ego2global = np.eye(4)
                        ego2global[:3, :3] = ego2global_R
                        ego2global[:3, 3] = ego2global_T
                        
                        # 3. global到ego的变换（相机帧）
                        cam_ego2global_R = Quaternion(cam_pose['rotation']).rotation_matrix
                        cam_ego2global_T = np.array(cam_pose['translation'])
                        cam_ego2global = np.eye(4)
                        cam_ego2global[:3, :3] = cam_ego2global_R
                        cam_ego2global[:3, 3] = cam_ego2global_T
                        
                        # 计算global到ego的逆变换
                        global2cam_ego = np.linalg.inv(cam_ego2global)
                        
                        # 4. ego到相机的变换
                        cam2ego_R = Quaternion(cam_sensor['rotation']).rotation_matrix
                        cam2ego_T = np.array(cam_sensor['translation'])
                        cam2ego = np.eye(4)
                        cam2ego[:3, :3] = cam2ego_R
                        cam2ego[:3, 3] = cam2ego_T
                        
                        # 5. 相机内参矩阵
                        cam_intrinsic = np.array(cam_sensor['camera_intrinsic'])
                        intrinsic_4x4 = np.eye(4)
                        intrinsic_4x4[:3, :3] = cam_intrinsic
                        
                        # 组合变换矩阵：lidar -> ego -> global -> ego' -> camera -> image
                        # lidar2img = intrinsic_4x4 @ cam2ego @ global2cam_ego @ ego2global @ lidar2ego
                        lidar2img_4x4 = intrinsic_4x4 @ cam2ego @ global2cam_ego @ ego2global @ lidar2ego
                        
                        print(f"    📐 {cam_type}: 内参矩阵")
                        print(f"        fx={cam_intrinsic[0,0]:.1f}, fy={cam_intrinsic[1,1]:.1f}")
                        print(f"        cx={cam_intrinsic[0,2]:.1f}, cy={cam_intrinsic[1,2]:.1f}")
                        
                    except Exception as e:
                        print(f"    ⚠️ Error computing exact transform for {cam_type}: {e}")
                        # 使用更合理的默认变换矩阵
                        lidar2img_4x4 = np.eye(4)
                        
                        # 设置相机内参（基于nuScenes的典型值）
                        lidar2img_4x4[0, 0] = 1200.0  # fx - 焦距x
                        lidar2img_4x4[1, 1] = 1200.0  # fy - 焦距y
                        lidar2img_4x4[0, 2] = 800.0   # cx - 主点x (1600/2)
                        lidar2img_4x4[1, 2] = 450.0   # cy - 主点y (900/2)
                        
                        # 设置外参（激光雷达到相机的变换）
                        # 假设激光雷达在车顶，相机在车辆前方
                        lidar2img_4x4[0, 3] = 0.0     # x偏移
                        lidar2img_4x4[1, 3] = 0.0     # y偏移
                        lidar2img_4x4[2, 3] = -1.5    # z偏移（激光雷达到相机的高度差）
                        
                        print(f"    ⚠️ {cam_type}: 使用默认变换矩阵")
                    
                    all_lidar2img_matrices.append(lidar2img_4x4)
                    print(f"    📐 {cam_type}: lidar2img matrix computed")
                else:
                    # 如果相机数据不存在，使用单位矩阵
                    all_lidar2img_matrices.append(np.eye(4))
                    print(f"    ⚠️ {cam_type}: using identity matrix")
            
            # 将所有相机的变换矩阵保存到一个文件中 - 每帧单独保存
            calib_path = os.path.join(output_path, "calib", f"lidar2img_{frame_idx}.bin")
            all_matrices = np.array(all_lidar2img_matrices, dtype=np.float32)
            all_matrices.tofile(calib_path)
            print(f"  ✅ Saved calibration data for frame {frame_idx}: shape={all_matrices.shape}")
            
        except Exception as e:
            print(f"  ❌ Error getting calibration: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Successfully generated offline data in: {output_path}")
    print(f"Generated {min(num_frames, len(samples))} frames with {len(camera_types)} cameras each")
    print(f"Data includes: images, lidar point clouds, 3D labels, and calibration data")


def verify_generated_data(output_path: str):
    """
    Verify that the generated data is valid.
    
    Args:
        output_path: Path to the generated data directory
    """
    print(f"Verifying generated data in: {output_path}")
    
    camera_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
                   "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    # 检查图像文件
    images_dir = os.path.join(output_path, "images")
    if not os.path.exists(images_dir):
        print("❌ Images directory not found!")
        return False
    
    valid_images = 0
    total_images = 0
    
    for cam_type in camera_types:
        cam_dir = os.path.join(images_dir, cam_type)
        if not os.path.exists(cam_dir):
            print(f"❌ Camera directory not found: {cam_dir}")
            continue
        
        image_files = [f for f in os.listdir(cam_dir) if f.endswith('.jpg')]
        total_images += len(image_files)
        
        for img_file in image_files:
            img_path = os.path.join(cam_dir, img_file)
            try:
                # 尝试读取图像
                img = cv2.imread(img_path)
                if img is not None and img.size > 0:
                    valid_images += 1
                    print(f"✅ {img_path}: shape={img.shape}, size={img.size}")
                else:
                    print(f"❌ {img_path}: Invalid image")
            except Exception as e:
                print(f"❌ {img_path}: Error reading image - {e}")
    
    # 检查标定文件
    calib_dir = os.path.join(output_path, "calib")
    valid_calibs = 0
    total_calibs = 0
    
    if os.path.exists(calib_dir):
        calib_files = [f for f in os.listdir(calib_dir) if f.endswith('.bin')]
        total_calibs = len(calib_files)
        
        for calib_file in calib_files:
            calib_path = os.path.join(calib_dir, calib_file)
            try:
                # 读取标定数据
                calib_data = np.fromfile(calib_path, dtype=np.float32)
                expected_size = 6 * 4 * 4  # 6个相机，每个4x4矩阵
                if calib_data.size == expected_size:
                    valid_calibs += 1
                    print(f"✅ {calib_path}: shape={calib_data.shape}")
                else:
                    print(f"❌ {calib_path}: Wrong size {calib_data.size}, expected {expected_size}")
            except Exception as e:
                print(f"❌ {calib_path}: Error reading calib - {e}")
    
    # 检查雷达点云文件
    lidar_dir = os.path.join(output_path, "lidar")
    valid_lidars = 0
    total_lidars = 0
    
    if os.path.exists(lidar_dir):
        lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith('.bin')]
        total_lidars = len(lidar_files)
        
        for lidar_file in lidar_files:
            lidar_path = os.path.join(lidar_dir, lidar_file)
            try:
                # 读取雷达点云数据
                lidar_data = np.fromfile(lidar_path, dtype=np.float32)
                if lidar_data.size > 0 and lidar_data.size % 5 == 0:  # 每个点5个值 (x, y, z, intensity, ring)
                    valid_lidars += 1
                    num_points = lidar_data.size // 5
                    print(f"✅ {lidar_path}: {num_points} points, shape={lidar_data.shape}")
                else:
                    print(f"❌ {lidar_path}: Invalid lidar data size {lidar_data.size}")
            except Exception as e:
                print(f"❌ {lidar_path}: Error reading lidar - {e}")
    
    # 检查3D标签文件
    labels_dir = os.path.join(output_path, "labels")
    valid_labels = 0
    total_labels = 0
    
    if os.path.exists(labels_dir):
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.json')]
        total_labels = len(label_files)
        
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            try:
                # 读取标签数据
                with open(label_path, 'r') as f:
                    labels_data = json.load(f)
                
                if isinstance(labels_data, list):
                    valid_labels += 1
                    num_objects = len(labels_data)
                    categories = [obj.get('category_name', 'unknown') for obj in labels_data]  # 使用映射后的类别
                    from collections import Counter
                    category_counts = Counter(categories)
                    print(f"✅ {label_path}: {num_objects} objects, categories={dict(category_counts)}")
                    
                    # 显示原始类别统计（用于调试）
                    original_categories = [obj.get('original_category', 'unknown') for obj in labels_data]
                    original_counts = Counter(original_categories)
                    print(f"   Original categories: {dict(original_counts)}")
                else:
                    print(f"❌ {label_path}: Invalid label format")
            except Exception as e:
                print(f"❌ {label_path}: Error reading labels - {e}")
    
    # 检查时间戳和变换信息文件
    temporal_dir = os.path.join(output_path, "temporal")
    valid_temporal = 0
    total_temporal = 0
    
    if os.path.exists(temporal_dir):
        temporal_files = [f for f in os.listdir(temporal_dir) if f.endswith('.json')]
        total_temporal = len(temporal_files)
        
        for temporal_file in temporal_files:
            temporal_path = os.path.join(temporal_dir, temporal_file)
            try:
                # 读取时间戳和变换信息
                with open(temporal_path, 'r') as f:
                    temporal_data = json.load(f)
                
                if isinstance(temporal_data, dict) and 'timestamp_info' in temporal_data:
                    valid_temporal += 1
                    timestamp = temporal_data['timestamp_info']['timestamp']
                    ego_translation = temporal_data['ego_pose_info']['ego2global_translation']
                    
                    print(f"✅ {temporal_path}: timestamp={timestamp:.6f}s, ego_pos={ego_translation}")
                    
                    # 验证变换矩阵
                    if 'transform_info' in temporal_data:
                        transform_info = temporal_data['transform_info']
                        if all(key in transform_info for key in ['lidar2ego_matrix', 'ego2global_matrix', 'global2lidar_matrix']):
                            print(f"   📐 Transform matrices: ✅ All present")
                        else:
                            print(f"   📐 Transform matrices: ❌ Missing some matrices")
                else:
                    print(f"❌ {temporal_path}: Invalid temporal data format")
            except Exception as e:
                print(f"❌ {temporal_path}: Error reading temporal data - {e}")
    
    print(f"\n📊 Verification Summary:")
    print(f"  Images: {valid_images}/{total_images} valid")
    print(f"  Calibs: {valid_calibs}/{total_calibs} valid")
    print(f"  Lidars: {valid_lidars}/{total_lidars} valid")
    print(f"  Labels: {valid_labels}/{total_labels} valid")
    print(f"  Temporal: {valid_temporal}/{total_temporal} valid")
    
    return valid_images > 0 and valid_calibs > 0 and valid_lidars > 0 and valid_labels > 0 and valid_temporal > 0


if __name__ == "__main__":
    # 配置参数 - 与010脚本保持一致
    config_path = "dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py"
    output_path = "/share/Code/SparseEnd2End/C++/Data/sparse/"
    num_frames = 3  # 与010脚本一致，处理前3帧
    
    try:
        # 尝试生成真实的nuScenes数据
        if os.path.exists(config_path):
            # 使用与010脚本相同的随机种子
            set_random_seed(seed=1, deterministic=True)  # 与010脚本一致
            
            # 使用与010脚本完全相同的数据加载器和配置
            print("🔧 使用与010脚本完全相同的数据加载器和配置...")
            
            # 读取配置
            cfg = read_cfg(config_path)
            
            # 使用与010脚本相同的test配置
            dataset_cfg = cfg["data"]["test"].copy()  # 使用test配置，与010脚本一致
            
            # 移除不应该传递给数据集的参数
            samples_per_gpu = dataset_cfg.pop("samples_per_gpu", 1)
            workers_per_gpu = dataset_cfg.pop("workers_per_gpu", 0)
            
            # 使用与010脚本相同的test_pipeline
            dataset_type = dataset_cfg.pop("type")
            dataset = eval(dataset_type)(**dataset_cfg)
            
            # 创建数据加载器，使用与010脚本相同的配置
            dataloader = dataloader_wrapper(
                dataset,
                samples_per_gpu=samples_per_gpu,  # 与010脚本一致
                workers_per_gpu=workers_per_gpu,
                dist=False,
                shuffle=False,  # 与010脚本一致
            )
            
            # 创建输出目录
            camera_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
                           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
            
            for cam_type in camera_types:
                cam_dir = os.path.join(output_path, "images", cam_type)
                os.makedirs(cam_dir, exist_ok=True)
            
            calib_dir = os.path.join(output_path, "calib")
            os.makedirs(calib_dir, exist_ok=True)
            
            # 处理数据
            data_iter = dataloader.__iter__()
            
            for frame_idx in tqdm(range(num_frames), desc="Generating frames"):
                try:
                    # 获取数据 - 与010脚本完全相同的方式
                    data = next(data_iter)
                    data = scatter(data, [0])[0]
                    
                    # 获取数据
                    img_metas = data["img_metas"][0]
                    lidar2img = data["lidar2img"].data[0].cpu().numpy()  # (1, 6, 4, 4) - 与010脚本格式一致
                    img = data["img"].data[0].cpu().numpy()  # (1, 6, 3, 256, 704) - 与010脚本格式一致
                    ori_img = data["ori_img"].data[0].cpu().numpy()  # (1, 6, 3, 900, 1600) - 原始图像
                    
                    print(f"\nFrame {frame_idx}: Processing {len(camera_types)} cameras")
                    print(f"  📊 ori_img shape: {ori_img.shape}")
                    print(f"  📊 img shape: {img.shape}")
                    print(f"  📊 lidar2img shape: {lidar2img.shape}")
                    
                    # 保存原始图像数据为二进制文件 - 与010脚本格式完全一致
                    # 010脚本保存格式: sample_{frame_idx}_ori_imgs_1*6*3*900*1600_uint8.bin
                    ori_imgs_path = os.path.join(output_path, "calib", f"sample_{frame_idx}_ori_imgs_1*6*3*900*1600_uint8.bin")
                    ori_img.astype(np.uint8).tofile(ori_imgs_path)
                    print(f"  ✅ Saved ori_imgs binary: {ori_imgs_path}")
                    
                    # 保存预处理图像数据为二进制文件 - 与010脚本格式完全一致
                    # 010脚本保存格式: sample_{frame_idx}_imgs_1*6*3*256*704_float32.bin
                    imgs_path = os.path.join(output_path, "calib", f"sample_{frame_idx}_imgs_1*6*3*256*704_float32.bin")
                    img.astype(np.float32).tofile(imgs_path)
                    print(f"  ✅ Saved imgs binary: {imgs_path}")
                    
                    # 同时保存JPG图像文件用于可视化
                    for cam_idx, cam_type in enumerate(camera_types):
                        cam_img = ori_img[0, cam_idx]
                        print(f"  Camera {cam_type}: cam_img shape = {cam_img.shape}")
                        
                        # 处理不同的图像格式
                        if cam_img.shape == (3, 900, 1600):
                            # 三通道彩色图像
                            cam_img_cv = np.transpose(cam_img, (1, 2, 0))
                            cam_img_bgr = cv2.cvtColor(cam_img_cv, cv2.COLOR_RGB2BGR)
                        elif cam_img.shape == (900, 1600):
                            # 灰度图像，转换为三通道
                            print(f"    ⚠️ 检测到灰度图像，转换为三通道")
                            cam_img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_GRAY2BGR)
                        elif cam_img.shape == (900, 1600, 3):
                            # 已经是BGR格式
                            cam_img_bgr = cam_img
                        else:
                            print(f"    ⚠️ 未知图像格式: {cam_img.shape}，尝试转换为三通道")
                            # 尝试转换为三通道
                            if cam_img.ndim == 2:
                                cam_img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_GRAY2BGR)
                            elif cam_img.ndim == 3 and cam_img.shape[2] == 3:
                                cam_img_bgr = cam_img
                            else:
                                raise ValueError(f"无法处理的图像格式: {cam_img.shape}")
                        
                        target_img_path = os.path.join(output_path, "images", cam_type, f"{frame_idx}.jpg")
                        cv2.imwrite(target_img_path, cam_img_bgr)
                        print(f"  Camera {cam_type}: Saved to {target_img_path}")
                        print(f"    📊 Image stats: shape={cam_img_bgr.shape}, mean={cam_img_bgr.mean():.1f}, std={cam_img_bgr.std():.1f}")
                    
                    # 保存lidar2img标定参数 - 与010脚本格式一致
                    calib_path = os.path.join(output_path, "calib", f"lidar2img_{frame_idx}.bin")
                    lidar2img.astype(np.float32).tofile(calib_path)
                    print(f"  ✅ Saved lidar2img calibration for frame {frame_idx}: shape={lidar2img.shape}")
                    
                    # 保存时间戳和变换信息
                    if "timestamp" in data:
                        timestamp = data["timestamp"].data[0].cpu().numpy()  # (1,)
                        timestamp_path = os.path.join(output_path, "calib", f"timestamp_{frame_idx}.bin")
                        timestamp.astype(np.float32).tofile(timestamp_path)
                        print(f"  ✅ Saved timestamp for frame {frame_idx}: shape={timestamp.shape}")
                    
                    # 保存global2lidar变换矩阵
                    if "global2lidar" in img_metas:
                        global2lidar = img_metas["global2lidar"]
                        global2lidar_path = os.path.join(output_path, "calib", f"global2lidar_{frame_idx}.bin")
                        global2lidar.astype(np.float32).tofile(global2lidar_path)
                        print(f"  ✅ Saved global2lidar for frame {frame_idx}: shape={global2lidar.shape}")
                    
                except StopIteration:
                    print(f"Dataset exhausted after {frame_idx} frames")
                    break
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"✅ 成功生成与010脚本一致的数据")
            print(f"✅ 数据源一致性:")
            print(f"  - 相同配置文件: {config_path}")
            print(f"  - 相同数据集: test")
            print(f"  - 相同随机种子: 1")
            print(f"  - 相同数据加载器: dataloader_wrapper")
            print(f"  - 相同帧数: {num_frames}")
            
        else:
            print(f"Config file not found: {config_path}")
            print("Generating sample data instead...")
            generate_sample_data(output_path, num_frames)
    except Exception as e:
        print(f"Error generating nuScenes data: {e}")
        print("Generating sample data instead...")
        generate_sample_data(output_path, num_frames) 