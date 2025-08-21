#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比030生成的第一帧数据和010生成的第一帧原始数据
检查数据格式和内容的一致性
"""

import os
import numpy as np
import json
import cv2
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any


def load_bin_file(file_path: str, dtype=np.float32) -> np.ndarray:
    """加载二进制文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    try:
        data = np.fromfile(file_path, dtype=dtype)
        print(f"✅ 加载成功: {file_path}, shape={data.shape}, dtype={data.dtype}")
        return data
    except Exception as e:
        print(f"❌ 加载失败: {file_path}, 错误: {e}")
        return None


def load_json_file(file_path: str) -> Dict:
    """加载JSON文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 加载成功: {file_path}, 数据类型: {type(data)}")
        return data
    except Exception as e:
        print(f"❌ 加载失败: {file_path}, 错误: {e}")
        return None


def load_image_file(file_path: str) -> np.ndarray:
    """加载图像文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"❌ 图像加载失败: {file_path}")
            return None
        print(f"✅ 加载成功: {file_path}, shape={img.shape}, dtype={img.dtype}")
        return img
    except Exception as e:
        print(f"❌ 加载失败: {file_path}, 错误: {e}")
        return None


def compare_numpy_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str, tolerance: float = 1e-6) -> bool:
    """对比两个numpy数组"""
    if arr1 is None or arr2 is None:
        print(f"❌ {name}: 其中一个数组为空")
        return False
    
    if arr1.shape != arr2.shape:
        print(f"❌ {name}: 形状不匹配 - {arr1.shape} vs {arr2.shape}")
        return False
    
    if arr1.dtype != arr2.dtype:
        print(f"❌ {name}: 数据类型不匹配 - {arr1.dtype} vs {arr2.dtype}")
        return False
    
    # 计算差异
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"📊 {name}:")
    print(f"   形状: {arr1.shape}")
    print(f"   数据类型: {arr1.dtype}")
    print(f"   最大值差异: {max_diff:.6f}")
    print(f"   平均差异: {mean_diff:.6f}")
    
    if max_diff > tolerance:
        print(f"   状态: ❌ 差异超过容差 {tolerance}")
        return False
    else:
        print(f"   状态: ✅ 数据一致")
        return True


def compare_images(img1: np.ndarray, img2: np.ndarray, name: str) -> bool:
    """对比两个图像"""
    if img1 is None or img2 is None:
        print(f"❌ {name}: 其中一个图像为空")
        return False
    
    if img1.shape != img2.shape:
        print(f"❌ {name}: 形状不匹配 - {img1.shape} vs {img2.shape}")
        return False
    
    # 计算差异
    diff = cv2.absdiff(img1, img2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"📊 {name}:")
    print(f"   形状: {img1.shape}")
    print(f"   数据类型: {img1.dtype}")
    print(f"   最大值差异: {max_diff}")
    print(f"   平均差异: {mean_diff:.2f}")
    
    if max_diff > 0:
        print(f"   状态: ⚠️ 图像存在差异")
        return False
    else:
        print(f"   状态: ✅ 图像完全一致")
        return True


def compare_json_data(data1: Dict, data2: Dict, name: str) -> bool:
    """对比两个JSON数据结构"""
    if data1 is None or data2 is None:
        print(f"❌ {name}: 其中一个数据为空")
        return False
    
    # 简单的键值对比
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    
    if keys1 != keys2:
        print(f"❌ {name}: 键不匹配")
        print(f"   data1 keys: {sorted(keys1)}")
        print(f"   data2 keys: {sorted(keys2)}")
        return False
    
    print(f"📊 {name}:")
    print(f"   键数量: {len(keys1)}")
    print(f"   键列表: {sorted(keys1)}")
    print(f"   状态: ✅ 结构一致")
    return True


def load_010_data(data_dir: str, frame_idx: int = 0) -> Dict[str, Any]:
    """加载010脚本生成的数据"""
    print(f"\n🔍 加载010脚本生成的数据 (frame {frame_idx})...")
    
    data = {}
    
    # 加载原始图像
    ori_img_path = os.path.join(data_dir, f"ori_imgs_{frame_idx}.bin")
    data['ori_img'] = load_bin_file(ori_img_path, dtype=np.uint8)
    
    # 加载预处理图像
    img_path = os.path.join(data_dir, f"imgs_{frame_idx}.bin")
    data['img'] = load_bin_file(img_path, dtype=np.float32)
    
    # 加载backbone特征
    feature_path = os.path.join(data_dir, f"feature_{frame_idx}.bin")
    data['feature'] = load_bin_file(feature_path, dtype=np.float32)
    
    # 加载head输入数据
    data['spatial_shapes'] = load_bin_file(os.path.join(data_dir, f"spatial_shapes_{frame_idx}.bin"), dtype=np.int32)
    data['level_start_index'] = load_bin_file(os.path.join(data_dir, f"level_start_index_{frame_idx}.bin"), dtype=np.int32)
    data['instance_feature'] = load_bin_file(os.path.join(data_dir, f"instance_feature_{frame_idx}.bin"), dtype=np.float32)
    data['anchor'] = load_bin_file(os.path.join(data_dir, f"anchor_{frame_idx}.bin"), dtype=np.float32)
    data['time_interval'] = load_bin_file(os.path.join(data_dir, f"time_interval_{frame_idx}.bin"), dtype=np.float32)
    data['image_wh'] = load_bin_file(os.path.join(data_dir, f"image_wh_{frame_idx}.bin"), dtype=np.float32)
    data['lidar2img'] = load_bin_file(os.path.join(data_dir, f"lidar2img_{frame_idx}.bin"), dtype=np.float32)
    
    # 加载head输出数据
    data['pred_instance_feature'] = load_bin_file(os.path.join(data_dir, f"pred_instance_feature_{frame_idx}.bin"), dtype=np.float32)
    data['pred_anchor'] = load_bin_file(os.path.join(data_dir, f"pred_anchor_{frame_idx}.bin"), dtype=np.float32)
    data['pred_class_score'] = load_bin_file(os.path.join(data_dir, f"pred_class_score_{frame_idx}.bin"), dtype=np.float32)
    data['pred_quality_score'] = load_bin_file(os.path.join(data_dir, f"pred_quality_score_{frame_idx}.bin"), dtype=np.float32)
    
    return data


def load_030_data(data_dir: str, frame_idx: int = 0) -> Dict[str, Any]:
    """加载030脚本生成的数据"""
    print(f"\n🔍 加载030脚本生成的数据 (frame {frame_idx})...")
    
    data = {}
    
    # 加载原始图像
    ori_img_path = os.path.join(data_dir, "images", f"ori_img_{frame_idx}.bin")
    data['ori_img'] = load_bin_file(ori_img_path, dtype=np.uint8)
    
    # 加载预处理图像
    img_path = os.path.join(data_dir, "images", f"img_{frame_idx}.bin")
    data['img'] = load_bin_file(img_path, dtype=np.float32)
    
    # 加载标定参数
    data['lidar2img'] = load_bin_file(os.path.join(data_dir, "calib", f"lidar2img_{frame_idx}.bin"), dtype=np.float32)
    data['image_wh'] = load_bin_file(os.path.join(data_dir, "calib", f"image_wh_{frame_idx}.bin"), dtype=np.float32)
    data['timestamp'] = load_bin_file(os.path.join(data_dir, "calib", f"timestamp_{frame_idx}.bin"), dtype=np.float32)
    
    # 加载相机图像
    camera_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
                   "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    data['camera_images'] = {}
    for cam_type in camera_types:
        img_path = os.path.join(data_dir, "images", cam_type, f"{frame_idx}.jpg")
        data['camera_images'][cam_type] = load_image_file(img_path)
    
    # 加载雷达点云
    lidar_path = os.path.join(data_dir, "lidar", f"lidar_{frame_idx}.bin")
    data['lidar'] = load_bin_file(lidar_path, dtype=np.float32)
    
    # 加载3D标签
    labels_path = os.path.join(data_dir, "labels", f"labels_{frame_idx}.json")
    data['labels'] = load_json_file(labels_path)
    
    # 加载时间戳信息
    temporal_path = os.path.join(data_dir, "temporal", f"temporal_{frame_idx}.json")
    data['temporal'] = load_json_file(temporal_path)
    
    return data


def compare_data_formats(data_010: Dict, data_030: Dict) -> bool:
    """对比数据格式"""
    print(f"\n📊 对比数据格式...")
    
    all_match = True
    
    # 对比原始图像
    if 'ori_img' in data_010 and 'ori_img' in data_030:
        print(f"\n🔍 对比原始图像数据...")
        match = compare_numpy_arrays(data_010['ori_img'], data_030['ori_img'], "ori_img", tolerance=0)
        all_match = all_match and match
    
    # 对比预处理图像
    if 'img' in data_010 and 'img' in data_030:
        print(f"\n🔍 对比预处理图像数据...")
        match = compare_numpy_arrays(data_010['img'], data_030['img'], "img", tolerance=1e-6)
        all_match = all_match and match
    
    # 对比lidar2img标定参数
    if 'lidar2img' in data_010 and 'lidar2img' in data_030:
        print(f"\n🔍 对比lidar2img标定参数...")
        match = compare_numpy_arrays(data_010['lidar2img'], data_030['lidar2img'], "lidar2img", tolerance=1e-6)
        all_match = all_match and match
    
    # 对比image_wh参数
    if 'image_wh' in data_010 and 'image_wh' in data_030:
        print(f"\n🔍 对比image_wh参数...")
        match = compare_numpy_arrays(data_010['image_wh'], data_030['image_wh'], "image_wh", tolerance=1e-6)
        all_match = all_match and match
    
    return all_match


def analyze_data_differences(data_010: Dict, data_030: Dict):
    """分析数据差异"""
    print(f"\n📈 数据差异分析...")
    
    # 分析原始图像差异
    if 'ori_img' in data_010 and 'ori_img' in data_030:
        print(f"\n📊 原始图像分析:")
        ori_010 = data_010['ori_img']
        ori_030 = data_030['ori_img']
        
        if ori_010 is not None and ori_030 is not None:
            print(f"   010数据: shape={ori_010.shape}, dtype={ori_010.dtype}, range=[{ori_010.min()}, {ori_010.max()}]")
            print(f"   030数据: shape={ori_030.shape}, dtype={ori_030.dtype}, range=[{ori_030.min()}, {ori_030.max()}]")
            
            if ori_010.shape == ori_030.shape:
                diff = np.abs(ori_010.astype(np.float32) - ori_030.astype(np.float32))
                print(f"   差异统计: max={diff.max()}, mean={diff.mean():.2f}, std={diff.std():.2f}")
    
    # 分析预处理图像差异
    if 'img' in data_010 and 'img' in data_030:
        print(f"\n📊 预处理图像分析:")
        img_010 = data_010['img']
        img_030 = data_030['img']
        
        if img_010 is not None and img_030 is not None:
            print(f"   010数据: shape={img_010.shape}, dtype={img_010.dtype}, range=[{img_010.min():.3f}, {img_010.max():.3f}]")
            print(f"   030数据: shape={img_030.shape}, dtype={img_030.dtype}, range=[{img_030.min():.3f}, {img_030.max():.3f}]")
            
            if img_010.shape == img_030.shape:
                diff = np.abs(img_010 - img_030)
                print(f"   差异统计: max={diff.max():.6f}, mean={diff.mean():.6f}, std={diff.std():.6f}")
    
    # 分析标定参数差异
    if 'lidar2img' in data_010 and 'lidar2img' in data_030:
        print(f"\n📊 lidar2img标定参数分析:")
        calib_010 = data_010['lidar2img']
        calib_030 = data_030['lidar2img']
        
        if calib_010 is not None and calib_030 is not None:
            print(f"   010数据: shape={calib_010.shape}, dtype={calib_010.dtype}")
            print(f"   030数据: shape={calib_030.shape}, dtype={calib_030.dtype}")
            
            if calib_010.shape == calib_030.shape:
                diff = np.abs(calib_010 - calib_030)
                print(f"   差异统计: max={diff.max():.6f}, mean={diff.mean():.6f}, std={diff.std():.6f}")
                
                # 显示每个相机的差异
                for i in range(min(6, calib_010.shape[0])):
                    cam_diff = np.abs(calib_010[i] - calib_030[i])
                    print(f"   相机{i}: max_diff={cam_diff.max():.6f}, mean_diff={cam_diff.mean():.6f}")


def compare_lidar2img_calibration(data_010_path: str, data_030_path: str, frame_idx: int = 0):
    """对比lidar2img标定参数"""
    print(f"\n🔍 对比lidar2img标定参数 (帧 {frame_idx})...")
    
    # 010脚本的lidar2img数据
    # 010脚本保存格式: sample_{frame_idx}_lidar2img_1*6*4*4_float32.bin
    lidar2img_010_path = os.path.join(data_010_path, f"sample_{frame_idx}_lidar2img_1*6*4*4_float32.bin")
    
    # 030脚本的lidar2img数据
    # 030脚本保存格式: lidar2img_{frame_idx}.bin
    lidar2img_030_path = os.path.join(data_030_path, "calib", f"lidar2img_{frame_idx}.bin")
    
    # 加载数据
    lidar2img_010 = load_bin_file(lidar2img_010_path, dtype=np.float32)
    lidar2img_030 = load_bin_file(lidar2img_030_path, dtype=np.float32)
    
    if lidar2img_010 is None or lidar2img_030 is None:
        print("❌ 无法加载lidar2img数据")
        return False
    
    # 重塑数据以进行比较
    # 010脚本: (1, 6, 4, 4) -> (6, 4, 4)
    # 030脚本: (6, 4, 4)
    if lidar2img_010.ndim == 4:
        lidar2img_010 = lidar2img_010.squeeze(0)  # 移除batch维度
    
    print(f"📊 数据形状对比:")
    print(f"  010脚本: {lidar2img_010.shape}")
    print(f"  030脚本: {lidar2img_030.shape}")
    
    if lidar2img_010.shape != lidar2img_030.shape:
        print("❌ 数据形状不匹配")
        return False
    
    # 比较数值差异
    diff = np.abs(lidar2img_010 - lidar2img_030)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"📊 数值差异统计:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    
    # 检查是否在可接受范围内
    tolerance = 1e-6
    if max_diff < tolerance:
        print("✅ lidar2img标定参数完全一致")
        return True
    else:
        print(f"⚠️ lidar2img标定参数存在差异 (容差: {tolerance})")
        
        # 显示具体差异
        for cam_idx in range(6):
            cam_diff = diff[cam_idx]
            cam_max_diff = np.max(cam_diff)
            if cam_max_diff > tolerance:
                print(f"  相机 {cam_idx}: 最大差异 = {cam_max_diff:.6f}")
        
        return False


def compare_image_data(data_010_path: str, data_030_path: str, frame_idx: int = 0):
    """对比图像数据"""
    print(f"\n🔍 对比图像数据 (帧 {frame_idx})...")
    
    # 010脚本的原始图像数据
    # 010脚本保存格式: sample_{frame_idx}_ori_imgs_1*6*3*900*1600_uint8.bin
    ori_imgs_010_path = os.path.join(data_010_path, f"sample_{frame_idx}_ori_imgs_1*6*3*900*1600_uint8.bin")
    
    # 030脚本的原始图像数据（JPG格式）
    camera_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
                   "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    # 加载010脚本的原始图像
    ori_imgs_010 = load_bin_file(ori_imgs_010_path, dtype=np.uint8)
    if ori_imgs_010 is None:
        print("❌ 无法加载010脚本的原始图像数据")
        return False
    
    # 重塑数据: (25920000,) -> (1, 6, 3, 900, 1600) -> (6, 3, 900, 1600)
    if ori_imgs_010.ndim == 1:
        ori_imgs_010 = ori_imgs_010.reshape(1, 6, 3, 900, 1600)
    if ori_imgs_010.ndim == 5:
        ori_imgs_010 = ori_imgs_010.squeeze(0)  # 移除batch维度
    
    print(f"📊 010脚本原始图像: shape={ori_imgs_010.shape}")
    
    # 加载030脚本的原始图像
    ori_imgs_030 = []
    for cam_idx, cam_type in enumerate(camera_types):
        img_path = os.path.join(data_030_path, "images", cam_type, f"{frame_idx}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # 转换为RGB格式 (BGR -> RGB)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 转换为 (3, 900, 1600) 格式
                img_transposed = np.transpose(img_rgb, (2, 0, 1))
                ori_imgs_030.append(img_transposed)
                print(f"✅ 加载030脚本图像 {cam_type}: shape={img_transposed.shape}")
            else:
                print(f"❌ 无法读取图像: {img_path}")
                return False
        else:
            print(f"❌ 图像文件不存在: {img_path}")
            return False
    
    ori_imgs_030 = np.array(ori_imgs_030)  # (6, 3, 900, 1600)
    print(f"📊 030脚本原始图像: shape={ori_imgs_030.shape}")
    
    # 比较图像数据
    if ori_imgs_010.shape != ori_imgs_030.shape:
        print("❌ 图像数据形状不匹配")
        return False
    
    # 比较数值差异
    diff = np.abs(ori_imgs_010.astype(np.int16) - ori_imgs_030.astype(np.int16))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"📊 图像差异统计:")
    print(f"  最大差异: {max_diff}")
    print(f"  平均差异: {mean_diff:.2f}")
    
    # 检查是否在可接受范围内（考虑到可能的压缩损失）
    tolerance = 5  # 允许5个像素值的差异
    if max_diff <= tolerance:
        print("✅ 图像数据基本一致")
        return True
    else:
        print(f"⚠️ 图像数据存在差异 (容差: {tolerance})")
        
        # 显示每个相机的差异
        for cam_idx, cam_type in enumerate(camera_types):
            cam_diff = diff[cam_idx]
            cam_max_diff = np.max(cam_diff)
            cam_mean_diff = np.mean(cam_diff)
            print(f"  相机 {cam_type}: 最大差异={cam_max_diff}, 平均差异={cam_mean_diff:.2f}")
        
        return False


def compare_temporal_data(data_010_path: str, data_030_path: str, frame_idx: int = 0):
    """对比时间戳数据"""
    print(f"\n🔍 对比时间戳数据 (帧 {frame_idx})...")
    
    # 010脚本的时间戳数据
    # 010脚本保存格式: sample_{frame_idx}_ibank_timestamp_1_float64.bin
    timestamp_010_path = os.path.join(data_010_path, f"sample_{frame_idx}_ibank_timestamp_1_float64.bin")
    
    # 030脚本的时间戳数据
    # 030脚本保存格式: temporal_{frame_idx}.json
    temporal_030_path = os.path.join(data_030_path, "temporal", f"temporal_{frame_idx}.json")
    
    # 加载数据
    timestamp_010 = load_bin_file(timestamp_010_path, dtype=np.float64)
    temporal_030 = load_json_file(temporal_030_path)
    
    if timestamp_010 is None or temporal_030 is None:
        print("❌ 无法加载时间戳数据")
        return False
    
    # 获取030脚本的时间戳
    timestamp_030 = temporal_030.get('timestamp_info', {}).get('timestamp', None)
    if timestamp_030 is None:
        print("❌ 030脚本时间戳数据格式错误")
        return False
    
    print(f"📊 时间戳对比:")
    print(f"  010脚本: {timestamp_010[0]:.6f}")
    print(f"  030脚本: {timestamp_030:.6f}")
    
    # 比较时间戳差异
    diff = abs(timestamp_010[0] - timestamp_030)
    print(f"  差异: {diff:.6f}")
    
    # 检查是否在可接受范围内
    tolerance = 1e-6
    if diff < tolerance:
        print("✅ 时间戳数据一致")
        return True
    else:
        print(f"⚠️ 时间戳数据存在差异 (容差: {tolerance})")
        return False


def compare_global2lidar_data(data_010_path: str, data_030_path: str, frame_idx: int = 0):
    """对比global2lidar变换矩阵"""
    print(f"\n🔍 对比global2lidar变换矩阵 (帧 {frame_idx})...")
    
    # 010脚本的global2lidar数据
    # 010脚本保存格式: sample_{frame_idx}_ibank_global2lidar_4*4_float32.bin
    global2lidar_010_path = os.path.join(data_010_path, f"sample_{frame_idx}_ibank_global2lidar_4*4_float32.bin")
    
    # 030脚本的global2lidar数据
    # 030脚本保存格式: temporal_{frame_idx}.json
    temporal_030_path = os.path.join(data_030_path, "temporal", f"temporal_{frame_idx}.json")
    
    # 加载数据
    global2lidar_010 = load_bin_file(global2lidar_010_path, dtype=np.float32)
    temporal_030 = load_json_file(temporal_030_path)
    
    if global2lidar_010 is None or temporal_030 is None:
        print("❌ 无法加载global2lidar数据")
        return False
    
    # 获取030脚本的global2lidar矩阵
    global2lidar_030 = temporal_030.get('transform_info', {}).get('global2lidar_matrix', None)
    if global2lidar_030 is None:
        print("❌ 030脚本global2lidar数据格式错误")
        return False
    
    global2lidar_030 = np.array(global2lidar_030, dtype=np.float32)
    
    # 重塑010脚本数据: (16,) -> (4, 4)
    if global2lidar_010.ndim == 1:
        global2lidar_010 = global2lidar_010.reshape(4, 4)
    
    print(f"📊 数据形状对比:")
    print(f"  010脚本: {global2lidar_010.shape}")
    print(f"  030脚本: {global2lidar_030.shape}")
    
    if global2lidar_010.shape != global2lidar_030.shape:
        print("❌ 数据形状不匹配")
        return False
    
    # 比较数值差异
    diff = np.abs(global2lidar_010 - global2lidar_030)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"📊 数值差异统计:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    
    # 检查是否在可接受范围内
    tolerance = 1e-6
    if max_diff < tolerance:
        print("✅ global2lidar变换矩阵完全一致")
        return True
    else:
        print(f"⚠️ global2lidar变换矩阵存在差异 (容差: {tolerance})")
        
        # 显示矩阵差异
        print("  矩阵差异:")
        for i in range(4):
            for j in range(4):
                if abs(diff[i, j]) > tolerance:
                    print(f"    [{i},{j}]: 010={global2lidar_010[i,j]:.6f}, 030={global2lidar_030[i,j]:.6f}, diff={diff[i,j]:.6f}")
        
        return False


def main():
    parser = argparse.ArgumentParser(description="对比030和010脚本生成的数据")
    parser.add_argument("--data-010", default="script/tutorial/asset", 
                       help="010脚本生成的数据路径")
    parser.add_argument("--data-030", default="/share/Code/SparseEnd2End/C++/Data/sparse", 
                       help="030脚本生成的数据路径")
    parser.add_argument("--frame-idx", type=int, default=0, 
                       help="要对比的帧索引")
    parser.add_argument("--output", default="compare_output.txt", 
                       help="输出结果文件")
    
    args = parser.parse_args()
    
    print("🔍 开始对比030和010脚本生成的数据...")
    print(f"010脚本数据路径: {args.data_010}")
    print(f"030脚本数据路径: {args.data_030}")
    print(f"对比帧索引: {args.frame_idx}")
    
    # 检查路径是否存在
    if not os.path.exists(args.data_010):
        print(f"❌ 010脚本数据路径不存在: {args.data_010}")
        return
    
    if not os.path.exists(args.data_030):
        print(f"❌ 030脚本数据路径不存在: {args.data_030}")
        return
    
    # 执行各项对比
    results = []
    
    # 1. 对比原始图像数据
    result1 = compare_image_data(args.data_010, args.data_030, args.frame_idx)
    results.append(("原始图像数据", result1))
    
    # 2. 对比时间戳数据
    result2 = compare_temporal_data(args.data_010, args.data_030, args.frame_idx)
    results.append(("时间戳数据", result2))
    
    # 3. 对比global2lidar变换矩阵
    result3 = compare_global2lidar_data(args.data_010, args.data_030, args.frame_idx)
    results.append(("global2lidar变换矩阵", result3))
    
    # 输出总结
    print(f"\n📊 对比结果总结:")
    print("=" * 50)
    for item, result in results:
        status = "✅ 一致" if result else "❌ 不一致"
        print(f"{item}: {status}")
    
    # 保存结果到文件
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("030和010脚本数据对比结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"010脚本数据路径: {args.data_010}\n")
        f.write(f"030脚本数据路径: {args.data_030}\n")
        f.write(f"对比帧索引: {args.frame_idx}\n\n")
        
        for item, result in results:
            status = "一致" if result else "不一致"
            f.write(f"{item}: {status}\n")
    
    print(f"\n📄 详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main() 