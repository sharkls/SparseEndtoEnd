#!/usr/bin/env python3
"""
激光雷达点云投影到相机图像的可视化脚本
基于lidar2img变换矩阵将3D点云投影到2D图像平面
支持多相机数据结构，从nuScenes数据集直接计算lidar2img变换矩阵
"""

import numpy as np
import cv2
import os
import struct
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from tqdm import tqdm

def load_lidar_points(lidar_path):
    """加载激光雷达点云数据"""
    with open(lidar_path, 'rb') as f:
        data = f.read()
    
    # 假设每个点有5个float值 (x, y, z, intensity, ring)
    num_points = len(data) // (5 * 4)  # 5个float，每个4字节
    points = np.frombuffer(data, dtype=np.float32).reshape(-1, 5)
    
    return points[:, :3]  # 只返回x, y, z坐标

def debug_calibration_matrix(matrix, camera_name):
    """调试标定矩阵"""
    print(f"    🔍 {camera_name} 标定矩阵调试信息:")
    print(f"      矩阵形状: {matrix.shape}")
    print(f"      矩阵内容:")
    print(f"        {matrix}")
    
    # 分析内参部分
    fx = matrix[0, 0]
    fy = matrix[1, 1]
    cx = matrix[0, 2]
    cy = matrix[1, 2]
    
    print(f"      内参分析:")
    print(f"        焦距 fx: {fx:.1f}")
    print(f"        焦距 fy: {fy:.1f}")
    print(f"        主点 cx: {cx:.1f}")
    print(f"        主点 cy: {cy:.1f}")
    
    # 分析外参部分
    tx = matrix[0, 3]
    ty = matrix[1, 3]
    tz = matrix[2, 3]
    
    print(f"      外参分析:")
    print(f"        平移 x: {tx:.3f}")
    print(f"        平移 y: {ty:.3f}")
    print(f"        平移 z: {tz:.3f}")
    
    # 检查矩阵是否合理
    if fx <= 0 or fy <= 0:
        print(f"      ❌ 焦距值异常: fx={fx}, fy={fy}")
    
    if abs(cx) > 2000 or abs(cy) > 2000:
        print(f"      ❌ 主点值异常: cx={cx}, cy={cy}")
    
    if abs(tx) > 100 or abs(ty) > 100 or abs(tz) > 100:
        print(f"      ❌ 平移值异常: tx={tx}, ty={ty}, tz={tz}")

def load_lidar2img_matrix(calib_path, camera_index=0):
    """加载lidar2img变换矩阵
    
    Args:
        calib_path: 标定文件路径（包含6个相机的变换矩阵）
        camera_index: 相机索引 (0-5，对应6个相机)
    """
    with open(calib_path, 'rb') as f:
        data = f.read()
    
    # 计算每个矩阵的大小
    total_size = len(data)
    num_cameras = 6  # 6个相机
    matrix_size = 4 * 4 * 4  # 4x4矩阵，每个元素4字节float
    
    # 检查文件大小是否合理
    expected_size = num_cameras * matrix_size
    if total_size != expected_size:
        print(f"警告: 标定文件大小不匹配。期望 {expected_size} 字节，实际 {total_size} 字节")
        # 尝试读取单个4x4矩阵
        if total_size >= matrix_size:
            matrix = np.frombuffer(data[:matrix_size], dtype=np.float32).reshape(4, 4)
            return matrix
        else:
            # 返回单位矩阵作为默认值
            return np.eye(4, dtype=np.float32)
    
    # 读取指定相机的变换矩阵
    start_offset = camera_index * matrix_size
    end_offset = start_offset + matrix_size
    
    if start_offset >= total_size:
        print(f"警告: 相机索引 {camera_index} 超出范围，使用单位矩阵")
        return np.eye(4, dtype=np.float32)
    
    matrix_data = data[start_offset:end_offset]
    matrix = np.frombuffer(matrix_data, dtype=np.float32).reshape(4, 4)
    
    return matrix

def load_image(image_path):
    """加载图像"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return img

def project_lidar_to_image(points_3d, lidar2img_matrix, img_shape):
    """将3D点云投影到2D图像平面"""
    # 添加齐次坐标
    points_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    
    # 应用变换矩阵
    points_cam = points_homo @ lidar2img_matrix.T
    
    # 透视除法
    points_cam[:, 0] /= points_cam[:, 2]
    points_cam[:, 1] /= points_cam[:, 2]
    
    # 提取2D坐标
    points_2d = points_cam[:, :2]
    
    # 过滤在图像范围内的点
    mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_shape[1]) & \
           (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_shape[0]) & \
           (points_cam[:, 2] > 0)  # 深度为正
    
    # 添加调试信息
    if len(points_2d) > 0:
        print(f"    投影统计:")
        print(f"      总点数: {len(points_3d)}")
        print(f"      有效投影点数: {np.sum(mask)}")
        print(f"      X范围: [{points_2d[:, 0].min():.1f}, {points_2d[:, 0].max():.1f}]")
        print(f"      Y范围: [{points_2d[:, 1].min():.1f}, {points_2d[:, 1].max():.1f}]")
        print(f"      深度范围: [{points_cam[:, 2].min():.1f}, {points_cam[:, 2].max():.1f}]")
        
        # 检查投影点是否集中在某个区域
        if np.sum(mask) > 0:
            valid_points = points_2d[mask]
            x_std = np.std(valid_points[:, 0])
            y_std = np.std(valid_points[:, 1])
            print(f"      X标准差: {x_std:.1f}, Y标准差: {y_std:.1f}")
            
            if x_std < 50 or y_std < 50:
                print(f"      ⚠️ 警告: 投影点分布过于集中，可能标定矩阵有问题")
    
    return points_2d[mask], mask

def create_color_map(depths, color_map_name='jet'):
    """根据深度创建颜色映射"""
    # 归一化深度到0-1
    depths_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-6)
    
    # 使用指定的颜色映射
    cmap = plt.get_cmap(color_map_name)
    colors = cmap(depths_norm)[:, :3] * 255
    return colors.astype(np.uint8)

def generate_lidar2img_from_nuscenes(config_path, output_dir, num_frames=10):
    """
    从nuScenes数据集生成lidar2img变换矩阵，每帧单独保存为一个bin文件
    
    Args:
        config_path: 数据集配置文件路径
        output_dir: 输出目录
        num_frames: 要生成的帧数
    """
    print(f"从nuScenes数据集生成lidar2img变换矩阵，共{num_frames}帧...")
    
    try:
        from nuscenes import NuScenes
        from pyquaternion import Quaternion
    except ImportError as e:
        print(f"❌ nuScenes或pyquaternion未安装: {e}")
        print("请安装: pip install nuscenes-devkit pyquaternion")
        return False
    
    # 读取配置
    try:
        from tool.utils.config import read_cfg
        cfg = read_cfg(config_path)
        dataset_cfg = cfg["data"]["val"].copy()
    except Exception as e:
        print(f"❌ 无法读取配置文件: {e}")
        return False
    
    # 获取nuScenes数据集路径
    data_root = dataset_cfg.get("data_root", "")
    version = dataset_cfg.get("version", "v1.0-trainval")
    
    print(f"数据根目录: {data_root}")
    print(f"数据集版本: {version}")
    
    if not data_root or not os.path.exists(data_root):
        print(f"❌ 数据根目录不存在: {data_root}")
        return False
    
    try:
        # 初始化nuScenes
        nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
        print(f"✅ NuScenes加载成功，共{len(nusc.sample)}个样本")
    except Exception as e:
        print(f"❌ 加载NuScenes失败: {e}")
        return False
    
    # 创建输出目录
    calib_dir = os.path.join(output_dir, "calib")
    os.makedirs(calib_dir, exist_ok=True)
    
    # 相机类型列表
    camera_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
                   "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    # 获取所有样本
    samples = list(nusc.sample)
    print(f"找到{len(samples)}个样本")
    
    # 处理前num_frames个样本
    for frame_idx in tqdm(range(min(num_frames, len(samples))), desc="生成lidar2img矩阵"):
        sample = samples[frame_idx]
        
        try:
            # 获取激光雷达的标定信息
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            
            # 为每个相机构建lidar2img变换矩阵
            all_lidar2img_matrices = []
            
            for cam_type in camera_types:
                if cam_type in sample['data']:
                    cam_data = nusc.get('sample_data', sample['data'][cam_type])
                    
                    try:
                        # 获取相机标定信息
                        cam_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                        cam_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
                        
                        # 构建完整的变换矩阵链
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
                        
                        # 计算ego到相机的逆变换
                        ego2cam = np.linalg.inv(cam2ego)
                        
                        # 5. 相机内参矩阵
                        cam_intrinsic = np.array(cam_sensor['camera_intrinsic'])
                        intrinsic_4x4 = np.eye(4)
                        intrinsic_4x4[:3, :3] = cam_intrinsic
                        
                        # 组合变换矩阵：lidar -> ego -> global -> ego' -> camera -> image
                        lidar2img_4x4 = intrinsic_4x4 @ ego2cam @ global2cam_ego @ ego2global @ lidar2ego
                        
                    except Exception as e:
                        print(f"    ⚠️ 计算{cam_type}的精确变换时出错: {e}")
                        # 使用合理的默认变换矩阵
                        lidar2img_4x4 = np.eye(4)
                        
                        # 设置相机内参（基于nuScenes的典型值）
                        lidar2img_4x4[0, 0] = 1200.0  # fx - 焦距x
                        lidar2img_4x4[1, 1] = 1200.0  # fy - 焦距y
                        lidar2img_4x4[0, 2] = 800.0   # cx - 主点x (1600/2)
                        lidar2img_4x4[1, 2] = 450.0   # cy - 主点y (900/2)
                        
                        # 设置外参（激光雷达到相机的变换）
                        lidar2img_4x4[0, 3] = 0.0     # x偏移
                        lidar2img_4x4[1, 3] = 0.0     # y偏移
                        lidar2img_4x4[2, 3] = -1.5    # z偏移（激光雷达到相机的高度差）
                    
                    all_lidar2img_matrices.append(lidar2img_4x4)
                else:
                    # 如果相机数据不存在，使用单位矩阵
                    all_lidar2img_matrices.append(np.eye(4))
            
            # 将6个相机的变换矩阵保存到一个文件中
            calib_path = os.path.join(calib_dir, f"lidar2img_{frame_idx}.bin")
            all_matrices = np.array(all_lidar2img_matrices, dtype=np.float32)
            all_matrices.tofile(calib_path)
            
            print(f"帧 {frame_idx}: 保存了{len(all_lidar2img_matrices)}个相机的标定数据到 {calib_path}")
            
        except Exception as e:
            print(f"处理帧 {frame_idx} 时出错: {e}")
            continue
    
    print(f"✅ 成功生成{min(num_frames, len(samples))}帧的lidar2img变换矩阵")
    return True

def visualize_projection(frame_idx, data_dir, output_dir, cameras, point_size=1, alpha=0.6, color_map='jet', debug=False):
    """可视化单帧的投影结果"""
    print(f"处理帧 {frame_idx}...")
    
    # 加载激光雷达点云
    lidar_path = os.path.join(data_dir, 'lidar', f'lidar_{frame_idx}.bin')
    if not os.path.exists(lidar_path):
        print(f"激光雷达文件不存在: {lidar_path}")
        return
    
    points_3d = load_lidar_points(lidar_path)
    print(f"加载了 {len(points_3d)} 个激光雷达点")
    
    # 创建大图用于显示所有相机的结果
    num_cams = len(cameras)
    cols = min(3, num_cams)
    rows = (num_cams + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    
    if num_cams == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for cam_idx, camera in enumerate(cameras):
        # 加载图像
        img_path = os.path.join(data_dir, 'images', camera, f'{frame_idx}.jpg')
        if not os.path.exists(img_path):
            print(f"图像文件不存在: {img_path}")
            continue
        
        img = load_image(img_path)
        print(f"  相机 {camera}: 图像尺寸 {img.shape}")
        
        # 加载对应的lidar2img变换矩阵
        calib_path = os.path.join(data_dir, 'calib', f'lidar2img_{frame_idx}.bin')
        if not os.path.exists(calib_path):
            print(f"标定文件不存在: {calib_path}")
            continue
        
        lidar2img_matrix = load_lidar2img_matrix(calib_path, cam_idx)
        
        # 调试标定矩阵
        if debug:
            debug_calibration_matrix(lidar2img_matrix, camera)
        
        # 投影点云到图像
        points_2d, mask = project_lidar_to_image(points_3d, lidar2img_matrix, img.shape)
        
        if len(points_2d) == 0:
            print(f"相机 {camera} 没有投影点")
            continue
        
        # 获取深度信息用于颜色映射
        depths = points_3d[mask, 2]  # z坐标作为深度
        colors = create_color_map(depths, color_map)
        
        # 绘制投影结果
        ax = axes[cam_idx]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 绘制投影点
        ax.scatter(points_2d[:, 0], points_2d[:, 1], c=colors/255, s=point_size, alpha=alpha)
        
        ax.set_title(f'{camera} - {len(points_2d)} points')
        ax.axis('off')
        
        # 保存单个相机的结果
        single_img = img.copy()
        for i, (point, color) in enumerate(zip(points_2d, colors)):
            cv2.circle(single_img, (int(point[0]), int(point[1])), point_size, color.tolist(), -1)
        
        single_output_path = os.path.join(output_dir, f'frame_{frame_idx}_{camera}.jpg')
        cv2.imwrite(single_output_path, single_img)
        
        print(f"  相机 {camera}: 投影了 {len(points_2d)} 个点，保存到 {single_output_path}")
    
    # 隐藏多余的子图
    for i in range(num_cams, len(axes)):
        axes[i].axis('off')
    
    # 保存组合图
    plt.tight_layout()
    combined_output_path = os.path.join(output_dir, f'frame_{frame_idx}_all_cameras.png')
    plt.savefig(combined_output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"帧 {frame_idx} 处理完成，结果保存到 {output_dir}")

def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='激光雷达点云投影到相机图像可视化')
    
    # 数据路径参数
    parser.add_argument('--data_dir', type=str, 
                       default="/share/Code/SparseEnd2End/C++/Data/sparse",
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str,
                       default="/share/Code/SparseEnd2End/C++/Output/val",
                       help='输出目录路径')
    parser.add_argument('--config_path', type=str,
                       default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
                       help='nuScenes数据集配置文件路径')
    
    # 处理参数
    parser.add_argument('--frames', type=str, default='all',
                       help='要处理的帧，格式: "0,1,2" 或 "all" 表示所有帧')
    parser.add_argument('--cameras', type=str, default='all',
                       help='要处理的相机，格式: "CAM_FRONT,CAM_BACK" 或 "all" 表示所有相机')
    parser.add_argument('--generate_calib', action='store_true',
                       help='是否从nuScenes数据集生成lidar2img标定文件')
    parser.add_argument('--num_frames', type=int, default=10,
                       help='生成标定文件时的帧数')
    
    # 可视化参数
    parser.add_argument('--point_size', type=int, default=1,
                       help='投影点的大小')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='投影点的透明度 (0-1)')
    parser.add_argument('--color_map', type=str, default='jet',
                       help='深度颜色映射 (jet, viridis, plasma, etc.)')
    
    # 其他参数
    parser.add_argument('--create_report', action='store_true',
                       help='是否创建HTML可视化报告')
    parser.add_argument('--verbose', action='store_true',
                       help='是否显示详细输出')
    parser.add_argument('--debug', action='store_true',
                       help='是否显示调试信息')
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果需要生成标定文件
    if args.generate_calib:
        print("开始从nuScenes数据集生成lidar2img标定文件...")
        success = generate_lidar2img_from_nuscenes(args.config_path, args.data_dir, args.num_frames)
        if not success:
            print("❌ 生成标定文件失败，退出程序")
            return
        print("✅ 标定文件生成完成")
    
    # 确定要处理的相机
    all_cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    if args.cameras.lower() == 'all':
        cameras = all_cameras
    else:
        cameras = [cam.strip() for cam in args.cameras.split(',')]
        # 验证相机名称
        for cam in cameras:
            if cam not in all_cameras:
                print(f"警告: 未知相机名称 {cam}")
    
    # 获取所有帧
    lidar_dir = os.path.join(args.data_dir, 'lidar')
    if not os.path.exists(lidar_dir):
        print(f"激光雷达目录不存在: {lidar_dir}")
        return
    
    # 获取所有帧索引
    frame_indices = []
    for file in os.listdir(lidar_dir):
        if file.startswith('lidar_') and file.endswith('.bin'):
            frame_idx = int(file.split('_')[1].split('.')[0])
            frame_indices.append(frame_idx)
    
    frame_indices.sort()
    
    # 确定要处理的帧
    if args.frames.lower() == 'all':
        frames_to_process = frame_indices
    else:
        frames_to_process = [int(f.strip()) for f in args.frames.split(',')]
        # 验证帧索引
        for frame_idx in frames_to_process:
            if frame_idx not in frame_indices:
                print(f"警告: 帧 {frame_idx} 不存在")
    
    if args.verbose:
        print(f"数据目录: {args.data_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"要处理的帧: {frames_to_process}")
        print(f"要处理的相机: {cameras}")
        print(f"点大小: {args.point_size}")
        print(f"透明度: {args.alpha}")
        print(f"颜色映射: {args.color_map}")
        print(f"调试模式: {args.debug}")
    
    print(f"找到 {len(frame_indices)} 帧数据，将处理 {len(frames_to_process)} 帧")
    
    # 处理每一帧
    for frame_idx in frames_to_process:
        try:
            visualize_projection(frame_idx, args.data_dir, args.output_dir, 
                               cameras, args.point_size, args.alpha, args.color_map, args.debug)
        except Exception as e:
            print(f"处理帧 {frame_idx} 时出错: {e}")
            continue
    
    print(f"所有帧处理完成，结果保存在: {args.output_dir}")
    
    # 创建可视化报告
    if args.create_report:
        create_visualization_report(args.output_dir, frames_to_process, cameras)

def create_visualization_report(output_dir, frame_indices, cameras):
    """创建可视化报告"""
    report_path = os.path.join(output_dir, 'visualization_report.html')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>激光雷达投影可视化报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .frame {{ margin-bottom: 30px; border: 1px solid #ccc; padding: 15px; }}
            .frame h3 {{ color: #333; }}
            .camera-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
            .camera-item {{ text-align: center; }}
            .camera-item img {{ max-width: 100%; height: auto; }}
            .camera-item p {{ margin: 5px 0; font-size: 12px; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>激光雷达投影可视化报告</h1>
        <div class="summary">
            <h3>处理摘要</h3>
            <p>处理了 {len(frame_indices)} 帧数据</p>
            <p>处理的相机: {', '.join(cameras)}</p>
            <p>输出目录: {output_dir}</p>
        </div>
    """
    
    for frame_idx in frame_indices:
        html_content += f"""
        <div class="frame">
            <h3>帧 {frame_idx}</h3>
            <div class="camera-grid">
        """
        
        for camera in cameras:
            img_path = f'frame_{frame_idx}_{camera}.jpg'
            if os.path.exists(os.path.join(output_dir, img_path)):
                html_content += f"""
                <div class="camera-item">
                    <img src="{img_path}" alt="{camera}">
                    <p>{camera}</p>
                </div>
                """
        
        html_content += """
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"可视化报告已生成: {report_path}")

if __name__ == "__main__":
    main()
