# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import argparse
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from typing import Optional, Dict, Any, List

from tool.trainer.utils import set_random_seed
from tool.utils.config import read_cfg
from tool.utils.data_parallel import E2EDataParallel
from tool.runner.fp16_utils import wrap_fp16_model
from tool.runner.checkpoint import load_checkpoint

from dataset.dataloader_wrapper import dataloader_wrapper_without_dist
from dataset import NuScenes4DDetTrackDataset
from modules.sparse4d_detector import Sparse4D


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse4D 预测脚本")
    parser.add_argument(
        "--config", 
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py", 
        help="配置文件路径"
    )
    parser.add_argument(
        "--checkpoint", 
        default="ckpt/sparse4dv3_r50.pth", 
        help="模型检查点文件路径"
    )
    parser.add_argument(
        "--output_dir", 
        default="predictions", 
        help="预测结果输出目录"
    )
    parser.add_argument(
        "--sample_idx", 
        type=int, 
        default=None, 
        help="指定预测的样本索引，如果不指定则预测所有样本"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=5, 
        help="预测的样本数量（当sample_idx未指定时）"
    )
    parser.add_argument(
        "--score_threshold", 
        type=float, 
        default=0.2, 
        help="检测置信度阈值"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="是否设置确定性选项用于CUDNN后端",
    )
    parser.add_argument(
        "--device", 
        default="cuda:0", 
        help="推理设备"
    )
    parser.add_argument(
        "--generate_input_bin",
        default=True,
        help="是否生成input_bin文件用于模型部署验证"
    )
    parser.add_argument(
        "--input_bin_samples",
        type=int,
        default=5,
        help="生成input_bin文件的样本数量"
    )
    parser.add_argument(
        "--generate_expected_output",
        default=False,
        help="是否生成预期输出文件用于模型部署验证"
    )
    parser.add_argument(
        "--expected_output_samples",
        type=int,
        default=5,
        help="生成预期输出文件的样本数量"
    )

    return parser.parse_args()


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    """构建模块的辅助函数"""
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def nms_3d(boxes, scores, iou_threshold):
    """简单的3D NMS实现"""
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    # 按置信度排序
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    sorted_boxes = boxes[sorted_indices]
    
    keep_indices = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    
    for i in range(len(sorted_boxes)):
        if suppressed[sorted_indices[i]]:
            continue
            
        keep_indices.append(sorted_indices[i])
        
        # 计算与当前框的IoU
        for j in range(i + 1, len(sorted_boxes)):
            if suppressed[sorted_indices[j]]:
                continue
                
            # 简化的IoU计算（基于中心点距离）
            current_box = sorted_boxes[i]
            other_box = sorted_boxes[j]
            
            # 计算中心点距离
            center_dist = torch.sqrt((current_box[0] - other_box[0])**2 + 
                                   (current_box[1] - other_box[1])**2)
            
            # 计算边界框对角线长度的一半作为阈值
            current_diag = torch.sqrt(current_box[3]**2 + current_box[4]**2) / 2
            other_diag = torch.sqrt(other_box[3]**2 + other_box[4]**2) / 2
            overlap_threshold = (current_diag + other_diag) * 0.5
            
            # 如果距离太近，认为是重叠
            if center_dist < overlap_threshold:
                suppressed[sorted_indices[j]] = True
    
    return torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)


def post_process_predictions(predictions: List[Dict], score_threshold: float = 0.2):
    """对预测结果进行后处理：只进行置信度过滤"""
    import torch
    
    processed_predictions = []
    
    for pred in predictions:
        if "img_bbox" in pred and pred["img_bbox"] is not None:
            bbox_pred = pred["img_bbox"]
            
            # 获取原始数据
            boxes_3d = bbox_pred["boxes_3d"]
            scores_3d = bbox_pred["scores_3d"]
            labels_3d = bbox_pred["labels_3d"]
            
            # 确保数据是tensor格式
            if not isinstance(boxes_3d, torch.Tensor):
                boxes_3d = torch.tensor(boxes_3d)
            if not isinstance(scores_3d, torch.Tensor):
                scores_3d = torch.tensor(scores_3d)
            if not isinstance(labels_3d, torch.Tensor):
                labels_3d = torch.tensor(labels_3d)
            
            # 置信度过滤
            valid_mask = scores_3d >= score_threshold
            if valid_mask.sum() == 0:
                # 如果没有目标通过置信度过滤，保留置信度最高的前10个
                top_k = min(10, len(scores_3d))
                _, top_indices = torch.topk(scores_3d, top_k)
                valid_mask = torch.zeros_like(scores_3d, dtype=torch.bool)
                valid_mask[top_indices] = True
            
            filtered_boxes = boxes_3d[valid_mask]
            filtered_scores = scores_3d[valid_mask]
            filtered_labels = labels_3d[valid_mask]
            
            # 按置信度排序
            sorted_indices = torch.argsort(filtered_scores, descending=True)
            final_boxes = filtered_boxes[sorted_indices]
            final_scores = filtered_scores[sorted_indices]
            final_labels = filtered_labels[sorted_indices]
            
            # 更新预测结果
            processed_pred = pred.copy()
            processed_pred["img_bbox"] = {
                "boxes_3d": final_boxes,
                "scores_3d": final_scores,
                "labels_3d": final_labels,
            }
            
            # 如果有track_ids，也需要过滤
            if "track_ids" in bbox_pred:
                original_track_ids = bbox_pred["track_ids"]
                if not isinstance(original_track_ids, torch.Tensor):
                    original_track_ids = torch.tensor(original_track_ids)
                filtered_track_ids = original_track_ids[valid_mask]
                final_track_ids = filtered_track_ids[sorted_indices]
                processed_pred["img_bbox"]["track_ids"] = final_track_ids
            
            processed_predictions.append(processed_pred)
        else:
            processed_predictions.append(pred)
    
    return processed_predictions


def draw_bev_image(boxes_3d, labels_3d, scores_3d, track_ids=None, class_names=None, 
                   output_path=None, size=50.0, resolution=0.1):
    """
    绘制BEV（Bird's Eye View）图像
    
    Args:
        boxes_3d: 3D边界框 [N, 9] (x, y, z, w, l, h, yaw, vx, vy)
        labels_3d: 类别标签 [N]
        scores_3d: 置信度分数 [N]
        track_ids: 跟踪ID [N] (可选)
        class_names: 类别名称列表
        output_path: 输出图像路径
        size: BEV图像大小（米）
        resolution: 分辨率（米/像素）
    """
    if len(boxes_3d) == 0:
        print("没有检测到目标，无法绘制BEV图像")
        return
    
    # 创建图像
    img_size = int(size / resolution)
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # 定义颜色映射
    colors = {
        'car': (255, 0, 0),      # 红色
        'truck': (255, 165, 0),   # 橙色
        'bus': (255, 255, 0),     # 黄色
        'trailer': (128, 0, 128), # 紫色
        'construction_vehicle': (0, 255, 255), # 青色
        'pedestrian': (0, 255, 0), # 绿色
        'motorcycle': (0, 0, 255), # 蓝色
        'bicycle': (128, 128, 0),  # 橄榄色
        'traffic_cone': (255, 192, 203), # 粉色
        'barrier': (165, 42, 42),  # 棕色
    }
    
    # 坐标转换：从世界坐标到图像坐标
    # 图像中心对应(0,0)，左上角为(-size/2, -size/2)
    center_x, center_y = img_size // 2, img_size // 2
    
    for i, (box, label, score) in enumerate(zip(boxes_3d, labels_3d, scores_3d)):
        x, y, z = box[0], box[1], box[2]  # 中心点坐标
        w, l = box[3], box[4]  # 宽度、长度
        # 从sin_yaw和cos_yaw计算yaw角
        sin_yaw = box[6]  # sin(yaw)
        cos_yaw = box[7]  # cos(yaw)
        yaw = np.arctan2(sin_yaw, cos_yaw)  # 计算实际的yaw角
        
        # 检查是否在图像范围内
        if abs(x) > size/2 or abs(y) > size/2:
            continue
        
        # 转换到图像坐标
        img_x = int((x + size/2) / resolution)
        img_y = int((y + size/2) / resolution)
        
        # 获取类别名称和颜色
        if class_names and label < len(class_names):
            class_name = class_names[label]
        else:
            class_name = f"class_{label}"
        
        color = colors.get(class_name, (128, 128, 128))  # 默认灰色
        
        # 绘制边界框
        # 计算边界框的四个角点
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # 边界框的半长和半宽
        half_l = l / 2
        half_w = w / 2
        
        # 四个角点的相对坐标
        corners = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ])
        
        # 旋转角点
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        rotated_corners = corners @ rotation_matrix.T
        
        # 转换到图像坐标
        img_corners = []
        for corner in rotated_corners:
            corner_x = int((corner[0] + x + size/2) / resolution)
            corner_y = int((corner[1] + y + size/2) / resolution)
            img_corners.append([corner_x, corner_y])
        
        img_corners = np.array(img_corners, dtype=np.int32)
        
        # 绘制填充的边界框
        cv2.fillPoly(img, [img_corners], color)
        
        # 绘制边界框轮廓
        cv2.polylines(img, [img_corners], True, (0, 0, 0), 2)
        
        # 绘制朝向指示线（加重朝向边框）
        # 计算朝向方向（车辆前进方向）
        # 在车辆坐标系中，车辆的朝向是沿着窄边方向
        # 如果l是前后方向，w是左右方向，那么朝向应该沿着l方向
        # 但是根据您的反馈，实际生活中车辆朝向应该在窄边
        # 所以我们需要判断哪个是窄边，然后沿着窄边绘制朝向
        if l < w:  # 如果长度小于宽度，说明l是窄边
            direction_x = cos_yaw * half_l  # 朝向方向的x分量（沿着窄边l）
            direction_y = sin_yaw * half_l  # 朝向方向的y分量（沿着窄边l）
        else:  # 如果长度大于宽度，说明w是窄边
            direction_x = cos_yaw * half_w  # 朝向方向的x分量（沿着窄边w）
            direction_y = sin_yaw * half_w  # 朝向方向的y分量（沿着窄边w）
        
        # 计算朝向线的起点和终点
        start_x = x
        start_y = y
        end_x = x + direction_x
        end_y = y + direction_y
        
        # 转换到图像坐标
        start_img_x = int((start_x + size/2) / resolution)
        start_img_y = int((start_y + size/2) / resolution)
        end_img_x = int((end_x + size/2) / resolution)
        end_img_y = int((end_y + size/2) / resolution)
        
        # 绘制朝向指示线（粗线，颜色更深）
        cv2.line(img, (start_img_x, start_img_y), (end_img_x, end_img_y), (0, 0, 0), 4)
        
        # 在朝向线末端绘制箭头
        arrow_length = 8
        arrow_angle = np.arctan2(direction_y, direction_x)
        
        # 箭头两个分支的角度
        arrow_angle1 = arrow_angle + np.pi/6
        arrow_angle2 = arrow_angle - np.pi/6
        
        # 计算箭头分支的端点
        arrow_x1 = end_img_x - arrow_length * np.cos(arrow_angle1)
        arrow_y1 = end_img_y - arrow_length * np.sin(arrow_angle1)
        arrow_x2 = end_img_x - arrow_length * np.cos(arrow_angle2)
        arrow_y2 = end_img_y - arrow_length * np.sin(arrow_angle2)
        
        # 绘制箭头分支
        cv2.line(img, (end_img_x, end_img_y), (int(arrow_x1), int(arrow_y1)), (0, 0, 0), 3)
        cv2.line(img, (end_img_x, end_img_y), (int(arrow_x2), int(arrow_y2)), (0, 0, 0), 3)
        
        # 添加文本标签
        track_id_text = f"ID:{track_ids[i]}" if track_ids is not None else ""
        label_text = f"{class_name}:{score:.2f} {track_id_text}"
        
        # 计算文本位置
        text_x = img_x
        text_y = img_y - 10 if img_y > 20 else img_y + 20
        
        # 绘制文本背景
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (text_x - 2, text_y - text_height - 2), 
                     (text_x + text_width + 2, text_y + 2), (255, 255, 255), -1)
        
        # 绘制文本
        cv2.putText(img, label_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 绘制坐标轴
    # X轴（红色）
    cv2.line(img, (center_x, center_y), (center_x + 50, center_y), (0, 0, 255), 2)
    cv2.putText(img, "X", (center_x + 55, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Y轴（绿色）
    cv2.line(img, (center_x, center_y), (center_x, center_y - 50), (0, 255, 0), 2)
    cv2.putText(img, "Y", (center_x - 5, center_y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 保存图像
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"BEV图像已保存到: {output_path}")
    
    return img


def generate_input_bin_files(model, dataset, output_dir: str, num_samples: int = 5):
    """
    生成模型推理所需的input_bin文件，用于模型部署验证
    
    Args:
        model: 训练好的模型
        dataset: 数据集
        output_dir: 输出目录
        num_samples: 生成的样本数量
    """
    import os
    import numpy as np
    from dataset.utils.collate import collate_fn
    
    # 创建输出目录
    bin_dir = os.path.join(output_dir, "input_bin")
    os.makedirs(bin_dir, exist_ok=True)
    
    print(f"生成input_bin文件到: {bin_dir}")
    print(f"生成 {num_samples} 个样本的输入文件...")
    
    model.eval()
    
    for sample_idx in range(min(num_samples, len(dataset))):
        print(f"处理样本 {sample_idx}...")
        
        # 获取数据并使用collate_fn处理
        single_data = [dataset[sample_idx]]  # 包装成list
        data = collate_fn(single_data, samples_per_gpu=1)
        
        # 处理DataContainer格式的数据
        processed_data = {}
        for key, value in data.items():
            if hasattr(value, 'data'):  # DataContainer
                if hasattr(value.data, 'cpu'):
                    processed_data[key] = value.data.cpu().numpy()
                else:
                    processed_data[key] = value.data
            elif isinstance(value, torch.Tensor):
                processed_data[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                processed_data[key] = value
            else:
                processed_data[key] = value
        
        # 提取需要的输入数据
        input_data = {}
        
        # 1. 原始图像数据 (img_inputs)
        if 'img_inputs' in processed_data:
            img_inputs = processed_data['img_inputs']
            if isinstance(img_inputs, list) and len(img_inputs) > 0:
                # 保存所有图像输入
                for img_idx, img_data in enumerate(img_inputs):
                    if isinstance(img_data, torch.Tensor):
                        img_data = img_data.cpu().numpy()
                    elif isinstance(img_data, np.ndarray):
                        img_data = img_data
                    else:
                        continue
                    
                    input_data[f'img_input_{img_idx}'] = img_data
                    
                    # 保存图像输入文件
                    img_shape = "*".join([str(it) for it in img_data.shape])
                    img_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_img_input_{img_idx}_{img_shape}_{img_data.dtype}.bin")
                    img_data.tofile(img_path)
                    print(f"  保存图像输入文件 {img_idx}: {img_path}")
        
        # 2. 空间形状 (spatial_shapes)
        if 'img_metas' in processed_data:
            img_metas = processed_data['img_metas']
            if hasattr(img_metas, 'data') and isinstance(img_metas.data, dict):
                if 'spatial_shapes' in img_metas.data:
                    spatial_shapes = img_metas.data['spatial_shapes']
                    if isinstance(spatial_shapes, torch.Tensor):
                        spatial_shapes = spatial_shapes.cpu().numpy()
                    input_data['spatial_shapes'] = spatial_shapes
                    
                    # 保存空间形状文件
                    spatial_shape = "*".join([str(it) for it in spatial_shapes.shape])
                    spatial_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_spatial_shapes_{spatial_shape}_{spatial_shapes.dtype}.bin")
                    spatial_shapes.tofile(spatial_path)
                    print(f"  保存空间形状文件: {spatial_path}")
        
        # 3. 层级起始索引 (level_start_index)
        if 'img_metas' in processed_data:
            img_metas = processed_data['img_metas']
            if hasattr(img_metas, 'data') and isinstance(img_metas.data, dict):
                if 'level_start_index' in img_metas.data:
                    level_start_index = img_metas.data['level_start_index']
                    if isinstance(level_start_index, torch.Tensor):
                        level_start_index = level_start_index.cpu().numpy()
                    input_data['level_start_index'] = level_start_index
                    
                    # 保存层级起始索引文件
                    level_shape = "*".join([str(it) for it in level_start_index.shape])
                    level_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_level_start_index_{level_shape}_{level_start_index.dtype}.bin")
                    level_start_index.tofile(level_path)
                    print(f"  保存层级起始索引文件: {level_path}")
        
        # 4. 时间间隔 (time_interval)
        if 'img_metas' in processed_data:
            img_metas = processed_data['img_metas']
            if hasattr(img_metas, 'data') and isinstance(img_metas.data, dict):
                if 'time_interval' in img_metas.data:
                    time_interval = img_metas.data['time_interval']
                    if isinstance(time_interval, torch.Tensor):
                        time_interval = time_interval.cpu().numpy()
                    input_data['time_interval'] = time_interval
                    
                    # 保存时间间隔文件
                    time_shape = "*".join([str(it) for it in time_interval.shape])
                    time_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_time_interval_{time_shape}_{time_interval.dtype}.bin")
                    time_interval.tofile(time_path)
                    print(f"  保存时间间隔文件: {time_path}")
        
        # 5. 图像尺寸 (image_wh)
        if 'img_metas' in processed_data:
            img_metas = processed_data['img_metas']
            if hasattr(img_metas, 'data') and isinstance(img_metas.data, dict):
                if 'img_shape' in img_metas.data:
                    img_shape = img_metas.data['img_shape']
                    if isinstance(img_shape, torch.Tensor):
                        img_shape = img_shape.cpu().numpy()
                    # 转换为图像宽高格式
                    image_wh = np.array([[shape[1], shape[0]] for shape in img_shape], dtype=np.float32)
                    input_data['image_wh'] = image_wh
                    
                    # 保存图像尺寸文件
                    wh_shape = "*".join([str(it) for it in image_wh.shape])
                    wh_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_image_wh_{wh_shape}_{image_wh.dtype}.bin")
                    image_wh.tofile(wh_path)
                    print(f"  保存图像尺寸文件: {wh_path}")
        
        # 6. 激光雷达到图像变换矩阵 (lidar2img)
        if 'lidar2img' in processed_data:
            lidar2img = processed_data['lidar2img']
            if isinstance(lidar2img, torch.Tensor):
                lidar2img = lidar2img.cpu().numpy()
            elif isinstance(lidar2img, memoryview):
                # 处理memoryview类型
                lidar2img = np.frombuffer(lidar2img, dtype=np.float32).reshape(-1, 4, 4)
            input_data['lidar2img'] = lidar2img
            
            # 保存激光雷达到图像变换矩阵文件
            lidar_shape = "*".join([str(it) for it in lidar2img.shape])
            lidar_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_lidar2img_{lidar_shape}_{lidar2img.dtype}.bin")
            lidar2img.tofile(lidar_path)
            print(f"  保存激光雷达到图像变换矩阵文件: {lidar_path}")
        
        # 7. 锚点数据 (anchor) - 从模型获取
        try:
            if hasattr(model.module, 'head') and hasattr(model.module.head, 'instance_bank'):
                instance_bank = model.module.head.instance_bank
                if hasattr(instance_bank, 'anchor'):
                    anchor = instance_bank.anchor.detach().cpu().numpy()
                    input_data['anchor'] = anchor
                    
                    # 保存锚点文件
                    anchor_shape = "*".join([str(it) for it in anchor.shape])
                    anchor_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_anchor_{anchor_shape}_{anchor.dtype}.bin")
                    anchor.tofile(anchor_path)
                    print(f"  保存锚点文件: {anchor_path}")
        except Exception as e:
            print(f"  获取锚点数据失败: {e}")
        
        # 8. 实例特征 (instance_feature) - 从模型获取
        try:
            with torch.no_grad():
                # 获取模型的实例特征
                if hasattr(model.module, 'head') and hasattr(model.module.head, 'instance_bank'):
                    instance_bank = model.module.head.instance_bank
                    if hasattr(instance_bank, 'query'):
                        instance_feature = instance_bank.query.detach().cpu().numpy()
                        input_data['instance_feature'] = instance_feature
                        
                        # 保存实例特征文件
                        instance_shape = "*".join([str(it) for it in instance_feature.shape])
                        instance_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_instance_feature_{instance_shape}_{instance_feature.dtype}.bin")
                        instance_feature.tofile(instance_path)
                        print(f"  保存实例特征文件: {instance_path}")
        except Exception as e:
            print(f"  获取实例特征失败: {e}")
        
        # 9. 其他元数据
        if 'img_metas' in processed_data:
            img_metas = processed_data['img_metas']
            if hasattr(img_metas, 'data') and isinstance(img_metas.data, dict):
                # 保存其他重要的元数据
                for meta_key in ['lidar2img', 'cam2img', 'box_type_3d', 'box_mode_3d']:
                    if meta_key in img_metas.data:
                        meta_value = img_metas.data[meta_key]
                        if isinstance(meta_value, torch.Tensor):
                            meta_value = meta_value.cpu().numpy()
                        elif isinstance(meta_value, list):
                            meta_value = np.array(meta_value)
                        
                        if isinstance(meta_value, np.ndarray):
                            input_data[meta_key] = meta_value
                            
                            # 保存元数据文件
                            meta_shape = "*".join([str(it) for it in meta_value.shape])
                            meta_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_{meta_key}_{meta_shape}_{meta_value.dtype}.bin")
                            meta_value.tofile(meta_path)
                            print(f"  保存{meta_key}文件: {meta_path}")
        
        # 保存输入数据的元信息
        meta_info = {
            'sample_idx': sample_idx,
            'input_files': {}
        }
        
        for key, value in input_data.items():
            if isinstance(value, np.ndarray):
                meta_info['input_files'][key] = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'file_path': f"sample_{sample_idx:03d}_{key}_{'*'.join([str(it) for it in value.shape])}_{value.dtype}.bin"
                }
        
        # 保存元信息文件
        import json
        meta_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, indent=2, ensure_ascii=False)
        print(f"  保存元信息文件: {meta_path}")
        
        print(f"  样本 {sample_idx} 处理完成")
    
    print(f"所有input_bin文件生成完成，保存在: {bin_dir}")


def generate_expected_output_files(model, dataset, output_dir: str, num_samples: int = 5):
    """
    生成预期的输出文件，用于与车载推理结果进行对比验证
    
    Args:
        model: 训练好的模型
        dataset: 数据集
        output_dir: 输出目录
        num_samples: 生成的样本数量
    """
    import os
    import numpy as np
    from dataset.utils.collate import collate_fn
    
    # 创建输出目录
    bin_dir = os.path.join(output_dir, "expected_output")
    os.makedirs(bin_dir, exist_ok=True)
    
    print(f"生成预期输出文件到: {bin_dir}")
    print(f"生成 {num_samples} 个样本的输出文件...")
    
    model.eval()
    
    # 获取设备
    device = next(model.parameters()).device
    
    for sample_idx in range(min(num_samples, len(dataset))):
        print(f"处理样本 {sample_idx}...")
        
        # 获取数据并使用collate_fn处理
        single_data = [dataset[sample_idx]]  # 包装成list
        data = collate_fn(single_data, samples_per_gpu=1)
        
        # 处理DataContainer格式的数据
        processed_data = {}
        for key, value in data.items():
            if hasattr(value, 'data'):  # DataContainer
                if hasattr(value.data, 'cpu'):
                    # 先转换为numpy，再转换为tensor
                    if isinstance(value.data, torch.Tensor):
                        processed_data[key] = value.data.cpu().numpy()
                    else:
                        processed_data[key] = value.data
                else:
                    processed_data[key] = value.data
            elif isinstance(value, torch.Tensor):
                processed_data[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                processed_data[key] = value
            else:
                processed_data[key] = value
        
        # 执行模型推理
        try:
            with torch.no_grad():
                # 将numpy数据转换为tensor并移到正确设备
                tensor_data = {}
                for key, value in processed_data.items():
                    if isinstance(value, np.ndarray):
                        tensor_data[key] = torch.from_numpy(value).to(device)
                    elif isinstance(value, torch.Tensor):
                        tensor_data[key] = value.to(device)
                    else:
                        tensor_data[key] = value
                
                results = model(return_loss=False, **tensor_data)
            
            # 提取输出结果
            if results and len(results) > 0:
                result = results[0]  # 取第一个结果
                
                # 1. 预测的实例特征 (pred_instance_feature) - 900个查询
                if hasattr(model.module, 'head') and hasattr(model.module.head, 'instance_bank'):
                    instance_bank = model.module.head.instance_bank
                    if hasattr(instance_bank, 'query'):
                        pred_instance_feature = instance_bank.query.detach().cpu().numpy()
                        
                        # 保存预测的实例特征文件
                        feature_shape = "*".join([str(it) for it in pred_instance_feature.shape])
                        feature_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_pred_instance_feature_{feature_shape}_{pred_instance_feature.dtype}.bin")
                        pred_instance_feature.tofile(feature_path)
                        print(f"  保存预测实例特征文件: {feature_path}")
                
                # 2. 预测的锚点 (pred_anchor) - 900个锚点
                if hasattr(model.module, 'head') and hasattr(model.module.head, 'instance_bank'):
                    instance_bank = model.module.head.instance_bank
                    if hasattr(instance_bank, 'anchor'):
                        pred_anchor = instance_bank.anchor.detach().cpu().numpy()
                        
                        # 保存预测的锚点文件
                        anchor_shape = "*".join([str(it) for it in pred_anchor.shape])
                        anchor_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_pred_anchor_{anchor_shape}_{pred_anchor.dtype}.bin")
                        pred_anchor.tofile(anchor_path)
                        print(f"  保存预测锚点文件: {anchor_path}")
                
                # === 新增：保存TensorRT引擎原始900个输出 ===
                # 需要从模型的head中获取原始预测结果（未经过后处理的900个）
                if hasattr(model.module, 'head'):
                    head = model.module.head
                    
                    # 尝试获取原始的类别分数 (900个)
                    if hasattr(head, 'cls_scores_raw') and head.cls_scores_raw is not None:
                        raw_cls_scores = head.cls_scores_raw.detach().cpu().numpy()
                        
                        # 保存原始类别分数文件 (900个)
                        cls_shape = "*".join([str(it) for it in raw_cls_scores.shape])
                        cls_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_raw_cls_scores_{cls_shape}_{raw_cls_scores.dtype}.bin")
                        raw_cls_scores.tofile(cls_path)
                        print(f"  保存原始类别分数文件 (900个): {cls_path}")
                    
                    # 尝试获取原始的质量分数 (900个)
                    if hasattr(head, 'quality_scores_raw') and head.quality_scores_raw is not None:
                        raw_quality_scores = head.quality_scores_raw.detach().cpu().numpy()
                        
                        # 保存原始质量分数文件 (900个)
                        quality_shape = "*".join([str(it) for it in raw_quality_scores.shape])
                        quality_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_raw_quality_scores_{quality_shape}_{raw_quality_scores.dtype}.bin")
                        raw_quality_scores.tofile(quality_path)
                        print(f"  保存原始质量分数文件 (900个): {quality_path}")
                    
                    # 尝试获取原始的边界框 (900个)
                    if hasattr(head, 'bbox_preds_raw') and head.bbox_preds_raw is not None:
                        raw_bbox_preds = head.bbox_preds_raw.detach().cpu().numpy()
                        
                        # 保存原始边界框文件 (900个)
                        bbox_shape = "*".join([str(it) for it in raw_bbox_preds.shape])
                        bbox_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_raw_bbox_preds_{bbox_shape}_{raw_bbox_preds.dtype}.bin")
                        raw_bbox_preds.tofile(bbox_path)
                        print(f"  保存原始边界框文件 (900个): {bbox_path}")
                    
                    # 尝试获取原始的标签预测 (900个)
                    if hasattr(head, 'labels_raw') and head.labels_raw is not None:
                        raw_labels = head.labels_raw.detach().cpu().numpy()
                        
                        # 保存原始标签文件 (900个)
                        labels_shape = "*".join([str(it) for it in raw_labels.shape])
                        labels_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_raw_labels_{labels_shape}_{raw_labels.dtype}.bin")
                        raw_labels.tofile(labels_path)
                        print(f"  保存原始标签文件 (900个): {labels_path}")
                    
                    # 尝试获取原始的跟踪ID (900个)
                    if hasattr(head, 'track_ids_raw') and head.track_ids_raw is not None:
                        raw_track_ids = head.track_ids_raw.detach().cpu().numpy()
                        
                        # 保存原始跟踪ID文件 (900个)
                        track_shape = "*".join([str(it) for it in raw_track_ids.shape])
                        track_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_raw_track_ids_{track_shape}_{raw_track_ids.dtype}.bin")
                        raw_track_ids.tofile(track_path)
                        print(f"  保存原始跟踪ID文件 (900个): {track_path}")
                    
                    # 如果上述属性不存在，尝试通过其他方式获取原始输出
                    # 可能需要从模型的最后一层输出中获取
                    if not any([hasattr(head, attr) for attr in ['cls_scores_raw', 'quality_scores_raw', 'bbox_preds_raw']]):
                        print(f"  警告: 未找到原始900个输出，可能需要修改模型代码来保存原始预测结果")
                        print(f"  当前head属性: {[attr for attr in dir(head) if not attr.startswith('_')]}")
                
                # === 继续保存过滤后的结果（300个）用于对比 ===
                # 3. 预测的类别分数 (pred_class_score) - 300个最终检测结果
                if 'img_bbox' in result and result['img_bbox'] is not None:
                    bbox_pred = result['img_bbox']
                    if 'cls_scores' in bbox_pred:
                        pred_class_score = bbox_pred['cls_scores'].cpu().numpy()
                        
                        # 保存预测的类别分数文件
                        score_shape = "*".join([str(it) for it in pred_class_score.shape])
                        score_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_pred_class_score_{score_shape}_{pred_class_score.dtype}.bin")
                        pred_class_score.tofile(score_path)
                        print(f"  保存预测类别分数文件 (300个): {score_path}")
                
                # 4. 预测的质量分数 (pred_quality_score) - 300个最终检测结果
                if 'img_bbox' in result and result['img_bbox'] is not None:
                    bbox_pred = result['img_bbox']
                    if 'quality_scores' in bbox_pred:
                        pred_quality_score = bbox_pred['quality_scores'].cpu().numpy()
                        
                        # 保存预测的质量分数文件
                        quality_shape = "*".join([str(it) for it in pred_quality_score.shape])
                        quality_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_pred_quality_score_{quality_shape}_{pred_quality_score.dtype}.bin")
                        pred_quality_score.tofile(quality_path)
                        print(f"  保存预测质量分数文件 (300个): {quality_path}")
                
                # 5. 预测的边界框 (pred_boxes_3d) - 300个最终检测结果
                if 'img_bbox' in result and result['img_bbox'] is not None:
                    bbox_pred = result['img_bbox']
                    if 'boxes_3d' in bbox_pred:
                        pred_boxes_3d = bbox_pred['boxes_3d'].cpu().numpy()
                        
                        # 保存预测的边界框文件
                        boxes_shape = "*".join([str(it) for it in pred_boxes_3d.shape])
                        boxes_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_pred_boxes_3d_{boxes_shape}_{pred_boxes_3d.dtype}.bin")
                        pred_boxes_3d.tofile(boxes_path)
                        print(f"  保存预测边界框文件 (300个): {boxes_path}")
                
                # 6. 预测的标签 (pred_labels_3d) - 300个最终检测结果
                if 'img_bbox' in result and result['img_bbox'] is not None:
                    bbox_pred = result['img_bbox']
                    if 'labels_3d' in bbox_pred:
                        pred_labels_3d = bbox_pred['labels_3d'].cpu().numpy()
                        
                        # 保存预测的标签文件
                        labels_shape = "*".join([str(it) for it in pred_labels_3d.shape])
                        labels_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_pred_labels_3d_{labels_shape}_{pred_labels_3d.dtype}.bin")
                        pred_labels_3d.tofile(labels_path)
                        print(f"  保存预测标签文件 (300个): {labels_path}")
                
                # 7. 预测的跟踪ID (pred_track_ids) - 300个最终检测结果
                if 'img_bbox' in result and result['img_bbox'] is not None:
                    bbox_pred = result['img_bbox']
                    if 'track_ids' in bbox_pred:
                        pred_track_ids = bbox_pred['track_ids'].cpu().numpy()
                        
                        # 保存预测的跟踪ID文件
                        track_shape = "*".join([str(it) for it in pred_track_ids.shape])
                        track_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_pred_track_ids_{track_shape}_{pred_track_ids.dtype}.bin")
                        pred_track_ids.tofile(track_path)
                        print(f"  保存预测跟踪ID文件 (300个): {track_path}")
                
                # 保存输出数据的元信息
                meta_info = {
                    'sample_idx': sample_idx,
                    'output_files': {},
                    'description': {
                        'raw_outputs': '原始TensorRT引擎输出 (900个)',
                        'processed_outputs': '经过后处理的输出 (300个)'
                    }
                }
                
                # 记录已保存的文件信息
                output_files = []
                
                # 检查并记录各个输出文件
                if 'pred_instance_feature' in locals():
                    output_files.append(('pred_instance_feature', pred_instance_feature))
                if 'pred_anchor' in locals():
                    output_files.append(('pred_anchor', pred_anchor))
                if 'pred_class_score' in locals():
                    output_files.append(('pred_class_score', pred_class_score))
                if 'pred_quality_score' in locals():
                    output_files.append(('pred_quality_score', pred_quality_score))
                
                for name, data in output_files:
                    if data is not None:
                        meta_info['output_files'][name] = {
                            'shape': list(data.shape),
                            'dtype': str(data.dtype),
                            'file_path': f"sample_{sample_idx:03d}_{name}_{'*'.join([str(it) for it in data.shape])}_{data.dtype}.bin"
                        }
                
                # 保存元信息文件
                import json
                meta_path = os.path.join(bin_dir, f"sample_{sample_idx:03d}_output_meta.json")
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta_info, f, indent=2, ensure_ascii=False)
                print(f"  保存输出元信息文件: {meta_path}")
                
        except Exception as e:
            print(f"  样本 {sample_idx} 推理失败: {e}")
        
        print(f"  样本 {sample_idx} 处理完成")
    
    print(f"所有预期输出文件生成完成，保存在: {bin_dir}")


def visualize_predictions(predictions: List[Dict], output_dir: str, class_names: List[str]):
    """可视化预测结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预测结果为JSON格式
    results = []
    for i, pred in enumerate(predictions):
        if "img_bbox" in pred:
            bbox_pred = pred["img_bbox"]
            if bbox_pred is not None:
                result = {
                    "sample_idx": i,
                    "boxes_3d": bbox_pred["boxes_3d"].cpu().numpy().tolist() if hasattr(bbox_pred["boxes_3d"], "cpu") else bbox_pred["boxes_3d"].tolist(),
                    "scores_3d": bbox_pred["scores_3d"].cpu().numpy().tolist() if hasattr(bbox_pred["scores_3d"], "cpu") else bbox_pred["scores_3d"].tolist(),
                    "labels_3d": bbox_pred["labels_3d"].cpu().numpy().tolist() if hasattr(bbox_pred["labels_3d"], "cpu") else bbox_pred["labels_3d"].tolist(),
                }
                if "track_ids" in bbox_pred:
                    result["track_ids"] = bbox_pred["track_ids"].tolist()
                results.append(result)
    
    # 保存详细结果
    with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 创建BEV图像目录
    bev_dir = os.path.join(output_dir, "bev_images")
    os.makedirs(bev_dir, exist_ok=True)
    
    # 保存统计信息
    stats = {
        "total_samples": len(predictions),
        "class_names": class_names,
        "detection_summary": {}
    }
    
    for class_name in class_names:
        stats["detection_summary"][class_name] = {
            "total_detections": 0,
            "avg_confidence": 0.0
        }
    
    total_detections = 0
    total_confidence = 0.0
    
    # 为每个样本绘制BEV图像
    for i, pred in enumerate(predictions):
        if "img_bbox" in pred and pred["img_bbox"] is not None:
            bbox_pred = pred["img_bbox"]
            labels = bbox_pred["labels_3d"].cpu().numpy() if hasattr(bbox_pred["labels_3d"], "cpu") else bbox_pred["labels_3d"].numpy()
            scores = bbox_pred["scores_3d"].cpu().numpy() if hasattr(bbox_pred["scores_3d"], "cpu") else bbox_pred["scores_3d"].numpy()
            boxes = bbox_pred["boxes_3d"].cpu().numpy() if hasattr(bbox_pred["boxes_3d"], "cpu") else bbox_pred["boxes_3d"].numpy()
            
            # 获取track_ids（如果存在）
            track_ids = None
            if "track_ids" in bbox_pred:
                track_ids = bbox_pred["track_ids"].cpu().numpy() if hasattr(bbox_pred["track_ids"], "cpu") else bbox_pred["track_ids"].numpy()
            
            # 绘制BEV图像
            bev_path = os.path.join(bev_dir, f"sample_{i:03d}_bev.jpg")
            draw_bev_image(boxes, labels, scores, track_ids, class_names, bev_path)
            
            # 统计信息
            for label, score in zip(labels, scores):
                if 0 <= label < len(class_names):
                    class_name = class_names[label]
                    stats["detection_summary"][class_name]["total_detections"] += 1
                    stats["detection_summary"][class_name]["avg_confidence"] += score
                    total_detections += 1
                    total_confidence += score
    
    # 计算平均置信度
    for class_name in class_names:
        if stats["detection_summary"][class_name]["total_detections"] > 0:
            stats["detection_summary"][class_name]["avg_confidence"] /= stats["detection_summary"][class_name]["total_detections"]
    
    stats["overall"] = {
        "total_detections": total_detections,
        "avg_confidence": total_confidence / max(1, total_detections)
    }
    
    with open(os.path.join(output_dir, "prediction_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"预测结果已保存到: {output_dir}")
    print(f"BEV图像已保存到: {bev_dir}")
    print(f"总检测数量: {total_detections}")
    print(f"平均置信度: {stats['overall']['avg_confidence']:.4f}")
    
    # 打印各类别检测统计
    print("\n各类别检测统计:")
    for class_name, info in stats["detection_summary"].items():
        if info["total_detections"] > 0:
            print(f"  {class_name}: {info['total_detections']} 个检测, 平均置信度: {info['avg_confidence']:.4f}")


def predict_single_sample(model, dataset, sample_idx: int, device: str):
    """预测单个样本，并打印详细shape和类型信息"""
    model.eval()
    
    # 获取数据
    data = dataset[sample_idx]
    print("==== data keys ====", list(data.keys()))
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"{k}: shape={v.shape}, type={type(v)}")
        elif isinstance(v, dict):
            print(f"{k}: dict, keys={list(v.keys())}")
        else:
            print(f"{k}: type={type(v)}")
    if 'img_metas' in data:
        print("img_metas内容:")
        img_metas = data['img_metas']
        if hasattr(img_metas, 'data'):
            print(f"  DataContainer, type={type(img_metas.data)}")
            if isinstance(img_metas.data, dict):
                for mk, mv in img_metas.data.items():
                    if hasattr(mv, 'shape'):
                        print(f"    {mk}: shape={mv.shape}, type={type(mv)}")
                    else:
                        print(f"    {mk}: type={type(mv)}")
            else:
                print(f"  img_metas.data: {img_metas.data}")
        else:
            print(f"  img_metas: {img_metas}")
    
    # 正确处理DataContainer格式的数据
    processed_data = {}
    for key, value in data.items():
        if hasattr(value, 'data'):  # DataContainer
            if hasattr(value.data, 'cpu'):
                processed_data[key] = value.data.cpu().to(device)
            else:
                processed_data[key] = value.data
        elif isinstance(value, torch.Tensor):
            processed_data[key] = value.to(device)
        elif isinstance(value, np.ndarray):
            processed_data[key] = torch.from_numpy(value).to(device)
        else:
            processed_data[key] = value
    
    # 执行预测
    try:
        with torch.no_grad():
            results = model(return_loss=False, **processed_data)
        return results
    except Exception as e:
        print(f"[模型推理报错] {e}")
        raise


def main():
    args = parse_args()
    
    # 读取配置
    cfg = read_cfg(args.config)
    cfg["model"]["img_backbone"]["init_cfg"] = {}
    
    # 设置随机种子
    set_random_seed(cfg.get("seed", 0), deterministic=args.deterministic)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 构建数据集
    print("构建数据集...")
    # 像test.py一样直接从配置文件读取数据集配置，并移除samples_per_gpu参数
    samples_per_gpu = cfg["data"]["test"].pop("samples_per_gpu", 1)
    dataset = build_module(cfg["data"]["test"])
    print(f"数据集大小: {len(dataset)}")
    
    # 构建模型
    print("构建模型...")
    model = build_module(cfg["model"])
    
    # 加载检查点
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    
    # 设置FP16
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    # 将模型移到设备上并包装为DataParallel
    model = model.to(device)
    model = E2EDataParallel(model, device_ids=[0])
    
    # 创建数据加载器，像test.py一样使用DataLoader
    print("创建数据加载器...")
    data_loader = dataloader_wrapper_without_dist(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg["data"]["workers_per_gpu"],
        dist=False,
        shuffle=False,
    )
    
    # 确定要预测的样本数量
    if args.sample_idx is not None:
        # 如果指定了特定样本，我们需要创建一个只包含该样本的数据加载器
        # 这里我们使用一个简单的方法：预测所有样本，然后只返回指定的样本
        num_samples_to_predict = len(dataset)
        print(f"预测所有样本，然后返回样本 {args.sample_idx}")
    else:
        num_samples_to_predict = min(args.num_samples, len(dataset))
        print(f"预测前 {num_samples_to_predict} 个样本")
    
    # 执行预测
    print("开始预测...")
    predictions = []
    
    # 使用DataLoader进行预测，像test.py一样
    model.eval()
    for i, data in enumerate(data_loader):
        if i >= num_samples_to_predict:
            break
            
        print(f"预测批次 {i+1}")
        
        try:
            with torch.no_grad():
                results = model(return_loss=False, **data)
            
            # 处理结果
            for j, result in enumerate(results):
                predictions.append(result)
                    
        except Exception as e:
            print(f"  预测批次 {i+1} 时出错: {e}")
            predictions.append({"error": str(e)})
    
    # 如果指定了特定样本，只返回该样本的结果
    if args.sample_idx is not None and args.sample_idx < len(predictions):
        predictions = [predictions[args.sample_idx]]
        print(f"返回指定样本 {args.sample_idx} 的结果")
    
    # 后处理：置信度过滤
    print(f"\n进行后处理：置信度过滤 (阈值: {args.score_threshold})...")
    processed_predictions = post_process_predictions(predictions, args.score_threshold)
    
    # 打印后处理统计和详细信息
    print("后处理统计:")
    for i, pred in enumerate(processed_predictions):
        if "img_bbox" in pred and pred["img_bbox"] is not None:
            bbox_pred = pred["img_bbox"]
            original_count = len(predictions[i]["img_bbox"]["boxes_3d"]) if "img_bbox" in predictions[i] else 0
            final_count = len(bbox_pred["boxes_3d"])
            print(f"  样本 {i}: {original_count} -> {final_count} 个目标 (过滤了 {original_count - final_count} 个)")
            
            # 显示后处理后的目标详细信息
            if final_count > 0:
                scores = bbox_pred["scores_3d"].cpu().numpy() if hasattr(bbox_pred["scores_3d"], "cpu") else bbox_pred["scores_3d"].numpy()
                labels = bbox_pred["labels_3d"].cpu().numpy() if hasattr(bbox_pred["labels_3d"], "cpu") else bbox_pred["labels_3d"].numpy()
                boxes = bbox_pred["boxes_3d"].cpu().numpy() if hasattr(bbox_pred["boxes_3d"], "cpu") else bbox_pred["boxes_3d"].numpy()
                
                # 获取track_ids（如果存在）
                track_ids = None
                if "track_ids" in bbox_pred:
                    track_ids = bbox_pred["track_ids"].cpu().numpy() if hasattr(bbox_pred["track_ids"], "cpu") else bbox_pred["track_ids"].numpy()
                
                print(f"    后处理后的目标详细信息:")
                for k in range(min(10, final_count)):  # 显示前10个
                    class_name = cfg["class_names"][labels[k]] if labels[k] < len(cfg["class_names"]) else f"未知({labels[k]})"
                    score = scores[k]
                    
                    # 解析3D边界框信息
                    box = boxes[k]
                    x, y, z = box[0], box[1], box[2]  # 中心点坐标
                    w, l, h = box[3], box[4], box[5]  # 宽度、长度、高度
                    # 从sin_yaw和cos_yaw计算yaw角
                    sin_yaw = box[6]  # sin(yaw)
                    cos_yaw = box[7]  # cos(yaw)
                    yaw = np.arctan2(sin_yaw, cos_yaw)  # 偏航角（弧度）
                    vx, vy = box[8], box[9]  # 速度分量
                    
                    # 转换角度为度
                    yaw_deg = np.degrees(yaw)
                    
                    # 获取track_id
                    track_id = track_ids[k] if track_ids is not None else -1
                    
                    print(f"      {k+1}. {class_name} (ID: {track_id})")
                    print(f"          置信度: {score:.4f}")
                    print(f"          位置: x={x:.2f}m, y={y:.2f}m, z={z:.2f}m")
                    print(f"          尺寸: 长={l:.2f}m, 宽={w:.2f}m, 高={h:.2f}m")
                    print(f"          角度: {yaw_deg:.1f}° ({yaw:.3f}rad)")
                    print(f"          速度: vx={vx:.2f}m/s, vy={vy:.2f}m/s")
                    print(f"          速度大小: {np.sqrt(vx**2 + vy**2):.2f}m/s")
                    
                if final_count > 10:
                    print(f"      ... 还有 {final_count - 10} 个目标")
        else:
            print(f"  样本 {i}: 无有效检测结果")
    
    # 可视化结果
    print("\n保存预测结果...")
    visualize_predictions(processed_predictions, args.output_dir, cfg["class_names"])
    
    # 生成input_bin文件（如果指定）
    if args.generate_input_bin:
        print("\n生成input_bin文件用于模型部署验证...")
        generate_input_bin_files(model, dataset, args.output_dir, args.input_bin_samples)
    
    # 生成预期输出文件（如果指定）
    if args.generate_expected_output:
        print("\n生成预期输出文件用于模型部署验证...")
        generate_expected_output_files(model, dataset, args.output_dir, args.expected_output_samples)
    
    print("预测完成!")


if __name__ == "__main__":
    main() 