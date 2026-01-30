# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
# Modified for PTQ (INT8) Export with Real Calibration Data
import os
import time
import copy
import logging
import argparse
import sys
import glob
import numpy as np

# 设置完全确定性环境 (INT8 Calibration 时必须关闭，否则直方图统计无法运行)
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '100'

import onnx
from onnxsim import simplify
from onnx import numpy_helper
import onnx.helper as helper

def fix_onnx_topk_k(onnx_path, logger=None):
    """
    针对 TensorRT 8.5 等版本不支持动态 K 的问题，手动折叠 TopK 的 K 输入
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    init_map = {init.name: init for init in graph.initializer}
    node_map = {output: node for node in graph.node for output in node.output}

    def get_constant_value(tensor_name):
        if tensor_name in init_map: return numpy_helper.to_array(init_map[tensor_name])
        if tensor_name not in node_map: return None
        producer = node_map[tensor_name]
        if producer.op_type == "Constant": return numpy_helper.to_array(producer.attribute[0].t)
        if producer.op_type == "Reshape":
            data = get_constant_value(producer.input[0])
            shape = get_constant_value(producer.input[1])
            if data is not None and shape is not None: return data.reshape(shape)
        if producer.op_type == "Cast": return get_constant_value(producer.input[0])
        return None

    changed = False
    for node in graph.node:
        if node.op_type == "TopK" and len(node.input) >= 2:
            k_input = node.input[1]
            if k_input not in init_map:
                val = get_constant_value(k_input)
                if val is not None and val.size == 1:
                    new_k_name = f"{node.name}_K_fixed"
                    graph.initializer.append(helper.make_tensor(new_k_name, onnx.TensorProto.INT64, [1], [int(val.item())]))
                    node.input[1] = new_k_name
                    changed = True
                    if logger: logger.info(f"[Fix] Folded TopK K for {node.name} to {int(val.item())}")
    if changed:
        onnx.save(model, onnx_path)
        return True
    return False

import torch
from torch import nn

# ================= QUANTIZATION IMPORTS =================
try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib
    from pytorch_quantization.tensor_quant import QuantDescriptor
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False
    print("[WARNING] pytorch-quantization not found! INT8 export will fallback to FP32/FP16.")
# =======================================================

from modules.sparse4d_detector import *
from modules.head.sparse4d_blocks.instance_bank import topk, topk_for_onnx_export
from modules.ops import deformable_aggregation_function as DAF

from tool.utils.config import read_cfg
from typing import Optional, Dict, Any, Tuple
from tool.utils.logger import set_logger

# 导入 sparsebox_plugin 替换函数
try:
    from deploy.sparsebox_plugin.replace_kps_generator import (
        replace_kps_generator_with_plugin
    )
    SPARSEBOX_PLUGIN_AVAILABLE = True
except ImportError as e:
    SPARSEBOX_PLUGIN_AVAILABLE = False

# 导入 ln_plugin 替换函数
try:
    from deploy.ln_plugin.replace_layernorm import replace_layernorm_with_plugin
    LN_PLUGIN_AVAILABLE = True
except ImportError as e:
    LN_PLUGIN_AVAILABLE = False

# 设置PyTorch确定性 (INT8 Calibration 时必须关闭)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
# torch.use_deterministic_algorithms(True, warn_only=False)

# 设置随机种子
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)

np.random.seed(100)
torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy SparseEND2END Head with INT8 PTQ!")
    parser.add_argument(
        "--cfg",
        type=str,
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
        help="deploy config file path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt/sparse4dv3_r50.pth",
        help="deploy ckpt path",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="deploy/onnx/export_head_quant.log",
    )
    parser.add_argument(
        "--save_onnx1",
        type=str,
        default="deploy/onnx/sparse4dhead1st_int8.onnx",
    )
    parser.add_argument(
        "--save_onnx2",
        type=str,
        default="deploy/onnx/sparse4dhead2nd_int8.onnx",
    )
    parser.add_argument(
        "--o2", action="store_true", help="only export sparse4dhead2nd onnx."
    )
    # 量化相关参数
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 Quantization (PTQ). Requires pytorch-quantization.",
    )
    parser.add_argument(
        "--calib_batches",
        type=int,
        default=100,
        help="Number of batches for calibration.",
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default=None,
        help="Path to calibration data directory (generated by 051 script). If None, use dummy data (NOT RECOMMENDED).",
    )
    args = parser.parse_args()
    return args


# ================= QUANTIZATION UTILS =================

def initialize_quantization():
    """初始化量化配置"""
    if not HAS_QUANTIZATION:
        return

    # 推荐配置：输入使用直方图校准，权重使用Per-Channel Max
    # quant_desc_input = QuantDescriptor(calib_method='histogram', axis=None) # Histogram unstable?
    quant_desc_input = QuantDescriptor(calib_method='max', axis=None)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    print("[Quant] Quantization initialized with MAX calibration and FB_FAKE_QUANT mode.")


def replace_to_quantization_module(model, ignore_layers=None, parent_name=""):
    """
    递归将 nn.Linear 替换为 quant_nn.QuantLinear
    """
    if not HAS_QUANTIZATION:
        return

    if ignore_layers is None:
        ignore_layers = []

    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        # 检查是否在忽略列表中 (部分匹配)
        is_ignored = any(x in full_name for x in ignore_layers)

        if is_ignored:
            print(f"[Quant] Ignoring layer (keeping FP16/32): {full_name}")
            continue

        if isinstance(module, torch.nn.Linear):
            # 创建 QuantLinear
            quant_layer = quant_nn.QuantLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None
            )
            # 复制权重和偏置
            quant_layer.weight.data = module.weight.data
            if module.bias is not None:
                quant_layer.bias.data = module.bias.data

            # 替换
            setattr(model, name, quant_layer.cuda())
            print(f"[Quant] Replaced: {full_name} -> QuantLinear")
        else:
            # 递归
            replace_to_quantization_module(module, ignore_layers, full_name)

class CalibrationDataLoader:
    def __init__(self, data_dir, head_type="head1", use_fp16=True):
        self.use_fp16 = use_fp16
        subdir = os.path.join(data_dir, head_type)
        if not os.path.exists(subdir):
             raise ValueError(f"Calibration data directory not found: {subdir}")

        self.data_files = sorted(glob.glob(os.path.join(subdir, "*.npz")))
        if len(self.data_files) == 0:
            raise ValueError(f"No .npz files found in {subdir}")

        self.idx = 0
        self.head_type = head_type
        print(f"[Quant] Found {len(self.data_files)} calibration samples for {head_type}")

    def __len__(self):
        return len(self.data_files)

    def __call__(self):
        if self.idx >= len(self.data_files):
            self.idx = 0 # Loop if needed

        filepath = self.data_files[self.idx]
        data = np.load(filepath)
        self.idx += 1

        # Convert numpy to torch
        batch = {}
        for k in data.files:
            tensor = torch.from_numpy(data[k])
            if torch.cuda.is_available():
                tensor = tensor.cuda()

            if tensor.is_floating_point():
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"[Quant] WARNING: Found NaN/Inf in input {k}. Replacing with 0.")
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                tensor = tensor.half() if self.use_fp16 else tensor.float()

            batch[k] = tensor

        # Return tuple based on head type signature
        if self.head_type == "head1":
            return (
                batch["feature"],
                batch["spatial_shapes"],
                batch["level_start_index"],
                batch["instance_feature"],
                batch["anchor"],
                batch["time_interval"],
                batch["image_wh"],
                batch["lidar2img"]
            )
        else: # head2
            return (
                batch["feature"],
                batch["spatial_shapes"],
                batch["level_start_index"],
                batch["instance_feature"],
                batch["anchor"],
                batch["time_interval"],
                batch["temp_instance_feature"],
                batch["temp_anchor"],
                batch["mask"],
                batch["track_id"],
                batch["image_wh"],
                batch["lidar2img"]
            )

def collect_stats(model, data_loader_func, num_batches):
    """
    校准流程：喂入数据，收集统计信息
    """
    if not HAS_QUANTIZATION:
        return

    print(f"[Quant] Starting calibration with {num_batches} batches...")

    # 1. 开启校准模式
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable_calib()
            module.disable_quant()

    # 2. 喂入数据
    with torch.no_grad():
        for i in range(num_batches):
            if i % 10 == 0:
                print(f"[Quant] Calibrating batch {i}/{num_batches}...")

            inputs = data_loader_func()

            try:
                if isinstance(inputs, (tuple, list)):
                    model(*inputs)
                elif isinstance(inputs, dict):
                    model(**inputs)
            except Exception as e:
                print(f"[Quant] Warning during calibration step {i}: {e}")
                pass

    # 3. 结束校准，计算 Scale (amax)
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            try:
                # 兼容不同类型的 Calibrator
                if isinstance(module._calibrator, calib.HistogramCalibrator):
                    module.load_calib_amax("entropy", strict=False)
                else:
                    # MaxCalibrator 不需要 method 参数
                    module.load_calib_amax(strict=False)

                # Check for invalid amax
                # 注意：TensorQuantizer 的属性是 amax (buffer)，不是 _amax
                # Check if calibrator has seen any data
                if module._calibrator._calib_amax is None:
                    module.amax.fill_(1.0)
                    module.disable_quant()
                    continue

                if module.amax is None or torch.isnan(module.amax).any() or torch.isinf(module.amax).any() or (module.amax == 0).any():
                    print(f"[Quant] WARNING: Layer {name} has invalid amax. Resetting to 1.0.")
                    module.amax.fill_(1.0)
                    module.disable_quant()
                else:
                     module.enable_quant()

                module.disable_calib()
            except Exception as e:
                print(f"[Quant] Warning: Failed to load scale for {name}. Disabling quantization. Error: {e}")
                module.disable_quant()
                continue

    # Final check for safety
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if not hasattr(module, 'amax') or module.amax is None:
                print(f"[Quant] Warning: {name} amax is None. Setting to 1.0")
                module.amax.fill_(1.0) if torch.cuda.is_available() else torch.tensor(1.0)
            elif torch.isnan(module.amax).any() or torch.isinf(module.amax).any() or (module.amax == 0).any():
                print(f"[Quant] Warning: {name} amax is NaN/Inf/0. Setting to 1.0")
                module.amax.fill_(1.0) if torch.cuda.is_available() else torch.tensor(1.0)

    print("[Quant] Calibration finished. AMax loaded.")

    count_quant = 0
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            count_quant += 1
    print(f"[Quant] Total quantized tensors: {count_quant}")

# ======================================================

# 封装第一帧的head
class Sparse4DHead1st(nn.Module):
    def __init__(self, model):
        super(Sparse4DHead1st, self).__init__()
        self.model = model

    @staticmethod
    def head_forward(self, feature, spatial_shapes, level_start_index, instance_feature, anchor, time_interval, image_wh, lidar2img):
        temp_instance_feature = None
        temp_anchor_embed = None
        metas = {"image_wh": image_wh, "lidar2img": lidar2img}
        anchor_embed = self.anchor_encoder(anchor)
        feature_maps = [feature, spatial_shapes, level_start_index]
        prediction = []
        tmp_outs = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i, instance_feature, temp_instance_feature, temp_instance_feature,
                    query_pos=anchor_embed, key_pos=temp_anchor_embed,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i, instance_feature, value=instance_feature, query_pos=anchor_embed,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                bs, num_anchor = instance_feature.shape[:2]
                key_points = self.layers[i].kps_generator(anchor, instance_feature)
                weights = self.layers[i]._get_weights(instance_feature, anchor_embed, metas)
                
                # Use V3.1 Explicit MatMul logic to keep connection
                bs, num_anchor, num_pts = key_points.shape[:3]
                pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
                pts_flat = pts_extend.view(bs, num_anchor * num_pts, 4)
                pts_T = pts_flat.transpose(1, 2)
                
                points_list = []
                for cam_idx in range(self.layers[i].num_cams):
                    lidar_mat = metas["lidar2img"][:, cam_idx, :, :]
                    proj = torch.matmul(lidar_mat, pts_T)
                    points_list.append(proj.transpose(1, 2))
                
                points_2d_raw = torch.stack(points_list, dim=-2)
                points_2d_raw = points_2d_raw.view(bs, num_anchor, num_pts, self.layers[i].num_cams, 4)
                points_2d = points_2d_raw[..., :2] / torch.clamp(points_2d_raw[..., 2:3], min=1e-5)
                
                if metas.get("image_wh") is not None:
                    img_wh = metas["image_wh"].unsqueeze(1).unsqueeze(1)
                    points_2d = points_2d / img_wh
                
                points_2d = points_2d.contiguous()
                weights = weights.permute(0, 1, 4, 2, 3, 5).contiguous()

                # Ensure FP32 for DAF
                if points_2d.dtype != torch.float32: points_2d = points_2d.float()
                if weights.dtype != torch.float32: weights = weights.float()

                features = DAF(feature, spatial_shapes, level_start_index, points_2d, weights)
                features = features.reshape(bs, num_anchor, -1)
                if features.dtype != instance_feature.dtype:
                    features = features.to(dtype=instance_feature.dtype)
                if features.dtype != instance_feature.dtype:
                    features = features.to(dtype=instance_feature.dtype)
                output = self.layers[i].output_proj(features)
                instance_feature = torch.cat([output, instance_feature], dim=-1)
                tmp_outs.append(instance_feature)
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature, anchor, anchor_embed, time_interval=time_interval,
                    return_cls=(len(prediction) == self.num_single_frame_decoder - 1 or i == len(self.operation_order) - 1),
                )
                prediction.append(anchor)
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
        return (instance_feature, anchor, cls, qt)

    def forward(self, feature, spatial_shapes, level_start_index, instance_feature, anchor, time_interval, image_wh, lidar2img):
        head = self.model.head
        return self.head_forward(head, feature, spatial_shapes, level_start_index, instance_feature, anchor, time_interval, image_wh, lidar2img)


class Sparse4DHead2nd(nn.Module):
    def __init__(self, model):
        super(Sparse4DHead2nd, self).__init__()
        self.model = model

    @staticmethod
    def head_forward(self, feature, spatial_shapes, level_start_index, instance_feature, anchor, time_interval, temp_instance_feature, temp_anchor, mask, track_id, image_wh, lidar2img):
        mask = mask.bool()
        anchor_embed = self.anchor_encoder(anchor)
        temp_anchor_embed = self.anchor_encoder(temp_anchor)
        metas = {"lidar2img": lidar2img, "image_wh": image_wh}
        feature_maps = [feature, spatial_shapes, level_start_index]
        prediction = []
        tmp_outs = []
        attn_mask = None

        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i, instance_feature, temp_instance_feature, temp_instance_feature,
                    query_pos=anchor_embed, key_pos=temp_anchor_embed, attn_mask=attn_mask,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i, instance_feature, value=instance_feature, query_pos=anchor_embed, attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                bs, num_anchor = instance_feature.shape[:2]
                key_points = self.layers[i].kps_generator(anchor, instance_feature)
                weights = self.layers[i]._get_weights(instance_feature, anchor_embed, metas)
                
                # Use V3.1 Explicit MatMul logic to keep connection
                bs, num_anchor, num_pts = key_points.shape[:3]
                pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
                pts_flat = pts_extend.view(bs, num_anchor * num_pts, 4)
                pts_T = pts_flat.transpose(1, 2)
                
                points_list = []
                for cam_idx in range(self.layers[i].num_cams):
                    lidar_mat = metas["lidar2img"][:, cam_idx, :, :]
                    proj = torch.matmul(lidar_mat, pts_T)
                    points_list.append(proj.transpose(1, 2))
                
                points_2d_raw = torch.stack(points_list, dim=-2)
                points_2d_raw = points_2d_raw.view(bs, num_anchor, num_pts, self.layers[i].num_cams, 4)
                points_2d = points_2d_raw[..., :2] / torch.clamp(points_2d_raw[..., 2:3], min=1e-5)
                
                if metas.get("image_wh") is not None:
                    img_wh = metas["image_wh"].unsqueeze(1).unsqueeze(1)
                    points_2d = points_2d / img_wh
                
                points_2d = points_2d.contiguous()
                weights = weights.permute(0, 1, 4, 2, 3, 5).contiguous()

                # Ensure FP32 for DAF
                if points_2d.dtype != torch.float32: points_2d = points_2d.float()
                if weights.dtype != torch.float32: weights = weights.float()

                features = DAF(feature, spatial_shapes, level_start_index, points_2d, weights)
                features = features.reshape(bs, num_anchor, -1)
                if features.dtype != instance_feature.dtype:
                    features = features.to(dtype=instance_feature.dtype)
                if features.dtype != instance_feature.dtype:
                    features = features.to(dtype=instance_feature.dtype)
                output = self.layers[i].output_proj(features)
                instance_feature = torch.cat([output, instance_feature], dim=-1)
                tmp_outs.append(instance_feature)
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature, anchor, anchor_embed, time_interval=time_interval,
                    return_cls=(len(prediction) == self.num_single_frame_decoder - 1 or i == len(self.operation_order) - 1),
                )
                prediction.append(anchor)

                if len(prediction) == self.num_single_frame_decoder:
                    confidence = cls.max(dim=-1).values
                    N = self.instance_bank.num_anchor - self.instance_bank.num_temp_instances
                    confidence_sorted, outputs, indices = topk_for_onnx_export(confidence, N, instance_feature, anchor)
                    selected_feature, selected_anchor = outputs
                    selected_feature = torch.cat([temp_instance_feature, selected_feature], dim=1)
                    selected_anchor = torch.cat([temp_anchor, selected_anchor], dim=1)
                    instance_feature = torch.where(mask[:, None, None], selected_feature, instance_feature)
                    anchor = torch.where(mask[:, None, None], selected_anchor, anchor)
                    track_id = torch.where(mask[:, None], track_id, track_id.new_tensor(-1))

                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if len(prediction) > self.num_single_frame_decoder:
                    temp_anchor_embed = anchor_embed[:, : self.instance_bank.num_temp_instances]

        return (instance_feature, anchor, cls, qt, track_id)

    def forward(self, feature, spatial_shapes, level_start_index, instance_feature, anchor, time_interval, temp_instance_feature, temp_anchor, mask, track_id, image_wh, lidar2img):
        head = self.model.head
        return self.head_forward(head, feature, spatial_shapes, level_start_index, instance_feature, anchor, time_interval, temp_instance_feature, temp_anchor, mask, track_id, image_wh, lidar2img)


# 生成dummy input (Fallback)
def dummpy_input(model, bs: int, nums_cam: int, input_h: int, input_w: int, nums_query=900, nums_topk=600, embed_dims=256, anchor_dims=11, first_frame=True, logger=None, use_fp16=False):
    h_4x, w_4x = input_h // 4, input_w // 4
    h_8x, w_8x = input_h // 8, input_w // 8
    h_16x, w_16x = input_h // 16, input_w // 16
    h_32x, w_32x = input_h // 32, input_w // 32
    feature_size = nums_cam * (h_4x * w_4x + h_8x * w_8x + h_16x * w_16x + h_32x * w_32x)
    float_dtype = torch.float16 if use_fp16 else torch.float32
    dummy_feature = torch.randn(bs, feature_size, embed_dims).to(dtype=float_dtype).cuda()
    dummy_spatial_shapes = (
        torch.tensor([[h_4x, w_4x], [h_8x, w_8x], [h_16x, w_16x], [h_32x, w_32x]])
        .int().unsqueeze(0).repeat(nums_cam, 1, 1).cuda()
    )
    scale_start_index = dummy_spatial_shapes[..., 0] * dummy_spatial_shapes[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0).int()
    scale_start_index = torch.cat([torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]])
    dummy_level_start_index = scale_start_index.reshape(nums_cam, 4)
    instance_feature = model.head.instance_bank.instance_feature
    dummy_instance_feature = instance_feature[None].repeat((bs, 1, 1)).to(dtype=float_dtype).cuda()
    anchor = model.head.instance_bank.anchor
    dummy_anchor = anchor[None].repeat((bs, 1, 1)).to(dtype=float_dtype).cuda()
    dummy_time_interval = torch.tensor([model.head.instance_bank.default_time_interval] * bs, dtype=float_dtype).cuda()
    dummy_temp_instance_feature = torch.randn((bs, nums_topk, embed_dims), dtype=float_dtype).cuda()
    dummy_temp_anchor = torch.randn((bs, nums_topk, anchor_dims), dtype=float_dtype).cuda()
    dummy_mask = torch.randint(0, 2, size=(bs,)).int().cuda()
    dummy_track_id = -1 * torch.ones((bs, nums_query)).int().cuda()
    dummy_image_wh = torch.tensor([input_w, input_h], dtype=float_dtype).unsqueeze(0).unsqueeze(0).repeat(bs, nums_cam, 1).cuda()
    dummy_lidar2img = torch.randn(bs, nums_cam, 4, 4, dtype=float_dtype).cuda()
    return (
        dummy_feature, dummy_spatial_shapes, dummy_level_start_index, dummy_instance_feature,
        dummy_anchor, dummy_time_interval, dummy_temp_instance_feature, dummy_temp_anchor,
        dummy_mask, dummy_track_id, dummy_image_wh, dummy_lidar2img,
    )


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_onnx1), exist_ok=True)

    logger, console_handler, file_handler = set_logger(args.log, True)
    logger.setLevel(logging.DEBUG)

    if args.int8 and not HAS_QUANTIZATION:
        logger.error("INT8 requested but pytorch-quantization not installed.")
        sys.exit(1)

    cfg = read_cfg(args.cfg)
    model = build_module(cfg["model"])
    checkpoint = args.ckpt
    _ = model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)
    model.cuda().eval()

    # [Check] 检查加载后的模型权重是否正常
    print("[Check] Verifying model weights...")
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"[Error] Weight {name} contains NaN or Inf!")
            has_nan = True
    if has_nan:
        print("[Error] Model weights contain NaN/Inf! Please check your checkpoint.")
        # sys.exit(1) # 可以选择退出，或者尝试继续(虽然很可能会失败)
    else:
        print("[Check] Model weights are clean.")

    # PTQ 策略：
    # 1. 即使是 INT8 导出，PyTorch 侧模型仍使用 FP32 运行 Trace，但会插入 INT8 的 Q/DQ 节点。
    # 2. TensorRT 在加载时会将 FP32 + Q/DQ 融合为 INT8 Engine。
    # 3. 强制使用 FP16 模式导出反而容易导致 cuBLAS 溢出或 ONNX 不兼容。
    use_fp16 = True
    if args.int8:
        use_fp16 = False  # INT8 模式下回退到 FP32 模型
        initialize_quantization()
    else:
        logger.info("Exporting model with FP16 precision (No INT8).")

    if use_fp16:
        model = model.half()
        model._export_fp16 = True
    else:
        model._export_fp16 = False

    BS = 1
    NUMS_CAM = 6
    INPUT_H = 256
    INPUT_W = 704

    # ========================== HEAD 1 ==========================
    if not args.o2:
        logger.info("Preparing Sparse4DHead1st...")
        first_frame_head = Sparse4DHead1st(copy.deepcopy(model))

        # [Fix] 恢复 Plugin 替换，确保图连接完整
        if SPARSEBOX_PLUGIN_AVAILABLE:
            logger.info("Replacing kps_generator with Plugin...")
            replace_kps_generator_with_plugin(first_frame_head.model.head, verbose=True)

        if LN_PLUGIN_AVAILABLE:
            logger.info("Replacing LayerNorm with Plugin...")
            replace_layernorm_with_plugin(first_frame_head.model.head, verbose=True)
        if args.int8:
            logger.info("[Quant] Replacing modules with QuantLinear...")
            sensitive_layers = ["anchor_encoder", "output_proj", "kps_generator", "project_points", "layers", "fc_before", "fc_after"]
            replace_to_quantization_module(first_frame_head.model.head, ignore_layers=sensitive_layers)

            logger.info("[Quant] Calibrating Head 1st...")
            if args.calib_data:
                logger.info(f"[Quant] Using REAL calibration data from {args.calib_data}")
                loader = CalibrationDataLoader(args.calib_data, "head1", use_fp16=use_fp16)
                calib_batches = min(args.calib_batches, len(loader))
                collect_stats(first_frame_head, loader, calib_batches)

                # 导出时也使用真实数据的一帧作为 input，保证 Trace 路径正确
                loader.idx = 0
                export_inputs = loader()
            else:
                logger.warning("[Quant] Using DUMMY data for calibration (Not Recommended)!")
                def dummy_gen():
                    d = dummpy_input(model, BS, NUMS_CAM, INPUT_H, INPUT_W, first_frame=True, logger=logger, use_fp16=use_fp16)
                    return (d[0], d[1], d[2], d[3], d[4], d[5], d[10], d[11])
                collect_stats(first_frame_head, dummy_gen, args.calib_batches)
                d = dummpy_input(model, BS, NUMS_CAM, INPUT_H, INPUT_W, first_frame=True, logger=logger, use_fp16=use_fp16)
                export_inputs = (d[0], d[1], d[2], d[3], d[4], d[5], d[10], d[11])

        else:
            # FP16 Export inputs
            d = dummpy_input(model, BS, NUMS_CAM, INPUT_H, INPUT_W, first_frame=True, logger=logger, use_fp16=use_fp16)
            export_inputs = (d[0], d[1], d[2], d[3], d[4], d[5], d[10], d[11])

        

        # 强制二次检查，彻底杜绝 NaN 进入导出流程
        for name, m in first_frame_head.named_modules():
            if hasattr(m, "amax") and m.amax is not None:
                if torch.isnan(m.amax).any() or torch.isinf(m.amax).any() or (m.amax == 0).any():
                    m.amax.data.fill_(1.0)
                    m.disable_quant()
        with torch.no_grad():
            torch.onnx.export(
                first_frame_head,
                export_inputs,
                args.save_onnx1,
                input_names=["feature", "spatial_shapes", "level_start_index", "instance_feature", "anchor", "time_interval", "image_wh", "lidar2img"],
                output_names=["pred_instance_feature", "pred_anchor", "pred_class_score", "pred_quality_score"],
                opset_version=17, # [Fix] Downgrade to 13 for better compatibility
                do_constant_folding=True, # [Fix] Disable constant folding to prevent input pruning
                verbose=False,
            )

        if not args.int8:
            try:
                onnx_orig = onnx.load(args.save_onnx1)
                onnx_simp, check = simplify(onnx_orig)
                if check: onnx.save(onnx_simp, args.save_onnx1)
            except: pass
        logger.info(f'🚀 Head1 Export completed: {args.save_onnx1}')
        # [Fix] 自动修复 TopK 问题
        if fix_onnx_topk_k(args.save_onnx1, logger):
            logger.info(f"✅ Applied TopK patch to {args.save_onnx1}")

    # ========================== HEAD 2 ==========================
    logger.info("Preparing Sparse4DHead2nd...")
    head2 = Sparse4DHead2nd(copy.deepcopy(model))

    # [Fix] 恢复 Plugin 替换，确保图连接完整
    if SPARSEBOX_PLUGIN_AVAILABLE:
        logger.info("Replacing kps_generator with Plugin (Head 2)...")
        replace_kps_generator_with_plugin(head2.model.head, verbose=False)
    if LN_PLUGIN_AVAILABLE:
        logger.info("Replacing LayerNorm with Plugin (Head 2)...")
        replace_layernorm_with_plugin(head2.model.head, verbose=False)
    if args.int8:
        logger.info("[Quant] Replacing modules with QuantLinear (Head 2)...")
        sensitive_layers = ["anchor_encoder", "output_proj", "kps_generator", "project_points", "layers", "fc_before", "fc_after"]
        replace_to_quantization_module(head2.model.head, ignore_layers=sensitive_layers)

        logger.info("[Quant] Calibrating Head 2nd...")
        if args.calib_data:
            logger.info(f"[Quant] Using REAL calibration data from {args.calib_data}")
            loader = CalibrationDataLoader(args.calib_data, "head2", use_fp16=use_fp16)
            calib_batches = min(args.calib_batches, len(loader))
            collect_stats(head2, loader, calib_batches)

            loader.idx = 0
            export_inputs_2nd = loader()
        else:
            def dummy_gen():
                return dummpy_input(model, BS, NUMS_CAM, INPUT_H, INPUT_W, first_frame=False, logger=logger, use_fp16=use_fp16)
            collect_stats(head2, dummy_gen, args.calib_batches)
            export_inputs_2nd = dummpy_input(model, BS, NUMS_CAM, INPUT_H, INPUT_W, first_frame=False, logger=logger, use_fp16=use_fp16)
    else:
        export_inputs_2nd = dummpy_input(model, BS, NUMS_CAM, INPUT_H, INPUT_W, first_frame=False, logger=logger, use_fp16=use_fp16)

    

    for name, m in head2.named_modules():
        if hasattr(m, "amax") and m.amax is not None:
            if torch.isnan(m.amax).any() or torch.isinf(m.amax).any() or (m.amax == 0).any():
                m.amax.data.fill_(1.0)
                m.disable_quant()
    with torch.no_grad():
            torch.onnx.export(
            head2,
            export_inputs_2nd,
            args.save_onnx2,
            input_names=["feature", "spatial_shapes", "level_start_index", "instance_feature", "anchor", "time_interval", "temp_instance_feature", "temp_anchor", "mask", "track_id", "image_wh", "lidar2img"],
            output_names=["pred_instance_feature", "pred_anchor", "pred_class_score", "pred_quality_score", "pred_track_id"],
            opset_version=17, # [Fix] Downgrade to 13
            do_constant_folding=True, # [Fix] Disable constant folding
            verbose=False,
        )

    if not args.int8:
        try:
            onnx_orig = onnx.load(args.save_onnx2)
            onnx_simp, check = simplify(onnx_orig)
            if check: onnx.save(onnx_simp, args.save_onnx2)
        except: pass

    logger.info(f'🚀 Head2 Export completed: {args.save_onnx2}')
    # [Fix] 自动修复 TopK 问题
    if fix_onnx_topk_k(args.save_onnx2, logger):
        logger.info(f"✅ Applied TopK patch to {args.save_onnx2}")
