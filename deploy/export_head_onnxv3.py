# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import time
import copy
import logging
import argparse

# 设置完全确定性环境
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '100'  # 与训练保持一致

import onnx
from onnxsim import simplify

import torch
from torch import nn

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
    print(f"[WARNING] SparseBox3DKeyPointsPlugin not available: {e}")
    print("[WARNING] Will export ONNX without plugin replacement.")

# 导入 ln_plugin 替换函数
try:
    from deploy.ln_plugin.replace_layernorm import replace_layernorm_with_plugin
    LN_PLUGIN_AVAILABLE = True
except ImportError as e:
    LN_PLUGIN_AVAILABLE = False
    print(f"[WARNING] LayerNormPlugin not available: {e}")

# 设置PyTorch确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

# 设置随机种子 - 与训练保持一致
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)

# 设置numpy随机种子 - 与训练保持一致
import numpy as np
np.random.seed(100)

# 设置CUDA确定性
torch.cuda.empty_cache()

# =========== 原有代码继续 ===========

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy SparseEND2END Head!")
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
        default="deploy/onnx/export_head_onnx_v3.log",
    )
    parser.add_argument(
        "--save_onnx1",
        type=str,
        default="deploy/onnx/sparse4dhead1st_v3.onnx",
    )
    parser.add_argument(
        "--save_onnx2",
        type=str,
        default="deploy/onnx/sparse4dhead2nd_v3.onnx",
    )
    parser.add_argument(
        "--o2", action="store_true", help="only export sparse4dhead2nd onnx."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export ONNX model with FP16 data types. This ensures TensorRT can properly allocate workspace for FP16 inference. Default: False (FP32). Specify --fp16 to export FP16.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Export ONNX model with FP32 data types. This is the default behavior if --fp16 is not specified.",
    )
    parser.add_argument(
        "--save_val_data",
        type=str,
        default=None,
        help="Directory to save validation data (inputs and outputs) for verification.",
    )
    args = parser.parse_args()
    return args


# 封装第一帧的head
class Sparse4DHead1st(nn.Module):
    def __init__(self, model):
        super(Sparse4DHead1st, self).__init__()
        self.model = model

    @staticmethod
    def head_forward(
        self,
        feature,
        spatial_shapes,
        level_start_index,
        instance_feature,
        anchor,
        time_interval,
        image_wh,
        lidar2img,
    ):

        # Instance bank get inputs
        temp_instance_feature = None
        temp_anchor_embed = None

        # DFA inputs
        metas = {
            "image_wh": image_wh,
            "lidar2img": lidar2img,
        }

        anchor_embed = self.anchor_encoder(anchor)

        feature_maps = [feature, spatial_shapes, level_start_index]
        prediction = []
        tmp_outs = []
        for i, op in enumerate(self.operation_order):
            print("i: ", i, "\top: ", op)
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                # instance_feature = self.layers[i](
                #     instance_feature,
                #     anchor,
                #     anchor_embed,
                #     feature_maps,
                #     metas,
                # )
                bs, num_anchor = instance_feature.shape[:2]
                key_points = self.layers[i].kps_generator(anchor, instance_feature)
                weights = self.layers[i]._get_weights(
                    instance_feature, anchor_embed, metas
                )
                
                # V3.1 Optimized logic: Explicit Loop MatMul to avoid TensorRT broadcasting issues on Orin
                # Original V3 broadcasting caused huge performance regression (87ms per layer -> 572ms total)
                
                bs, num_anchor, num_pts = key_points.shape[:3]
                pts_extend = torch.cat(
                    [key_points, torch.ones_like(key_points[..., :1])], dim=-1
                ) # [B, N, P, 4]
                
                pts_flat = pts_extend.view(bs, num_anchor * num_pts, 4) # [B, N*P, 4]
                pts_T = pts_flat.transpose(1, 2) # [B, 4, N*P]
                
                points_list = []
                for cam_idx in range(self.layers[i].num_cams):
                    # [B, 4, 4]
                    lidar_mat = metas["lidar2img"][:, cam_idx, :, :]
                    # [B, 4, 4] @ [B, 4, N*P] -> [B, 4, N*P]
                    proj = torch.matmul(lidar_mat, pts_T)
                    # [B, N*P, 4]
                    points_list.append(proj.transpose(1, 2))
                
                # [B, N*P, C, 4]
                points_2d_raw = torch.stack(points_list, dim=-2)
                # [B, N, P, C, 4]
                points_2d_raw = points_2d_raw.view(bs, num_anchor, num_pts, self.layers[i].num_cams, 4)
                
                # Normalize
                points_2d = points_2d_raw[..., :2] / torch.clamp(points_2d_raw[..., 2:3], min=1e-5)
                
                if metas.get("image_wh") is not None:
                    img_wh = metas["image_wh"].unsqueeze(1).unsqueeze(1)
                    points_2d = points_2d / img_wh
                
                points_2d = points_2d.contiguous()
                
                weights = weights.permute(0, 1, 4, 2, 3, 5).contiguous()

                # CRITICAL FIX for TensorRT Reformatting:
                if points_2d.dtype != torch.float32:
                    points_2d = points_2d.float()
                
                if weights.dtype != torch.float32:
                    weights = weights.float()

                features = DAF(*feature_maps, points_2d, weights)
                features = features.reshape(bs, num_anchor, self.layers[i].embed_dims)
                # DAF函数返回FP32，需要转换为与模型相同的精度
                if features.dtype != instance_feature.dtype:
                    features = features.to(dtype=instance_feature.dtype)
                output = self.layers[i].output_proj(features)
                assert self.layers[i].residual_mode == "cat"
                instance_feature = torch.cat([output, instance_feature], dim=-1)
                tmp_outs.append(instance_feature)
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
        return (
            instance_feature,
            anchor,
            cls,
            qt,
        )

    def forward(
        self,
        feature,
        spatial_shapes,
        level_start_index,
        instance_feature,
        anchor,
        time_interval,
        image_wh,
        lidar2img,
    ):
        head = self.model.head
        return self.head_forward(
            head,
            feature,
            spatial_shapes,
            level_start_index,
            instance_feature,
            anchor,
            time_interval,
            image_wh,
            lidar2img,
        )


# 封装第二帧的head
class Sparse4DHead2nd(nn.Module):
    def __init__(self, model):
        super(Sparse4DHead2nd, self).__init__()
        self.model = model

    @staticmethod
    def head_forward(
        self,
        feature,
        spatial_shapes,
        level_start_index,
        instance_feature,
        anchor,
        time_interval,
        temp_instance_feature,
        temp_anchor,
        mask,
        track_id,
        image_wh,
        lidar2img,
    ):
        mask = mask.bool()  # TensorRT binding type for bool input is NoneType.
        anchor_embed = self.anchor_encoder(anchor)
        temp_anchor_embed = self.anchor_encoder(temp_anchor)

        # DFA inputs
        metas = {
            "lidar2img": lidar2img,
            "image_wh": image_wh,
        }

        feature_maps = [feature, spatial_shapes, level_start_index]
        prediction = []
        tmp_outs = []

        attn_mask = None  # 第二帧不使用attn_mask
        for i, op in enumerate(self.operation_order):
            print("op:  ", op)
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                # instance_feature = self.layers[i](
                #     instance_feature,
                #     anchor,
                #     anchor_embed,
                #     feature_maps,
                #     metas,
                # )
                bs, num_anchor = instance_feature.shape[:2]
                key_points = self.layers[i].kps_generator(anchor, instance_feature)
                weights = self.layers[i]._get_weights(
                    instance_feature, anchor_embed, metas
                )
                
                # V3.1 Optimized logic for Sparse4DHead2nd (Explicit Loop MatMul)
                bs, num_anchor, num_pts = key_points.shape[:3]
                pts_extend = torch.cat(
                    [key_points, torch.ones_like(key_points[..., :1])], dim=-1
                ) # [B, N, P, 4]
                
                pts_flat = pts_extend.view(bs, num_anchor * num_pts, 4) # [B, N*P, 4]
                pts_T = pts_flat.transpose(1, 2) # [B, 4, N*P]
                
                points_list = []
                for cam_idx in range(self.layers[i].num_cams):
                    # [B, 4, 4]
                    lidar_mat = metas["lidar2img"][:, cam_idx, :, :]
                    # [B, 4, 4] @ [B, 4, N*P] -> [B, 4, N*P]
                    proj = torch.matmul(lidar_mat, pts_T)
                    # [B, N*P, 4]
                    points_list.append(proj.transpose(1, 2))
                
                # [B, N*P, C, 4]
                points_2d_raw = torch.stack(points_list, dim=-2)
                # [B, N, P, C, 4]
                points_2d_raw = points_2d_raw.view(bs, num_anchor, num_pts, self.layers[i].num_cams, 4)
                
                points_2d = points_2d_raw[..., :2] / torch.clamp(points_2d_raw[..., 2:3], min=1e-5)
                
                if metas.get("image_wh") is not None:
                    img_wh = metas["image_wh"].unsqueeze(1).unsqueeze(1)
                    points_2d = points_2d / img_wh
                
                points_2d = points_2d.contiguous()
                
                weights = weights.permute(0, 1, 4, 2, 3, 5).contiguous()

                # CRITICAL FIX for TensorRT Reformatting:
                if points_2d.dtype != torch.float32:
                    points_2d = points_2d.float()
                
                if weights.dtype != torch.float32:
                    weights = weights.float()

                features = DAF(*feature_maps, points_2d, weights)
                features = features.reshape(bs, num_anchor, self.layers[i].embed_dims)
                # DAF函数返回FP32，需要转换为与模型相同的精度
                if features.dtype != instance_feature.dtype:
                    features = features.to(dtype=instance_feature.dtype)
                output = self.layers[i].output_proj(features)
                assert self.layers[i].residual_mode == "cat"
                instance_feature = torch.cat([output, instance_feature], dim=-1)
                tmp_outs.append(instance_feature)
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)

                # 初始化update_comparison为None
                update_comparison = None

                # 修复：在第二帧的第一个refine后执行InstanceBank更新逻辑
                if len(prediction) == self.num_single_frame_decoder:
                    # 保存update前的状态
                    instance_feature_before_update = instance_feature.clone()
                    anchor_before_update = anchor.clone()
                    
                    # TopK选择新实例 - 使用优化的topk实现
                    N = self.instance_bank.num_anchor - self.instance_bank.num_temp_instances
                    
                    # 使用优化的topk函数
                    confidence = cls.max(dim=-1).values
                    
                    # 调用ONNX兼容的topk函数
                    confidence_sorted, outputs, indices = topk_for_onnx_export(
                        confidence, N, instance_feature, anchor
                    )
                    selected_feature, selected_anchor = outputs
                    
                    # 融合历史实例和新实例
                    selected_feature = torch.cat([temp_instance_feature, selected_feature], dim=1)
                    selected_anchor = torch.cat([temp_anchor, selected_anchor], dim=1)
                    
                    # 根据mask条件更新
                    instance_feature = torch.where(
                        mask[:, None, None], selected_feature, instance_feature
                    )
                    anchor = torch.where(mask[:, None, None], selected_anchor, anchor)
                    track_id = torch.where(mask[:, None], track_id, track_id.new_tensor(-1))
                    
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if len(prediction) > self.num_single_frame_decoder:
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]

        return (
            instance_feature,
            anchor,
            cls,
            qt,
            track_id,
        )

    def forward(
        self,
        feature,
        spatial_shapes,
        level_start_index,
        instance_feature,
        anchor,
        time_interval,
        temp_instance_feature,
        temp_anchor,
        mask,
        track_id,
        image_wh,
        lidar2img,
    ):
        head = self.model.head
        (
            instance_feature,
            anchor,
            cls,
            qt,
            track_id,
        ) = self.head_forward(
            head,
            feature,
            spatial_shapes,
            level_start_index,
            instance_feature,
            anchor,
            time_interval,
            temp_instance_feature,
            temp_anchor,
            mask,
            track_id,
            image_wh,
            lidar2img,
        )
        return (
            instance_feature,
            anchor,
            cls,
            qt,
            track_id,
        )


# 生成dummy input
def dummpy_input(
    model,
    bs: int,               # batch size
    nums_cam: int,         # 相机数量
    input_h: int,          # 输入高度
    input_w: int,          # 输入宽度
    nums_query=900,        # 查询数量
    nums_topk=600,         # 候选数量
    embed_dims=256,        # 特征维度
    anchor_dims=11,        # 锚点维度
    first_frame=True,      # 是否为第一帧
    logger=None,           # 日志记录器
    use_fp16=False,        # 是否使用FP16精度
):
    h_4x, w_4x = input_h // 4, input_w // 4
    h_8x, w_8x = input_h // 8, input_w // 8
    h_16x, w_16x = input_h // 16, input_w // 16
    h_32x, w_32x = input_h // 32, input_w // 32
    feature_size = nums_cam * (          # 特征维度
        h_4x * w_4x + h_8x * w_8x + h_16x * w_16x + h_32x * w_32x
    )
    # 根据精度选择数据类型
    float_dtype = torch.float16 if use_fp16 else torch.float32
    dummy_feature = torch.randn(bs, feature_size, embed_dims).to(dtype=float_dtype).cuda()  # 生成随机特征

    # 生成空间形状[6, 4, 2]
    # [64, 176],    # 4倍下采样
    # [32, 88],     # 8倍下采样
    # [16, 44],     # 16倍下采样
    # [8, 22]       # 32倍下采样
    dummy_spatial_shapes = (
        torch.tensor([[h_4x, w_4x], [h_8x, w_8x], [h_16x, w_16x], [h_32x, w_32x]])
        .int()
        .unsqueeze(0)
        .repeat(nums_cam, 1, 1)
        .cuda()
    )

    # 计算每个尺度的面积 [6, 4] # = [11264, 2816, 704, 176]
    scale_start_index = dummy_spatial_shapes[..., 0] * dummy_spatial_shapes[..., 1]
    # 计算累积和 [6, 4] = [11264, 14080, 14784, 14960]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0).int()
    # 计算每个尺度的起始索引 [6, 4] = [0, 11264, 14080, 14784]
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    dummy_level_start_index = scale_start_index.reshape(nums_cam, 4)

    # 生成实例特征
    instance_feature = model.head.instance_bank.instance_feature  # (900, 256)
    dummy_instance_feature = (
        instance_feature[None].repeat((bs, 1, 1)).to(dtype=float_dtype).cuda()
    )  # (bs, 900, 256)

    # 生成锚点
    anchor = model.head.instance_bank.anchor  # (900, 11)
    dummy_anchor = anchor[None].repeat((bs, 1, 1)).to(dtype=float_dtype).cuda()  # (bs, 900, 11)

    # 生成时间间隔
    dummy_time_interval = torch.tensor(
        [model.head.instance_bank.default_time_interval] * bs, dtype=float_dtype
    ).cuda()

    # 生成临时实例特征 [bs, nums_topk, embed_dims]
    dummy_temp_instance_feature = (
        torch.zeros((bs, nums_topk, embed_dims), dtype=float_dtype).cuda())
    # 生成临时锚点 [bs, nums_topk, anchor_dims]
    dummy_temp_anchor = torch.zeros((bs, nums_topk, anchor_dims), dtype=float_dtype).cuda()
    # 生成掩码 [bs]
    dummy_mask = torch.randint(0, 2, size=(bs,)).int().cuda()
    # 生成跟踪ID [bs, nums_query]
    dummy_track_id = -1 * torch.ones((bs, nums_query)).int().cuda()

    # 生成图像宽高 [bs, nums_cam, 2]
    dummy_image_wh = (
        torch.tensor([input_w, input_h], dtype=float_dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(bs, nums_cam, 1)
        .cuda()
    )

    # 生成lidar2img [bs, nums_cam, 4, 4]
    dummy_lidar2img = torch.randn(bs, nums_cam, 4, 4, dtype=float_dtype).cuda()

    logger.debug(f"Dummy input : hape&Type&Device Msg >>>>>>")
    roi_x = [
        "dummy_feature",                # 虚拟特征 （bs, feature_size, embed_dims）
        "dummy_spatial_shapes",         # 空间形状 [nums_cam, feature_scales, feature_size]
        "dummy_level_start_index",      # 每个尺度的起始索引 [nums_cam, 4]
        "dummy_instance_feature",       # 实例特征 （bs, nums_query, embed_dims）
        "dummy_anchor",                 # 锚点 （bs, nums_query, anchor_dims）
        "dummy_time_interval",          # 时间间隔 （bs,）
        "dummy_image_wh",               # 图像宽高 （bs, 2）
        "dummy_lidar2img",              # 雷达到图像的变换矩阵 （bs, nums_cam, 4, 4）
    ]
    for x in roi_x:
        logger.debug(
            f"{x}\t:\tshape={eval(x).shape},\tdtype={eval(x).dtype},\tdevice={eval(x).device}"
        )

    if first_frame:
        logger.debug(f"Frame > 1: Extra dummy input is needed >>>>>>>")
        roi_y = [
            "dummy_temp_instance_feature",
            "dummy_temp_anchor",
            "dummy_mask",
            "dummy_track_id",
        ]
        for y in roi_y:
            logger.debug(
                f"{y}\t:\tshape={eval(y).shape},\tdtype={eval(y).dtype},\tdevice={eval(y).device}"
            )

    return (
        dummy_feature,                   # 虚拟特征 （bs, feature_size, embed_dims）
        dummy_spatial_shapes,            # 空间形状 [nums_cam, feature_scales, feature_size]
        dummy_level_start_index,         # 每个尺度的起始索引 [nums_cam, 4]
        dummy_instance_feature,          # 实例特征 （bs, nums_query, embed_dims）
        dummy_anchor,                    # 锚点 （bs, nums_query, anchor_dims）
        dummy_time_interval,             # 时间间隔 （bs,）
        dummy_temp_instance_feature,     # 临时实例特征 [bs, nums_topk, embed_dims]
        dummy_temp_anchor,               # 临时锚点 [bs, nums_topk, anchor_dims]
        dummy_mask,                      # 掩码 [bs]
        dummy_track_id,                  # 跟踪ID [bs, nums_query]
        dummy_image_wh,                  # 图像宽高 [bs, 2]
        dummy_lidar2img,                 # 雷达到图像的变换矩阵 [bs, nums_cam, 4, 4]
    )


# 构建模块
def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


# 验证 ONNX 模型的精度
def verify_onnx_precision(onnx_path: str, use_fp16: bool, logger):
    """验证 ONNX 模型的精度类型（只验证浮点类型的输入/输出）"""
    logger.info("Verifying ONNX model precision...")
    onnx_model = onnx.load(onnx_path)
    
    # ONNX TensorProto.DataType: FLOAT=1, FLOAT16=10, INT32=6, INT64=7
    # 只检查浮点类型的输入和输出
    float_input_types = []
    float_output_types = []
    
    for input_tensor in onnx_model.graph.input:
        input_type = input_tensor.type.tensor_type.elem_type
        # 只检查浮点类型（FLOAT=1, FLOAT16=10）
        if input_type in [1, 10]:
            float_input_types.append(input_type)
    
    for output_tensor in onnx_model.graph.output:
        output_type = output_tensor.type.tensor_type.elem_type
        # 只检查浮点类型（FLOAT=1, FLOAT16=10）
        if output_type in [1, 10]:
            float_output_types.append(output_type)
    
    expected_type = 10 if use_fp16 else 1  # FLOAT16=10, FLOAT32=1
    
    all_inputs_correct = all(t == expected_type for t in float_input_types) if float_input_types else True
    all_outputs_correct = all(t == expected_type for t in float_output_types) if float_output_types else True
    
    if all_inputs_correct and all_outputs_correct:
        precision_str = "FP16" if use_fp16 else "FP32"
        logger.info(f"✓ ONNX model verified as {precision_str} (all float inputs and outputs are {precision_str})")
    else:
        precision_str = "FP16" if use_fp16 else "FP32"
        logger.warning(
            f"⚠ ONNX model precision mismatch! "
            f"Float input types: {float_input_types}, Float output types: {float_output_types} "
            f"(expected: {expected_type} for {precision_str})"
        )


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_onnx1), exist_ok=True)

    # 设置日志
    logger, console_handler, file_handler = set_logger(args.log, True)  #  # 创建logger, 控制台处理器, 文件处理器
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    cfg = read_cfg(args.cfg)
    model = build_module(cfg["model"])
    checkpoint = args.ckpt
    _ = model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)
    model.cuda().eval()
    
    # 如果指定了--fp16，将模型转换为FP16（如果同时指定--fp16和--fp32，--fp16优先）
    if args.fp16 and not args.fp32:
        logger.info("Converting model to FP16 for ONNX export...")
        model = model.half()  # 将模型转换为FP16
        model._export_fp16 = True  # 标记模型为FP16导出模式
        logger.info("Model converted to FP16. All floating-point inputs will use FP16 dtype.")
    else:
        logger.info("Exporting model with FP32 precision.")
        model._export_fp16 = False

    BS = 1
    NUMS_CAM = 6
    INPUT_H = 256
    INPUT_W = 704
    first_frame = True
    
    # 根据导出精度选择数据类型
    use_fp16 = getattr(model, '_export_fp16', False)
    (
        dummy_feature,
        dummy_spatial_shapes,
        dummy_level_start_index,
        dummy_instance_feature,
        dummy_anchor,
        dummy_time_interval,
        dummy_temp_instance_feature,
        dummy_temp_anchor,
        dummy_mask,
        dummy_track_id,
        dummy_image_wh,
        dummy_lidar2img,
    ) = dummpy_input(
        model, BS, NUMS_CAM, INPUT_H, INPUT_W, first_frame=first_frame, logger=logger, use_fp16=use_fp16
    )

    if not args.o2:
        first_frame_head = Sparse4DHead1st(copy.deepcopy(model))
        
        # 1. 替换 kps_generator 为 Plugin 版本
        if SPARSEBOX_PLUGIN_AVAILABLE:
            logger.info("Replacing kps_generator with SparseBox3DKeyPointsPlugin...")
            replaced_count = replace_kps_generator_with_plugin(
                first_frame_head.model.head, 
                verbose=True
            )
            logger.info(f"Replaced {replaced_count} kps_generator(s) with Plugin version")
        else:
            logger.warning("SparseBox3DKeyPointsPlugin not available, using original kps_generator")
            
        # 2. 替换 LayerNorm 为 Plugin 版本
        if LN_PLUGIN_AVAILABLE:
            logger.info("Replacing LayerNorm with LayerNormPlugin...")
            replaced_ln_count = replace_layernorm_with_plugin(
                first_frame_head.model.head, 
                verbose=True
            )
            logger.info(f"Replaced {replaced_ln_count} LayerNorm(s) with Plugin version")
        else:
            logger.warning("LayerNormPlugin not available, using original LayerNorm")
        
        # Save validation data if requested
        if args.save_val_data and not args.o2:
            val_dir = os.path.join(args.save_val_data, "head1")
            os.makedirs(val_dir, exist_ok=True)
            logger.info(f"Saving validation data to {val_dir}...")
            
            # 1. Save Inputs
            input_map = {
                "feature": dummy_feature,
                "spatial_shapes": dummy_spatial_shapes,
                "level_start_index": dummy_level_start_index,
                "instance_feature": dummy_instance_feature,
                "anchor": dummy_anchor,
                "time_interval": dummy_time_interval,
                "image_wh": dummy_image_wh,
                "lidar2img": dummy_lidar2img,
            }
            
            for name, tensor in input_map.items():
                data = tensor.detach().cpu().numpy()
                data.tofile(os.path.join(val_dir, f"input_{name}.bin"))
                np.array(data.shape, dtype=np.int32).tofile(os.path.join(val_dir, f"input_{name}.shape"))
            
            # 2. Run Inference & Save Outputs
            with torch.no_grad():
                outputs = first_frame_head(
                    dummy_feature,
                    dummy_spatial_shapes,
                    dummy_level_start_index,
                    dummy_instance_feature,
                    dummy_anchor,
                    dummy_time_interval,
                    dummy_image_wh,
                    dummy_lidar2img,
                )
                
            output_names = [
                "pred_instance_feature",
                "pred_anchor",
                "pred_class_score",
                "pred_quality_score",
            ]
            
            for name, tensor in zip(output_names, outputs):
                data = tensor.detach().cpu().numpy()
                data.tofile(os.path.join(val_dir, f"output_{name}.bin"))
                np.array(data.shape, dtype=np.int32).tofile(os.path.join(val_dir, f"output_{name}.shape"))
            
            logger.info("Validation data saved.")

        logger.info("Export Sparse4DHead1st Onnx >>>>>>>>>>>>>>>>")
        time.sleep(2)
        with torch.no_grad():
            torch.onnx.export(
                first_frame_head,
                (
                    dummy_feature,                  #  多视角多尺度特征（bs, feature_size, embed_dims）
                    dummy_spatial_shapes,           # 空间形状 [nums_cam, feature_scales, feature_size]
                    dummy_level_start_index,        # 每个尺度的起始索引 [nums_cam, 4]
                    dummy_instance_feature,         # 实例特征 （bs, nums_query, embed_dims）
                    dummy_anchor,                   # 锚点 （bs, nums_query, anchor_dims）
                    dummy_time_interval,            # 时间间隔 （bs,）
                    dummy_image_wh,                 # 图像宽高 （bs, 2）
                    dummy_lidar2img,                # 雷达到图像的变换矩阵 （bs, nums_cam, 4, 4）
                ),
                args.save_onnx1,
                input_names=[
                    "feature",
                    "spatial_shapes",
                    "level_start_index",
                    "instance_feature",
                    "anchor",
                    "time_interval",
                    "image_wh",
                    "lidar2img",
                ],
                output_names=[
                    "pred_instance_feature",
                    "pred_anchor",
                    "pred_class_score",
                    "pred_quality_score",
                ],
                opset_version=17,
                do_constant_folding=True,
                verbose=False,
            )

            # 
            onnx_orig = onnx.load(args.save_onnx1)
            # FP16模式下，ONNX简化可能会失败（某些操作符如Range不支持FP16）
            # 如果简化失败，直接使用原始模型
            try:
                onnx_simp, check = simplify(onnx_orig)
                if check:
                    onnx.save(onnx_simp, args.save_onnx1)
                    logger.info("ONNX model simplified successfully.")
                else:
                    logger.warning("ONNX simplification failed validation, using original model.")
                    onnx.save(onnx_orig, args.save_onnx1)
            except Exception as e:
                logger.warning(f"ONNX simplification failed: {e}, using original model.")
                onnx.save(onnx_orig, args.save_onnx1)
            
            # 验证 ONNX 模型的精度
            verify_onnx_precision(args.save_onnx1, use_fp16, logger)
            
            logger.info(
                f'🚀 Export onnx completed. ONNX saved in "{args.save_onnx1}" 🤗.'
            )

    head = Sparse4DHead2nd(copy.deepcopy(model))
    
    # 1. 替换 kps_generator 为 Plugin 版本
    if SPARSEBOX_PLUGIN_AVAILABLE:
        logger.info("Replacing kps_generator with SparseBox3DKeyPointsPlugin...")
        replaced_count = replace_kps_generator_with_plugin(
            head.model.head, 
            verbose=True
        )
        logger.info(f"Replaced {replaced_count} kps_generator(s) with Plugin version")
    else:
        logger.warning("SparseBox3DKeyPointsPlugin not available, using original kps_generator")
        
    # 2. 替换 LayerNorm 为 Plugin 版本
    if LN_PLUGIN_AVAILABLE:
        logger.info("Replacing LayerNorm with LayerNormPlugin...")
        replaced_ln_count = replace_layernorm_with_plugin(
            head.model.head, 
            verbose=True
        )
        logger.info(f"Replaced {replaced_ln_count} LayerNorm(s) with Plugin version")
    else:
        logger.warning("LayerNormPlugin not available, using original LayerNorm")
    
    # Save validation data for 2nd head
    if args.save_val_data:
        val_dir = os.path.join(args.save_val_data, "head2")
        os.makedirs(val_dir, exist_ok=True)
        logger.info(f"Saving validation data (2nd head) to {val_dir}...")
        
        # 1. Save Inputs
        input_map = {
            "feature": dummy_feature,
            "spatial_shapes": dummy_spatial_shapes,
            "level_start_index": dummy_level_start_index,
            "instance_feature": dummy_instance_feature,
            "anchor": dummy_anchor,
            "time_interval": dummy_time_interval,
            "temp_instance_feature": dummy_temp_instance_feature,
            "temp_anchor": dummy_temp_anchor,
            "mask": dummy_mask,
            "track_id": dummy_track_id,
            "image_wh": dummy_image_wh,
            "lidar2img": dummy_lidar2img,
        }
        
        for name, tensor in input_map.items():
            data = tensor.detach().cpu().numpy()
            data.tofile(os.path.join(val_dir, f"input_{name}.bin"))
            np.array(data.shape, dtype=np.int32).tofile(os.path.join(val_dir, f"input_{name}.shape"))
        
        # 2. Run Inference & Save Outputs
        with torch.no_grad():
            outputs = head(
                dummy_feature,
                dummy_spatial_shapes,
                dummy_level_start_index,
                dummy_instance_feature,
                dummy_anchor,
                dummy_time_interval,
                dummy_temp_instance_feature,
                dummy_temp_anchor,
                dummy_mask,
                dummy_track_id,
                dummy_image_wh,
                dummy_lidar2img,
            )
            
        output_names = [
            "pred_instance_feature",
            "pred_anchor",
            "pred_class_score",
            "pred_quality_score",
            "pred_track_id",
        ]
        
        for name, tensor in zip(output_names, outputs):
            data = tensor.detach().cpu().numpy()
            data.tofile(os.path.join(val_dir, f"output_{name}.bin"))
            np.array(data.shape, dtype=np.int32).tofile(os.path.join(val_dir, f"output_{name}.shape"))
        
        logger.info("Validation data (2nd head) saved.")

    logger.info("Export Sparse4DHead2nd Onnx >>>>>>>>>>>>>>>>")
    time.sleep(2)
    with torch.no_grad():
        torch.onnx.export(
            head,
            (
                dummy_feature,
                dummy_spatial_shapes,
                dummy_level_start_index,
                dummy_instance_feature,
                dummy_anchor,
                dummy_time_interval,
                dummy_temp_instance_feature,
                dummy_temp_anchor,
                dummy_mask,
                dummy_track_id,
                dummy_image_wh,
                dummy_lidar2img,
            ),
            args.save_onnx2,
            input_names=[
                "feature",
                "spatial_shapes",
                "level_start_index",
                "instance_feature",
                "anchor",
                "time_interval",
                "temp_instance_feature",
                "temp_anchor",
                "mask",
                "track_id",
                "image_wh",
                "lidar2img",
            ],
            output_names=[
                "pred_instance_feature",
                "pred_anchor",
                "pred_class_score",
                "pred_quality_score",
                "pred_track_id",
            ],
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
        )

        onnx_orig = onnx.load(args.save_onnx2)
        # FP16模式下，ONNX简化可能会失败（某些操作符如Range不支持FP16）
        # 如果简化失败，直接使用原始模型
        try:
            onnx_simp, check = simplify(onnx_orig)
            if check:
                onnx.save(onnx_simp, args.save_onnx2)
                logger.info("ONNX model simplified successfully.")
            else:
                logger.warning("ONNX simplification failed validation, using original model.")
                onnx.save(onnx_orig, args.save_onnx2)
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}, using original model.")
            onnx.save(onnx_orig, args.save_onnx2)
        
        # 验证 ONNX 模型的精度
        verify_onnx_precision(args.save_onnx2, use_fp16, logger)
        
        logger.info(f'🚀 Export onnx completed. ONNX saved in "{args.save_onnx2}" 🤗.')
