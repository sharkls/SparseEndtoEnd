# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import argparse
import torch
import torch.nn as nn
import logging
import numpy as np

# 设置完全确定性环境
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '100'

from inspect import signature
from typing import Union, Optional, Any, Dict, List

from tool.utils.config import read_cfg
from tool.utils.logger import set_logger

# from tool.runner.fp16_utils import auto_fp16
from tool.runner.fp16_utils import wrap_fp16_model
from tool.runner.checkpoint import load_checkpoint
from tool.trainer.utils import set_random_seed
from tool.utils.save_bin import save_bins

from modules.sparse4d_detector import Sparse4D
from dataset.dataloader_wrapper import dataloader_wrapper
from dataset import NuScenes4DDetTrackDataset
from dataset.utils.scatter_gather import scatter

# 设置PyTorch确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

# 设置随机种子
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)
np.random.seed(100)

# 设置CUDA确定性
torch.cuda.empty_cache()


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def parse_args():
    parser = argparse.ArgumentParser(description="Export each module bin file!")
    parser.add_argument(
        "--config",
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
        help="inference config file path",
    )
    parser.add_argument(
        "--checkpoint", default="ckpt/sparse4dv3_r50.pth", help="checkpoint file"
    )
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="script/tutorial/save_bin.log",
    )
    args = parser.parse_args()
    return args


class Sparse4D_backbone(nn.Module):
    def __init__(self, model):
        super(Sparse4D_backbone, self).__init__()
        self._model = model

    def io_hook(self):
        """IO tensor is converted to numpy type."""
        img = self._img.detach().cpu().numpy()
        feature = self._feature.detach().cpu().numpy()
        return [img], [feature]

    def feature_maps_format(self, feature_maps):

        bs, num_cams = feature_maps[0].shape[:2]
        spatial_shape = []

        col_feats = []  # (bs, n, c, -1)
        for i, feat in enumerate(feature_maps):
            spatial_shape.append(feat.shape[-2:])
            col_feats.append(torch.reshape(feat, (bs, num_cams, feat.shape[2], -1)))

        # (bs, n, c', c) => (bs, n*c', c)
        col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
        spatial_shape = [spatial_shape] * num_cams
        spatial_shape = torch.tensor(
            spatial_shape,
            dtype=torch.int64,
            device=col_feats.device,
        )

        scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
        scale_start_index = scale_start_index.flatten().cumsum(dim=0)
        scale_start_index = torch.cat(
            [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
        )
        scale_start_index = scale_start_index.reshape(num_cams, -1)

        feature_maps = [
            col_feats,
            spatial_shape,
            scale_start_index,
        ]
        return feature_maps

    # @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]       # 1* 6 * 3 * 256 * 704
        self._img = img.clone()
        if img.dim() == 5:
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self._model.use_grid_mask:
            img = self._model.grid_mask(img)
        if "metas" in signature(self._model.img_backbone.forward).parameters:
            feature_maps = self._model.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self._model.img_backbone(img)
        if self._model.img_neck is not None:
            feature_maps = list(self._model.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(feat, (bs, num_cams) + feat.shape[1:])
        if return_depth and self._model.depth_branch is not None:
            depths = self._model.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self._model.use_deformable_func:
            feature_maps = self.feature_maps_format(feature_maps)

        self._feature = (feature_maps[0]).clone()

        if return_depth:
            return feature_maps, depths
        # # 打印特征图的数值范围        数值范围: [-1741.969482, 1685.179565]
        # feature_tensor = feature_maps[0]
        # if isinstance(feature_tensor, torch.Tensor):
        #     print(f"  数值范围: [{torch.min(feature_tensor).item():.6f}, {torch.max(feature_tensor).item():.6f}]")
        # else:
        #     print(f"  数值范围: [{np.min(feature_tensor):.6f}, {np.max(feature_tensor):.6f}]")
        return feature_maps

    def forward(self, img):
        return self.extract_feat(img)


class Sparse4D_head(nn.Module):
    def __init__(self, model):
        super(Sparse4D_head, self).__init__()
        self._head = model
        self._first_frame = True
        # 添加缓存来保存第一帧的推理结果
        self._cached_pred_instance_feature = None
        self._cached_pred_anchor = None
        # 添加temp_gnn输出缓存
        self._temp_gnn_output = None
        # 添加tmp_outs缓存
        self._tmp_outs = []
        # 添加refine_outs缓存
        self._refine_outs = []

    def head_io_hook(
        self,
    ):
        """Head common input tensor names."""
        # feature = self._feature.detach().cpu().numpy()
        spatial_shapes = (
            self._spatial_shapes.int().detach().cpu().numpy()
        )  # int64->in32
        level_start_index = (
            self._level_start_index.int().detach().cpu().numpy()
        )  # int64->in32
        instance_feature = self._instance_feature.detach().cpu().numpy()
        anchor = self._anchor.detach().cpu().numpy()
        time_interval = self._time_interval.detach().cpu().numpy()
        image_wh = self._image_wh.detach().cpu().numpy()
        lidar2img = self._lidar2img.detach().cpu().numpy()

        """ Head common output tensor names. """
        pred_instance_feature = self._pred_instance_feature.detach().cpu().numpy()
        pred_anchor = self._pred_anchor.detach().cpu().numpy()
        pred_class_score = self._pred_class_score.detach().cpu().numpy()
        pred_quality_score = self._pred_quality_score.detach().cpu().numpy()

        if (
            self._temp_instance_feature is not None
            and self._temp_anchor is not None
            and self._mask is not None
            and self._track_id is not None
            and self._pred_track_id is not None
        ):
            """Head frame > 1 input tensor names."""
            temp_instance_feature = self._temp_instance_feature.detach().cpu().numpy()
            temp_anchor = self._temp_anchor.detach().cpu().numpy()
            mask = self._mask.int().detach().cpu().numpy()  # int64->in32
            track_id = self._track_id.int().detach().cpu().numpy()  # int64->in32

            """ Head frame > 1 output tensor names. """
            pred_track_id = (
                self._pred_track_id.int().detach().cpu().numpy()
            )  # int64->in32

            # 添加temp_gnn输出
            temp_gnn_output = self._temp_gnn_output.detach().cpu().numpy() if self._temp_gnn_output is not None else None

        if self._first_frame:
            inputs = [
                # feature, # repeat with the backbone output
                spatial_shapes,
                level_start_index,
                instance_feature,
                anchor,
                time_interval,
                image_wh,
                lidar2img,
            ]
            
            # 正确展开refine_outs字典，包括update_comparison
            refine_outputs = []
            for refine_out in self._refine_outs:
                refine_outputs.extend([
                    refine_out['anchor'],
                    refine_out['instance_feature'],
                    refine_out['anchor_embed'],
                    refine_out['temp_anchor_embed']
                ])
                
                # 添加update_comparison的输出（如果存在）
                if refine_out['update_comparison'] is not None:
                    refine_outputs.extend([
                        refine_out['update_comparison']['instance_feature_before_update'],
                        refine_out['update_comparison']['anchor_before_update'],
                        refine_out['update_comparison']['temp_instance_feature'],
                        refine_out['update_comparison']['temp_anchor'],
                        refine_out['update_comparison']['confidence_sorted'],      # 新增：confidence_sorted
                        refine_out['update_comparison']['indices'],              # 新增：indices
                        refine_out['update_comparison']['selected_feature'],      # 新增：selected_feature
                        refine_out['update_comparison']['selected_anchor'],      # 新增：selected_anchor
                        refine_out['update_comparison']['instance_feature_after_update'],
                        refine_out['update_comparison']['anchor_after_update']
                    ])
                else:
                    # 如果update_comparison为None，创建零张量
                    refine_outputs.extend([
                        np.zeros_like(refine_out['instance_feature']),  # instance_feature_before_update
                        np.zeros_like(refine_out['anchor']),            # anchor_before_update
                        np.zeros_like(refine_out['instance_feature']),  # temp_instance_feature (新增)
                        np.zeros_like(refine_out['anchor']),            # temp_anchor (新增)
                        np.zeros((refine_out['instance_feature'].shape[0], self._head.instance_bank.num_anchor - self._head.instance_bank.num_temp_instances)),  # confidence_sorted (新增)
                        np.zeros((refine_out['instance_feature'].shape[0], self._head.instance_bank.num_anchor - self._head.instance_bank.num_temp_instances), dtype=np.int32),  # indices (新增)
                        np.zeros_like(refine_out['instance_feature']),  # selected_feature (新增)
                        np.zeros_like(refine_out['anchor']),            # selected_anchor (新增)
                        np.zeros_like(refine_out['instance_feature']),  # instance_feature_after_update
                        np.zeros_like(refine_out['anchor'])             # anchor_after_update
                    ])
            
            outputs = [
                pred_instance_feature,
                pred_anchor,
                pred_class_score,
                pred_quality_score,
                *self._tmp_outs,   # 展开tmp_outs列表
                *refine_outputs,   # 展开refine_outs字典
            ]
            return inputs, outputs

        inputs = [
            temp_instance_feature,
            temp_anchor,
            mask,
            track_id,
            instance_feature,
            anchor,
            time_interval,
            # feature, # repeat with the backbone output
            spatial_shapes,
            level_start_index,
            image_wh,
            lidar2img,
        ]
        
        # 正确展开refine_outs字典，包括update_comparison
        refine_outputs = []
        for refine_out in self._refine_outs:
            refine_outputs.extend([
                refine_out['anchor'],
                refine_out['instance_feature'],
                refine_out['anchor_embed'],
                refine_out['temp_anchor_embed']
            ])
            
            # 添加update_comparison的输出（如果存在）
            if refine_out['update_comparison'] is not None:
                refine_outputs.extend([
                    refine_out['update_comparison']['instance_feature_before_update'],
                    refine_out['update_comparison']['anchor_before_update'],
                    refine_out['update_comparison']['temp_instance_feature'],
                    refine_out['update_comparison']['temp_anchor'],
                    refine_out['update_comparison']['confidence_sorted'],
                    refine_out['update_comparison']['indices'],
                    refine_out['update_comparison']['selected_feature'],
                    refine_out['update_comparison']['selected_anchor'],
                    refine_out['update_comparison']['instance_feature_after_update'],
                    refine_out['update_comparison']['anchor_after_update']
                ])
            else:
                # 如果update_comparison为None，创建零张量
                refine_outputs.extend([
                    np.zeros_like(refine_out['instance_feature']),  # instance_feature_before_update
                    np.zeros_like(refine_out['anchor']),            # anchor_before_update
                    np.zeros_like(refine_out['instance_feature']),  # temp_instance_feature (新增)
                    np.zeros_like(refine_out['anchor']),            # temp_anchor (新增)
                                            np.zeros((refine_out['instance_feature'].shape[0], self._head.instance_bank.num_anchor - self._head.instance_bank.num_temp_instances)),  # confidence_sorted (新增)
                                            np.zeros((refine_out['instance_feature'].shape[0], self._head.instance_bank.num_anchor - self._head.instance_bank.num_temp_instances), dtype=np.int32),  # indices (新增)
                    np.zeros_like(refine_out['instance_feature']),  # selected_feature (新增)
                    np.zeros_like(refine_out['anchor']),            # selected_anchor (新增)
                    np.zeros_like(refine_out['instance_feature']),  # instance_feature_after_update
                    np.zeros_like(refine_out['anchor'])             # anchor_after_update
                ])
        
        outputs = [
            pred_instance_feature,
            pred_anchor,
            pred_class_score,
            pred_quality_score,
            pred_track_id,
            temp_gnn_output,  # temp_gnn输出
            *self._tmp_outs,   # 展开tmp_outs列表
            *refine_outputs,   # 展开refine_outs字典
        ]
        return inputs, outputs

    def instance_bank_io_hook(self):
        """InstanceBank::get() input tensor names."""
        ibank_timestamp = self._ibank_timestamp.detach().cpu().numpy()
        # ibank_global2lidar = self._ibank_global2lidar.astype(
        #     np.float32
        # )  # float64 -> float32

        ibank_global2lidar = self._ibank_global2lidar

        """InstanceBank::get() output tensor names. """
        # self._instance_feature
        # self._anchor
        # self._time_interval
        # self._feature
        # self._spatial_shapes
        # sefl._level_start_index

        """InstanceBank::cache() input tensor names. """
        # self._pred_instance_feature
        # self._pred_instance_feature

        """InstanceBank::cache() output tensor names. """
        ibank_temp_confidence = self._ibank_temp_confidence.detach().cpu().numpy()
        ibank_confidence = self._ibank_confidence.detach().cpu().numpy()
        ibank_cached_feature = self._ibank_cached_feature.detach().cpu().numpy()
        ibank_cached_anchor = self._ibank_cached_anchor.detach().cpu().numpy()

        """InstanceBank GetTrackId() output tensor names. """
        ibank_prev_id = np.array(
            [self._ibank_prev_id.detach().cpu()], dtype=np.int32
        )  # int64 -> int32
        ibank_updated_cur_track_id = (
            self._ibank_updated_cur_track_id.int()
            .detach()
            .cpu()
            .numpy()  # int64 -> int32
        )
        ibank_updated_temp_track_id = (
            self._ibank_updated_temp_track_id.int()
            .detach()
            .cpu()
            .numpy()  # int64 -> int32
        )

        inputs = [
            ibank_timestamp,
            ibank_global2lidar,
        ]
        outputs = [
            ibank_temp_confidence,
            ibank_confidence,
            ibank_cached_feature,
            ibank_cached_anchor,
            ibank_prev_id,
            ibank_updated_cur_track_id,
            ibank_updated_temp_track_id,
        ]
        return inputs, outputs

    def post_process_io_hook(self):
        decoder_boxes_3d = self._decoder_boxes_3d.detach().cpu().numpy()
        decoder_scores_3d = self._decoder_scores_3d.detach().cpu().numpy()
        decoder_labels_3d = (
            self._decoder_labels_3d.int().detach().cpu().numpy()
        )  # int64 -> int32
        decoder_cls_scores = self._decoder_cls_scores.detach().cpu().numpy()
        decoder_track_ids = self._decoder_track_ids.int().detach().cpu().numpy()

        inputs = []
        outputs = [
            decoder_boxes_3d,
            decoder_scores_3d,
            decoder_labels_3d,
            decoder_cls_scores,
            decoder_track_ids,
        ]
        return inputs, outputs

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # 确保instance_feature被正确初始化（第一帧和第二帧都需要）
        if not hasattr(self._head.instance_bank, '_initialized'):
            self._head.instance_bank.init_weight()
            # 强制初始化instance_feature，即使requires_grad为False
            if torch.all(self._head.instance_bank.instance_feature == 0):
                torch.nn.init.xavier_uniform_(self._head.instance_bank.instance_feature.data, gain=1)
            self._head.instance_bank._initialized = True
        
        # 使用与ONNX导出一致的逻辑
        if self._first_frame:
            # 第一帧：使用InstanceBank的默认逻辑
            (
                instance_feature,
                anchor,
                temp_instance_feature,
                temp_anchor,
                time_interval,
            ) = self._head.instance_bank.get(
                batch_size, metas, dn_metas=self._head.sampler.dn_metas
            )
        else:
            # 第二帧：使用正确的temporal数据逻辑
            # 使用InstanceBank的get方法获取当前帧的instance_feature和anchor
            (
                instance_feature,
                anchor,
                temp_instance_feature,
                temp_anchor,
                time_interval,
            ) = self._head.instance_bank.get(
                batch_size, metas, dn_metas=self._head.sampler.dn_metas
            )
            
            # 使用InstanceBank返回的真实temporal数据，而不是随机数据
            # temp_instance_feature和temp_anchor现在来自第一帧的缓存结果

        """InstanceBank::get() input hook. """
        self._ibank_timestamp = metas["timestamp"].clone()
        self._ibank_global2lidar = metas["img_metas"][0]["global2lidar"].copy()

        """Head input hook. """
        # self._feature = feature_maps[0].clone() # it repeats in backbone io_hook.
        self._spatial_shapes = feature_maps[1].clone()
        self._level_start_index = feature_maps[2].clone()

        self._instance_feature = instance_feature.clone()
        self._anchor = anchor.clone()
        self._time_interval = time_interval.clone()
        if self._first_frame:
            assert temp_instance_feature is None
            assert temp_anchor is None
            assert self._head.instance_bank.mask is None
            assert self._head.instance_bank.track_id is None

            self._temp_instance_feature = None
            self._temp_anchor = None
            self._mask = None
            self._track_id = None
        else:
            assert temp_instance_feature is not None
            assert temp_anchor is not None
            assert self._head.instance_bank.mask is not None
            assert self._head.instance_bank.track_id is not None

            self._temp_instance_feature = temp_instance_feature.clone()
            self._temp_anchor = temp_anchor.clone()
            self._mask = self._head.instance_bank.mask.clone()
            self._track_id = self._head.instance_bank.track_id.clone()

        self._image_wh = metas["image_wh"].clone()
        self._lidar2img = metas["lidar2img"].clone()

        attn_mask = None
        temp_dn_reg_target = None

        anchor_embed = self._head.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self._head.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        prediction = []
        classification = []
        quality = []
        self._temp_gnn_output = None  # 清空之前的temp_gnn输出
        temp_gnn_count = 0  # 添加计数器来跟踪temp_gnn操作
        self._tmp_outs = []  # 清空之前的tmp_outs
        self._refine_outs = []  # 清空之前的refine_outs

        for i, op in enumerate(self._head.operation_order):
            if self._head.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self._head.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask if temp_instance_feature is None else None,
                )
                 # 只保存第一个temp_gnn模块的输出
                if not self._first_frame and temp_gnn_count == 0:  # 只在第二帧且是第一个temp_gnn时保存
                    self._temp_gnn_output = instance_feature.clone()
                    temp_gnn_count += 1
                elif not self._first_frame:
                    temp_gnn_count += 1  # 继续计数，但不保存
            elif op == "gnn":
                instance_feature = self._head.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self._head.layers[i](instance_feature)
            elif op == "deformable":
                # 保存deformable操作前的instance_feature维度
                original_shape = instance_feature.shape
                
                instance_feature = self._head.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
                #直接保存feature和instance_feature合并后的结果 [1, 900, 512]
                self._tmp_outs.append(instance_feature.detach().cpu().numpy())
            elif op == "refine":
                # 保存refine操作前的状态
                instance_feature_before = instance_feature.clone()
                anchor_before = anchor.clone()
                anchor_embed_before = anchor_embed.clone()
                temp_anchor_embed_before = temp_anchor_embed.clone() if temp_anchor_embed is not None else None
                
                anchor, cls, qt = self._head.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self._head.training
                        or len(prediction) == self._head.num_single_frame_decoder - 1
                        or i == len(self._head.operation_order) - 1
                    ),
                )
                
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)

                # 初始化update_comparison为None
                update_comparison = None
                if len(prediction) == self._head.num_single_frame_decoder:
                    # 保存update前的状态
                    instance_feature_before_update = instance_feature.clone()
                    anchor_before_update = anchor.clone()

                    instance_feature, anchor = self._head.instance_bank.update(
                        instance_feature, anchor, cls
                    )

                    # 可以选择保存update前后的对比数据
                    update_comparison = {
                        'instance_feature_before_update': instance_feature_before_update.detach().cpu().numpy(),
                        'anchor_before_update': anchor_before_update.detach().cpu().numpy(),
                        'temp_instance_feature': self._head.instance_bank.cached_feature.detach().cpu().numpy() if self._head.instance_bank.cached_feature is not None else np.zeros_like(instance_feature_before_update[:, : self._head.instance_bank.num_temp_instances].detach().cpu().numpy()),
                        'temp_anchor': self._head.instance_bank.cached_anchor.detach().cpu().numpy() if self._head.instance_bank.cached_anchor is not None else np.zeros_like(anchor_before_update[:, : self._head.instance_bank.num_temp_instances].detach().cpu().numpy()),
                        'confidence_sorted': self._head.instance_bank.confidence_sorted.detach().cpu().numpy() if self._head.instance_bank.confidence_sorted is not None else np.zeros((instance_feature_before_update.shape[0], self._head.instance_bank.num_anchor - self._head.instance_bank.num_temp_instances)),
                        'indices': self._head.instance_bank.indices.int().detach().cpu().numpy().astype(np.int32) if self._head.instance_bank.indices is not None else np.zeros((instance_feature_before_update.shape[0], self._head.instance_bank.num_anchor - self._head.instance_bank.num_temp_instances), dtype=np.int32),
                        'selected_feature': self._head.instance_bank.selected_feature.detach().cpu().numpy() if self._head.instance_bank.selected_feature is not None else np.zeros_like(instance_feature_before_update.detach().cpu().numpy()),
                        'selected_anchor': self._head.instance_bank.selected_anchor.detach().cpu().numpy() if self._head.instance_bank.selected_anchor is not None else np.zeros_like(anchor_before_update.detach().cpu().numpy()),
                        'instance_feature_after_update': instance_feature.detach().cpu().numpy(),
                        'anchor_after_update': anchor.detach().cpu().numpy(),
                    }

                if i != len(self._head.operation_order) - 1:
                    anchor_embed = self._head.anchor_encoder(anchor)
                if (
                    len(prediction) > self._head.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self._head.instance_bank.num_temp_instances
                    ]

                # 第一帧和第二帧都保存refine模块的输出
                # 处理temp_anchor_embed，如果为None则创建零张量
                if temp_anchor_embed is not None:
                    temp_anchor_embed_np = temp_anchor_embed.detach().cpu().numpy()
                else:
                    # 创建一个与anchor_embed形状相同的零张量
                    temp_anchor_embed_np = np.zeros_like(anchor_embed[:, : self._head.instance_bank.num_temp_instances].detach().cpu().numpy())
                
                # 保存refine模块的四个输出，以及update_comparison
                refine_output = {
                    'anchor': anchor.detach().cpu().numpy(),
                    'instance_feature': instance_feature.detach().cpu().numpy(),
                    'anchor_embed': anchor_embed.detach().cpu().numpy(),
                    'temp_anchor_embed': temp_anchor_embed_np,
                    'update_comparison': update_comparison  # 添加update_comparison
                }
                self._refine_outs.append(refine_output)

        """Head output hook. """
        self._pred_instance_feature = instance_feature.clone()
        self._pred_anchor = anchor.clone()
        self._pred_class_score = cls.clone()
        self._pred_quality_score = qt.clone()
        if self._head.instance_bank.track_id is not None:
            self._pred_track_id = self._head.instance_bank.track_id.clone()

        output = {}
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )

        # 缓存当前帧结果用于下一帧的推理
        if self._first_frame:
            self._cached_pred_instance_feature = instance_feature.clone()
            self._cached_pred_anchor = anchor.clone()

        self._head.instance_bank.cache(instance_feature, anchor, cls, metas)

        """InstanceBank::cache() output hook. """
        self._ibank_temp_confidence = self._head.instance_bank.temp_confidence.clone()
        self._ibank_confidence = self._head.instance_bank.confidence.clone()
        self._ibank_cached_feature = self._head.instance_bank.cached_feature.clone()
        self._ibank_cached_anchor = self._head.instance_bank.cached_anchor.clone()

        track_id = self._head.instance_bank.get_track_id(
            cls, self._head.decoder.score_threshold
        )
        output["track_id"] = track_id  # [1, 900], int64

        """InstanceBank::get_track_id() output hook. """
        self._ibank_prev_id = self._head.instance_bank.prev_id.clone()
        self._ibank_updated_cur_track_id = track_id.clone()
        self._ibank_updated_temp_track_id = self._head.instance_bank.track_id.clone()

        """Postprocessor output  hook. """
        output = self._head.decoder.decode(
            output["classification"],
            output["prediction"],
            output.get("track_id"),
            output.get("quality"),
            output_idx=-1,
        )[batch_size - 1]
        
        # 置信度过滤
        score_threshold = 0.3  # 置信度阈值
        boxes_3d = output["boxes_3d"]
        scores_3d = output["scores_3d"]
        labels_3d = output["labels_3d"]
        cls_scores = output["cls_scores"]
        track_ids = output["track_ids"]
        
        # 应用置信度过滤
        valid_mask = scores_3d >= score_threshold
        filtered_count = valid_mask.sum().item()
        total_count = len(scores_3d)
        
        print(f"样本 {i}: 总目标数 {total_count}, 置信度过滤后 {filtered_count} 个目标 (阈值: {score_threshold})")
        
        # 保存过滤后的结果
        self._decoder_boxes_3d = boxes_3d[valid_mask]
        self._decoder_scores_3d = scores_3d[valid_mask]
        self._decoder_labels_3d = labels_3d[valid_mask]
        self._decoder_cls_scores = cls_scores[valid_mask]
        self._decoder_track_ids = track_ids[valid_mask]


def main():
    set_random_seed(seed=1, deterministic=True)

    args = parse_args()
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logger, console_handler, file_handler = set_logger(args.log, save_file=True)
    logger.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    cfg = read_cfg(args.config)  # dict

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg["data"]["test"]["test_mode"] = True

    # build the dataloader
    samples_per_gpu = cfg["data"]["test"].pop("samples_per_gpu", 1)
    dataset = build_module(cfg["data"]["test"])
    data_loader = dataloader_wrapper(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg["data"]["workers_per_gpu"],
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    model = build_module(cfg["model"])
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")

    model.eval().cuda()

    backbone_hook = Sparse4D_backbone(model)
    head_hook = Sparse4D_head(model.head)
    
    # 统计变量
    total_samples = 0
    total_detections = 0
    total_filtered_detections = 0

    for i, data in enumerate(data_loader):
        if i == 3:
            break
        with torch.no_grad():
            data = scatter(data, [0])[0]
            ori_imgs = data["ori_img"].detach().cpu().numpy()
            imgs = data["img"].detach().cpu().numpy()
            save_bins(
                inputs=[ori_imgs],
                outputs=[imgs],
                names=["ori_imgs", "imgs"],
                sample_index=i,
                logger=logger,
            )

            feature_maps = backbone_hook(img=data.pop("img"))
            logger.info(
                f"Start to save bin for Sparse4dBackbone, sampleindex={i} >>>>>>>>>>>>>>>>"
            )
            save_bins(
                inputs=backbone_hook.io_hook()[0],
                outputs=backbone_hook.io_hook()[1],
                names=["imgs", "feature"],
                sample_index=i,
                logger=logger,
            )

            head_hook(feature_maps, data)
            
            # 获取过滤统计信息
            if hasattr(head_hook, '_decoder_scores_3d') and head_hook._decoder_scores_3d is not None:
                total_count = len(head_hook._decoder_scores_3d)
                score_threshold = 0.2
                valid_mask = head_hook._decoder_scores_3d >= score_threshold
                filtered_count = valid_mask.sum().item()
                
                # 更新统计信息
                total_samples += 1
                total_detections += total_count
                total_filtered_detections += filtered_count
            
            inputs, outputs = head_hook.head_io_hook()
            if head_hook._first_frame:
                head_hook._first_frame = False
                logger.info(
                    f"Start to save bin for first frame Sparse4dHead, sampleindex={i} >>>>>>>>>>>>>>>>"
                )

                # 构建输出名称列表
                output_names = [
                    "pred_instance_feature",
                    "pred_anchor",
                    "pred_class_score",
                    "pred_quality_score",
                ]
                
                # 添加tmp_outs名称
                for j in range(len(head_hook._tmp_outs)):
                    output_names.append(f"tmp_outs{j}")
                    
                # 添加refine_outs名称 - 第一帧也保存
                for j in range(len(head_hook._refine_outs)):
                    output_names.append(f"refine_outs{j}_anchor")
                    output_names.append(f"refine_outs{j}_instance_feature")
                    output_names.append(f"refine_outs{j}_anchor_embed")
                    output_names.append(f"refine_outs{j}_temp_anchor_embed")
                    # 添加update_comparison的名称
                    output_names.append(f"refine_outs{j}_instance_feature_before_update")
                    output_names.append(f"refine_outs{j}_anchor_before_update")
                    output_names.append(f"refine_outs{j}_temp_instance_feature")  # 新增
                    output_names.append(f"refine_outs{j}_temp_anchor")            # 新增
                    output_names.append(f"refine_outs{j}_confidence_sorted")     # 新增：confidence_sorted
                    output_names.append(f"refine_outs{j}_indices")               # 新增：indices
                    output_names.append(f"refine_outs{j}_selected_feature")      # 新增：selected_feature
                    output_names.append(f"refine_outs{j}_selected_anchor")       # 新增：selected_anchor
                    output_names.append(f"refine_outs{j}_instance_feature_after_update")
                    output_names.append(f"refine_outs{j}_anchor_after_update")
                
                save_bins(
                    inputs=inputs,
                    outputs=outputs,
                    names=[
                        "spatial_shapes",
                        "level_start_index",
                        "instance_feature",
                        "anchor",
                        "time_interval",
                        "image_wh",
                        "lidar2img",
                    ] + output_names,  # 合并输入和输出名称
                    logger=logger,
                    sample_index=i,
                )
            else:
                logger.info(
                    f"Start to save bin for frame > 1 Sparse4dHead, sampleindex={i} >>>>>>>>>>>>>>>>"
                )
                
                # 构建输出名称列表
                output_names = [
                    "pred_instance_feature",
                    "pred_anchor",
                    "pred_class_score",
                    "pred_quality_score",
                    "pred_track_id",
                    "temp_gnn_output",
                ]
                
                # 添加tmp_outs名称
                for j in range(len(head_hook._tmp_outs)):
                    output_names.append(f"tmp_outs{j}")
                
                # 添加refine_outs名称 - 第二帧也保存
                for j in range(len(head_hook._refine_outs)):
                    output_names.append(f"refine_outs{j}_anchor")
                    output_names.append(f"refine_outs{j}_instance_feature")
                    output_names.append(f"refine_outs{j}_anchor_embed")
                    output_names.append(f"refine_outs{j}_temp_anchor_embed")
                    # 添加update_comparison的名称
                    output_names.append(f"refine_outs{j}_instance_feature_before_update")
                    output_names.append(f"refine_outs{j}_anchor_before_update")
                    output_names.append(f"refine_outs{j}_temp_instance_feature")  # 新增
                    output_names.append(f"refine_outs{j}_temp_anchor")            # 新增
                    output_names.append(f"refine_outs{j}_confidence_sorted")     # 新增：confidence_sorted
                    output_names.append(f"refine_outs{j}_indices")               # 新增：indices
                    output_names.append(f"refine_outs{j}_selected_feature")      # 新增：selected_feature
                    output_names.append(f"refine_outs{j}_selected_anchor")       # 新增：selected_anchor
                    output_names.append(f"refine_outs{j}_instance_feature_after_update")
                    output_names.append(f"refine_outs{j}_anchor_after_update")
                
                save_bins(
                    inputs=inputs,
                    outputs=outputs,
                    names=[
                        "temp_instance_feature",
                        "temp_anchor",
                        "mask",
                        "track_id",
                        "instance_feature",
                        "anchor",
                        "time_interval",
                        "spatial_shapes",
                        "level_start_index",
                        "image_wh",
                        "lidar2img",
                    ] + output_names,  # 合并输入和输出名称
                    logger=logger,
                    sample_index=i,
                )

            logger.info(
                f"Start to save bin for InstanceBank, sampleindex={i} >>>>>>>>>>>>>>>>"
            )
            save_bins(
                inputs=head_hook.instance_bank_io_hook()[0],
                outputs=head_hook.instance_bank_io_hook()[1],
                names=[
                    "ibank_timestamp",
                    "ibank_global2lidar",
                    "ibank_temp_confidence",
                    "ibank_confidence",
                    "ibank_cached_feature",
                    "ibank_cached_anchor",
                    "ibank_prev_id",
                    "ibank_updated_cur_track_id",
                    "ibank_updated_temp_track_id",
                ],
                logger=logger,
                sample_index=i,
            )

            logger.info(
                f"Start to save bin for Postprocessor, sampleindex={i} >>>>>>>>>>>>>>>>"
            )
            save_bins(
                inputs=head_hook.post_process_io_hook()[0],
                outputs=head_hook.post_process_io_hook()[1],
                names=[
                    "decoder_boxes_3d",
                    "decoder_scores_3d",
                    "decoder_labels_3d",
                    "decoder_cls_scores",
                    "decoder_track_ids",
                ],
                logger=logger,
                sample_index=i,
            )
    
    # 打印总体统计信息
    print(f"\n=== 置信度过滤统计 ===")
    print(f"处理样本数: {total_samples}")
    print(f"总检测目标数: {total_detections}")
    print(f"过滤后目标数: {total_filtered_detections}")
    if total_detections > 0:
        print(f"过滤率: {((total_detections - total_filtered_detections) / total_detections * 100):.1f}%")
        print(f"平均每样本过滤后目标数: {total_filtered_detections / total_samples:.1f}")


if __name__ == "__main__":
    main()