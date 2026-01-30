
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import argparse
import torch
import torch.nn as nn
import logging
import numpy as np
from inspect import signature
from typing import Union, Optional, Any, Dict, List

from tool.utils.config import read_cfg
from tool.utils.logger import set_logger
from tool.runner.fp16_utils import wrap_fp16_model
from tool.runner.checkpoint import load_checkpoint
from tool.trainer.utils import set_random_seed
from dataset.dataloader_wrapper import dataloader_wrapper
from dataset.utils.scatter_gather import scatter
from dataset import NuScenes4DDetTrackDataset
from modules.sparse4d_detector import Sparse4D

def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Calibration Data for INT8 Quantization")
    parser.add_argument("--config", default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py")
    parser.add_argument("--checkpoint", default="ckpt/sparse4dv3_r50.pth")
    parser.add_argument("--log", type=str, default="script/tutorial/generate_calibration_data.log")
    parser.add_argument("--save-dir", type=str, default="deploy/calibration_data")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--head", type=str, default="both", choices=["head1", "head2", "both", "backbone", "all"])
    args = parser.parse_args()
    return args

class Sparse4D_backbone(nn.Module):
    def __init__(self, model):
        super(Sparse4D_backbone, self).__init__()
        self._model = model

    def feature_maps_format(self, feature_maps):
        bs, num_cams = feature_maps[0].shape[:2]
        spatial_shape = []
        col_feats = []
        for i, feat in enumerate(feature_maps):
            spatial_shape.append(feat.shape[-2:])
            col_feats.append(torch.reshape(feat, (bs, num_cams, feat.shape[2], -1)))

        col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
        spatial_shape = [spatial_shape] * num_cams
        spatial_shape = torch.tensor(spatial_shape, dtype=torch.int64, device=col_feats.device)

        scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
        scale_start_index = scale_start_index.flatten().cumsum(dim=0)
        scale_start_index = torch.cat([torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]])
        scale_start_index = scale_start_index.reshape(num_cams, -1)

        feature_maps = [col_feats, spatial_shape, scale_start_index]
        return feature_maps

    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
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
        
        if self._model.use_deformable_func:
            feature_maps = self.feature_maps_format(feature_maps)

        return feature_maps

    def forward(self, img):
        return self.extract_feat(img)

class Sparse4D_head(nn.Module):
    def __init__(self, model):
        super(Sparse4D_head, self).__init__()
        self._head = model
        self._first_frame = True
        
        # Temporary storage for captured inputs
        self._feature = None
        self._spatial_shapes = None
        self._level_start_index = None
        self._instance_feature = None
        self._anchor = None
        self._time_interval = None
        self._temp_instance_feature = None
        self._temp_anchor = None
        self._mask = None
        self._track_id = None
        self._image_wh = None
        self._lidar2img = None

    def save_calibration_data(self, save_dir, sample_idx, head_mode="both"):
        # head1_dir is no longer managed by this script for primary calibration
        head2_dir = os.path.join(save_dir, "head2")
        os.makedirs(head2_dir, exist_ok=True)

        def to_numpy(x):
            if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
            if isinstance(x, np.ndarray): return x
            return np.array(x)

        # Head2 inference happens at frames > 0, utilizing temporal history.
        if not self._first_frame and head_mode in ["head2", "both", "all"]:
            head2_data = {
                "feature": to_numpy(self._feature), 
                "spatial_shapes": to_numpy(self._spatial_shapes),
                "level_start_index": to_numpy(self._level_start_index),
                "instance_feature": to_numpy(self._instance_feature),
                "anchor": to_numpy(self._anchor),
                "time_interval": to_numpy(self._time_interval),
                "image_wh": to_numpy(self._image_wh),
                "lidar2img": to_numpy(self._lidar2img),
                "temp_instance_feature": to_numpy(self._temp_instance_feature),
                "temp_anchor": to_numpy(self._temp_anchor),
                "mask": to_numpy(self._mask.int()), 
                "track_id": to_numpy(self._track_id.int()),
            }
            np.savez(os.path.join(head2_dir, f"{sample_idx:06d}.npz"), **head2_data)

    def forward(self, feature_maps, metas):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self._head.instance_bank.get(
            batch_size, metas, dn_metas=self._head.sampler.dn_metas
        )

        self._feature = feature_maps[0]
        self._spatial_shapes = feature_maps[1]
        self._level_start_index = feature_maps[2]
        
        self._instance_feature = instance_feature
        self._anchor = anchor
        self._time_interval = time_interval
        
        self._image_wh = metas["image_wh"] 
        self._lidar2img = metas["lidar2img"]

        if self._first_frame:
            self._temp_instance_feature = None
            self._temp_anchor = None
            self._mask = None
            self._track_id = None
        else:
            self._temp_instance_feature = temp_instance_feature
            self._temp_anchor = temp_anchor
            self._mask = self._head.instance_bank.mask
            self._track_id = self._head.instance_bank.track_id

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
                instance_feature = self._head.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas, 
                )
            elif op == "refine":
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
                if len(prediction) == self._head.num_single_frame_decoder:
                    instance_feature, anchor = self._head.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                        
                if i != len(self._head.operation_order) - 1:
                    anchor_embed = self._head.anchor_encoder(anchor)
                if (
                    len(prediction) > self._head.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self._head.instance_bank.num_temp_instances
                    ]

        self._head.instance_bank.cache(instance_feature, anchor, cls, metas)
        
        track_id = self._head.instance_bank.get_track_id(
            cls, self._head.decoder.score_threshold
        )
        
        return None 

def main():
    set_random_seed(seed=1, deterministic=True)
    args = parse_args()
    
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logger, console_handler, file_handler = set_logger(args.log, save_file=True)
    
    cfg = read_cfg(args.config)
    cfg["data"]["test"]["test_mode"] = False
    cfg["data"]["test"].pop("samples_per_gpu", None)
    
    dataset = build_module(cfg["data"]["test"])
    
    loader = dataloader_wrapper(
        dataset, 
        samples_per_gpu=1, 
        workers_per_gpu=cfg["data"]["workers_per_gpu"], 
        dist=False, 
        shuffle=False
    )
    
    model = build_module(cfg["model"])
    if args.checkpoint: load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval().cuda()
    
    # Reset InstanceBank state
    if hasattr(model.head, "instance_bank"):
        model.head.instance_bank.reset()
    
    # WRAP MODULES
    backbone_wrapper = Sparse4D_backbone(model)
    head_wrapper = Sparse4D_head(model.head)
    
    try:
        from tqdm import tqdm
        pbar = tqdm(total=args.num_samples)
    except: pbar = None

    for i, data in enumerate(loader):
        if i >= args.num_samples: break
        
        with torch.no_grad():
            data = scatter(data, [0])[0]
            img = data["img"]
            img_metas = data["img_metas"]
            
            # Ensure img has batch dim [1, N, C, H, W]
            if img.dim() == 4: img = img.unsqueeze(0)
            
            # Construct a rich Meta Dict for Head
            head_input_metas = {
                "img_metas": img_metas
            }
            
            # 1. lidar2img
            global_l2i = None
            if "lidar2img" in data:
                l2i = data["lidar2img"]
                if isinstance(l2i, list): l2i = np.array(l2i)
                if isinstance(l2i, np.ndarray): global_l2i = torch.tensor(l2i, dtype=torch.float32).cuda()
                else: global_l2i = l2i.cuda().float()
                
                # Ensure [B, N, 4, 4]
                if global_l2i.dim() == 3: global_l2i = global_l2i.unsqueeze(0)
                head_input_metas["lidar2img"] = global_l2i

            # 2. image_wh
            wh_list = []
            for m in img_metas:
                # Preprocess internal keys for consistency
                if "img_shape" in m and "image_wh" not in m:
                    h, w = m["img_shape"][:2]
                    m["image_wh"] = torch.tensor([w, h], dtype=torch.float32).cuda()
                elif "image_wh" not in m:
                    m["image_wh"] = torch.tensor([704, 256], dtype=torch.float32).cuda()
                
                if not isinstance(m["image_wh"], torch.Tensor):
                    m["image_wh"] = torch.tensor(m["image_wh"], dtype=torch.float32).cuda()
                
                wh_list.append(m["image_wh"])
            
            if wh_list:
                wh_stack = torch.stack(wh_list) # [B, 2]
                # Expand to [B, N, 2]
                num_cams = global_l2i.shape[1] if global_l2i is not None else 6
                wh_stack = wh_stack.unsqueeze(1).expand(-1, num_cams, -1)
                head_input_metas["image_wh"] = wh_stack

            # 3. timestamp - CRITICAL FIX for InstanceBank mask
            if "timestamp" in img_metas[0]:
                ts = img_metas[0]["timestamp"]
                if isinstance(ts, torch.Tensor):
                    ts = ts.cuda().float()
                    if ts.dim() == 0: ts = ts.unsqueeze(0)
                else:
                    ts = torch.tensor([ts], dtype=torch.float32).cuda()
                head_input_metas["timestamp"] = ts

            # Fix nested metas for compatibility
            for m in img_metas:
                if "lidar2img" not in m and global_l2i is not None:
                    m["lidar2img"] = global_l2i[0] 
                if "timestamp" not in m and "timestamp" in head_input_metas:
                    # Keep as scalar/float inside img_metas if that's what original code did?
                    # But head_input_metas["timestamp"] is the key one.
                    pass

            # Save Backbone Input
            if args.head in ["backbone", "all"]:
                save_path = os.path.join(args.save_dir, "backbone", f"{i:06d}.npz")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez(save_path, img=img.detach().cpu().numpy())

            # 1. Run Backbone
            feature_maps = backbone_wrapper(img)
            
            # 2. Run Head
            head_wrapper(feature_maps, head_input_metas)
            
            # 3. Save Data
            head_wrapper.save_calibration_data(args.save_dir, i, args.head)
            
            if head_wrapper._first_frame:
                head_wrapper._first_frame = False
            
            if pbar: pbar.update(1)

    if pbar: pbar.close()
    logger.info("Done.")

if __name__ == "__main__":
    main()
