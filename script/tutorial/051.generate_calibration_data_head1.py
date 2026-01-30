
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from inspect import signature
from typing import Optional, Any, Dict

from tool.utils.config import read_cfg
from tool.utils.logger import set_logger
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
    parser = argparse.ArgumentParser(description="Generate Calibration Data specifically for Head1")
    parser.add_argument("--config", default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py")
    parser.add_argument("--checkpoint", default="ckpt/sparse4dv3_r50.pth")
    parser.add_argument("--log", type=str, default="script/tutorial/generate_head1_calibration_data.log")
    parser.add_argument("--save-dir", type=str, default="deploy/calibration_data")
    parser.add_argument("--num-samples", type=int, default=100)
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
        return [col_feats, spatial_shape, scale_start_index]

    def forward(self, img):
        bs = img.shape[0]
        if img.dim() == 5:
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self._model.use_grid_mask:
            img = self._model.grid_mask(img)
        if "metas" in signature(self._model.img_backbone.forward).parameters:
            feature_maps = self._model.img_backbone(img, num_cams)
        else:
            feature_maps = self._model.img_backbone(img)
        if self._model.img_neck is not None:
            feature_maps = list(self._model.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(feat, (bs, num_cams) + feat.shape[1:])
        if self._model.use_deformable_func:
            feature_maps = self.feature_maps_format(feature_maps)
        return feature_maps

class Sparse4D_head1_extractor(nn.Module):
    def __init__(self, model):
        super(Sparse4D_head1_extractor, self).__init__()
        self._head = model

    def forward(self, feature_maps, metas, sample_idx, save_dir):
        # Every call here is treated as a FIRST FRAME
        self._head.instance_bank.reset()
        batch_size = feature_maps[0].shape[0]

        (instance_feature, anchor, _, _, time_interval) = self._head.instance_bank.get(
            batch_size, metas, dn_metas=self._head.sampler.dn_metas
        )

        # Capture Head 1 Inputs
        data_to_save = {
            "feature": feature_maps[0].detach().cpu().numpy(),
            "spatial_shapes": feature_maps[1].detach().cpu().numpy(),
            "level_start_index": feature_maps[2].detach().cpu().numpy(),
            "instance_feature": instance_feature.detach().cpu().numpy(),
            "anchor": anchor.detach().cpu().numpy(),
            "time_interval": time_interval.detach().cpu().numpy(),
            "image_wh": metas["image_wh"].detach().cpu().numpy(),
            "lidar2img": metas["lidar2img"].detach().cpu().numpy(),
        }
        
        save_path = os.path.join(save_dir, "head1", f"{sample_idx:06d}.npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **data_to_save)

def main():
    set_random_seed(seed=1, deterministic=True)
    args = parse_args()
    logger, _, _ = set_logger(args.log, save_file=True)
    
    cfg = read_cfg(args.config)
    cfg["data"]["test"]["test_mode"] = False
    cfg["data"]["test"].pop("samples_per_gpu", None)
    dataset = build_module(cfg["data"]["test"])
    loader = dataloader_wrapper(dataset, samples_per_gpu=1, workers_per_gpu=cfg["data"]["workers_per_gpu"], dist=False, shuffle=False)
    
    model = build_module(cfg["model"])
    if args.checkpoint: load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval().cuda()
    
    backbone_wrapper = Sparse4D_backbone(model)
    head1_extractor = Sparse4D_head1_extractor(model.head)
    
    from tqdm import tqdm
    pbar = tqdm(total=args.num_samples)

    for i, data in enumerate(loader):
        if i >= args.num_samples: break
        with torch.no_grad():
            data = scatter(data, [0])[0]
            img = data["img"]
            if img.dim() == 4: img = img.unsqueeze(0)
            
            # Construct Metas
            img_metas = data["img_metas"]
            global_l2i = data["lidar2img"]
            if not isinstance(global_l2i, torch.Tensor):
                global_l2i = torch.tensor(global_l2i, dtype=torch.float32).cuda()
            if global_l2i.dim() == 3: global_l2i = global_l2i.unsqueeze(0)
            
            wh_list = []
            for m in img_metas:
                if "image_wh" not in m:
                    if "img_shape" in m:
                        h, w = m["img_shape"][:2]
                        m["image_wh"] = torch.tensor([w, h], dtype=torch.float32).cuda()
                    else:
                        # Fallback to default resolution if keys are missing
                        m["image_wh"] = torch.tensor([704, 256], dtype=torch.float32).cuda()
                
                if not isinstance(m["image_wh"], torch.Tensor):
                    m["image_wh"] = torch.tensor(m["image_wh"], dtype=torch.float32).cuda()
                wh_list.append(m["image_wh"])
            wh_stack = torch.stack(wh_list).unsqueeze(1).expand(-1, 6, -1)
            
            ts = img_metas[0]["timestamp"]
            ts_tensor = torch.tensor([ts], dtype=torch.float32).cuda() if not isinstance(ts, torch.Tensor) else ts.cuda().float().reshape(1)

            head_metas = {
                "img_metas": img_metas,
                "lidar2img": global_l2i,
                "image_wh": wh_stack,
                "timestamp": ts_tensor
            }

            feature_maps = backbone_wrapper(img)
            head1_extractor(feature_maps, head_metas, i, args.save_dir)
            pbar.update(1)

    pbar.close()
    logger.info(f"Successfully generated Head1 calibration data in {args.save_dir}/head1")

if __name__ == "__main__":
    main()
