# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import argparse
import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Optional, Dict, Any
from inspect import signature

from tool.utils.config import read_cfg
from tool.utils.logger import set_logger
from tool.runner.fp16_utils import wrap_fp16_model
from tool.runner.checkpoint import load_checkpoint
from tool.trainer.utils import set_random_seed
from tool.utils.save_bin import save_bins

from dataset.dataloader_wrapper import dataloader_wrapper
from dataset import NuScenes4DDetTrackDataset
from dataset.utils.scatter_gather import scatter

# Import modules to patch/hook
import modules.ops as ops_module
from modules.head.sparse4d_blocks.sparse3d_embedding import SparseBox3DKeyPointsGenerator
from modules.sparse4d_detector import Sparse4D

def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)

def parse_args():
    parser = argparse.ArgumentParser(description="Export Plugin IO Data (Real Data)")
    parser.add_argument(
        "--config",
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
        help="inference config file path",
    )
    parser.add_argument(
        "--checkpoint", default="ckpt/sparse4dv3_r50.pth", help="checkpoint file"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="deploy/val_data_plugin/real_data",
        help="Directory to save plugin IO data",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="script/tutorial/export_plugin_real.log",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 model weights and inputs",
    )
    args = parser.parse_args()
    return args

class PluginDataCollector:
    def __init__(self, save_dir, logger, fp16=False):
        self.save_dir = save_dir
        self.logger = logger
        self.fp16 = fp16
        self.sample_idx = 0
        self.ln_count = 0
        self.kps_count = 0
        self.dfa_count = 0
        self.original_daf = ops_module.deformable_aggregation_function

    def reset_counts(self):
        self.ln_count = 0
        self.kps_count = 0
        self.dfa_count = 0

    def set_sample_idx(self, idx):
        self.sample_idx = idx
        self.reset_counts()

    def get_save_path(self, plugin_name, count):
        path = os.path.join(self.save_dir, f"sample_{self.sample_idx}", f"{plugin_name}_{count}")
        os.makedirs(path, exist_ok=True)
        return path

    def ln_hook(self, module, input, output):
        """Hook for nn.LayerNorm"""
        x = input[0]
        save_path = self.get_save_path("ln", self.ln_count)
        # self.logger.debug(f"Saving LayerNorm {self.ln_count} data to {save_path}")
        
        save_bins([x.detach().cpu().numpy()], [], ["input"], self.sample_idx, self.logger, save_path)
        
        params = []
        param_names = []
        if module.weight is not None:
            params.append(module.weight.detach().cpu().numpy())
            param_names.append("weight")
        if module.bias is not None:
            params.append(module.bias.detach().cpu().numpy())
            param_names.append("bias")
        
        if params:
            save_bins(params, [], param_names, self.sample_idx, self.logger, save_path)
            
        with open(os.path.join(save_path, "attr.txt"), "w") as f:
            f.write(f"epsilon:{module.eps}\n")
            
        save_bins([], [output.detach().cpu().numpy()], ["output"], self.sample_idx, self.logger, save_path)
        self.ln_count += 1

    def kps_hook(self, module, input, output):
        """Hook for SparseBox3DKeyPointsGenerator"""
        anchor = input[0]
        feature = input[1]
        save_path = self.get_save_path("sparsebox", self.kps_count)
        # self.logger.debug(f"Saving SparseBox {self.kps_count} data to {save_path}")
        
        save_bins(
            [anchor.detach().cpu().numpy(), feature.detach().cpu().numpy()], 
            [output.detach().cpu().numpy()], 
            ["anchor", "feature", "keypoints"], 
            self.sample_idx, 
            self.logger, 
            save_path
        )
        self.kps_count += 1

    def dfa_wrapper(self, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
        """Wrapper for DAF function"""
        output = self.original_daf(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        
        save_path = self.get_save_path("dfa", self.dfa_count)
        # self.logger.debug(f"Saving DFA {self.dfa_count} data to {save_path}")
        
        inputs = [
            value.detach().cpu().numpy(),
            spatial_shapes.detach().cpu().numpy(),
            level_start_index.detach().cpu().numpy(),
            sampling_locations.detach().cpu().numpy(),
            attention_weights.detach().cpu().numpy()
        ]
        input_names = ["value", "spatial_shapes", "level_start_index", "sampling_locations", "attention_weights"]
        
        save_bins(inputs, [output.detach().cpu().numpy()], input_names + ["output"], self.sample_idx, self.logger, save_path)
        self.dfa_count += 1
        
        # If running in FP16 mode but output is FP32 (which DAF often is), cast to FP16 
        # to ensure compatibility with subsequent FP16 layers.
        if self.fp16 and output.dtype == torch.float32:
            return output.half()
            
        return output

# Re-using Sparse4D_backbone from 010 script to handle input formatting
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

    def forward(self, img, metas=None):
        return self.extract_feat(img, metas=metas)

def main():
    set_random_seed(seed=1, deterministic=True)
    args = parse_args()
    
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logger, console_handler, file_handler = set_logger(args.log, save_file=True)
    logger.setLevel(logging.INFO)
    
    cfg = read_cfg(args.config)
    
    # Enable CUDNN Benchmark if configured
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg["data"]["test"]["test_mode"] = True

    # Build dataloader
    samples_per_gpu = cfg["data"]["test"].pop("samples_per_gpu", 1)
    dataset = build_module(cfg["data"]["test"])
    data_loader = dataloader_wrapper(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg["data"]["workers_per_gpu"],
        dist=False,
        shuffle=False,
    )

    # Build model
    model = build_module(cfg["model"])
    
    if args.fp16:
        logger.info("Enabling FP16 mode...")
        wrap_fp16_model(model)
        model.half()
    
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    
    model.eval().cuda()

    # Setup Collector
    collector = PluginDataCollector(args.save_dir, logger, fp16=args.fp16)

    # Register Hooks
    logger.info("Registering hooks...")
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            module.register_forward_hook(collector.ln_hook)
        if isinstance(module, SparseBox3DKeyPointsGenerator):
            module.register_forward_hook(collector.kps_hook)
    
    # Patch DFA
    # ops_module.deformable_aggregation_function = collector.dfa_wrapper
    
    # We need to patch the function where it is used.
    # It is used in modules/head/sparse4d_blocks/core_blocks.py as DAF
    import modules.head.sparse4d_blocks.core_blocks as core_blocks
    if hasattr(core_blocks, 'DAF'):
        print("Patching core_blocks.DAF")
        collector.original_daf = core_blocks.DAF
        core_blocks.DAF = collector.dfa_wrapper
    else:
        print("Warning: DAF not found in core_blocks, patching ops_module")
        ops_module.deformable_aggregation_function = collector.dfa_wrapper

    # Wrapper for backbone
    backbone_wrapper = Sparse4D_backbone(model)

    logger.info(f"Starting inference loop for {args.samples} samples...")
    
    for i, data in enumerate(data_loader):
        if i >= args.samples:
            break
            
        collector.set_sample_idx(i)
        logger.info(f"Processing sample {i}...")
        
        with torch.no_grad():
            data = scatter(data, [0])[0]
            img = data["img"]
            metas = data["img_metas"]
            
            # 1. Run Backbone to get features
            # Note: 010 script calls extract_feat directly on wrapper
            if args.fp16:
                img = img.half()
                # Convert relevant tensors in data (metas) to half
                if "lidar2img" in data:
                    data["lidar2img"] = data["lidar2img"].half()
                if "image_wh" in data:
                    data["image_wh"] = data["image_wh"].half()
                
            feature_maps = backbone_wrapper(img, metas=data["img_metas"]) 
            
            # 2. Run Head
            # This triggers the hooks in DFA, LayerNorm, SparseBox
            # Pass full data dict because Sparse4DHead accesses data["lidar2img"] via metas argument
            model.head(feature_maps, data)
            
    # Restore DAF
    if hasattr(core_blocks, 'DAF'):
        core_blocks.DAF = collector.original_daf
    ops_module.deformable_aggregation_function = collector.original_daf
    logger.info(f"Done. Data saved to {args.save_dir}")

if __name__ == "__main__":
    main()

