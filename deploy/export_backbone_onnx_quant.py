# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import time
import copy
import logging
import argparse
import sys
import glob
import numpy as np

import onnx
from onnxsim import simplify
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
    print("[WARNING] pytorch-quantization not found!")
# =======================================================

from modules.sparse4d_detector import *
from tool.utils.logger import set_logger
from tool.utils.config import read_cfg
from typing import Optional, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy Sparse4D Backbone INT8!")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--log", type=str, default="deploy/onnx/export_backbone_quant.log")
    parser.add_argument("--save_onnx", type=str, default="deploy/onnx/sparse4dbackbone_int8.onnx")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--calib_batches", type=int, default=100)
    parser.add_argument("--calib_data", type=str, default=None)
    return parser.parse_args()

def initialize_quantization():
    quant_desc_input = QuantDescriptor(calib_method='max', axis=None)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    # Enable ONNX export support for QDQ nodes
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

def replace_to_quantization_module(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Conv2d):
            quant_layer = quant_nn.QuantConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding, dilation=module.dilation,
                groups=module.groups, bias=module.bias is not None
            )
            quant_layer.weight.data = module.weight.data
            if module.bias is not None:
                quant_layer.bias.data = module.bias.data
            setattr(model, name, quant_layer.cuda())
        elif isinstance(module, torch.nn.Linear):
            quant_layer = quant_nn.QuantLinear(module.in_features, module.out_features, bias=module.bias is not None)
            quant_layer.weight.data = module.weight.data
            if module.bias is not None:
                quant_layer.bias.data = module.bias.data
            setattr(model, name, quant_layer.cuda())
        else:
            replace_to_quantization_module(module)

class CalibrationDataLoader:
    def __init__(self, data_dir):
        subdir = os.path.join(data_dir, "backbone")
        if not os.path.exists(subdir):
            raise ValueError(f"Calibration data directory not found: {subdir}")
        self.data_files = sorted(glob.glob(os.path.join(subdir, "*.npz")))
        print(f"[Quant] Found {len(self.data_files)} samples for backbone.")
        self.idx = 0
    def __len__(self): return len(self.data_files)
    def __call__(self, device="cuda"):
        if self.idx >= len(self.data_files): self.idx = 0
        data = np.load(self.data_files[self.idx])
        self.idx += 1
        img = torch.from_numpy(data["img"]).to(device).float()
        return (torch.nan_to_num(img, nan=0.0),)

def collect_stats(model, data_loader_func, num_batches):
    print(f"[Quant] Calibration starting for {num_batches} batches...")
    device = next(model.parameters()).device
    model.eval()
    for m in model.modules():
        if isinstance(m, quant_nn.TensorQuantizer):
            m.enable_calib()
            m.disable_quant()
    with torch.no_grad():
        for i in range(num_batches):
            model(*data_loader_func(device=device))
            if i % 20 == 0: print(f"Batch {i}/{num_batches}")
    
    print("[Quant] Calibration finished. Loading amax and syncing devices...")
    for name, m in model.named_modules():
        if isinstance(m, quant_nn.TensorQuantizer):
            m.load_calib_amax(strict=False)
            if m.amax is not None:
                new_amax = m.amax.detach().to(device)
                new_amax = torch.where(torch.isnan(new_amax), torch.ones_like(new_amax), new_amax)
                new_amax = torch.where(new_amax == 0, torch.ones_like(new_amax), new_amax)
                
                # Directly update _amax to bypass property setter warnings and device stickiness
                if hasattr(m, '_amax'):
                    m._amax = new_amax
                else:
                    m.register_buffer('_amax', new_amax)
            
            m.enable_quant()
            m.disable_calib()
    model.to(device)

class Sparse4DBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Physical isolation: only keep backbone and neck
        self.img_backbone = model.img_backbone
        self.img_neck = model.img_neck
        
    def forward(self, img):
        # Input: [BS, N, C, H, W] -> [BS*N, C, H, W]
        if img.dim() == 5:
            bs, n, c, h, w = img.shape
            img = img.reshape(bs * n, c, h, w)
        else:
            # Fallback if input is already 4D, assuming bs=1, n=6 from dimensions
            # But safer to just process as is if 4D
            bs = 1
            # Infer n from shape if possible, or just treat 0-dim as batch
            n = img.shape[0] // bs 

        # 1. FPN Forward
        # Output: List of [BS*N, 256, H_i, W_i]
        feats = self.img_neck(self.img_backbone(img))
        
        # 2. Flatten & Permute & Concat (Standard Sparse4D Pre-processing)
        # Target: [BS, N * Sum(H_i*W_i), 256]
        feature_maps = []
        for feat in feats:
            # [BS*N, C, H, W] -> [BS*N, C, H*W] -> [BS*N, H*W, C]
            feat = feat.flatten(2).transpose(1, 2)
            feature_maps.append(feat)
            
        # Concat multi-scale features: [BS*N, Sum(H*W), C]
        feature = torch.cat(feature_maps, dim=1)
        
        # Reshape to separate Batch and Camera: [BS, N, Sum(H*W), C]
        # Then flatten Camera and Spatial: [BS, N * Sum(H*W), C]
        feature = feature.reshape(bs, n, -1, feature.shape[-1]).flatten(1, 2)
        
        return feature

if __name__ == "__main__":
    args = parse_args()
    logger, _, _ = set_logger(args.log, True)
    cfg = read_cfg(args.cfg)
    
    # Critical Fix: Disable Gradient Checkpointing (with_cp) for ONNX export
    if "img_backbone" in cfg["model"] and "with_cp" in cfg["model"]["img_backbone"]:
        cfg["model"]["img_backbone"]["with_cp"] = False
        logger.info("Disabled img_backbone.with_cp for ONNX export safety.")

    # 1. Load full model
    full_model = eval(cfg["model"]["type"])(**{k:v for k,v in cfg["model"].items() if k != "type"})
    full_model.load_state_dict(torch.load(args.ckpt)["state_dict"], strict=False)
    
    # 2. Extract Backbone wrapper
    backbone_wrapper = Sparse4DBackbone(full_model).cuda().eval()
    del full_model # Release memory
    
    if args.int8:
        initialize_quantization()
        replace_to_quantization_module(backbone_wrapper)
        if args.calib_data:
            loader = CalibrationDataLoader(args.calib_data)
            collect_stats(backbone_wrapper, loader, min(args.calib_batches, len(loader)))
            loader.idx = 0
            export_inputs = loader()
        else:
            logger.warning("Using dummy data for calibration!")
            export_inputs = (torch.randn(1, 6, 3, 256, 704).cuda(),)
    else:
        export_inputs = (torch.randn(1, 6, 3, 256, 704).cuda(),)

    # 3. Export ONNX
    # We move to CPU for export to avoid RuntimeError: Expected all tensors to be on the same device
    # which often happens during ONNX export optimization passes (eval_peephole)
    logger.info("Moving model to CPU for final export...")
    backbone_wrapper.cpu()
    export_inputs = tuple(x.cpu() for x in export_inputs)
    
    # Ensure all quantizer scales are on CPU
    for m in backbone_wrapper.modules():
        if isinstance(m, quant_nn.TensorQuantizer):
            if hasattr(m, '_amax') and m._amax is not None:
                m._amax = m._amax.cpu()

    logger.info("Exporting ONNX...")
    torch.onnx.export(
        backbone_wrapper, export_inputs, args.save_onnx,
        input_names=["img"], output_names=["feature"],
        opset_version=13, 
        do_constant_folding=True, # 尝试开启，减少冗余 Cast
        training=torch.onnx.TrainingMode.EVAL,
        dynamic_axes={'img': {0: 'batch_size'}, 'feature': {0: 'batch_size'}}
    )
    
    # 尝试使用 onnx-simplifier 进一步清理模型
    try:
        import onnxsim
        logger.info("Simplifying ONNX with onnxsim...")
        model_onnx = onnx.load(args.save_onnx)
        # skip_constant_folding=True 保护 QDQ 的 scale
        model_simp, check = onnxsim.simplify(model_onnx, skip_constant_folding=False)
        if check:
            onnx.save(model_simp, args.save_onnx)
            logger.info("Simplified ONNX saved.")
    except Exception as e:
        logger.warning(f"Failed to simplify ONNX: {e}")

    logger.info(f"Done: {args.save_onnx}")
