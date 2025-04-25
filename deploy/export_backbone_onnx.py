# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import logging
import argparse

import onnx
from onnxsim import simplify

import torch
from torch import nn
from typing import Optional, Dict, Any

from modules.sparse4d_detector import *
from tool.utils.logger import set_logger
from tool.utils.config import read_cfg


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Deploy SparseEND2END Backbone!")
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
        default="deploy/onnx/export_backbone_onnx.log",
    )
    parser.add_argument(
        "--save_onnx",
        type=str,
        default="deploy/onnx/sparse4dbackbone.onnx",
    )
    args = parser.parse_args()
    return args


# 封装Sparse4DBackbone
class Sparse4DBackbone(nn.Module):
    def __init__(self, model):
        super(Sparse4DBackbone, self).__init__()
        self.model = model

    def forward(self, img):

        feature, spatial_shapes, level_start_index = self.model.extract_feat(img)

        return feature


# 构建模型
def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 创建保存onnx模型的目录
    os.makedirs(os.path.dirname(args.save_onnx), exist_ok=True)
    # 设置日志
    logger, console_handler, file_handler = set_logger(args.log, True)  # 创建logger, 控制台处理器, 文件处理器
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    logger.info("Export Sparse4d Backbone Onnx...")

    # 读取配置文件
    cfg = read_cfg(args.cfg)
    # 构建模型
    model = build_module(cfg["model"])
    # 加载模型参数
    checkpoint = args.ckpt
    _ = model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)
    model.eval()

    # 设置输入参数
    BS = 1
    NUMS_CAM = 6
    C = 3
    INPUT_H = 256
    INPUT_W = 704
    dummy_img = torch.randn(BS, NUMS_CAM, C, INPUT_H, INPUT_W).cuda()

    # 封装Sparse4DBackbone
    backbone = Sparse4DBackbone(model).cuda()

    # 导出onnx模型
    with torch.no_grad():
        torch.onnx.export(
            backbone,                   # 模型
            (dummy_img,),               # 输入
            args.save_onnx,             # 导出onnx的保存路径
            input_names=["img"],        # 输入名称
            output_names=[              # 输出名称
                "feature",
            ],
            opset_version=15,           # onnx算子集版本
            do_constant_folding=True,  # 常量折叠
            verbose=False,             # 是否打印详细信息
        )
        # 简化onnx模型
        onnx_orig = onnx.load(args.save_onnx)
        onnx_simp, check = simplify(onnx_orig)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_simp, args.save_onnx)
        logger.info(f'🚀 Export onnx completed. ONNX saved in "{args.save_onnx}" 🤗.')
