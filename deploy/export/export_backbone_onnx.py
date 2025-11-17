# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import sys
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
    
    # 如果指定了--fp16，将模型转换为FP16（如果同时指定--fp16和--fp32，--fp16优先）
    if args.fp16 and not args.fp32:
        logger.info("Converting model to FP16 for ONNX export...")
        model = model.half()  # 将模型转换为FP16
        model._export_fp16 = True  # 标记模型为FP16导出模式
        logger.info("Model converted to FP16. All floating-point inputs will use FP16 dtype.")
    else:
        logger.info("Exporting model with FP32 precision.")
        model._export_fp16 = False

    # 设置输入参数
    BS = 1
    NUMS_CAM = 6
    C = 3
    INPUT_H = 256
    INPUT_W = 704
    
    # 根据导出精度选择数据类型
    use_fp16 = getattr(model, '_export_fp16', False)
    float_dtype = torch.float16 if use_fp16 else torch.float32
    dummy_img = torch.randn(BS, NUMS_CAM, C, INPUT_H, INPUT_W).to(dtype=float_dtype).cuda()

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
            dynamic_axes={              # 动态轴设置
                'img': {0: 'batch_size'},
                'feature': {0: 'batch_size'}
            },
            keep_initializers_as_inputs=False,  # 不保留初始化器作为输入
            export_params=True,         # 导出模型参数
            training=torch.onnx.TrainingMode.EVAL,  # 设置为评估模式
        )
        # 简化onnx模型
        onnx_orig = onnx.load(args.save_onnx)
        onnx_simp, check = simplify(onnx_orig)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_simp, args.save_onnx)
        logger.info(f'🚀 Export onnx completed. ONNX saved in "{args.save_onnx}" 🤗.')
        
        # 验证导出的 ONNX 精度
        if args.fp16 and not args.fp32:
            logger.info("验证导出的 ONNX 是否为 FP16 精度...")
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "verify_onnx_fp16.py"), args.save_onnx],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    logger.info("✓ ONNX 模型验证为 FP16 精度")
                else:
                    logger.warning("⚠ ONNX 模型精度验证失败，请手动检查")
                    logger.warning(result.stderr)
            except Exception as e:
                logger.warning(f"⚠ 无法验证 ONNX 精度: {e}")
                logger.info("提示: 可以使用 'python deploy/verify_onnx_fp16.py <onnx_path>' 手动验证")
