#!/usr/bin/env python3
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

import os
import sys
import argparse
import tensorrt as trt
import logging
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Build Sparse4D Backbone TensorRT Engine with High Precision")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="deploy/onnx/sparse4dbackbone.onnx",
        help="ONNX模型路径"
    )
    parser.add_argument(
        "--engine_path",
        type=str,
        default="deploy/engine/sparse4dbackbone_highprec.engine",
        help="TensorRT引擎保存路径"
    )
    parser.add_argument(
        "--workspace_size",
        type=int,
        default=8192,
        help="工作空间大小(MB)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="精度模式"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["verbose", "info", "warning", "error"],
        default="info",
        help="日志级别"
    )
    return parser.parse_args()

def set_logger(log_level):
    """设置日志级别"""
    level_map = {
        "verbose": trt.Logger.VERBOSE,
        "info": trt.Logger.INFO,
        "warning": trt.Logger.WARNING,
        "error": trt.Logger.ERROR
    }
    return trt.Logger(level_map[log_level])

def build_engine(onnx_path, engine_path, workspace_size, precision, logger):
    """构建TensorRT引擎"""
    
    # 创建builder
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # 设置工作空间大小
    config.max_workspace_size = workspace_size * (1 << 20)  # 转换为字节
    
    # 设置精度模式
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ 启用FP16精度模式")
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        print("✓ 启用INT8精度模式")
    else:
        # 强制使用FP32
        config.set_flag(trt.BuilderFlag.DIRECT_IO)
        print("✓ 使用FP32精度模式")
    
    # 禁用TF32以提高精度
    config.set_flag(trt.BuilderFlag.DIRECT_IO)
    
    # 设置优化级别
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    
    # 解析ONNX模型
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("✗ ONNX解析失败:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print(f"✓ ONNX模型解析成功: {onnx_path}")
    
    # 设置输入输出形状
    input_tensor = network.get_input(0)
    input_tensor.shape = [1, 6, 3, 256, 704]
    
    # 构建引擎
    print("开始构建TensorRT引擎...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("✗ 引擎构建失败")
        return False
    
    # 保存引擎
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"✓ TensorRT引擎构建成功: {engine_path}")
    print(f"引擎大小: {engine.num_layers} 层")
    print(f"最大工作空间: {workspace_size} MB")
    
    return True

def main():
    args = parse_args()
    
    # 检查ONNX文件是否存在
    if not os.path.exists(args.onnx_path):
        print(f"✗ ONNX文件不存在: {args.onnx_path}")
        return 1
    
    # 设置日志
    logger = set_logger(args.log_level)
    
    print("=== Sparse4D Backbone TensorRT引擎构建 ===")
    print(f"ONNX路径: {args.onnx_path}")
    print(f"引擎路径: {args.engine_path}")
    print(f"工作空间: {args.workspace_size} MB")
    print(f"精度模式: {args.precision}")
    
    # 构建引擎
    success = build_engine(
        args.onnx_path,
        args.engine_path,
        args.workspace_size,
        args.precision,
        logger
    )
    
    if success:
        print("\n🎉 引擎构建完成!")
        return 0
    else:
        print("\n❌ 引擎构建失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 