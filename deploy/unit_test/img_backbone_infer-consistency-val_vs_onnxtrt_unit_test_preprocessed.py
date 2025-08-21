# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
"""
基于预处理后数据的 Backbone 推理一致性验证脚本
使用真实的图像数据与期望的特征输出进行误差计算
"""
import os
import torch
import argparse
import numpy as np
import tensorrt as trt
from cuda import cudart

import logging
from tool.utils.logger import set_logger

from modules.sparse4d_detector import *
from typing import Optional, Dict, Any
from tool.utils.config import read_cfg

# 修复NumPy兼容性问题
if not hasattr(np, 'bool'):
    np.bool = bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于预处理后数据的 Backbone 推理一致性验证!"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt/sparse4dv3_r50.pth",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="deploy/onnx/trt_consistencycheck_preprocessed.log",
    )
    parser.add_argument(
        "--trtengine",
        type=str,
        default="deploy/engine/sparse4dbackbone.engine",
    )
    parser.add_argument(
        "--input_img",
        type=str,
        default="script/tutorial/asset/sample_0_imgs_1*6*3*256*704_float32.bin",
        help="预处理后的图像数据文件路径"
    )
    parser.add_argument(
        "--expected_feature",
        type=str,
        default="script/tutorial/asset/sample_0_feature_1*89760*256_float32.bin",
        help="期望的特征输出文件路径"
    )
    args = parser.parse_args()
    return args


def load_preprocessed_data(input_img_path, expected_feature_path):
    """
    加载预处理后的数据
    
    Args:
        input_img_path: 预处理后的图像数据路径
        expected_feature_path: 期望的特征输出路径
    
    Returns:
        input_img: 图像数据 (1, 6, 3, 256, 704)
        expected_feature: 期望特征 (1, 89760, 256)
    """
    print(f"加载预处理后的图像数据: {input_img_path}")
    if not os.path.exists(input_img_path):
        raise FileNotFoundError(f"图像数据文件不存在: {input_img_path}")
    
    # 加载图像数据: (1, 6, 3, 256, 704)
    input_img = np.fromfile(input_img_path, dtype=np.float32).reshape(1, 6, 3, 256, 704)
    print(f"图像数据形状: {input_img.shape}")
    print(f"图像数据范围: [{input_img.min():.6f}, {input_img.max():.6f}]")
    
    print(f"加载期望特征数据: {expected_feature_path}")
    if not os.path.exists(expected_feature_path):
        raise FileNotFoundError(f"期望特征文件不存在: {expected_feature_path}")
    
    # 加载期望特征: (1, 89760, 256)
    expected_feature = np.fromfile(expected_feature_path, dtype=np.float32).reshape(1, 89760, 256)
    print(f"期望特征形状: {expected_feature.shape}")
    print(f"期望特征范围: [{expected_feature.min():.6f}, {expected_feature.max():.6f}]")
    
    return input_img, expected_feature


def build_network(trtFile):
    """构建TensorRT网络"""
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            return None
        return engine
    return None


def inference(
    engine,
    input,
    trt_old,
    logger,
):
    """TensorRT推理执行"""
    if trt_old:
        nIO = engine.num_bindings
        lTensorName = [engine.get_binding_name(i) for i in range(nIO)]
        nInput = sum([engine.binding_is_input(lTensorName[i]) for i in range(nIO)])

        bufferH = []
        bufferH.append(input)
        for i in range(nInput, nIO):
            bufferH.append(
                np.zeros(
                    engine.get_binding_shape(lTensorName[i]),
                    dtype=trt.nptype(engine.get_binding_dtype(lTensorName[i])),
                )
            )

        for j in range(nIO):
            logger.debug(
                f"Engine Binding name:{lTensorName[j]}, shape:{engine.get_binding_shape(lTensorName[j])}, type:{engine.get_binding_dtype(lTensorName[j])} ."
            )
            logger.debug(
                f"Compared Input Data{lTensorName[j]} shape:{bufferH[j].shape}, type:{bufferH[j].dtype}, nbytes:{bufferH[j].nbytes} ."
            )

    else:
        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(
            trt.TensorIOMode.INPUT
        )

        bufferH = []
        bufferH.append(input)
        context = engine.create_execution_context()
        for i in range(nInput, nIO):
            bufferH.append(
                np.zeros(
                    context.get_tensor_shape(lTensorName[i]),
                    dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i])),
                )
            )

        for j in range(nIO):
            logger.debug(
                f"Engine Binding name:{lTensorName[j]}, shape:{context.get_tensor_shape(lTensorName[j])}, type:{trt.nptype(engine.get_tensor_dtype(lTensorName[j]))} ."
            )
            logger.debug(
                f"Compared Input Data:{lTensorName[j]} shape:{bufferH[j].shape}, type:{bufferH[j].dtype} ."
            )

    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(
            bufferD[i],
            bufferH[i].ctypes.data,
            bufferH[i].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

    if trt_old:
        binding_addrs = [int(bufferD[i]) for i in range(nIO)]
        context.execute_v2(binding_addrs)
    else:
        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(
            bufferH[i].ctypes.data,
            bufferD[i],
            bufferH[i].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )

    for b in bufferD:
        cudart.cudaFree(b)

    return bufferH[-1]


def printArrayInformation(x, logger, info: str):
    """打印数组信息"""
    logger.debug(f"Name={info}")
    logger.debug(
        "\tSumAbs=%.3f, Max=%.3f, Min=%.3f"
        % (
            np.sum(abs(x)),
            np.max(x),
            np.min(x),
        )
    )


def trt_inference(
    engine,
    input,
    trt_old,
    logger,
):
    """TensorRT推理包装函数"""
    output = inference(
        engine,
        input,
        trt_old,
        logger,
    )
    return output


def model_infer(model, input_img):
    """PyTorch模型推理"""
    with torch.no_grad():
        feature, spatial_shapes, level_start_index = model.extract_feat(input_img)
    return feature.cpu().numpy()


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    """构建模块"""
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def calculate_errors(output1, output2, logger):
    """
    计算两种误差指标
    
    Args:
        output1: PyTorch输出
        output2: TensorRT输出
        logger: 日志记录器
    
    Returns:
        cosine_distance: 余弦距离
        max_abs_distance: 最大绝对距离
    """
    # 1. 余弦距离 (cosine_distance)
    cosine_distance = 1 - np.dot(output1.flatten(), output2.flatten()) / (
        np.linalg.norm(output1.flatten()) * np.linalg.norm(output2.flatten())
    )
    
    # 2. 最大绝对距离 (max_abs_distance)
    max_abs_distance = float((np.abs(output1 - output2)).max())
    
    logger.info(f"cosine_distance = {float(cosine_distance)}")
    logger.info(f"max(abs(a-b))   = {max_abs_distance}")
    
    return cosine_distance, max_abs_distance


def main():
    args = parse_args()
    logger, _, _ = set_logger(args.log, save_file=False)
    logger.setLevel(logging.DEBUG)

    logger.info("基于预处理后数据的 Sparse4d Backbone 推理一致性验证!")
    logger.info(f"输入图像: {args.input_img}")
    logger.info(f"期望特征: {args.expected_feature}")

    # 加载预处理后的数据
    try:
        input_img, expected_feature = load_preprocessed_data(args.input_img, args.expected_feature)
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return

    # 构建PyTorch模型
    cfg = read_cfg(args.cfg)
    model = build_module(cfg["model"])
    checkpoint = args.ckpt
    _ = model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)
    model = model.cuda().eval()

    # 构建TensorRT引擎
    trt_engine = build_network(args.trtengine)
    if trt_engine is None:
        logger.error(f"TensorRT引擎加载失败: {args.trtengine}")
        return

    # 执行推理
    logger.info("开始执行推理...")
    
    # PyTorch推理
    logger.info("执行PyTorch推理...")
    output1 = model_infer(model, torch.from_numpy(input_img).cuda())
    printArrayInformation(output1, logger, "PyTorch输出")
    
    # TensorRT推理
    logger.info("执行TensorRT推理...")
    output2 = trt_inference(trt_engine, input_img, False, logger)
    printArrayInformation(output2, logger, "TensorRT输出")

    # 计算误差
    logger.info("计算推理一致性误差...")
    cosine_distance, max_abs_distance = calculate_errors(output1, output2, logger)

    # 验证一致性
    logger.info("验证推理一致性...")
    
    # 余弦距离验证 (参考原始代码的阈值 1e-3)
    cosine_threshold = 1e-3
    if cosine_distance < cosine_threshold:
        logger.info(f"✓ 余弦距离验证通过: {cosine_distance:.6f} < {cosine_threshold}")
    else:
        logger.error(f"✗ 余弦距离验证失败: {cosine_distance:.6f} >= {cosine_threshold}")
    
    # 最大绝对距离验证 (参考原始代码的阈值 0.11)
    max_abs_threshold = 0.11
    if max_abs_distance < max_abs_threshold:
        logger.info(f"✓ 最大绝对距离验证通过: {max_abs_distance:.6f} < {max_abs_threshold}")
    else:
        logger.error(f"✗ 最大绝对距离验证失败: {max_abs_distance:.6f} >= {max_abs_threshold}")

    # 与期望特征比较
    logger.info("与期望特征进行比较...")
    if output1.shape == expected_feature.shape:
        # 计算与期望特征的误差
        expected_cosine_distance, expected_max_abs_distance = calculate_errors(
            output1, expected_feature, logger
        )
        
        logger.info("=== 与期望特征的比较结果 ===")
        logger.info(f"PyTorch输出与期望特征的余弦距离: {expected_cosine_distance:.6f}")
        logger.info(f"PyTorch输出与期望特征的最大绝对距离: {expected_max_abs_distance:.6f}")
        
        # 判断是否在可接受范围内
        if expected_max_abs_distance < 0.1:  # 设置一个合理的阈值
            logger.info("✓ PyTorch输出与期望特征基本一致")
        else:
            logger.warning("⚠ PyTorch输出与期望特征存在差异")
    else:
        logger.warning(f"输出形状不匹配: PyTorch {output1.shape} vs 期望 {expected_feature.shape}")

    logger.info("推理一致性验证完成!")


if __name__ == "__main__":
    main()