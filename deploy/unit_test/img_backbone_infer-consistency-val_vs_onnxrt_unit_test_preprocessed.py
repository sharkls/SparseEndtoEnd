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
        # default="script/tutorial/asset/sample_0_imgs_1*6*3*256*704_float32.bin",
        default="C++/Output/val_bin/sample_0_imgs_1*6*3*256*704_float32.bin",
        help="预处理后的图像数据文件路径"
    )
    parser.add_argument(
        "--expected_feature",
        type=str,
        default="C++/Output/val_bin/sample_0_feature_1*89760*256_float32.bin",
        help="期望的特征输出文件路径"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.11,
        help="最大绝对距离容差阈值"
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


def detailed_error_analysis(output1, output2, logger, name1="Output1", name2="Output2"):
    """
    详细的误差分析
    
    Args:
        output1: 第一个输出
        output2: 第二个输出
        logger: 日志记录器
        name1: 第一个输出的名称
        name2: 第二个输出的名称
    """
    logger.info(f"\n=== {name1} vs {name2} 详细误差分析 ===")
    
    # 基本统计
    diff = output1 - output2
    abs_diff = np.abs(diff)
    
    logger.info(f"数据形状: {output1.shape}")
    logger.info(f"总元素数量: {output1.size}")
    
    # 误差统计
    logger.info(f"最大绝对误差: {np.max(abs_diff):.6f}")
    logger.info(f"最小绝对误差: {np.min(abs_diff):.6f}")
    logger.info(f"平均绝对误差: {np.mean(abs_diff):.6f}")
    logger.info(f"绝对误差标准差: {np.std(abs_diff):.6f}")
    
    # 误差分布
    error_ranges = [
        (0, 1e-6, "极小误差 (0-1e-6)"),
        (1e-6, 1e-5, "很小误差 (1e-6-1e-5)"),
        (1e-5, 1e-4, "小误差 (1e-5-1e-4)"),
        (1e-4, 1e-3, "中等误差 (1e-4-1e-3)"),
        (1e-3, 1e-2, "较大误差 (1e-3-1e-2)"),
        (1e-2, 1e-1, "大误差 (1e-2-1e-1)"),
        (1e-1, float('inf'), "极大误差 (>1e-1)")
    ]
    
    logger.info(f"\n误差分布统计:")
    for min_val, max_val, desc in error_ranges:
        if max_val == float('inf'):
            count = np.sum(abs_diff >= min_val)
        else:
            count = np.sum((abs_diff >= min_val) & (abs_diff < max_val))
        percentage = count / output1.size * 100
        logger.info(f"  {desc}: {count} 个元素 ({percentage:.2f}%)")
    
    # 找出最大误差的位置
    max_error_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    max_error_value = abs_diff[max_error_idx]
    logger.info(f"\n最大误差位置: {max_error_idx}")
    logger.info(f"最大误差值: {max_error_value:.6f}")
    logger.info(f"对应值: {name1}[{max_error_idx}] = {output1[max_error_idx]:.6f}")
    logger.info(f"对应值: {name2}[{max_error_idx}] = {output2[max_error_idx]:.6f}")
    
    return diff, abs_diff


def provide_optimization_suggestions(cosine_distance, max_abs_distance, logger):
    """
    提供优化建议
    
    Args:
        cosine_distance: 余弦距离
        max_abs_distance: 最大绝对距离
        logger: 日志记录器
    """
    logger.info(f"\n=== 优化建议 ===")
    
    if cosine_distance < 1e-6 and max_abs_distance < 0.01:
        logger.info("✓ 推理一致性很好，无需额外优化")
        return
    
    if cosine_distance >= 1e-3:
        logger.warning("⚠ 余弦距离较大，可能存在数值稳定性问题")
        logger.info("建议:")
        logger.info("  1. 检查TensorRT引擎的精度设置")
        logger.info("  2. 考虑使用FP32而不是FP16")
        logger.info("  3. 检查模型权重是否一致")
    
    if max_abs_distance >= 0.1:
        logger.warning("⚠ 最大绝对距离较大，可能存在精度损失")
        logger.info("建议:")
        logger.info("  1. 检查TensorRT的精度模式 (FP16/FP32)")
        logger.info("  2. 调整TensorRT的优化级别")
        logger.info("  3. 检查是否有数值溢出")
        logger.info("  4. 考虑重新构建TensorRT引擎")
    
    logger.info("\nTensorRT精度调优建议:")
    logger.info("  1. 使用 builder.max_workspace_size 增加工作空间")
    logger.info("  2. 设置 builder.fp16_mode = False 强制使用FP32")
    logger.info("  3. 调整 builder.optimization_level")
    logger.info("  4. 检查输入数据的数值范围")


def main():
    args = parse_args()
    logger, _, _ = set_logger(args.log, save_file=False)
    logger.setLevel(logging.DEBUG)

    logger.info("基于预处理后数据的 Sparse4d Backbone 推理一致性验证!")
    logger.info(f"输入图像: {args.input_img}")
    logger.info(f"期望特征: {args.expected_feature}")
    logger.info(f"容差阈值: {args.tolerance}")

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

    # 详细误差分析
    diff_pytorch_trt, abs_diff_pytorch_trt = detailed_error_analysis(
        output1, output2, logger, "PyTorch", "TensorRT"
    )

    # 验证一致性
    logger.info("验证推理一致性...")
    
    # 余弦距离验证 (参考原始代码的阈值 1e-3)
    cosine_threshold = 1e-3
    if cosine_distance < cosine_threshold:
        logger.info(f"✓ 余弦距离验证通过: {cosine_distance:.6f} < {cosine_threshold}")
    else:
        logger.error(f"✗ 余弦距离验证失败: {cosine_distance:.6f} >= {cosine_threshold}")
    
    # 最大绝对距离验证 (使用用户指定的阈值)
    max_abs_threshold = args.tolerance
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
            
        # 详细分析PyTorch与期望特征的差异
        diff_pytorch_expected, abs_diff_pytorch_expected = detailed_error_analysis(
            output1, expected_feature, logger, "PyTorch", "期望特征"
        )
    else:
        logger.warning(f"输出形状不匹配: PyTorch {output1.shape} vs 期望 {expected_feature.shape}")

    # 提供优化建议
    provide_optimization_suggestions(cosine_distance, max_abs_distance, logger)

    logger.info("推理一致性验证完成!")


if __name__ == "__main__":
    main() 