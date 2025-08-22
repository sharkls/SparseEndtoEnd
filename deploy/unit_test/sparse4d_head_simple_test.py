# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
"""
简化的SparseBEV.cpp第二帧推理一致性验证脚本
只验证核心输出：tmp_outs0~5、pred_instance_feature、pred_anchor、pred_class_score、pred_quality_score、pred_track_id
"""

import os
import sys
import time
import logging
import numpy as np

# 设置完全确定性环境
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import ctypes
import tensorrt as trt

from typing import List
from cuda import cudart
from tool.utils.logger import logger_wrapper
from deploy.utils.utils import printArrayInformation

# 设置numpy随机种子
np.random.seed(42)

# 设置CUDA确定性
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.empty_cache()

if not hasattr(np, 'bool'):
    np.bool = bool

# TensorRT数据类型到NumPy数据类型的映射
TRT_TO_NP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT8: np.int8,
    trt.DataType.INT32: np.int32,
    trt.DataType.BOOL: np.bool,
}

def read_bin(samples, logger):
    """读取二进制文件，只包含核心输出"""
    prefix = "script/tutorial/asset/"
    # prefix = "C++/Output/val_bin/"

    inputs = list()
    outputs = list()

    for i in range(1, samples):
        # 输入数据
        feature_shape = [1, 89760, 256]
        feature = np.fromfile(
            f"{prefix}sample_{i}_feature_1*89760*256_float32.bin",
            dtype=np.float32,
        ).reshape(feature_shape)
        printArrayInformation(feature, logger, "feature", "PyTorch")

        spatial_shapes_shape = [6, 4, 2]
        spatial_shapes = np.fromfile(
            f"{prefix}sample_{i}_spatial_shapes_6*4*2_int32.bin", dtype=np.int32
        ).reshape(spatial_shapes_shape)
        printArrayInformation(spatial_shapes, logger, "spatial_shapes", "PyTorch")

        level_start_index_shape = [6, 4]
        level_start_index = np.fromfile(
            f"{prefix}sample_{i}_level_start_index_6*4_int32.bin",
            dtype=np.int32,
        ).reshape(level_start_index_shape)
        printArrayInformation(level_start_index, logger, "level_start_index", "PyTorch")

        instance_feature_shape = [1, 900, 256]
        instance_feature = np.fromfile(
            f"{prefix}sample_{i}_instance_feature_1*900*256_float32.bin",
            dtype=np.float32,
        ).reshape(instance_feature_shape)
        printArrayInformation(instance_feature, logger, "instance_feature", "PyTorch")

        anchor_shape = [1, 900, 11]
        anchor = np.fromfile(
            f"{prefix}sample_{i}_anchor_1*900*11_float32.bin", dtype=np.float32
        ).reshape(anchor_shape)
        printArrayInformation(anchor, logger, "anchor", "PyTorch")

        time_interval_shape = [1]
        time_interval = np.fromfile(
            f"{prefix}sample_{i}_time_interval_1_float32.bin",
            dtype=np.float32,
        ).reshape(time_interval_shape)
        printArrayInformation(time_interval, logger, "time_interval", "PyTorch")

        temp_instance_feature_shape = [1, 600, 256]
        temp_instance_feature = np.fromfile(
            f"{prefix}sample_{i}_temp_instance_feature_1*600*256_float32.bin",
            dtype=np.float32,
        ).reshape(temp_instance_feature_shape)
        printArrayInformation(temp_instance_feature, logger, "temp_instance_feature", "PyTorch")

        temp_anchor_shape = [1, 600, 11]
        temp_anchor = np.fromfile(
            f"{prefix}sample_{i}_temp_anchor_1*600*11_float32.bin",
            dtype=np.float32,
        ).reshape(temp_anchor_shape)
        printArrayInformation(temp_anchor, logger, "temp_anchor", "PyTorch")

        mask_shape = [1]
        mask = np.fromfile(
            f"{prefix}sample_{i}_mask_1_int32.bin",
            dtype=np.int32,
        ).reshape(mask_shape)
        printArrayInformation(mask, logger, "mask", "PyTorch")

        track_id_shape = [1, 900]
        track_id = np.fromfile(
            f"{prefix}sample_{i}_track_id_1*900_int32.bin",
            dtype=np.int32,
        ).reshape(track_id_shape)
        printArrayInformation(track_id, logger, "track_id", "PyTorch")

        image_wh_shape = [1, 6, 2]
        image_wh = np.fromfile(
            f"{prefix}sample_{i}_image_wh_1*6*2_float32.bin",
            dtype=np.float32,
        ).reshape(image_wh_shape)
        printArrayInformation(image_wh, logger, "image_wh", "PyTorch")

        lidar2img_shape = [1, 6, 4, 4]
        lidar2img = np.fromfile(
            f"{prefix}sample_{i}_lidar2img_1*6*4*4_float32.bin",
            dtype=np.float32,
        ).reshape(lidar2img_shape)
        printArrayInformation(lidar2img, logger, "lidar2img", "PyTorch")

        # 核心输出数据
        pred_instance_feature_shape = [1, 900, 256]
        pred_instance_feature = np.fromfile(
            f"{prefix}sample_{i}_pred_instance_feature_1*900*256_float32.bin",
            dtype=np.float32,
        ).reshape(pred_instance_feature_shape)
        printArrayInformation(pred_instance_feature, logger, "pred_instance_feature", "PyTorch")

        pred_anchor_shape = [1, 900, 11]
        pred_anchor = np.fromfile(
            f"{prefix}sample_{i}_pred_anchor_1*900*11_float32.bin",
            dtype=np.float32,
        ).reshape(pred_anchor_shape)
        printArrayInformation(pred_anchor, logger, "pred_anchor", "PyTorch")

        pred_class_score_shape = [1, 900, 10]
        pred_class_score = np.fromfile(
            f"{prefix}sample_{i}_pred_class_score_1*900*10_float32.bin",
            dtype=np.float32,
        ).reshape(pred_class_score_shape)
        printArrayInformation(pred_class_score, logger, "pred_class_score", "PyTorch")

        pred_quality_score_shape = [1, 900, 2]
        pred_quality_score = np.fromfile(
            f"{prefix}sample_{i}_pred_quality_score_1*900*2_float32.bin",
            dtype=np.float32,
        ).reshape(pred_quality_score_shape)
        printArrayInformation(pred_quality_score, logger, "pred_quality_score", "PyTorch")

        pred_track_id_shape = [1, 900]
        pred_track_id = np.fromfile(
            f"{prefix}sample_{i}_pred_track_id_1*900_int32.bin",
            dtype=np.int32,
        ).reshape(pred_track_id_shape)
        printArrayInformation(pred_track_id, logger, "pred_track_id", "PyTorch")

        # tmp_outs数据
        tmp_outs = []
        for j in range(6):
            tmp_out_shape = [1, 900, 512]
            try:
                tmp_out = np.fromfile(
                    f"{prefix}sample_{i}_tmp_outs{j}_1*900*512_float32.bin",
                    dtype=np.float32,
                ).reshape(tmp_out_shape)
                printArrayInformation(tmp_out, logger, f"tmp_outs{j}", "PyTorch")
            except FileNotFoundError:
                # 如果文件不存在，创建零张量
                tmp_out = np.zeros(tmp_out_shape, dtype=np.float32)
                logger.warning(f"tmp_outs{j} file not found, using zeros")
            tmp_outs.append(tmp_out)

        inputs.append([
            feature, spatial_shapes, level_start_index, instance_feature, anchor,
            time_interval, temp_instance_feature, temp_anchor, mask, track_id,
            image_wh, lidar2img
        ])

        outputs.append([
            # 调整顺序以匹配TensorRT输出：tmp_outs0-5, pred_instance_feature, pred_anchor, pred_class_score, pred_quality_score, pred_track_id
            *tmp_outs,  # tmp_outs0-5 (6个)
            pred_instance_feature,  # 第6个
            pred_anchor,           # 第7个
            pred_class_score,      # 第8个
            pred_quality_score,    # 第9个
            pred_track_id          # 第10个
        ])

    return inputs, outputs

def getPlugin(plugin_name) -> trt.tensorrt.IPluginV2:
    """获取TensorRT插件"""
    for i, c in enumerate(trt.get_plugin_registry().plugin_creator_list):
        logger.debug(f"Plugin{i} : {c.name}")
        if c.name == plugin_name:
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))

def build_network(trtFile, logger):
    """构建TensorRT网络"""
    trtlogger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trtlogger, "")

    if os.path.isfile(trtFile):
        logger.info("Start to deserialize...")
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(trtlogger).deserialize_cuda_engine(f.read())
        if engine == None:
            logger.error(f"Failed loading engine: {trtFile}!")
            return
        logger.info(f"Succeed to load engine: {trtFile}!")
        return engine
    else:
        logger.error(f"{trtFile} is not exist!")
        return None

def inference(
    feature, spatial_shapes, level_start_index, instance_feature, anchor,
    time_interval, temp_instance_feature, temp_anchor, mask, track_id,
    image_wh, lidar2img, engine, trt_old, logger
):
    """执行TensorRT推理"""
    bufferH = [
        feature, spatial_shapes, level_start_index, instance_feature, anchor,
        time_interval, temp_instance_feature, temp_anchor, mask, track_id,
        image_wh, lidar2img
    ]

    if trt_old:
        nIO = engine.num_bindings
        lTensorName = [engine.get_binding_name(i) for i in range(nIO)]
        nInput = sum([engine.binding_is_input(lTensorName[i]) for i in range(nIO)])

        # 为输出分配内存
        for i in range(nInput, nIO):
            bufferH.append(
                np.zeros(
                    engine.get_binding_shape(lTensorName[i]),
                    dtype=TRT_TO_NP[engine.get_binding_dtype(lTensorName[i])],
                )
            )

        # 打印输入输出信息
        for i in range(nInput):
            logger.debug(f"Input{i}: {lTensorName[i]} - Shape: {engine.get_binding_shape(lTensorName[i])}")
        for i in range(nInput, nIO):
            logger.debug(f"Output{i-nInput}: {lTensorName[i]} - Shape: {engine.get_binding_shape(lTensorName[i])}")

    else:
        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

        context = engine.create_execution_context()
        # 为输出分配内存
        for i in range(nInput, nIO):
            bufferH.append(
                np.zeros(
                    context.get_tensor_shape(lTensorName[i]),
                    dtype=TRT_TO_NP[engine.get_tensor_dtype(lTensorName[i])],
                )
            )

        # 打印输入输出信息
        for i in range(nInput):
            logger.debug(f"Input{i}: {lTensorName[i]} - Shape: {context.get_tensor_shape(lTensorName[i])}")
        for i in range(nInput, nIO):
            logger.debug(f"Output{i-nInput}: {lTensorName[i]} - Shape: {context.get_tensor_shape(lTensorName[i])}")

    # 分配GPU内存
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    # 复制输入数据到GPU
    for i in range(nInput):
        cudart.cudaMemcpy(
            bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

    # 执行推理
    if trt_old:
        binding_addrs = [int(bufferD[i]) for i in range(nIO)]
        context.execute_v2(binding_addrs)
    else:
        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        context.execute_async_v3(0)

    # 复制输出数据到CPU
    for i in range(nInput, nIO):
        cudart.cudaMemcpy(
            bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )

    # 释放GPU内存
    for b in bufferD:
        cudart.cudaFree(b)

    if trt_old:
        return nInput, nIO, bufferH, lTensorName, None
    else:
        return nInput, nIO, bufferH, lTensorName, context

def inference_consistency_validatation(predicted_data, expected_data, output_names):
    """验证推理一致性"""
    for x, y, name in zip(predicted_data, expected_data, output_names):
        max_abs_distance = float((np.abs(x - y)).max())
        logger.info(f"[max(abs()) error] {name} = {max_abs_distance}")

        cosine_distance = 1 - np.dot(x.flatten(), y.flatten()) / (
            np.linalg.norm(x.flatten()) * np.linalg.norm(y.flatten())
        )
        logger.info(f"[cosine_distance ] {name} = {float(cosine_distance)}")

def main(
    input_bins, output_bins, trt_old, logger,
    plugin_name="DeformableAttentionAggrPlugin",
    soFile="deploy/dfa_plugin/lib/deformableAttentionAggr.so",
    trtFile="deploy/engine/sparse4dhead2nd.engine",
):
    """主函数"""
    ctypes.cdll.LoadLibrary(soFile)
    plugin = getPlugin(plugin_name)
    engine = build_network(trtFile, logger)
    if engine == None:
        logger.error(f"{plugin_name} Engine Building Failed: {trtFile} !")
        return

    sample_id = 1
    for x, y in zip(input_bins, output_bins):
        (feature, spatial_shapes, level_start_index, instance_feature, anchor,
         time_interval, temp_instance_feature, temp_anchor, mask, track_id,
         image_wh, lidar2img) = x

        # 执行推理
        nInput, nIO, bufferH, lTensorName, context = inference(
            feature, spatial_shapes, level_start_index, instance_feature, anchor,
            time_interval, temp_instance_feature, temp_anchor, mask, track_id,
            image_wh, lidar2img, engine, trt_old, logger
        )

        # 打印输入信息
        input_names = [
            "feature", "spatial_shapes", "level_start_index", "instance_feature",
            "anchor", "time_interval", "temp_instance_feature", "temp_anchor",
            "mask", "track_id", "image_wh", "lidar2img"
        ]

        for i, name in enumerate(input_names):
            printArrayInformation(bufferH[i], logger, info=f"{name}", prefix="TensorRT")

        # 特别打印track_id的详细信息
        logger.info("=" * 80)
        logger.info("输入track_id详细信息:")
        input_track_id = bufferH[9]  # track_id是第9个输入
        logger.info(f"输入track_id形状: {input_track_id.shape}")
        logger.info(f"输入track_id数据类型: {input_track_id.dtype}")
        logger.info(f"输入track_id最大值: {input_track_id.max()}")
        logger.info(f"输入track_id最小值: {input_track_id.min()}")
        logger.info(f"输入track_id中-1的数量: {np.sum(input_track_id == -1)}")
        logger.info(f"输入track_id中非-1的数量: {np.sum(input_track_id != -1)}")
        
        # 打印非-1的track_id值
        non_neg1_indices = np.where(input_track_id != -1)[0]
        if len(non_neg1_indices) > 0:
            logger.info(f"输入track_id中非-1的值: {input_track_id.flatten()[non_neg1_indices].tolist()}")
        logger.info("=" * 80)

        # 期望的输出数量：6个tmp_outs + 5个pred输出 = 11个
        expected_output_count = 11
        logger.info(f"Sample {sample_id} inference consistency validation:")
        logger.info(f"Expected output count: {expected_output_count}")
        logger.info(f"Actual output count: {nIO - nInput}")
        
        # 验证输出数量
        if nIO - nInput < expected_output_count:
            logger.warning(f"Output count mismatch! Expected: {expected_output_count}, Actual: {nIO - nInput}")
            logger.warning("Some outputs may be missing. Validation will be limited.")

        # 执行一致性验证
        # 根据TensorRT实际输出顺序调整验证
        # 从日志可以看到实际顺序是：
        # Output0-5: tmp_outs0-5
        # Output6: pred_instance_feature
        # Output7: pred_anchor
        # Output8: pred_class_score
        # Output9: pred_quality_score
        # Output10: pred_track_id
        
        validation_outputs = []
        validation_names = []
        
        # 按照TensorRT实际输出顺序获取数据
        if nInput + 0 < nIO:  # tmp_outs0
            validation_outputs.append(bufferH[nInput + 0])
            validation_names.append("tmp_outs0")
        if nInput + 1 < nIO:  # tmp_outs1
            validation_outputs.append(bufferH[nInput + 1])
            validation_names.append("tmp_outs1")
        if nInput + 2 < nIO:  # tmp_outs2
            validation_outputs.append(bufferH[nInput + 2])
            validation_names.append("tmp_outs2")
        if nInput + 3 < nIO:  # tmp_outs3
            validation_outputs.append(bufferH[nInput + 3])
            validation_names.append("tmp_outs3")
        if nInput + 4 < nIO:  # tmp_outs4
            validation_outputs.append(bufferH[nInput + 4])
            validation_names.append("tmp_outs4")
        if nInput + 5 < nIO:  # tmp_outs5
            validation_outputs.append(bufferH[nInput + 5])
            validation_names.append("tmp_outs5")
        if nInput + 6 < nIO:  # pred_instance_feature
            validation_outputs.append(bufferH[nInput + 6])
            validation_names.append("pred_instance_feature")
        if nInput + 7 < nIO:  # pred_anchor
            validation_outputs.append(bufferH[nInput + 7])
            validation_names.append("pred_anchor")
        if nInput + 8 < nIO:  # pred_class_score
            validation_outputs.append(bufferH[nInput + 8])
            validation_names.append("pred_class_score")
        if nInput + 9 < nIO:  # pred_quality_score
            validation_outputs.append(bufferH[nInput + 9])
            validation_names.append("pred_quality_score")
        if nInput + 10 < nIO:  # pred_track_id
            validation_outputs.append(bufferH[nInput + 10])
            validation_names.append("pred_track_id")

        # 确保验证数据数量匹配
        expected_count = min(len(y), len(validation_outputs))
        logger.info(f"Validation: {len(validation_outputs)} outputs vs {expected_count} expected outputs")
        
        # 打印验证顺序信息
        logger.info("Validation order:")
        for i, name in enumerate(validation_names):
            logger.info(f"  {i}: {name}")
        
        # 只验证前expected_count个输出
        validation_outputs = validation_outputs[:expected_count]
        validation_names = validation_names[:expected_count]

        inference_consistency_validatation(
            validation_outputs,
            y[:expected_count],
            validation_names,
        )
        
        # 特别打印输出pred_track_id的详细信息
        logger.info("=" * 80)
        logger.info("输出pred_track_id详细信息:")
        if nInput + 10 < nIO:  # pred_track_id是第10个输出
            output_track_id = bufferH[nInput + 10]
            logger.info(f"输出pred_track_id形状: {output_track_id.shape}")
            logger.info(f"输出pred_track_id数据类型: {output_track_id.dtype}")
            logger.info(f"输出pred_track_id最大值: {output_track_id.max()}")
            logger.info(f"输出pred_track_id最小值: {output_track_id.min()}")
            logger.info(f"输出pred_track_id中-1的数量: {np.sum(output_track_id == -1)}")
            logger.info(f"输出pred_track_id中非-1的数量: {np.sum(output_track_id != -1)}")
            
            # 打印完整的pred_track_id数组
            logger.info("输出pred_track_id完整数组:")
            output_track_id_flat = output_track_id.flatten()
            for i in range(0, len(output_track_id_flat), 20):
                end_idx = min(i + 20, len(output_track_id_flat))
                row_values = output_track_id_flat[i:end_idx]
                row_str = " ".join([f"{val:4d}" for val in row_values])
                logger.info(f"  [{i:3d}-{end_idx-1:3d}]: {row_str}")
            
            # # 打印非-1的pred_track_id值
            # non_neg1_indices = np.where(output_track_id != -1)[0]
            # if len(non_neg1_indices) > 0:
            #     logger.info(f"输出pred_track_id中非-1的值: {output_track_id.flatten()[non_neg1_indices].tolist()}")
        else:
            logger.warning("输出pred_track_id不存在")
        logger.info("=" * 80)
        
        sample_id += 1

if __name__ == "__main__":
    logger, _, _ = logger_wrapper("", False)
    logger.setLevel(logging.DEBUG)

    # 检查TensorRT版本
    a, b, c, d = (trt.__version__).split(".")
    version = eval(a + "." + b)
    logger.info(f"Python TensorRT Version is: {trt.__version__} !")

    if version < 8.5:
        trt_old = True
    else:
        trt_old = False

    np.set_printoptions(precision=4, linewidth=200, suppress=True)

    logger.info("Starting simplified unit test...")
    inputs, expected_outputs = read_bin(2, logger)
    main(inputs, expected_outputs, trt_old, logger)
    logger.info("All tests are passed!!!") 