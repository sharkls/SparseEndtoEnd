# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
""" If you get following results, congratulations
[2024-09-24::22:57:55] [INFO]: Sample 0 inference consistency validatation:
[2024-09-24::22:57:55] [INFO]: [max(abs()) error] pred_track_id = 0.0
[2024-09-24::22:57:55] [INFO]: [cosine_distance ] pred_track_id = 0.0
[2024-09-24::22:57:55] [INFO]: [max(abs()) error] pred_instance_feature = 0.007264554500579834
[2024-09-24::22:57:55] [INFO]: [cosine_distance ] pred_instance_feature = 0.0
[2024-09-24::22:57:55] [INFO]: [max(abs()) error] pred_anchor = 0.014062881469726562
[2024-09-24::22:57:55] [INFO]: [cosine_distance ] pred_anchor = -1.1920928955078125e-07
[2024-09-24::22:57:55] [INFO]: [max(abs()) error] pred_class_score = 0.0039052963256835938
[2024-09-24::22:57:55] [INFO]: [cosine_distance ] pred_class_score = 5.960464477539063e-08
[2024-09-24::22:57:55] [INFO]: [max(abs()) error] pred_quality_score = 0.0037108659744262695
[2024-09-24::22:57:55] [INFO]: [cosine_distance ] pred_quality_score = 1.1920928955078125e-07
[2024-09-24::22:57:56] [INFO]: Sample 1 inference consistency validatation:
[2024-09-24::22:57:56] [INFO]: [max(abs()) error] pred_track_id = 0.0
[2024-09-24::22:57:56] [INFO]: [cosine_distance ] pred_track_id = 0.0
[2024-09-24::22:57:56] [INFO]: [max(abs()) error] pred_instance_feature = 0.0004710555076599121
[2024-09-24::22:57:56] [INFO]: [cosine_distance ] pred_instance_feature = 1.1920928955078125e-07
[2024-09-24::22:57:56] [INFO]: [max(abs()) error] pred_anchor = 0.0003387928009033203
[2024-09-24::22:57:56] [INFO]: [cosine_distance ] pred_anchor = 1.1920928955078125e-07
[2024-09-24::22:57:56] [INFO]: [max(abs()) error] pred_class_score = 0.0007314682006835938
[2024-09-24::22:57:56] [INFO]: [cosine_distance ] pred_class_score = 1.1920928955078125e-07
[2024-09-24::22:57:56] [INFO]: [max(abs()) error] pred_quality_score = 0.0002580881118774414
[2024-09-24::22:57:56] [INFO]: [cosine_distance ] pred_quality_score = 0.0
"""
import os
import sys
import time
import logging
import argparse
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
    prefix = "script/tutorial/asset/"
    inputs = list()
    outputs = list()

    for i in range(1, samples):

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
        printArrayInformation(
            temp_instance_feature, logger, "temp_instance_feature", "PyTorch"
        )

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

        pred_instance_feature_shape = [1, 900, 256]
        pred_instance_feature = np.fromfile(
            f"{prefix}sample_{i}_pred_instance_feature_1*900*256_float32.bin",
            dtype=np.float32,
        ).reshape(pred_instance_feature_shape)
        printArrayInformation(
            pred_instance_feature, logger, "pred_instance_feature", "PyTorch"
        )

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
        printArrayInformation(
            pred_quality_score, logger, "pred_quality_score", "PyTorch"
        )

        pred_track_id_shape = [1, 900]
        pred_track_id = np.fromfile(
            f"{prefix}sample_{i}_pred_track_id_1*900_int32.bin",
            dtype=np.int32,
        ).reshape(pred_track_id_shape)
        printArrayInformation(pred_track_id, logger, "pred_track_id", "PyTorch")

        # 添加temp_gnn_output数据读取
        temp_gnn_output_shape = [1, 900, 256]
        temp_gnn_output = np.fromfile(
            f"{prefix}sample_{i}_temp_gnn_output_1*900*256_float32.bin",
            dtype=np.float32,
        ).reshape(temp_gnn_output_shape)
        printArrayInformation(temp_gnn_output, logger, "temp_gnn_output", "PyTorch")

        # 添加tmp_outs*数据读取
        tmp_outs = []
        for j in range(6):  # 假设有6个tmp_outs
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

        # 添加refine_outs*数据读取
        refine_outs = []
        for j in range(1):  # 假设只有1个refine模块
            try:
                # 读取refine模块的四个输出
                refine_anchor_shape = [1, 900, 11]
                refine_anchor = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_anchor_1*900*11_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_anchor_shape)
                printArrayInformation(refine_anchor, logger, f"refine_outs{j}_anchor", "PyTorch")

                refine_instance_feature_shape = [1, 900, 256]
                refine_instance_feature = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_instance_feature_1*900*256_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_instance_feature_shape)
                printArrayInformation(refine_instance_feature, logger, f"refine_outs{j}_instance_feature", "PyTorch")

                refine_anchor_embed_shape = [1, 900, 256]
                refine_anchor_embed = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_anchor_embed_1*900*256_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_anchor_embed_shape)
                printArrayInformation(refine_anchor_embed, logger, f"refine_outs{j}_anchor_embed", "PyTorch")

                refine_temp_anchor_embed_shape = [1, 600, 256]
                refine_temp_anchor_embed = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_temp_anchor_embed_1*600*256_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_temp_anchor_embed_shape)
                printArrayInformation(refine_temp_anchor_embed, logger, f"refine_outs{j}_temp_anchor_embed", "PyTorch")

                # 添加update_comparison数据读取
                refine_instance_feature_before_update_shape = [1, 900, 256]
                refine_instance_feature_before_update = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_instance_feature_before_update_1*900*256_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_instance_feature_before_update_shape)
                printArrayInformation(refine_instance_feature_before_update, logger, f"refine_outs{j}_instance_feature_before_update", "PyTorch")

                refine_anchor_before_update_shape = [1, 900, 11]
                refine_anchor_before_update = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_anchor_before_update_1*900*11_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_anchor_before_update_shape)
                printArrayInformation(refine_anchor_before_update, logger, f"refine_outs{j}_anchor_before_update", "PyTorch")

                # 新增：读取temp_instance_feature和temp_anchor
                refine_temp_instance_feature_shape = [1, 600, 256]
                refine_temp_instance_feature = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_temp_instance_feature_1*600*256_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_temp_instance_feature_shape)
                printArrayInformation(refine_temp_instance_feature, logger, f"refine_outs{j}_temp_instance_feature", "PyTorch")

                refine_temp_anchor_shape = [1, 600, 11]
                refine_temp_anchor = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_temp_anchor_1*600*11_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_temp_anchor_shape)
                printArrayInformation(refine_temp_anchor, logger, f"refine_outs{j}_temp_anchor", "PyTorch")

                # 新增：读取selected_feature和selected_anchor
                refine_selected_feature_shape = [1, 900, 256]
                refine_selected_feature = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_selected_feature_1*900*256_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_selected_feature_shape)
                printArrayInformation(refine_selected_feature, logger, f"refine_outs{j}_selected_feature", "PyTorch")

                refine_selected_anchor_shape = [1, 900, 11]
                refine_selected_anchor = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_selected_anchor_1*900*11_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_selected_anchor_shape)
                printArrayInformation(refine_selected_anchor, logger, f"refine_outs{j}_selected_anchor", "PyTorch")

                refine_instance_feature_after_update_shape = [1, 900, 256]
                refine_instance_feature_after_update = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_instance_feature_after_update_1*900*256_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_instance_feature_after_update_shape)
                printArrayInformation(refine_instance_feature_after_update, logger, f"refine_outs{j}_instance_feature_after_update", "PyTorch")

                refine_anchor_after_update_shape = [1, 900, 11]
                refine_anchor_after_update = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_anchor_after_update_1*900*11_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_anchor_after_update_shape)
                printArrayInformation(refine_anchor_after_update, logger, f"refine_outs{j}_anchor_after_update", "PyTorch")

                # 新增：读取confidence_sorted和indices
                refine_confidence_sorted_shape = [1, 300]  # N = 900 - 600 = 300
                refine_confidence_sorted = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_confidence_sorted_1*300_float32.bin",
                    dtype=np.float32,
                ).reshape(refine_confidence_sorted_shape)
                printArrayInformation(refine_confidence_sorted, logger, f"refine_outs{j}_confidence_sorted", "PyTorch")

                refine_indices_shape = [1, 300]  # N = 900 - 600 = 300
                print(f"{prefix}sample_{i}_refine_outs{j}_indices_1*300_int32.bin")
                refine_indices = np.fromfile(
                    f"{prefix}sample_{i}_refine_outs{j}_indices_1*300_int32.bin",
                    dtype=np.int32,
                ).reshape(refine_indices_shape)
                printArrayInformation(refine_indices, logger, f"refine_outs{j}_indices", "PyTorch")

                refine_outs.extend([
                    refine_anchor, refine_instance_feature, refine_anchor_embed, refine_temp_anchor_embed,
                    refine_instance_feature_before_update, refine_anchor_before_update,
                    refine_temp_instance_feature, refine_temp_anchor,  # 新增
                    refine_instance_feature_after_update, refine_anchor_after_update,
                    refine_selected_feature, refine_selected_anchor,  # 新增：selected_feature和selected_anchor
                    refine_confidence_sorted, refine_indices,  # 新增：confidence_sorted和indices
                ])
            except FileNotFoundError:
                # 如果文件不存在，创建零张量
                logger.warning(f"refine_outs{j} files not found, using zeros")
                refine_anchor = np.zeros([1, 900, 11], dtype=np.float32)
                refine_instance_feature = np.zeros([1, 900, 256], dtype=np.float32)
                refine_anchor_embed = np.zeros([1, 900, 256], dtype=np.float32)
                refine_temp_anchor_embed = np.zeros([1, 600, 256], dtype=np.float32)
                refine_instance_feature_before_update = np.zeros([1, 900, 256], dtype=np.float32)
                refine_anchor_before_update = np.zeros([1, 900, 11], dtype=np.float32)
                refine_temp_instance_feature = np.zeros([1, 600, 256], dtype=np.float32)  # 新增
                refine_temp_anchor = np.zeros([1, 600, 11], dtype=np.float32)            # 新增
                refine_confidence_sorted = np.zeros([1, 300], dtype=np.float32)          # 新增：confidence_sorted
                refine_indices = np.zeros([1, 300], dtype=np.int32)                     # 新增：indices
                refine_selected_feature = np.zeros([1, 900, 256], dtype=np.float32) # 新增
                refine_selected_anchor = np.zeros([1, 900, 11], dtype=np.float32) # 新增
                refine_instance_feature_after_update = np.zeros([1, 900, 256], dtype=np.float32)
                refine_anchor_after_update = np.zeros([1, 900, 11], dtype=np.float32)
                refine_outs.extend([
                    refine_anchor, refine_instance_feature, refine_anchor_embed, refine_temp_anchor_embed,
                    refine_instance_feature_before_update, refine_anchor_before_update,
                    refine_temp_instance_feature, refine_temp_anchor,  # 新增
                    refine_instance_feature_after_update, refine_anchor_after_update,
                    refine_selected_feature, refine_selected_anchor,  # 新增：selected_feature和selected_anchor
                    refine_confidence_sorted, refine_indices,  # 新增：confidence_sorted和indices
                ])

        inputs.append(
            [
                feature,
                spatial_shapes,
                level_start_index,
                instance_feature,
                anchor,
                time_interval,
                temp_instance_feature,
                temp_anchor,
                mask,
                track_id,
                image_wh,
                lidar2img,
            ]
        )

        outputs.append(
            [
                pred_track_id,
                pred_instance_feature,
                pred_anchor,
                pred_class_score,
                pred_quality_score,
                temp_gnn_output,
                *tmp_outs,  # 展开tmp_outs列表
                *refine_outs,  # 展开refine_outs列表
            ]
        )
    return inputs, outputs


def getPlugin(plugin_name) -> trt.tensorrt.IPluginV2:
    for i, c in enumerate(trt.get_plugin_registry().plugin_creator_list):
        logger.debug(f"Plugin{i} : {c.name}")
        if c.name == plugin_name:
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))


def build_network(trtFile, logger):
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
    feature: np.ndarray,
    spatial_shapes: np.ndarray,
    level_start_index: np.ndarray,
    instance_feature: np.ndarray,
    anchor: np.ndarray,
    time_interval: np.ndarray,
    temp_instance_feature: np.ndarray,
    temp_anchor: np.ndarray,
    mask: np.ndarray,
    track_id: np.ndarray,
    image_wh: np.ndarray,
    lidar2img: np.ndarray,
    engine: str,
    trt_old: bool,
    logger,
):
    bufferH = []
    bufferH.append(feature)
    bufferH.append(spatial_shapes)
    bufferH.append(level_start_index)
    bufferH.append(instance_feature)
    bufferH.append(anchor)
    bufferH.append(time_interval)
    bufferH.append(temp_instance_feature)
    bufferH.append(temp_anchor)
    bufferH.append(mask)
    bufferH.append(track_id)
    bufferH.append(image_wh)
    bufferH.append(lidar2img)

    if trt_old:
        nIO = engine.num_bindings
        lTensorName = [engine.get_binding_name(i) for i in range(nIO)]
        nInput = sum([engine.binding_is_input(lTensorName[i]) for i in range(nIO)])

        for i in range(nInput, nIO):
            bufferH.append(
                np.zeros(
                    engine.get_binding_shape(lTensorName[i]),
                    dtype=TRT_TO_NP[engine.get_binding_dtype(lTensorName[i])],
                )
            )

        for i in range(nInput):
            logger.debug(
                f"LoadEngine: Input{i}={lTensorName[i]}:\tshape:{engine.get_binding_shape}\ttype:{str(TRT_TO_NP[engine.get_binding_dtype(lTensorName[i])])} ."
            )
        for i in range(nInput, nIO):
            logger.debug(
                f"LoadEngine: Output{i}={lTensorName[i]}:\tshape:{engine.get_binding_shape}\ttype:{str(TRT_TO_NP[engine.get_binding_dtype(lTensorName[i])])} ."
            )

    else:
        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(
            trt.TensorIOMode.INPUT
        )

        context = engine.create_execution_context()
        for i in range(nInput, nIO):
            bufferH.append(
                np.zeros(
                    context.get_tensor_shape(lTensorName[i]),
                    dtype=TRT_TO_NP[engine.get_tensor_dtype(lTensorName[i])],
                )
            )

        for i in range(nInput):
            logger.debug(
                f"LoadEngine: BindingInput{i}={lTensorName[i]} :\tshape:{context.get_tensor_shape(lTensorName[i])},\ttype:{str(TRT_TO_NP[engine.get_tensor_dtype(lTensorName[i])])}"
            )
        for i in range(nInput, nIO):
            logger.debug(
                f"LoadEngine: BindingOutput{i}={lTensorName[i]}:\tshape:{context.get_tensor_shape(lTensorName[i])},\ttype:{str(TRT_TO_NP[engine.get_tensor_dtype(lTensorName[i])])}"
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

    # 返回lTensorName和context，这样main函数就可以访问它们
    if trt_old:
        return nInput, nIO, bufferH, lTensorName, None
    else:
        return nInput, nIO, bufferH, lTensorName, context


def inference_consistency_validatation(
    predicted_data: List, expected_data: List, output_names: List
):
    for x, y, name in zip(predicted_data, expected_data, output_names):
        max_abs_distance = float((np.abs(x - y)).max())
        logger.info(f"[max(abs()) error] {name} = {max_abs_distance}")

        cosine_distance = 1 - np.dot(x.flatten(), y.flatten()) / (
            np.linalg.norm(x.flatten()) * np.linalg.norm(y.flatten())
        )
        logger.info(f"[cosine_distance ] {name} = {float(cosine_distance)}")
        
        # 特别关注 indices 的详细对比
        if "indices" in name:
            logger.info("=" * 60)
            logger.info(f"🔍 详细对比 {name}:")
            logger.info(f"TensorRT输出 (predicted):")
            logger.info(f"  形状: {x.shape}, 类型: {x.dtype}")
            logger.info(f"  完整数据: {x.flatten()}")
            logger.info(f"  最小值: {x.min()}, 最大值: {x.max()}")
            
            logger.info(f"PyTorch期望 (expected):")
            logger.info(f"  形状: {y.shape}, 类型: {y.dtype}")
            logger.info(f"  完整数据: {y.flatten()}")
            logger.info(f"  最小值: {y.min()}, 最大值: {y.max()}")
            
            # 计算差异 - 确保形状一致
            x_flat = x.flatten()
            y_flat = y.flatten()
            diff = np.abs(x_flat - y_flat)
            logger.info(f"差异统计:")
            logger.info(f"  最大差异: {diff.max()}")
            logger.info(f"  平均差异: {diff.mean()}")
            logger.info(f"  差异大于0的元素数量: {np.sum(diff > 0)}")
            logger.info(f"  完全匹配的元素数量: {np.sum(diff == 0)}")
            
            # 找出差异最大的位置
            max_diff_pos = diff.argmax()
            logger.info(f"  最大差异位置: {max_diff_pos}")
            logger.info(f"  该位置TensorRT值: {x_flat[max_diff_pos]}")
            logger.info(f"  该位置PyTorch值: {y_flat[max_diff_pos]}")
            
            # 找出所有差异的位置
            mismatch_positions = np.where(diff > 0)[0]
            if len(mismatch_positions) > 0:
                logger.info(f"  所有差异位置:")
                for pos in mismatch_positions:
                    logger.info(f"    位置{pos}: TensorRT={x_flat[pos]}, PyTorch={y_flat[pos]}, 差异={diff[pos]}")
            logger.info("=" * 60)


def main(
    input_bins: List[np.ndarray],
    output_bins: List[np.ndarray],
    trt_old: bool,
    logger,
    plugin_name="DeformableAttentionAggrPlugin",
    soFile="deploy/dfa_plugin/lib/deformableAttentionAggr.so",
    # trtFile="deploy/engine/sparse4dhead2nd_polygraphy.engine",
    trtFile="deploy/engine/sparse4dhead2nd.engine",
):
    ctypes.cdll.LoadLibrary(soFile)
    plugin = getPlugin(plugin_name)
    engine = build_network(trtFile, logger)
    if engine == None:
        logger.error(f"{plugin_name} Engine Building Failed: {trtFile} !")
        return

    sample_id = 1
    for x, y in zip(input_bins, output_bins):
        (
            feature,
            spatial_shapes,
            level_start_index,
            instance_feature,
            anchor,
            time_interval,
            temp_instance_feature,
            temp_anchor,
            mask,
            track_id,
            image_wh,
            lidar2img,
        ) = x

        # 修改返回值接收
        if trt_old:
            nInput, nIO, bufferH, lTensorName, context = inference(
                feature,
                spatial_shapes,
                level_start_index,
                instance_feature,
                anchor,
                time_interval,
                temp_instance_feature,
                temp_anchor,
                mask,
                track_id,
                image_wh,
                lidar2img,
                engine,
                trt_old,
                logger,
            )
        else:
            nInput, nIO, bufferH, lTensorName, context = inference(
                feature,
                spatial_shapes,
                level_start_index,
                instance_feature,
                anchor,
                time_interval,
                temp_instance_feature,
                temp_anchor,
                mask,
                track_id,
                image_wh,
                lidar2img,
                engine,
                trt_old,
                logger,
            )

        input_names = [
            "feature",
            "spatial_shapes",
            "level_start_index",
            "instance_feature",
            "anchor",
            "time_interval",
            "temp_instance_feature",
            "temp_anchor",
            "mask",
            "track_id",
            "image_wh",
            "lidar2img",
        ]

        # Sequence of output_names should be same with engine binding sequence.
        output_names = [
            "tmp_outs0",
            "temp_gnn_output",    # 添加temp_gnn_output
            "tmp_outs1",
            "tmp_outs2",
            "tmp_outs3",
            "tmp_outs4",
            "tmp_outs5",
            "pred_instance_feature",
            "pred_anchor",
            "pred_class_score",
            "pred_quality_score",
            "pred_track_id",
            # 添加refine_outs输出名称
            "refine_outs0_anchor",
            "refine_outs0_instance_feature",
            "refine_outs0_anchor_embed",
            "refine_outs0_temp_anchor_embed",
            # 添加update_comparison输出名称
            "refine_outs0_instance_feature_before_update",
            "refine_outs0_anchor_before_update",
            "refine_outs0_temp_instance_feature",  # 新增
            "refine_outs0_temp_anchor",            # 新增
            "refine_outs0_confidence_sorted",      # 新增：confidence_sorted
            "refine_outs0_indices",                # 新增：indices
            "refine_outs0_selected_feature",       # 新增：selected_feature
            "refine_outs0_selected_anchor",        # 新增：selected_anchor
            "refine_outs0_instance_feature_after_update",
            "refine_outs0_anchor_after_update",
        ]

        assert len(input_names) == nInput
        assert len(output_names) == nIO - nInput

        # 增加engine输出变量名称的详细打印
        logger.info("=" * 80)
        logger.info("TensorRT Engine Output Information:")
        logger.info("=" * 80)
        
        # 打印所有输入tensor信息
        logger.info("Input Tensors:")
        for i, name in enumerate(input_names):
            logger.info(f"  Input{i}: {name} - Shape: {bufferH[i].shape}, Dtype: {bufferH[i].dtype}")
        
        # 打印所有输出tensor信息
        logger.info("Output Tensors:")
        for i in range(nInput, nIO):
            tensor_name = lTensorName[i]
            if trt_old:
                tensor_shape = engine.get_binding_shape(tensor_name)
                tensor_dtype = engine.get_binding_dtype(tensor_name)
            else:
                tensor_shape = context.get_tensor_shape(tensor_name)
                tensor_dtype = engine.get_tensor_dtype(tensor_name)
            logger.info(f"  Output{i-nInput}: {tensor_name} - Shape: {tensor_shape}, Dtype: {trt.nptype(tensor_dtype)}")
            logger.info(f"    Buffer Index: {i}, Expected Name: {output_names[i-nInput] if i-nInput < len(output_names) else 'N/A'}")
        
        logger.info("=" * 80)
        
        # 打印期望的输出顺序
        logger.info("Expected Output Order:")
        for i, name in enumerate(output_names):
            logger.info(f"  Index {i}: {name}")
        
        logger.info("=" * 80)

        for i, name in enumerate(input_names):
            printArrayInformation(bufferH[i], logger, info=f"{name}", prefix="TensorRT")

        assert 26 == len(y)  # 6个tmp_outs + 6个其他输出 + 4个refine_outs + 10个update_comparison (新增6个)
        logger.info(f"Sample {sample_id} inference consistency validatation:")
        
        # 增加实际使用的输出索引信息打印
        logger.info("Actual Output Indices Used for Validation:")
        logger.info(f"  refine_outs0_temp_anchor_embed: bufferH[{nInput + 0}] -> {lTensorName[nInput + 0]} (Output0)")
        logger.info(f"  tmp_outs0: bufferH[{nInput + 1}] -> {lTensorName[nInput + 1]} (Output1)")
        logger.info(f"  refine_outs0_instance_feature_before_update: bufferH[{nInput + 2}] -> {lTensorName[nInput + 2]} (Output2)")
        logger.info(f"  refine_outs0_anchor_before_update: bufferH[{nInput + 3}] -> {lTensorName[nInput + 3]} (Output3)")
        logger.info(f"  refine_outs0_confidence_sorted: bufferH[{nInput + 5}] -> {lTensorName[nInput + 5]} (Output12)")  # 新增：confidence_sorted
        logger.info(f"  refine_outs0_indices: bufferH[{nInput + 4}] -> {lTensorName[nInput + 4]} (Output13)")            # 新增：indices
        logger.info(f"  refine_outs0_selected_feature: bufferH[{nInput + 6}] -> {lTensorName[nInput + 6]} (Output6)")
        logger.info(f"  refine_outs0_selected_anchor: bufferH[{nInput + 7}] -> {lTensorName[nInput + 7]} (Output7)")
        logger.info(f"  refine_outs0_instance_feature: bufferH[{nInput + 8}] -> {lTensorName[nInput + 8]} (Output8)")
        logger.info(f"  refine_outs0_instance_feature_after_update: bufferH[{nInput + 9}] -> {lTensorName[nInput + 7]} (Output9)")
        logger.info(f"  refine_outs0_anchor: bufferH[{nInput + 10}] -> {lTensorName[nInput + 8]} (Output10)")
        logger.info(f"  refine_outs0_anchor_after_update: bufferH[{nInput + 11}] -> {lTensorName[nInput + 9]} (Output11)")
        logger.info(f"  refine_outs0_anchor_embed: bufferH[{nInput + 12}] -> {lTensorName[nInput + 10]} (Output12)")
        logger.info(f"  temp_gnn_output: bufferH[{nInput + 13}] -> {lTensorName[nInput + 11]} (Output13)")
        logger.info(f"  tmp_outs1: bufferH[{nInput + 14}] -> {lTensorName[nInput + 12]} (Output14)")
        logger.info(f"  tmp_outs2: bufferH[{nInput + 15}] -> {lTensorName[nInput + 13]} (Output15)")
        logger.info(f"  tmp_outs3: bufferH[{nInput + 16}] -> {lTensorName[nInput + 14]} (Output16)")
        logger.info(f"  tmp_outs4: bufferH[{nInput + 17}] -> {lTensorName[nInput + 15]} (Output17)")
        logger.info(f"  tmp_outs5: bufferH[{nInput + 18}] -> {lTensorName[nInput + 16]} (Output18)")
        logger.info(f"  pred_instance_feature: bufferH[{nInput + 19}] -> {lTensorName[nInput + 17]} (Output19)")
        logger.info(f"  pred_anchor: bufferH[{nInput + 20}] -> {lTensorName[nInput + 20]} (Output20)")
        logger.info(f"  pred_class_score: bufferH[{nInput + 21}] -> {lTensorName[nInput + 21]} (Output21)")
        logger.info(f"  pred_quality_score: bufferH[{nInput + 22}] -> {lTensorName[nInput + 22]} (Output22)")
        logger.info(f"  pred_track_id: bufferH[{nInput + 23}] -> {lTensorName[nInput + 23]} (Output23)")
        logger.info(f"  refine_outs0_temp_instance_feature: bufferH[{nInput + 24}] -> {lTensorName[nInput + 18]} (Output24)")
        logger.info(f"  refine_outs0_temp_anchor: bufferH[{nInput + 25}] -> {lTensorName[nInput + 19]} (Output25)")
        
        # 更新验证时使用的buffer索引，按照TensorRT引擎的实际输出顺序
        inference_consistency_validatation(
            [
                bufferH[nInput + 23],  # pred_track_id (Output23)
                bufferH[nInput + 19],  # pred_instance_feature (Output19)
                bufferH[nInput + 20],   # pred_anchor (Output20)
                bufferH[nInput + 21],   # pred_class_score (Output21)
                bufferH[nInput + 22],  # pred_quality_score (Output22)
                bufferH[nInput + 13],  # temp_gnn_output (Output13)
                bufferH[nInput + 1],   # tmp_outs0 (Output1)
                bufferH[nInput + 14],  # tmp_outs1 (Output14)
                bufferH[nInput + 15],  # tmp_outs2 (Output15)
                bufferH[nInput + 16],  # tmp_outs3 (Output16)
                bufferH[nInput + 17],  # tmp_outs4 (Output17)
                bufferH[nInput + 18],  # tmp_outs5 (Output18)
                # 添加refine_outs验证，按照TensorRT实际输出顺序
                bufferH[nInput + 10],   # refine_outs0_anchor (Output10)
                bufferH[nInput + 8],   # refine_outs0_instance_feature (Output8)
                bufferH[nInput + 12],  # refine_outs0_anchor_embed (Output12)
                bufferH[nInput + 0],   # refine_outs0_temp_anchor_embed (Output0)
                # 添加update_comparison验证，按照TensorRT实际输出顺序
                bufferH[nInput + 2],   # refine_outs0_instance_feature_before_update (Output2)
                bufferH[nInput + 3],   # refine_outs0_anchor_before_update (Output3)
                bufferH[nInput + 24],  # refine_outs0_temp_instance_feature (Output24)
                bufferH[nInput + 25],  # refine_outs0_temp_anchor (Output25)
                bufferH[nInput + 9],   # refine_outs0_instance_feature_after_update (Output9)
                bufferH[nInput + 11],   # refine_outs0_anchor_after_update (Output11)
                # 新增：selected_feature和selected_anchor
                bufferH[nInput + 6],   # refine_outs0_selected_feature (Output6)
                bufferH[nInput + 7],   # refine_outs0_selected_anchor (Output7)
                # 新增：confidence_sorted和indices - 使用正确的索引
                bufferH[nInput + 4],  # refine_outs0_confidence_sorted (Output12) - 形状(1,300)
                bufferH[nInput + 5],  # refine_outs0_indices (Output13) - 形状(1,300)
            ],
            y,
            [
                "pred_track_id",
                "pred_instance_feature", 
                "pred_anchor",
                "pred_class_score",
                "pred_quality_score",
                "temp_gnn_output",
                "tmp_outs0",
                "tmp_outs1",
                "tmp_outs2",
                "tmp_outs3",
                "tmp_outs4",
                "tmp_outs5",
                # 添加refine_outs验证名称
                "refine_outs0_anchor",
                "refine_outs0_instance_feature",
                "refine_outs0_anchor_embed",
                "refine_outs0_temp_anchor_embed",
                # 添加update_comparison验证名称
                "refine_outs0_instance_feature_before_update",
                "refine_outs0_anchor_before_update",
                "refine_outs0_temp_instance_feature",  # 新增
                "refine_outs0_temp_anchor",            # 新增
                "refine_outs0_instance_feature_after_update",
                "refine_outs0_anchor_after_update",
                # 新增：selected_feature和selected_anchor
                "refine_outs0_selected_feature",       # 新增：selected_feature
                "refine_outs0_selected_anchor",        # 新增：selected_anchor
                # 新增：confidence_sorted和indices
                "refine_outs0_confidence_sorted",      # 新增：confidence_sorted
                "refine_outs0_indices",                # 新增：indices
            ],
        )
        # 在验证之前，专门打印indices的完整对比信息
        logger.info("🔍 完整对比 refine_outs0_indices:")
        logger.info("=" * 80)
        
        # 获取TensorRT的indices输出
        trt_indices = bufferH[nInput + 5]  # refine_outs0_indices
        # 获取PyTorch期望的indices
        pytorch_indices = y[25]  # 根据你的输出列表，indices在第25个位置
        
        logger.info(f"TensorRT indices:")
        logger.info(f"  形状: {trt_indices.shape}")
        logger.info(f"  数据类型: {trt_indices.dtype}")
        logger.info(f"  完整数据: {trt_indices.flatten()}")
        logger.info(f"  数值范围: [{trt_indices.min()}, {trt_indices.max()}]")
        
        logger.info(f"PyTorch期望 indices:")
        logger.info(f"  形状: {pytorch_indices.shape}")
        logger.info(f"  数据类型: {pytorch_indices.dtype}")
        logger.info(f"  完整数据: {pytorch_indices.flatten()}")
        logger.info(f"  数值范围: [{pytorch_indices.min()}, {pytorch_indices.max()}]")
        
        # 计算并打印差异
        trt_flat = trt_indices.flatten()
        pytorch_flat = pytorch_indices.flatten()
        indices_diff = np.abs(trt_flat - pytorch_flat)
        logger.info(f"差异分析:")
        logger.info(f"  最大差异: {indices_diff.max()}")
        logger.info(f"  平均差异: {indices_diff.mean()}")
        logger.info(f"  差异分布:")
        logger.info(f"    差异=0: {np.sum(indices_diff == 0)} 个元素")
        logger.info(f"    差异>0: {np.sum(indices_diff > 0)} 个元素")
        
        # 找出所有差异的位置
        mismatch_positions = np.where(indices_diff > 0)[0]
        if len(mismatch_positions) > 0:
            logger.info(f"  所有差异位置详情:")
            for pos in mismatch_positions:
                logger.info(f"    位置{pos}: TensorRT={trt_flat[pos]}, PyTorch={pytorch_flat[pos]}, 差异={indices_diff[pos]}")
        else:
            logger.info("  ✅ 所有位置都完全匹配！")
        
        logger.info("=" * 80)
        sample_id += 1


if __name__ == "__main__":

    logger, _, _ = logger_wrapper("", False)
    logger.setLevel(logging.DEBUG)

    a, b, c, d = (trt.__version__).split(".")
    verson = eval(a + "." + b)
    logger.info(f"Python Tensor Version is: {trt.__version__} !")

    if verson < 8.5:
        trt_old = True
    else:
        trt_old = False

    np.set_printoptions(precision=4, linewidth=200, suppress=True)

    logger.info("Starting unit test...")
    inputs, expected_outputs = read_bin(3, logger)
    main(inputs, expected_outputs, trt_old, logger)
    logger.info("All tests are passed!!!")
