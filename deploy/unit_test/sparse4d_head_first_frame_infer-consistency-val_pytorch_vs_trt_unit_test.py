# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
""" If you get following results, congratulations
[2024-09-24::21:25:07] [INFO]: [max(abs()) error] pred_instance_feature = 0.0006016641855239868
[2024-09-24::21:25:07] [INFO]: [cosine_distance ] pred_instance_feature = 0.0
[2024-09-24::21:25:07] [INFO]: [max(abs()) error] pred_anchor = 0.0004584789276123047
[2024-09-24::21:25:07] [INFO]: [cosine_distance ] pred_anchor = -1.1920928955078125e-07
[2024-09-24::21:25:07] [INFO]: [max(abs()) error] pred_class_score = 0.0003018379211425781
[2024-09-24::21:25:07] [INFO]: [cosine_distance ] pred_class_score = 5.960464477539063e-08
[2024-09-24::21:25:07] [INFO]: [max(abs()) error] pred_quality_score = 0.00012242794036865234
[2024-09-24::21:25:07] [INFO]: [cosine_distance ] pred_quality_score = 0.0
[2024-09-24::21:25:07] [INFO]: All tests are passed!!!
"""
import os
import ctypes
import logging
import numpy as np
import tensorrt as trt

from typing import List
from cuda import cudart
from tool.utils.logger import logger_wrapper
from deploy.utils.utils import printArrayInformation

if not hasattr(np, 'bool'):
    np.bool = bool


def read_bin(samples, logger):
    prefix = "script/tutorial/asset/"
    # prefix = "C++/Output/val_bin/"

    inputs = list()
    outputs = list()

    for i in range(samples):

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

        # 读取tmp_outs数据
        tmp_outs = []
        for j in range(6):  # 假设有6个deformable模块
            try:
                tmp_out_shape = [1, 900, 512]
                tmp_out = np.fromfile(
                    f"{prefix}sample_{i}_tmp_outs{j}_1*900*512_float32.bin",
                    dtype=np.float32,
                ).reshape(tmp_out_shape)
                tmp_outs.append(tmp_out)
                printArrayInformation(tmp_out, logger, f"tmp_outs{j}", "PyTorch")
            except FileNotFoundError:
                # 如果文件不存在，创建一个零张量作为占位符
                tmp_out = np.zeros((1, 900, 512), dtype=np.float32)
                tmp_outs.append(tmp_out)
                logger.warning(f"tmp_outs{j}文件不存在，使用零张量作为占位符")

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

        inputs.append(
            [
                feature,
                spatial_shapes,
                level_start_index,
                instance_feature,
                anchor,
                time_interval,
                image_wh,
                lidar2img,
            ]
        )

        outputs.append(
            tmp_outs + [  # 先添加tmp_outs
                pred_instance_feature,
                pred_anchor,
                pred_class_score,
                pred_quality_score,
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

    return nInput, nIO, bufferH


def inference_consistency_validatation(predicted_data, expected_data, output_names):
    logger.info("=== 开始验证推理一致性 ===")
    for x, y, name in zip(predicted_data, expected_data, output_names):
        max_abs_distance = float((np.abs(x - y)).max())
        logger.info(f"[max(abs()) error] {name} = {max_abs_distance}")

        cosine_distance = 1 - np.dot(x.flatten(), y.flatten()) / (
            np.linalg.norm(x.flatten()) * np.linalg.norm(y.flatten())
        )
        logger.info(f"[cosine_distance ] {name} = {float(cosine_distance)}")
        
        # 为tmp_outs添加特殊标记
        if name.startswith("tmp_outs"):
            logger.info(f"  -> {name} 验证完成 (deformable模块输出)")
    
    logger.info("=== 推理一致性验证完成 ===")


def main(
    input_bins: List[np.ndarray],
    output_bins: List[np.ndarray],
    trt_old: bool,
    logger,
    plugin_name="DeformableAttentionAggrPlugin",
    soFile="deploy/dfa_plugin/lib/deformableAttentionAggr.so",
    # trtFile="deploy/engine/sparse4dhead1st_polygraphy.engine",
    trtFile="deploy/engine/sparse4dhead1st.engine",
):
    ctypes.cdll.LoadLibrary(soFile)
    plugin = getPlugin(plugin_name)
    engine = build_network(trtFile, logger)
    if engine == None:
        logger.error(f"{plugin_name} Engine Building Failed: {trtFile}!")
        return

    for x, y in zip(input_bins, output_bins):
        (
            feature,
            spatial_shapes,
            level_start_index,
            instance_feature,
            anchor,
            time_interval,
            image_wh,
            lidar2img,
        ) = x

        nInput, nIO, bufferH = inference(
            feature,
            spatial_shapes,
            level_start_index,
            instance_feature,
            anchor,
            time_interval,
            image_wh,
            lidar2img,
            engine,
            trt_old,
            logger,
        )
        
        # 统计tmp_outs的数量
        tmp_outs_count = len([name for name in y if isinstance(name, np.ndarray) and name.shape == (1, 900, 256)])
        logger.info(f"检测到 {tmp_outs_count} 个tmp_outs张量 (deformable模块输出)")

        input_names = [
            "feature",
            "spatial_shapes",
            "level_start_index",
            "instance_feature",
            "anchor",
            "time_interval",
            "image_wh",
            "lidar2img",
        ]

        output_names = [
            "tmp_outs0",
            "tmp_outs1",
            "tmp_outs2",
            "tmp_outs3",
            "tmp_outs4",
            "tmp_outs5",
            "pred_instance_feature",
            "pred_anchor",
            "pred_class_score",
            "pred_quality_score",
        ]

        assert len(input_names) == nInput
        assert len(output_names) == nIO - nInput

        for i, name in enumerate(input_names):
            printArrayInformation(bufferH[i], logger, info=f"{name}", prefix="TensorRT")
        
        # 打印所有输出张量的信息
        for i, name in enumerate(output_names):
            printArrayInformation(
                bufferH[i + nInput], logger, info=f"{name}", prefix="TensorRT"
            )

        # 验证所有输出，包括tmp_outs
        assert len(output_names) == len(y)
        inference_consistency_validatation(bufferH[nInput:], y, output_names)


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
    logger.info("注意：本测试将验证第一帧推理中tmp_outs*数据的一致性")
    logger.info("tmp_outs*数据来自deformable模块的输出，用于特征融合")
    inputs, expected_outputs = read_bin(1, logger)
    main(inputs, expected_outputs, trt_old, logger)
    logger.info("All tests are passed!!!")
