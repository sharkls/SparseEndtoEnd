#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import onnxruntime as ort
import logging
from tool.utils.logger import logger_wrapper

def debug_onnx_vs_trt_direct():
    """直接对比ONNX模型和TensorRT Engine的输入输出"""
    logger, _, _ = logger_wrapper("", False)
    logger.setLevel(logging.INFO)
    
    # 加载第二帧的输入数据
    prefix = "script/tutorial/asset/"
    sample_id = 1
    
    # 加载输入数据
    feature = np.fromfile(
        f"{prefix}sample_{sample_id}_feature_1*89760*256_float32.bin",
        dtype=np.float32,
    ).reshape(1, 89760, 256)
    
    spatial_shapes = np.fromfile(
        f"{prefix}sample_{sample_id}_spatial_shapes_6*4*2_int32.bin", dtype=np.int32
    ).reshape(6, 4, 2)
    
    level_start_index = np.fromfile(
        f"{prefix}sample_{sample_id}_level_start_index_6*4_int32.bin",
        dtype=np.int32,
    ).reshape(6, 4)
    
    instance_feature = np.fromfile(
        f"{prefix}sample_{sample_id}_instance_feature_1*900*256_float32.bin",
        dtype=np.float32,
    ).reshape(1, 900, 256)
    
    anchor = np.fromfile(
        f"{prefix}sample_{sample_id}_anchor_1*900*11_float32.bin", dtype=np.float32
    ).reshape(1, 900, 11)
    
    time_interval = np.fromfile(
        f"{prefix}sample_{sample_id}_time_interval_1_float32.bin",
        dtype=np.float32,
    ).reshape(1)
    
    temp_instance_feature = np.fromfile(
        f"{prefix}sample_{sample_id}_temp_instance_feature_1*600*256_float32.bin",
        dtype=np.float32,
    ).reshape(1, 600, 256)
    
    temp_anchor = np.fromfile(
        f"{prefix}sample_{sample_id}_temp_anchor_1*600*11_float32.bin", dtype=np.float32
    ).reshape(1, 600, 11)
    
    mask = np.fromfile(
        f"{prefix}sample_{sample_id}_mask_1_int32.bin",
        dtype=np.int32,
    ).reshape(1)
    
    track_id = np.fromfile(
        f"{prefix}sample_{sample_id}_track_id_1*900_int32.bin",
        dtype=np.int32,
    ).reshape(1, 900)
    
    image_wh = np.fromfile(
        f"{prefix}sample_{sample_id}_image_wh_1*6*2_float32.bin",
        dtype=np.float32,
    ).reshape(1, 6, 2)
    
    lidar2img = np.fromfile(
        f"{prefix}sample_{sample_id}_lidar2img_1*6*4*4_float32.bin",
        dtype=np.float32,
    ).reshape(1, 6, 4, 4)
    
    # 准备输入数据
    inputs = {
        "feature": feature,
        "spatial_shapes": spatial_shapes,
        "level_start_index": level_start_index,
        "instance_feature": instance_feature,
        "anchor": anchor,
        "time_interval": time_interval,
        "temp_instance_feature": temp_instance_feature,
        "temp_anchor": temp_anchor,
        "mask": mask,
        "track_id": track_id,
        "image_wh": image_wh,
        "lidar2img": lidar2img,
    }
    
    logger.info("=== ONNX模型推理 ===")
    try:
        # 加载ONNX模型
        onnx_model_path = "deploy/onnx/sparse4dhead2nd.onnx"
        session = ort.InferenceSession(onnx_model_path)
        
        # 获取输入输出信息
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        logger.info(f"ONNX输入名称: {input_names}")
        logger.info(f"ONNX输出名称: {output_names}")
        
        # 执行ONNX推理
        onnx_outputs = session.run(output_names, inputs)
        
        logger.info("ONNX推理成功")
        for i, (name, output) in enumerate(zip(output_names, onnx_outputs)):
            logger.info(f"  {name}: shape={output.shape}, max={output.max()}, min={output.min()}")
            
    except Exception as e:
        logger.error(f"ONNX推理失败: {e}")
    
    logger.info("=== TensorRT Engine推理 ===")
    try:
        import tensorrt as trt
        import ctypes
        from cuda import cudart
        
        # 加载TensorRT Engine
        trtFile = "deploy/engine/sparse4dhead2nd.engine"
        trtlogger = trt.Logger(trt.Logger.INFO)
        
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(trtlogger).deserialize_cuda_engine(f.read())
        
        # 获取输入输出信息
        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(
            trt.TensorIOMode.INPUT
        )
        
        logger.info(f"TensorRT输入数量: {nInput}")
        logger.info(f"TensorRT输出数量: {nIO - nInput}")
        logger.info(f"TensorRT输入名称: {lTensorName[:nInput]}")
        logger.info(f"TensorRT输出名称: {lTensorName[nInput:]}")
        
        # 准备输入数据
        bufferH = []
        for i in range(nInput):
            tensor_name = lTensorName[i]
            if tensor_name in inputs:
                bufferH.append(inputs[tensor_name])
            else:
                logger.error(f"找不到输入张量: {tensor_name}")
                return
        
        # 准备输出缓冲区
        context = engine.create_execution_context()
        for i in range(nInput, nIO):
            bufferH.append(
                np.zeros(
                    context.get_tensor_shape(lTensorName[i]),
                    dtype=np.float32,
                )
            )
        
        # 分配GPU内存
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        
        # 复制输入数据到GPU
        for i in range(nInput):
            cudart.cudaMemcpy(
                bufferD[i],
                bufferH[i].ctypes.data,
                bufferH[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        
        # 设置张量地址
        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        
        # 执行推理
        context.execute_async_v3(0)
        
        # 复制输出数据到CPU
        for i in range(nInput, nIO):
            cudart.cudaMemcpy(
                bufferH[i].ctypes.data,
                bufferD[i],
                bufferH[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
        
        # 释放GPU内存
        for b in bufferD:
            cudart.cudaFree(b)
        
        logger.info("TensorRT推理成功")
        for i in range(nInput, nIO):
            output_name = lTensorName[i]
            output_data = bufferH[i]
            logger.info(f"  {output_name}: shape={output_data.shape}, max={output_data.max()}, min={output_data.min()}")
            
    except Exception as e:
        logger.error(f"TensorRT推理失败: {e}")

if __name__ == "__main__":
    debug_onnx_vs_trt_direct() 