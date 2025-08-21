#!/usr/bin/env python3
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

import tensorrt as trt
import numpy as np
import os
import sys

def test_tensorrt_engine_output():
    """测试TensorRT引擎的输出，检查是否包含sigmoid激活函数"""
    
    # 引擎文件路径
    engine_path = "/share/Code/SparseEnd2End/onboard/assets/trt_engine/sparse4dhead1st.engine"
    
    if not os.path.exists(engine_path):
        print(f"错误：引擎文件不存在: {engine_path}")
        return
    
    print(f"测试TensorRT引擎: {engine_path}")
    
    # 创建TensorRT运行时
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    # 加载引擎
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    
    # 获取引擎信息
    print(f"引擎输入数量: {engine.num_bindings}")
    
    # 检查输出绑定
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        is_input = engine.binding_is_input(i)
        dtype = engine.get_binding_dtype(i)
        shape = engine.get_binding_shape(i)
        
        print(f"绑定 {i}: {name} ({'输入' if is_input else '输出'})")
        print(f"  形状: {shape}")
        print(f"  数据类型: {dtype}")
        
        if not is_input:
            # 检查是否是分类得分或质量得分输出
            if "pred_class_score" in name or "pred_quality_score" in name:
                print(f"  *** 这是分类/质量得分输出 ***")
    
    # 创建虚拟输入数据
    input_data = {}
    input_shapes = {
        "feature": (1, 89760, 256),
        "spatial_shapes": (6, 4, 2),
        "level_start_index": (6, 4),
        "instance_feature": (1, 900, 256),
        "anchor": (1, 900, 11),
        "time_interval": (1,),
        "image_wh": (1, 6, 2),
        "lidar2img": (1, 6, 4, 4)
    }
    
    # 准备输入数据
    for name, shape in input_shapes.items():
        if "spatial_shapes" in name or "level_start_index" in name:
            input_data[name] = np.random.randint(0, 100, shape, dtype=np.int32)
        else:
            input_data[name] = np.random.randn(*shape).astype(np.float32)
    
    # 准备输出缓冲区
    output_buffers = {}
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if not engine.binding_is_input(i):
            shape = engine.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            output_buffers[name] = np.zeros(shape, dtype=dtype)
    
    # 执行推理
    print("\n执行推理...")
    try:
        # 准备输入输出缓冲区
        buffers = []
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            if engine.binding_is_input(i):
                buffers.append(input_data[name])
            else:
                buffers.append(output_buffers[name])
        
        # 执行推理
        context.execute_v2(buffers)
        
        print("推理成功!")
        
        # 分析输出数据
        for name, output in output_buffers.items():
            if "pred_class_score" in name or "pred_quality_score" in name:
                print(f"\n=== {name} 分析 ===")
                print(f"形状: {output.shape}")
                print(f"数据类型: {output.dtype}")
                print(f"数值范围: [{output.min():.6f}, {output.max():.6f}]")
                print(f"平均值: {output.mean():.6f}")
                print(f"标准差: {output.std():.6f}")
                
                # 检查是否所有值都在(0,1)范围内（sigmoid特征）
                if output.min() >= 0 and output.max() <= 1:
                    print("*** 检测到sigmoid激活函数特征：所有值都在[0,1]范围内 ***")
                else:
                    print("*** 未检测到sigmoid激活函数：值范围超出[0,1] ***")
                
                # 检查是否有负值
                negative_count = np.sum(output < 0)
                positive_count = np.sum(output > 0)
                print(f"负值数量: {negative_count}")
                print(f"正值数量: {positive_count}")
                
    except Exception as e:
        print(f"推理失败: {e}")
        return
    
    # 清理
    context = None
    engine = None
    runtime = None

if __name__ == "__main__":
    test_tensorrt_engine_output() 