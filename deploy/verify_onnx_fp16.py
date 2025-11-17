#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 ONNX 模型是否真的是 FP16 精度
使用方法: python verify_onnx_fp16.py <onnx_path>
"""

import os
import sys
import argparse

def verify_onnx_fp16(onnx_path):
    """验证 ONNX 模型是否使用 FP16 精度"""
    
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX 文件不存在: {onnx_path}")
        return False
    
    try:
        import onnx
        import numpy as np
    except ImportError:
        print("[ERROR] 请安装 onnx 和 numpy: pip install onnx numpy")
        return False
    
    print("=" * 80)
    print(f"验证 ONNX 模型: {onnx_path}")
    print("=" * 80)
    
    # 加载 ONNX 模型
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("✓ ONNX 模型加载成功，格式验证通过")
    except Exception as e:
        print(f"✗ ONNX 模型加载失败: {e}")
        return False
    
    # ONNX 数据类型映射
    # FLOAT = 1, FLOAT16 = 10, INT32 = 6, INT64 = 7
    TYPE_MAP = {
        1: "FLOAT32",
        10: "FLOAT16",
        6: "INT32",
        7: "INT64",
    }
    
    # 检查输入数据类型
    print("\n" + "-" * 80)
    print("输入信息:")
    print("-" * 80)
    fp16_inputs = []
    fp32_inputs = []
    int_inputs = []
    
    for input_tensor in model.graph.input:
        tensor_type = input_tensor.type.tensor_type
        elem_type = tensor_type.elem_type
        dtype_name = TYPE_MAP.get(elem_type, f"UNKNOWN({elem_type})")
        
        # 获取形状
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        
        print(f"  输入: {input_tensor.name:30s} dtype={dtype_name:10s} shape={shape}")
        
        if elem_type == 10:  # FLOAT16
            fp16_inputs.append(input_tensor.name)
        elif elem_type == 1:  # FLOAT32
            fp32_inputs.append(input_tensor.name)
        else:
            int_inputs.append(input_tensor.name)
    
    # 检查输出数据类型
    print("\n" + "-" * 80)
    print("输出信息:")
    print("-" * 80)
    fp16_outputs = []
    fp32_outputs = []
    
    for output_tensor in model.graph.output:
        tensor_type = output_tensor.type.tensor_type
        elem_type = tensor_type.elem_type
        dtype_name = TYPE_MAP.get(elem_type, f"UNKNOWN({elem_type})")
        
        # 获取形状
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        
        print(f"  输出: {output_tensor.name:30s} dtype={dtype_name:10s} shape={shape}")
        
        if elem_type == 10:  # FLOAT16
            fp16_outputs.append(output_tensor.name)
        elif elem_type == 1:  # FLOAT32
            fp32_outputs.append(output_tensor.name)
    
    # 检查初始值（权重）的数据类型
    print("\n" + "-" * 80)
    print("权重/初始值信息 (采样前10个):")
    print("-" * 80)
    fp16_weights = []
    fp32_weights = []
    
    for i, initializer in enumerate(model.graph.initializer[:10]):  # 只显示前10个
        elem_type = initializer.data_type
        dtype_name = TYPE_MAP.get(elem_type, f"UNKNOWN({elem_type})")
        
        # 获取形状
        shape = list(initializer.dims)
        
        print(f"  权重: {initializer.name:30s} dtype={dtype_name:10s} shape={shape}")
        
        if elem_type == 10:  # FLOAT16
            fp16_weights.append(initializer.name)
        elif elem_type == 1:  # FLOAT32
            fp32_weights.append(initializer.name)
    
    if len(model.graph.initializer) > 10:
        print(f"  ... (还有 {len(model.graph.initializer) - 10} 个权重未显示)")
    
    # 统计所有权重
    for initializer in model.graph.initializer:
        if initializer.data_type == 10:  # FLOAT16
            fp16_weights.append(initializer.name)
        elif initializer.data_type == 1:  # FLOAT32
            fp32_weights.append(initializer.name)
    
    # 总结
    print("\n" + "=" * 80)
    print("精度检查结果:")
    print("=" * 80)
    
    # 检查输入
    if fp16_inputs:
        print(f"✓ FP16 输入: {len(fp16_inputs)} 个")
        for name in fp16_inputs[:5]:  # 只显示前5个
            print(f"    - {name}")
        if len(fp16_inputs) > 5:
            print(f"    ... (还有 {len(fp16_inputs) - 5} 个)")
    
    if fp32_inputs:
        print(f"⚠ FP32 输入: {len(fp32_inputs)} 个")
        for name in fp32_inputs[:5]:
            print(f"    - {name}")
        if len(fp32_inputs) > 5:
            print(f"    ... (还有 {len(fp32_inputs) - 5} 个)")
    
    if int_inputs:
        print(f"ℹ 整数输入: {len(int_inputs)} 个 (spatial_shapes, level_start_index, mask, track_id 等)")
    
    # 检查输出
    print()
    if fp16_outputs:
        print(f"✓ FP16 输出: {len(fp16_outputs)} 个")
        for name in fp16_outputs:
            print(f"    - {name}")
    
    if fp32_outputs:
        print(f"⚠ FP32 输出: {len(fp32_outputs)} 个")
        for name in fp32_outputs:
            print(f"    - {name}")
    
    # 检查权重
    print()
    print(f"权重统计:")
    print(f"  - FP16 权重: {len(fp16_weights)} 个")
    print(f"  - FP32 权重: {len(fp32_weights)} 个")
    
    # 最终判断
    print("\n" + "=" * 80)
    print("最终结论:")
    print("=" * 80)
    
    is_fp16_model = False
    
    # 判断标准：
    # 1. 所有浮点输入都是 FP16
    # 2. 所有浮点输出都是 FP16
    # 3. 大部分权重是 FP16（允许少量 FP32，因为某些操作可能需要）
    
    float_inputs_total = len(fp16_inputs) + len(fp32_inputs)
    float_outputs_total = len(fp16_outputs) + len(fp32_outputs)
    weights_total = len(fp16_weights) + len(fp32_weights)
    
    if float_inputs_total > 0:
        fp16_input_ratio = len(fp16_inputs) / float_inputs_total
        if fp16_input_ratio >= 0.8:  # 80% 以上是 FP16
            print("✓ 输入: 主要是 FP16 精度")
            is_fp16_model = True
        else:
            print("✗ 输入: 主要是 FP32 精度（不是 FP16 模型）")
            is_fp16_model = False
    
    if float_outputs_total > 0:
        fp16_output_ratio = len(fp16_outputs) / float_outputs_total
        if fp16_output_ratio >= 0.8:  # 80% 以上是 FP16
            print("✓ 输出: 主要是 FP16 精度")
        else:
            print("⚠ 输出: 主要是 FP32 精度")
            is_fp16_model = False
    
    if weights_total > 0:
        fp16_weight_ratio = len(fp16_weights) / weights_total
        if fp16_weight_ratio >= 0.5:  # 50% 以上是 FP16
            print(f"✓ 权重: {fp16_weight_ratio*100:.1f}% 是 FP16 精度")
        else:
            print(f"⚠ 权重: 只有 {fp16_weight_ratio*100:.1f}% 是 FP16 精度")
    
    print()
    if is_fp16_model and len(fp16_inputs) > 0 and len(fp16_outputs) > 0:
        print("🎉 结论: 这是一个 FP16 精度的 ONNX 模型 ✓")
        print("   可以用于构建 FP16 TensorRT engine")
    else:
        print("❌ 结论: 这不是一个 FP16 精度的 ONNX 模型")
        print("   建议:")
        print("   1. 检查导出时是否使用了 --fp16 参数")
        print("   2. 确认模型和输入都已转换为 FP16")
        print("   3. 某些操作可能不支持 FP16，导致自动转换为 FP32")
    
    return is_fp16_model


def main():
    parser = argparse.ArgumentParser(description="验证 ONNX 模型是否使用 FP16 精度")
    parser.add_argument("onnx_path", help="ONNX 文件路径")
    
    args = parser.parse_args()
    
    success = verify_onnx_fp16(args.onnx_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

