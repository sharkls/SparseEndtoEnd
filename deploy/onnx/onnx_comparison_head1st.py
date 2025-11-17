#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 FP32 和 FP16 head1st TensorRT Engine 的推理结果差异
使用方法: python compare_head1st_engine_fp32_fp16.py [--input-dir DIR] [--output-dir DIR]
"""

import os
import sys
import argparse
import numpy as np
import ctypes
from typing import Dict, Tuple, List

try:
    import tensorrt as trt
    from cuda import cudart
except ImportError as e:
    print(f"[ERROR] 无法导入必要的库: {e}")
    print("请确保已安装: tensorrt, cuda-python")
    sys.exit(1)


def load_plugin(plugin_path: str = None):
    """
    加载 TensorRT 自定义插件
    
    Args:
        plugin_path: 插件库路径
    """
    if plugin_path is None:
        # 默认插件路径
        plugin_path = "/share/Code/Sparse4dE2E/deploy/dfa_plugin/lib/deformableAttentionAggr.so"
    
    if os.path.exists(plugin_path):
        print(f"加载 Plugin: {plugin_path}")
        try:
            ctypes.CDLL(plugin_path)
            print("✓ Plugin 加载成功")
        except Exception as e:
            print(f"⚠ 警告: Plugin 加载失败: {e}")
    else:
        print(f"⚠ 警告: Plugin 文件不存在: {plugin_path}")


def load_head1st_inputs(input_dir: str = None) -> Dict[str, np.ndarray]:
    """
    加载 head1st 模型的输入数据
    
    Args:
        input_dir: 输入数据目录（可选，如果不提供则生成随机数据）
    
    Returns:
        inputs: 输入字典
    """
    inputs = {}
    
    # head1st 的输入形状和类型
    input_specs = {
        "feature": {"shape": (1, 89760, 256), "dtype": np.float32},
        "spatial_shapes": {"shape": (6, 4, 2), "dtype": np.int32},
        "level_start_index": {"shape": (6, 4), "dtype": np.int32},
        "instance_feature": {"shape": (1, 900, 256), "dtype": np.float32},
        "anchor": {"shape": (1, 900, 11), "dtype": np.float32},
        "time_interval": {"shape": (1,), "dtype": np.float32},
        "image_wh": {"shape": (1, 6, 2), "dtype": np.float32},
        "lidar2img": {"shape": (1, 6, 4, 4), "dtype": np.float32},
    }
    
    if input_dir and os.path.exists(input_dir):
        print(f"从目录加载输入数据: {input_dir}")
        for name, spec in input_specs.items():
            file_path = os.path.join(input_dir, f"{name}.bin")
            if os.path.exists(file_path):
                data = np.fromfile(file_path, dtype=spec["dtype"])
                data = data.reshape(spec["shape"])
                inputs[name] = data
                print(f"  ✓ {name}: {data.shape}, {data.dtype}")
            else:
                print(f"  ⚠ {name}: 文件不存在，生成随机数据")
                np.random.seed(42)
                if spec["dtype"] == np.int32:
                    inputs[name] = np.random.randint(0, 100, size=spec["shape"], dtype=spec["dtype"])
                else:
                    inputs[name] = np.random.randn(*spec["shape"]).astype(spec["dtype"])
    else:
        print("生成随机输入数据...")
        np.random.seed(42)
        
        for name, spec in input_specs.items():
            if spec["dtype"] == np.int32:
                if name == "spatial_shapes":
                    inputs[name] = np.array([[[64, 176], [32, 88], [16, 44], [8, 22]]] * 6, dtype=spec["dtype"])
                elif name == "level_start_index":
                    inputs[name] = np.array([[0, 11264, 14080, 14784, 14960, 26224, 29040, 29744, 44880, 56144, 58960, 59664, 59840, 71104, 73920, 74624, 74800, 86064, 88880, 89584]] * 6, dtype=spec["dtype"])
                else:
                    inputs[name] = np.random.randint(0, 100, size=spec["shape"], dtype=spec["dtype"])
            else:
                inputs[name] = np.random.randn(*spec["shape"]).astype(spec["dtype"])
                if name in ["feature", "instance_feature"]:
                    inputs[name] = (inputs[name] - inputs[name].mean()) / (inputs[name].std() + 1e-8)
                elif name == "anchor":
                    inputs[name] = (inputs[name] - inputs[name].min()) / (inputs[name].max() - inputs[name].min() + 1e-8)
                elif name == "time_interval":
                    inputs[name] = np.array([0.1], dtype=spec["dtype"])
                elif name == "image_wh":
                    inputs[name] = np.array([[[704, 256]]] * 6, dtype=spec["dtype"])
                elif name == "lidar2img":
                    inputs[name] = np.eye(4, dtype=spec["dtype"]).reshape(1, 1, 4, 4).repeat(6, axis=1)
    
    print(f"\n输入数据信息:")
    for name, data in inputs.items():
        print(f"  {name}: 形状={data.shape}, 类型={data.dtype}")
        if data.dtype != np.int32:
            print(f"    范围: [{data.min():.6f}, {data.max():.6f}]")
    
    return inputs


def load_engine(engine_path: str, plugin_path: str = None) -> trt.ICudaEngine:
    """
    加载 TensorRT Engine 文件
    
    Args:
        engine_path: Engine 文件路径
        plugin_path: 插件库路径
    
    Returns:
        engine: TensorRT Engine
    """
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine 文件不存在: {engine_path}")
    
    # 加载插件
    if plugin_path:
        load_plugin(plugin_path)
    
    # 初始化 TensorRT
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    
    # 加载 Engine
    print(f"加载 Engine: {engine_path}")
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        raise RuntimeError("Engine 加载失败")
    
    print("✓ Engine 加载成功")
    return engine


def run_tensorrt_inference(engine: trt.ICudaEngine, inputs: Dict[str, np.ndarray], 
                          use_fp16: bool = False) -> Dict[str, np.ndarray]:
    """
    使用 TensorRT Engine 进行推理
    
    Args:
        engine: TensorRT Engine
        inputs: 输入字典
        use_fp16: 是否使用 FP16 精度（用于确定输入数据类型）
    
    Returns:
        outputs: 输出字典
    """
    print(f"\n{'='*80}")
    print(f"TensorRT 推理 ({'FP16' if use_fp16 else 'FP32'})")
    print(f"{'='*80}")
    
    # 检测 TensorRT 版本
    trt_old = hasattr(engine, 'num_bindings') and hasattr(engine, 'get_binding_name')
    
    if trt_old:
        # TensorRT 8.x
        num_bindings = engine.num_bindings
        input_names = []
        output_names = []
        
        for i in range(num_bindings):
            name = engine.get_binding_name(i)
            if engine.binding_is_input(i):
                input_names.append(name)
            else:
                output_names.append(name)
    else:
        # TensorRT 10.x
        num_io = engine.num_io_tensors
        input_names = []
        output_names = []
        
        for i in range(num_io):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
    
    print(f"\n输入: {input_names}")
    print(f"输出: {output_names}")
    
    # 创建执行上下文
    context = engine.create_execution_context()
    
    # 准备输入输出缓冲区
    host_buffers = []
    device_buffers = []
    
    # 准备输入缓冲区
    print(f"\n准备输入数据 ({'FP16' if use_fp16 else 'FP32'}):")
    for name in input_names:
        if name not in inputs:
            raise ValueError(f"缺少输入: {name}")
        
        data = inputs[name]
        
        # 如果是浮点类型且使用 FP16，转换为 FP16
        if use_fp16 and data.dtype in [np.float32, np.float64]:
            data = data.astype(np.float16)
        
        host_buffers.append(data)
        print(f"  {name}: {data.shape}, {data.dtype}")
        
        # 分配 GPU 内存
        err, d_ptr = cudart.cudaMalloc(data.nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"无法分配 GPU 内存: {err}")
        device_buffers.append(d_ptr)
        
        # 复制到 GPU
        err = cudart.cudaMemcpy(
            d_ptr, data.ctypes.data, data.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        if isinstance(err, tuple):
            err = err[0]
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"无法复制数据到 GPU: {err}")
    
    # 准备输出缓冲区
    print(f"\n准备输出缓冲区:")
    output_shapes = {}
    for name in output_names:
        if trt_old:
            shape = tuple(engine.get_binding_shape(engine.get_binding_index(name)))
        else:
            shape = tuple(context.get_tensor_shape(name))
        
        output_shapes[name] = shape
        
        # 根据精度确定输出类型（从 Engine 获取实际类型）
        if trt_old:
            dtype_trt = engine.get_binding_dtype(engine.get_binding_index(name))
        else:
            dtype_trt = engine.get_tensor_dtype(name)
        
        # 转换为 numpy 类型
        if dtype_trt == trt.DataType.HALF:
            dtype = np.float16
        elif dtype_trt == trt.DataType.FLOAT:
            dtype = np.float32
        elif dtype_trt == trt.DataType.INT32:
            dtype = np.int32
        else:
            # 默认根据 use_fp16 判断
            dtype = np.float16 if use_fp16 else np.float32
        
        output_data = np.zeros(shape, dtype=dtype)
        host_buffers.append(output_data)
        print(f"  {name}: {shape}, {dtype}")
        
        # 分配 GPU 内存
        err, d_ptr = cudart.cudaMalloc(output_data.nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"无法分配 GPU 内存: {err}")
        device_buffers.append(d_ptr)
    
    # 设置 tensor 地址（TensorRT 10.x）或准备 binding 地址（TensorRT 8.x）
    if trt_old:
        # TensorRT 8.x: 使用 binding 地址
        binding_addrs = [int(ptr) for ptr in device_buffers]
    else:
        # TensorRT 10.x: 设置 tensor 地址
        for i, name in enumerate(input_names + output_names):
            context.set_tensor_address(name, int(device_buffers[i]))
    
    # 执行推理
    print("\n执行推理...")
    if trt_old:
        context.execute_async_v2(binding_addrs, 0)
    else:
        context.execute_async_v3(0)
    
    # 同步
    cudart.cudaDeviceSynchronize()
    
    # 复制输出数据回 CPU
    output_dict = {}
    for i, name in enumerate(output_names):
        output_idx = len(input_names) + i
        output_data = host_buffers[output_idx]
        
        err = cudart.cudaMemcpy(
            output_data.ctypes.data, device_buffers[output_idx], output_data.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
        if isinstance(err, tuple):
            err = err[0]
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"无法复制数据回 CPU: {err}")
        
        output_dict[name] = output_data
    
    # 清理 GPU 内存
    for d_ptr in device_buffers:
        cudart.cudaFree(d_ptr)
    
    # 打印输出信息
    print("\n推理结果:")
    for name, output in output_dict.items():
        print(f"  {name}:")
        print(f"    形状: {output.shape}")
        print(f"    类型: {output.dtype}")
        if output.dtype != np.int32:
            output_fp32 = output.astype(np.float32) if output.dtype == np.float16 else output
            print(f"    范围: [{output_fp32.min():.6f}, {output_fp32.max():.6f}]")
            print(f"    均值: {output_fp32.mean():.6f}")
            print(f"    标准差: {output_fp32.std():.6f}")
    
    return output_dict


def compare_outputs(output_fp32: Dict[str, np.ndarray], 
                   output_fp16: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """
    对比 FP32 和 FP16 的输出结果
    """
    print(f"\n{'='*80}")
    print("对比 FP32 和 FP16 输出结果")
    print(f"{'='*80}")
    
    comparison = {}
    
    common_outputs = set(output_fp32.keys()) & set(output_fp16.keys())
    if not common_outputs:
        raise ValueError("FP32 和 FP16 模型的输出名称不一致！")
    
    for output_name in common_outputs:
        fp32_output = output_fp32[output_name]
        fp16_output = output_fp16[output_name]
        
        # 转换为 FP32 进行对比
        if fp32_output.dtype == np.float16:
            fp32_output = fp32_output.astype(np.float32)
        if fp16_output.dtype == np.float16:
            fp16_output_fp32 = fp16_output.astype(np.float32)
        else:
            fp16_output_fp32 = fp16_output
        
        if fp32_output.shape != fp16_output_fp32.shape:
            print(f"⚠️  警告: {output_name} 的形状不一致!")
            print(f"    FP32: {fp32_output.shape}")
            print(f"    FP16: {fp16_output_fp32.shape}")
            continue
        
        # 计算差异
        diff = fp32_output - fp16_output_fp32
        abs_diff = np.abs(diff)
        relative_diff = np.abs(diff) / (np.abs(fp32_output) + 1e-8)
        
        stats = {
            'max_abs_diff': float(np.max(abs_diff)),
            'mean_abs_diff': float(np.mean(abs_diff)),
            'std_abs_diff': float(np.std(abs_diff)),
            'max_relative_diff': float(np.max(relative_diff)),
            'mean_relative_diff': float(np.mean(relative_diff)),
            'cosine_similarity': float(np.dot(fp32_output.flatten(), fp16_output_fp32.flatten()) / 
                                      (np.linalg.norm(fp32_output.flatten()) * np.linalg.norm(fp16_output_fp32.flatten()) + 1e-8)),
            'fp32_range': [float(np.min(fp32_output)), float(np.max(fp32_output))],
            'fp16_range': [float(np.min(fp16_output_fp32)), float(np.max(fp16_output_fp32))],
            'rmse': float(np.sqrt(np.mean(diff ** 2))),
        }
        
        comparison[output_name] = stats
        
        print(f"\n输出: {output_name}")
        print(f"  最大绝对差异: {stats['max_abs_diff']:.6e}")
        print(f"  平均绝对差异: {stats['mean_abs_diff']:.6e}")
        print(f"  标准差: {stats['std_abs_diff']:.6e}")
        print(f"  均方根误差 (RMSE): {stats['rmse']:.6e}")
        print(f"  最大相对差异: {stats['max_relative_diff']:.6e}")
        print(f"  平均相对差异: {stats['mean_relative_diff']:.6e}")
        print(f"  余弦相似度: {stats['cosine_similarity']:.8f}")
        print(f"  FP32 范围: [{stats['fp32_range'][0]:.6f}, {stats['fp32_range'][1]:.6f}]")
        print(f"  FP16 范围: [{stats['fp16_range'][0]:.6f}, {stats['fp16_range'][1]:.6f}]")
        
        diff_percentiles = np.percentile(abs_diff, [50, 75, 90, 95, 99, 99.9])
        print(f"  差异百分位数:")
        print(f"    50%: {diff_percentiles[0]:.6e}")
        print(f"    75%: {diff_percentiles[1]:.6e}")
        print(f"    90%: {diff_percentiles[2]:.6e}")
        print(f"    95%: {diff_percentiles[3]:.6e}")
        print(f"    99%: {diff_percentiles[4]:.6e}")
        print(f"    99.9%: {diff_percentiles[5]:.6e}")
    
    return comparison


def save_comparison_results(comparison: Dict, output_dir: str = None):
    """保存对比结果到文件"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "head1st_engine_comparison_results.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("FP32 vs FP16 head1st TensorRT Engine 推理结果对比\n")
            f.write("="*80 + "\n\n")
            
            for output_name, stats in comparison.items():
                f.write(f"输出: {output_name}\n")
                f.write(f"  最大绝对差异: {stats['max_abs_diff']:.6e}\n")
                f.write(f"  平均绝对差异: {stats['mean_abs_diff']:.6e}\n")
                f.write(f"  标准差: {stats['std_abs_diff']:.6e}\n")
                f.write(f"  均方根误差 (RMSE): {stats['rmse']:.6e}\n")
                f.write(f"  最大相对差异: {stats['max_relative_diff']:.6e}\n")
                f.write(f"  平均相对差异: {stats['mean_relative_diff']:.6e}\n")
                f.write(f"  余弦相似度: {stats['cosine_similarity']:.8f}\n")
                f.write(f"  FP32 范围: [{stats['fp32_range'][0]:.6f}, {stats['fp32_range'][1]:.6f}]\n")
                f.write(f"  FP16 范围: [{stats['fp16_range'][0]:.6f}, {stats['fp16_range'][1]:.6f}]\n")
                f.write("\n")
        
        print(f"\n对比结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="对比 FP32 和 FP16 head1st TensorRT Engine 的推理结果")
    parser.add_argument(
        "--fp32-engine",
        type=str,
        default="/share/Code/Sparse4dE2E/deploy/engine/enginev1/engine32/sparse4dhead1st.engine",
        help="FP32 Engine 文件路径"
    )
    parser.add_argument(
        "--fp16-engine",
        type=str,
        default="/share/Code/Sparse4dE2E/deploy/engine/sparse4dhead1st.engine",
        help="FP16 Engine 文件路径"
    )
    parser.add_argument(
        "--plugin-path",
        type=str,
        default="/share/Code/Sparse4dE2E/deploy/dfa_plugin/lib/deformableAttentionAggr.so",
        help="插件库路径"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="输入数据目录（可选，如果不提供则生成随机数据）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（可选，保存对比结果）"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("FP32 vs FP16 head1st TensorRT Engine 推理结果对比工具")
    print("="*80)
    
    # 加载输入数据
    inputs = load_head1st_inputs(args.input_dir)
    
    # 加载 FP32 Engine 并推理
    print(f"\n{'='*80}")
    print("FP32 Engine 推理")
    print(f"{'='*80}")
    engine_fp32 = load_engine(args.fp32_engine, args.plugin_path)
    output_fp32 = run_tensorrt_inference(engine_fp32, inputs, use_fp16=False)
    
    # 加载 FP16 Engine 并推理
    print(f"\n{'='*80}")
    print("FP16 Engine 推理")
    print(f"{'='*80}")
    engine_fp16 = load_engine(args.fp16_engine, args.plugin_path)
    output_fp16 = run_tensorrt_inference(engine_fp16, inputs, use_fp16=True)
    
    # 对比结果
    comparison = compare_outputs(output_fp32, output_fp16)
    
    # 保存结果
    if args.output_dir:
        save_comparison_results(comparison, args.output_dir)
    
    # 总结
    print(f"\n{'='*80}")
    print("对比总结")
    print(f"{'='*80}")
    
    for output_name, stats in comparison.items():
        print(f"\n{output_name}:")
        if stats['cosine_similarity'] > 0.99:
            print(f"  ✅ 余弦相似度很高 ({stats['cosine_similarity']:.8f})，结果基本一致")
        elif stats['cosine_similarity'] > 0.95:
            print(f"  ⚠️  余弦相似度较高 ({stats['cosine_similarity']:.8f})，存在一定差异")
        else:
            print(f"  ❌ 余弦相似度较低 ({stats['cosine_similarity']:.8f})，差异较大")
        
        if stats['max_relative_diff'] < 0.01:
            print(f"  ✅ 最大相对差异很小 ({stats['max_relative_diff']:.6e})")
        elif stats['max_relative_diff'] < 0.1:
            print(f"  ⚠️  最大相对差异中等 ({stats['max_relative_diff']:.6e})")
        else:
            print(f"  ❌ 最大相对差异较大 ({stats['max_relative_diff']:.6e})")
        
        if stats['rmse'] < 0.01:
            print(f"  ✅ RMSE 很小 ({stats['rmse']:.6e})")
        elif stats['rmse'] < 0.1:
            print(f"  ⚠️  RMSE 中等 ({stats['rmse']:.6e})")
        else:
            print(f"  ❌ RMSE 较大 ({stats['rmse']:.6e})")
    
    print("\n对比完成！")


if __name__ == "__main__":
    main()