#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 TensorRT Engine 是否支持 FP16 并测试 warmup
使用方法: python check_engine_fp16.py <engine_path> [--warmup-iterations N]
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def check_engine_fp16(engine_path, warmup_iterations=100):
    """检查 engine 是否支持 FP16"""
    
    if not os.path.exists(engine_path):
        print(f"[ERROR] Engine 文件不存在: {engine_path}")
        return False
    
    try:
        import tensorrt as trt
        import numpy as np
        from cuda import cudart
    except ImportError as e:
        print(f"[ERROR] 无法导入必要的库: {e}")
        print("请确保已安装: tensorrt, numpy, cuda-python")
        return False
    
    print("=" * 80)
    print(f"检查 Engine: {engine_path}")
    print("=" * 80)
    
    # 加载 engine
    logger = trt.Logger(trt.Logger.WARNING)
    try:
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            print("[ERROR] 无法反序列化 engine")
            return False
        
        print(f"\n✓ Engine 加载成功")
        print(f"  - TensorRT 版本: {trt.__version__}")
        print(f"  - Engine 层数: {engine.num_layers}")
        
        # 检查 TensorRT 版本并获取 bindings
        if hasattr(engine, 'num_bindings'):
            # TensorRT 8.x
            num_bindings = engine.num_bindings
            get_binding_name = lambda i: engine.get_binding_name(i)
            binding_is_input = lambda i: engine.binding_is_input(i)
            get_binding_dtype = lambda i: engine.get_binding_dtype(i)
            get_binding_shape = lambda i: engine.get_binding_shape(i)
        else:
            # TensorRT 10.x
            num_bindings = engine.num_io_tensors
            get_binding_name = lambda i: engine.get_tensor_name(i)
            binding_is_input = lambda i: engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
            get_binding_dtype = lambda i: engine.get_tensor_dtype(engine.get_tensor_name(i))
            get_binding_shape = lambda i: None  # 需要从 context 获取
        
        print(f"  - Bindings 数量: {num_bindings}")
        print("-" * 80)
        
        # 检查数据类型
        fp16_found = False
        fp32_found = False
        input_info = []
        output_info = []
        
        print("\n输入/输出信息:")
        for i in range(num_bindings):
            name = get_binding_name(i)
            is_input = binding_is_input(i)
            dtype = get_binding_dtype(i)
            
            dtype_name = dtype.name if hasattr(dtype, 'name') else str(dtype)
            
            if dtype_name == 'HALF':
                fp16_found = True
                dtype_str = "FP16 (HALF) ✓"
            elif dtype_name == 'FLOAT':
                fp32_found = True
                dtype_str = "FP32 (FLOAT)"
            else:
                dtype_str = dtype_name
            
            io_type = "INPUT" if is_input else "OUTPUT"
            shape_str = ""
            try:
                shape = get_binding_shape(i)
                if shape:
                    shape_str = f" shape={tuple(shape)}"
            except:
                pass
            
            print(f"  [{i}] {io_type:6s} '{name:30s}' dtype={dtype_str:15s}{shape_str}")
            
            if is_input:
                input_info.append((name, dtype_name, dtype))
            else:
                output_info.append((name, dtype_name, dtype))
        
        print("-" * 80)
        
        # 总结
        print("\n精度检查结果:")
        if fp16_found:
            print("  ✓ Engine 包含 FP16 (HALF) 数据类型 - 支持 FP16 推理")
            if fp32_found:
                print("  ⚠ 同时包含 FP32 类型，可能是混合精度")
        elif fp32_found:
            print("  ✗ Engine 仅包含 FP32 (FLOAT) 数据类型 - 不支持 FP16 推理")
        else:
            print("  ? Engine 数据类型未知")
        
        # 测试 warmup
        if warmup_iterations > 0 and fp16_found:
            print(f"\n{'=' * 80}")
            print(f"测试 FP16 Warmup ({warmup_iterations} 次迭代)")
            print("=" * 80)
            
            try:
                context = engine.create_execution_context()
                
                # 准备输入数据（使用 FP16）
                gpu_buffers = []
                host_buffers = []
                
                for i, (name, dtype_name, dtype) in enumerate(input_info):
                    # 创建虚拟输入数据
                    if dtype_name == 'HALF':
                        # FP16 输入
                        shape = get_binding_shape(i) if hasattr(engine, 'get_binding_shape') else (1,)
                        if shape:
                            host_data = np.random.randn(*shape).astype(np.float16)
                        else:
                            host_data = np.random.randn(1).astype(np.float16)
                    else:
                        # FP32 或其他类型
                        shape = get_binding_shape(i) if hasattr(engine, 'get_binding_shape') else (1,)
                        if shape:
                            host_data = np.random.randn(*shape).astype(np.float32)
                        else:
                            host_data = np.random.randn(1).astype(np.float32)
                    
                    host_buffers.append(host_data)
                    
                    # 分配 GPU 内存
                    err, gpu_ptr = cudart.cudaMalloc(host_data.nbytes)
                    if err != 0:
                        print(f"[ERROR] 无法分配 GPU 内存: {cudart.cudaGetErrorString(err)}")
                        return False
                    gpu_buffers.append(gpu_ptr)
                    
                    # 复制到 GPU
                    err = cudart.cudaMemcpy(
                        gpu_ptr, host_data.ctypes.data, host_data.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
                    )
                    if err != 0:
                        print(f"[ERROR] 无法复制数据到 GPU: {cudart.cudaGetErrorString(err)}")
                        return False
                
                # 准备输出缓冲区
                for i, (name, dtype_name, dtype) in enumerate(output_info):
                    shape = get_binding_shape(i + len(input_info)) if hasattr(engine, 'get_binding_shape') else (1,)
                    if shape:
                        if dtype_name == 'HALF':
                            host_data = np.zeros(shape, dtype=np.float16)
                        else:
                            host_data = np.zeros(shape, dtype=np.float32)
                    else:
                        host_data = np.zeros(1, dtype=np.float32)
                    
                    host_buffers.append(host_data)
                    
                    err, gpu_ptr = cudart.cudaMalloc(host_data.nbytes)
                    if err != 0:
                        print(f"[ERROR] 无法分配 GPU 内存: {cudart.cudaGetErrorString(err)}")
                        return False
                    gpu_buffers.append(gpu_ptr)
                
                # 设置 tensor 地址（TensorRT 10.x）或使用 bindings（TensorRT 8.x）
                if hasattr(engine, 'num_io_tensors'):
                    # TensorRT 10.x
                    for i in range(num_bindings):
                        name = get_binding_name(i)
                        context.set_tensor_address(name, int(gpu_buffers[i]))
                else:
                    # TensorRT 8.x - 使用 bindings
                    pass
                
                # 执行 warmup
                print(f"\n执行 {warmup_iterations} 次 warmup 迭代...")
                import time
                start_time = time.time()
                
                for iteration in range(warmup_iterations):
                    if hasattr(engine, 'num_io_tensors'):
                        # TensorRT 10.x
                        context.execute_async_v3(0)
                    else:
                        # TensorRT 8.x
                        context.execute_async_v2(gpu_buffers, 0)
                    
                    if (iteration + 1) % 10 == 0:
                        print(f"  完成 {iteration + 1}/{warmup_iterations} 次迭代...", end='\r')
                
                # 同步
                cudart.cudaDeviceSynchronize()
                elapsed_time = time.time() - start_time
                
                print(f"\n✓ Warmup 完成！")
                print(f"  - 总时间: {elapsed_time:.3f} 秒")
                print(f"  - 平均每次: {elapsed_time/warmup_iterations*1000:.3f} ms")
                
                # 清理
                for gpu_ptr in gpu_buffers:
                    cudart.cudaFree(gpu_ptr)
                
                return True
                
            except Exception as e:
                print(f"\n✗ Warmup 测试失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="检查 TensorRT Engine 是否支持 FP16")
    parser.add_argument("engine_path", help="Engine 文件路径")
    parser.add_argument("--warmup-iterations", type=int, default=100,
                       help="Warmup 迭代次数 (默认: 100, 设为 0 跳过 warmup 测试)")
    parser.add_argument("--no-warmup", action="store_true",
                       help="跳过 warmup 测试")
    
    args = parser.parse_args()
    
    warmup_iter = 0 if args.no_warmup else args.warmup_iterations
    
    success = check_engine_fp16(args.engine_path, warmup_iter)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

