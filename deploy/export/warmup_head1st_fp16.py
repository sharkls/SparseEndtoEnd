#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载 sparse4dhead1st.engine 并执行 FP16 warmup
使用方法: python warmup_head1st_fp16.py [--engine-path PATH] [--iterations N]
"""

import os
import sys
import argparse
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)

def warmup_head1st_fp16(engine_path, iterations=100):
    """加载 head1st engine 并执行 FP16 warmup"""
    
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
    print(f"加载 Engine: {engine_path}")
    print("=" * 80)
    
    # 加载 plugin（如果存在）
    plugin_path = os.path.join(os.path.dirname(engine_path), "..", "dfa_plugin", "lib", "deformableAttentionAggr.so")
    plugin_path = os.path.normpath(plugin_path)
    
    if os.path.exists(plugin_path):
        print(f"\n加载 Plugin: {plugin_path}")
        # 使用 ctypes 加载 plugin（TensorRT 会自动注册）
        import ctypes
        try:
            ctypes.CDLL(plugin_path)
            print("✓ Plugin 加载成功")
        except Exception as e:
            print(f"⚠ 警告: Plugin 加载失败: {e}")
            print("  尝试继续运行（如果 engine 不需要 plugin）...")
    else:
        print(f"\n⚠ 警告: Plugin 文件不存在: {plugin_path}")
        print("  尝试继续运行（如果 engine 不需要 plugin）...")
    
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
        # TensorRT 8.x 使用 num_bindings，TensorRT 10.x 使用 num_io_tensors
        # 更准确的判断：检查是否有 get_binding_name 方法（TensorRT 8.x）
        trt_old = hasattr(engine, 'num_bindings') and hasattr(engine, 'get_binding_name')
        
        if trt_old:
            # TensorRT 8.x - 使用固定形状（静态形状）
            num_bindings = engine.num_bindings
            get_binding_name = lambda i: engine.get_binding_name(i)
            binding_is_input = lambda i: engine.binding_is_input(i)
            get_binding_dtype = lambda i: engine.get_binding_dtype(i)
            get_binding_shape = lambda i: engine.get_binding_shape(i)
        else:
            # TensorRT 10.x - 可能使用动态形状
            num_bindings = engine.num_io_tensors
            get_binding_name = lambda i: engine.get_tensor_name(i)
            binding_is_input = lambda i: engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
            get_binding_dtype = lambda i: engine.get_tensor_dtype(engine.get_tensor_name(i))
            get_binding_shape = lambda i: None  # 需要从 context 获取
        
        print(f"  - Bindings 数量: {num_bindings}")
        print("-" * 80)
        
        # 获取输入输出信息
        input_info = []
        output_info = []
        
        print("\n输入/输出信息:")
        for i in range(num_bindings):
            name = get_binding_name(i)
            is_input = binding_is_input(i)
            dtype = get_binding_dtype(i)
            
            dtype_name = dtype.name if hasattr(dtype, 'name') else str(dtype)
            
            if dtype_name == 'HALF':
                dtype_str = "FP16 (HALF) ✓"
            elif dtype_name == 'FLOAT':
                dtype_str = "FP32 (FLOAT)"
            elif dtype_name == 'INT32':
                dtype_str = "INT32"
            else:
                dtype_str = dtype_name
            
            io_type = "INPUT" if is_input else "OUTPUT"
            shape_str = ""
            try:
                if trt_old:
                    shape = get_binding_shape(i)
                else:
                    shape = None
                if shape:
                    shape_str = f" shape={tuple(shape)}"
            except:
                pass
            
            print(f"  [{i}] {io_type:6s} '{name:30s}' dtype={dtype_str:15s}{shape_str}")
            
            if is_input:
                input_info.append((i, name, dtype_name, dtype))
            else:
                output_info.append((i, name, dtype_name, dtype))
        
        print("-" * 80)
        
        # 创建执行上下文
        context = engine.create_execution_context()
        
        # 显示输入形状信息
        if trt_old:
            # TensorRT 8.x - 固定形状（静态形状）
            print("\n输入形状信息（固定形状）:")
            for i, (idx, name, dtype_name, dtype) in enumerate(input_info):
                shape = get_binding_shape(idx)
                if shape:
                    print(f"  Input '{name}': {tuple(shape)}")
        else:
            # TensorRT 10.x - 可能使用动态形状
            print("\n输入形状信息:")
            for i, (idx, name, dtype_name, dtype) in enumerate(input_info):
                shape = context.get_tensor_shape(name)
                if shape:
                    # 检查是否是动态形状（包含 -1）
                    is_dynamic = any(s < 0 for s in shape)
                    shape_type = "动态形状" if is_dynamic else "固定形状"
                    print(f"  Input '{name}' {shape_type}: {tuple(shape)}")
        
        # 准备输入数据（根据 head1st 的实际输入）
        print(f"\n{'=' * 80}")
        print(f"准备 FP16 输入数据")
        print("=" * 80)
        
        # head1st 的输入形状（batch=1）
        input_shapes = {
            "feature": (1, 89760, 256),
            "spatial_shapes": (6, 4, 2),
            "level_start_index": (6, 4),
            "instance_feature": (1, 900, 256),
            "anchor": (1, 900, 11),
            "time_interval": (1,),
            "image_wh": (1, 6, 2),
            "lidar2img": (1, 6, 4, 4),
        }
        
        gpu_buffers = []
        host_buffers = []
        
        # 准备输入数据
        for idx, name, dtype_name, dtype in input_info:
            # 获取形状
            if trt_old:
                shape = get_binding_shape(idx)
            else:
                shape = context.get_tensor_shape(name)
            
            # 如果形状未知，尝试使用预定义的形状
            if not shape or any(s < 0 for s in shape):
                if name in input_shapes:
                    shape = input_shapes[name]
                    print(f"  使用预定义形状: {name} = {shape}")
                else:
                    # 尝试推断形状
                    shape = (1,)  # 默认形状
                    print(f"  警告: 无法确定 '{name}' 的形状，使用默认形状 {shape}")
            
            # 根据数据类型创建数据
            if dtype_name == 'HALF':
                # FP16 输入
                host_data = np.random.randn(*shape).astype(np.float16)
            elif dtype_name == 'INT32':
                # INT32 输入（如 spatial_shapes, level_start_index）
                if name == "spatial_shapes":
                    # 预定义的 spatial_shapes
                    host_data = np.array([[[64, 176], [32, 88], [16, 44], [8, 22]]] * 6, dtype=np.int32)
                elif name == "level_start_index":
                    # 预定义的 level_start_index
                    host_data = np.array([[0, 11264, 14080, 14784, 14960, 26224, 29040, 29744, 29920, 41184, 44000, 44704, 44880, 56144, 58960, 59664, 59840, 71104, 73920, 74624, 74800, 86064, 88880, 89584]] * 6, dtype=np.int32)
                else:
                    host_data = np.random.randint(0, 100, size=shape, dtype=np.int32)
            else:
                # FP32 或其他类型
                host_data = np.random.randn(*shape).astype(np.float32)
            
            print(f"  [{idx}] Input '{name:30s}': shape={tuple(host_data.shape)}, dtype={host_data.dtype}")
            host_buffers.append(host_data)
            
            # 分配 GPU 内存
            err, gpu_ptr = cudart.cudaMalloc(host_data.nbytes)
            if err != cudart.cudaError_t.cudaSuccess:
                print(f"[ERROR] 无法分配 GPU 内存: {err}")
                return False
            gpu_buffers.append(gpu_ptr)
            
            # 复制到 GPU
            err = cudart.cudaMemcpy(
                gpu_ptr, host_data.ctypes.data, host_data.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            )
            # cudaMemcpy 返回 (cudaError_t,)
            if isinstance(err, tuple):
                err = err[0]
            if err != cudart.cudaError_t.cudaSuccess:
                print(f"[ERROR] 无法复制数据到 GPU: {err}")
                return False
        
        # 准备输出缓冲区
        print(f"\n准备输出缓冲区:")
        for idx, name, dtype_name, dtype in output_info:
            # 获取形状
            if trt_old:
                shape = get_binding_shape(idx)
            else:
                shape = context.get_tensor_shape(name)
            
            if not shape or any(s < 0 for s in shape):
                # 尝试使用预定义的输出形状（根据 trtexec 输出）
                if name == "pred_instance_feature":
                    shape = (1, 900, 256)
                elif name == "pred_anchor":
                    shape = (1, 900, 11)
                elif name == "pred_class_score":
                    shape = (1, 900, 10)
                elif name == "pred_quality_score":
                    shape = (1, 900, 2)  # 根据 trtexec 输出，实际是 1x900x2
                else:
                    shape = (1,)  # 默认形状
                print(f"  警告: 无法确定 '{name}' 的形状，使用推断形状 {shape}")
            
            # 根据数据类型创建输出缓冲区
            if dtype_name == 'HALF':
                host_data = np.zeros(shape, dtype=np.float16)
            elif dtype_name == 'INT32':
                host_data = np.zeros(shape, dtype=np.int32)
            else:
                host_data = np.zeros(shape, dtype=np.float32)
            
            print(f"  [{idx}] Output '{name:30s}': shape={tuple(host_data.shape)}, dtype={host_data.dtype}")
            host_buffers.append(host_data)
            
            err, gpu_ptr = cudart.cudaMalloc(host_data.nbytes)
            if err != cudart.cudaError_t.cudaSuccess:
                print(f"[ERROR] 无法分配 GPU 内存: {err}")
                return False
            gpu_buffers.append(gpu_ptr)
        
        # 设置 tensor 地址
        print(f"\n{'=' * 80}")
        print(f"设置 Tensor 地址并执行 Warmup ({iterations} 次迭代)")
        print("=" * 80)
        
        if trt_old:
            # TensorRT 8.x - 使用 bindings
            binding_addrs = [int(ptr) for ptr in gpu_buffers]
        else:
            # TensorRT 10.x - 设置 tensor 地址
            for i, (idx, name, dtype_name, dtype) in enumerate(input_info):
                context.set_tensor_address(name, int(gpu_buffers[i]))
            for i, (idx, name, dtype_name, dtype) in enumerate(output_info):
                context.set_tensor_address(name, int(gpu_buffers[len(input_info) + i]))
        
        # 执行 warmup
        print(f"\n执行 {iterations} 次 warmup 迭代...")
        start_time = time.time()
        
        for iteration in range(iterations):
            if trt_old:
                # TensorRT 8.x
                context.execute_async_v2(binding_addrs, 0)
            else:
                # TensorRT 10.x
                context.execute_async_v3(0)
            
            if (iteration + 1) % 10 == 0:
                print(f"  完成 {iteration + 1}/{iterations} 次迭代...", end='\r')
        
        # 同步
        cudart.cudaDeviceSynchronize()
        elapsed_time = time.time() - start_time
        
        print(f"\n\n✓ FP16 Warmup 完成！")
        print(f"  - 总时间: {elapsed_time:.3f} 秒")
        print(f"  - 平均每次: {elapsed_time/iterations*1000:.3f} ms")
        print(f"  - 吞吐量: {iterations/elapsed_time:.2f} iterations/sec")
        
        # 清理
        for gpu_ptr in gpu_buffers:
            cudart.cudaFree(gpu_ptr)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Warmup 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="加载 sparse4dhead1st.engine 并执行 FP16 warmup")
    parser.add_argument("--engine-path", type=str,
                       default="/share/Code/Sparse4dE2E/deploy/engine/sparse4dhead1st.engine",
                       help="Engine 文件路径 (默认: deploy/engine/sparse4dhead1st.engine)")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Warmup 迭代次数 (默认: 100)")
    
    args = parser.parse_args()
    
    success = warmup_head1st_fp16(args.engine_path, args.iterations)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

