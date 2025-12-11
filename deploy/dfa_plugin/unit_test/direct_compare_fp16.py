#!/usr/bin/env python3
"""
直接对比 PyTorch DFA CUDA 实现和 TRT 插件实现 (FP16模式)
"""
import os
import sys
import numpy as np
import torch
import ctypes
import tensorrt as trt

# 添加项目根目录到 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from modules.ops import e2e_deformable_aggregation_ext


def load_bin(path, shape, dtype=np.float32):
    """加载 bin 文件"""
    data = np.fromfile(path, dtype=dtype)
    return torch.from_numpy(data.reshape(shape)).cuda()


def build_and_run_trt_dfa(value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
    """构建并运行 TRT DFA 插件"""
    # 加载插件
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    ctypes.CDLL("deploy/dfa_plugin/lib/deformableAttentionAggr.so")
    
    # 创建简单的 ONNX wrapper 来调用插件
    from torch.autograd.function import Function
    
    class DFAFunction(Function):
        @staticmethod
        def symbolic(g, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
            return g.op("custom::DeformableAttentionAggrPlugin",
                        mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)
        @staticmethod
        def forward(ctx, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
            return mc_ms_feat[:, :sampling_location.shape[1], :]  # dummy output with correct shape
    
    class DFAWrapper(torch.nn.Module):
        def forward(self, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
            return DFAFunction.apply(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)
    
    model = DFAWrapper().eval().cuda()
    
    # 导出 ONNX
    onnx_file = "/tmp/tmp_dfa_direct_fp16.onnx"
    torch.onnx.export(model, (value, spatial_shapes.int(), level_start_index.int(), sampling_locations, attention_weights),
                      onnx_file, input_names=["value", "spatial_shapes", "level_start_index", "sampling_locations", "attention_weights"],
                      output_names=["output"], opset_version=13)
    
    # 构建 TRT 引擎
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16) # Enable FP16
    
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_file, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    plan = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    
    # 准备输入输出
    inputs = {
        "value": value.contiguous(),
        "spatial_shapes": spatial_shapes.int().contiguous(),
        "level_start_index": level_start_index.int().contiguous(),
        "sampling_locations": sampling_locations.contiguous(),
        "attention_weights": attention_weights.contiguous()
    }
    
    for name, tensor in inputs.items():
        idx = engine.get_binding_index(name)
        context.set_binding_shape(idx, tensor.shape)
    
    # 分配输出
    output_shape = context.get_binding_shape(engine.get_binding_index("output"))
    output = torch.empty(tuple(output_shape), dtype=torch.float16, device='cuda')
    
    # 设置绑定
    bindings = [None] * engine.num_bindings
    for name, tensor in inputs.items():
        idx = engine.get_binding_index(name)
        bindings[idx] = tensor.data_ptr()
    bindings[engine.get_binding_index("output")] = output.data_ptr()
    
    # 执行
    context.execute_v2(bindings)
    torch.cuda.synchronize()
    
    return output


def benchmark_trt_dfa(engine, context, inputs, output_ptr, num_iterations=100):
    """Benchmark TRT DFA Plugin"""
    # Warmup
    bindings = [None] * engine.num_bindings
    for name, tensor in inputs.items():
        idx = engine.get_binding_index(name)
        bindings[idx] = tensor.data_ptr()
    bindings[engine.get_binding_index("output")] = output_ptr
    
    for _ in range(10):
        context.execute_v2(bindings)
    torch.cuda.synchronize()
    
    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        context.execute_v2(bindings)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_time = start_event.elapsed_time(end_event) / num_iterations
    return avg_time


def main():
    # 加载数据
    data_dir = "deploy/val_data_plugin/real_data/sample_0/dfa_0"
    
    # 加载输入 (FP16)
    try:
        value = load_bin(f"{data_dir}/sample_0_value_1*89760*256_float16.bin", (1, 89760, 256), np.float16)
        spatial_shapes = load_bin(f"{data_dir}/sample_0_spatial_shapes_6*4*2_int64.bin", (6, 4, 2), np.int64)
        level_start_index = load_bin(f"{data_dir}/sample_0_level_start_index_6*4_int64.bin", (6, 4), np.int64)
        sampling_locations = load_bin(f"{data_dir}/sample_0_sampling_locations_1*900*13*6*2_float16.bin", (1, 900, 13, 6, 2), np.float16)
        attention_weights = load_bin(f"{data_dir}/sample_0_attention_weights_1*900*13*6*4*8_float16.bin", (1, 900, 13, 6, 4, 8), np.float16)
    except FileNotFoundError as e:
        print(f"Error loading FP16 data: {e}")
        return

    print("=" * 60)
    print("Input shapes:")
    print(f"  value: {value.shape}, dtype={value.dtype}")
    print(f"  spatial_shapes: {spatial_shapes.shape}, dtype={spatial_shapes.dtype}")
    print(f"  level_start_index: {level_start_index.shape}, dtype={level_start_index.dtype}")
    print(f"  sampling_locations: {sampling_locations.shape}, dtype={sampling_locations.dtype}")
    print(f"  attention_weights: {attention_weights.shape}, dtype={attention_weights.dtype}")
    
    # 转换为 PyTorch 期望的类型
    spatial_shapes_int = spatial_shapes.int()
    level_start_index_int = level_start_index.int()
    
    print("\n" + "=" * 60)
    print("Running PyTorch CUDA implementation (FP16 inputs converted to FP32)...")
    
    # 调用 PyTorch CUDA 实现 (PyTorch implementation only supports FP32)
    pytorch_output = e2e_deformable_aggregation_ext.deformable_aggregation_forward(
        value.float().contiguous(),
        spatial_shapes_int.contiguous(),
        level_start_index_int.contiguous(),
        sampling_locations.float().contiguous(),
        attention_weights.float().contiguous()
    )
    
    print(f"  PyTorch output shape: {pytorch_output.shape}")
    print(f"  PyTorch output range: [{pytorch_output.min().item():.6f}, {pytorch_output.max().item():.6f}]")
    
    print("\n" + "=" * 60)
    print("Running TRT DFA Plugin (FP16 with Gather Optimization)...")
    
    # Build engine for benchmarking
    # 加载插件
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    ctypes.CDLL("deploy/dfa_plugin/lib/deformableAttentionAggr.so")
    
    # 创建简单的 ONNX wrapper 来调用插件
    from torch.autograd.function import Function
    
    class DFAFunction(Function):
        @staticmethod
        def symbolic(g, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
            return g.op("custom::DeformableAttentionAggrPlugin",
                        mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)
        @staticmethod
        def forward(ctx, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
            return mc_ms_feat[:, :sampling_location.shape[1], :]  # dummy output with correct shape
    
    class DFAWrapper(torch.nn.Module):
        def forward(self, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
            return DFAFunction.apply(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)
    
    model = DFAWrapper().eval().cuda()
    
    # 导出 ONNX
    onnx_file = "/tmp/tmp_dfa_direct_fp16.onnx"
    torch.onnx.export(model, (value, spatial_shapes.int(), level_start_index.int(), sampling_locations, attention_weights),
                      onnx_file, input_names=["value", "spatial_shapes", "level_start_index", "sampling_locations", "attention_weights"],
                      output_names=["output"], opset_version=13)
    
    # 构建 TRT 引擎
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16) # Enable FP16
    
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_file, 'rb') as f:
        parser.parse(f.read())
    
    plan = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    
    # 准备输入输出
    inputs = {
        "value": value.contiguous(),
        "spatial_shapes": spatial_shapes.int().contiguous(),
        "level_start_index": level_start_index.int().contiguous(),
        "sampling_locations": sampling_locations.contiguous(),
        "attention_weights": attention_weights.contiguous()
    }
    
    for name, tensor in inputs.items():
        idx = engine.get_binding_index(name)
        context.set_binding_shape(idx, tensor.shape)
    
    # 分配输出
    output_shape = context.get_binding_shape(engine.get_binding_index("output"))
    output = torch.empty(tuple(output_shape), dtype=torch.float16, device='cuda')
    
    # 设置绑定
    bindings = [None] * engine.num_bindings
    for name, tensor in inputs.items():
        idx = engine.get_binding_index(name)
        bindings[idx] = tensor.data_ptr()
    bindings[engine.get_binding_index("output")] = output.data_ptr()
    
    # 执行一次验证
    context.execute_v2(bindings)
    torch.cuda.synchronize()
    trt_output = output
    
    if trt_output is not None:
        print(f"  TRT output shape: {trt_output.shape}")
        print(f"  TRT output range: [{trt_output.min().item():.6f}, {trt_output.max().item():.6f}]")
        
        # 对比 PyTorch 和 TRT 输出
        # Convert TRT output to float for comparison
        diff = (trt_output.float() - pytorch_output).abs()
        print(f"\n  TRT vs PyTorch output:")
        print(f"    Max Diff: {diff.max().item():.10f}")
        print(f"    Mean Diff: {diff.mean().item():.10f}")
        
        # 找出最大差异的位置
        max_idx = diff.argmax().item()
        print(f"    Max diff at index {max_idx}: TRT={trt_output.flatten()[max_idx].item():.6f}, PyTorch={pytorch_output.flatten()[max_idx].item():.6f}")
        
        # 打印一些样本值对比
        print(f"\n  Sample values comparison (first 10):")
        for i in range(10):
            print(f"    [{i}] TRT={trt_output.flatten()[i].item():.6f}, PyTorch={pytorch_output.flatten()[i].item():.6f}, Diff={diff.flatten()[i].item():.6f}")

        # Benchmark
        print("\n" + "=" * 60)
        print("Benchmarking TRT DFA Plugin...")
        avg_latency = benchmark_trt_dfa(engine, context, inputs, output.data_ptr())
        print(f"  Average Latency: {avg_latency:.4f} ms")
    else:
        print("  TRT engine build failed!")



if __name__ == "__main__":
    main()

