import tensorrt as trt
import numpy as np
import os
import ctypes
import sys
import argparse
import time

# Use custom cuda_utils to avoid dependency on cuda-python
try:
    from cuda import cudart
except ImportError:
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            import cuda_utils as cudart
        except ImportError:
            from deploy.tools import cuda_utils as cudart

def parse_args():
    parser = argparse.ArgumentParser(description="Test TensorRT Engine Latency")
    parser.add_argument("--engine", type=str, required=True, help="Path to the TensorRT engine file")
    parser.add_argument("--profile", action="store_true", help="Enable per-layer profiling")
    return parser.parse_args()

class Profiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layers = {}

    def report_layer_time(self, layer_name, ms):
        if layer_name not in self.layers:
            self.layers[layer_name] = []
        self.layers[layer_name].append(ms)

    def print_layer_times(self):
        print("\n=== Layer Profiling Results ===")
        # Sort by average time descending
        sorted_layers = sorted(self.layers.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
        
        print(f"{'Layer Name':<80} | {'Avg Time (ms)':<15} | {'% of Total':<10}")
        print("-" * 110)
        
        total_time = sum([sum(times)/len(times) for _, times in self.layers.items()])
        
        for name, times in sorted_layers:
            avg_time = sum(times) / len(times)
            percent = (avg_time / total_time) * 100
            print(f"{name[:77]+'...' if len(name)>77 else name:<80} | {avg_time:.4f}          | {percent:.2f}%")
        print("-" * 110)
        print(f"{'Total':<80} | {total_time:.4f}          | 100.00%")

# 加载插件
# 尝试从常见位置加载插件，或者通过环境变量
PLUGIN_PATHS = [
    "deploy/dfa_plugin/lib/deformableAttentionAggr.so",
    "deploy/ln_plugin/lib/customLayerNorm.so",
    "deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so"
]

# 获取当前脚本所在目录的父目录的父目录作为项目根目录假设
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_plugins():
    loaded = 0
    for path in PLUGIN_PATHS:
        # 尝试相对路径
        full_path = path
        if not os.path.exists(full_path):
            # 尝试相对于项目根目录
            full_path = os.path.join(PROJECT_ROOT, path)
        
        if os.path.exists(full_path):
            try:
                ctypes.CDLL(full_path)
                print(f"[INFO] Loaded plugin: {full_path}")
                loaded += 1
            except Exception as e:
                print(f"[ERROR] Failed to load plugin {full_path}: {e}")
        else:
            print(f"[WARNING] Plugin not found: {path}")
    return loaded

def validate_engine(engine_path, enable_profile=False):
    if not os.path.exists(engine_path):
        print(f"[ERROR] Engine file not found at {engine_path}")
        return

    load_plugins()

    logger = trt.Logger(trt.Logger.WARNING) # 减少日志干扰
    try:
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            print(f"[INFO] Deserializing engine from {engine_path}...")
            engine = runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        print(f"[ERROR] Exception during deserialization: {e}")
        return

    if engine is None:
        print(f"[ERROR] Failed to load engine: {engine_path}")
        return

    print(f"[INFO] Successfully loaded engine.")
    
    try:
        context = engine.create_execution_context()
        profiler = None
        if enable_profile:
            profiler = Profiler()
            context.profiler = profiler
            # Ensure profiling is enabled in the context
            # In some TRT versions, this is all that's needed.
            # But we must ensure 'execute_async_v3' or similar calls trigger it.
    except Exception as e:
        print(f"[ERROR] Exception during context creation: {e}")
        return
    
    # 简单的 dummy input 推理测试
    allocations = []
    stream = 0
    
    try:
        err, stream = cudart.cudaStreamCreate()
        
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            
            # 处理动态维度 (-1) - 这里假设 batch size 为 1
            start_dims = [1 if x == -1 else x for x in shape]
            
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                 context.set_input_shape(name, start_dims)
            
            vol = 1
            for d in start_dims:
                vol *= d
            
            size = vol * np.dtype(trt.nptype(dtype)).itemsize
            # Align size to 256 bytes
            size = (size + 255) // 256 * 256
            
            err, ptr = cudart.cudaMalloc(size)
            if err != cudart.cudaError_t.cudaSuccess:
                 print(f"[ERROR] cudaMalloc failed for {name}")
                 return
            allocations.append(ptr)
            context.set_tensor_address(name, ptr)
            
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                # Initialize with zeros
                cudart.cudaMemset(ptr, 0, size) 

        print("[INFO] Starting Warmup (20 iterations)...")
        for _ in range(20):
            context.execute_async_v3(stream)
        cudart.cudaStreamSynchronize(stream)
        
        print("[INFO] Starting Benchmark (100 iterations)...")
        iterations = 100
        start_time = time.time()
        for _ in range(iterations):
            context.execute_async_v3(stream)
        cudart.cudaStreamSynchronize(stream)
        end_time = time.time()
        
        avg_latency = (end_time - start_time) * 1000 / iterations
        print(f"============================================")
        print(f"Engine: {os.path.basename(engine_path)}")
        print(f"Average Latency: {avg_latency:.4f} ms")
        print(f"============================================")
        
        if enable_profile and profiler:
            profiler.print_layer_times()
        
    except Exception as e:
        print(f"[ERROR] Exception during inference: {e}")
    finally:
        # 清理
        for ptr in allocations:
            cudart.cudaFree(ptr)
        cudart.cudaStreamDestroy(stream)

if __name__ == "__main__":
    args = parse_args()
    validate_engine(args.engine, args.profile)
