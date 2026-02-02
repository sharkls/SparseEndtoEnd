import tensorrt as trt
import os
import numpy as np
import torch
import sys
import time

# 1. 日志记录辅助类
class Tee(object):
    def __init__(self, name, mode='w'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

# 2. 层级耗时统计类
class SimpleProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.layers = {}
    def report_layer_time(self, layer_name, ms):
        if layer_name not in self.layers:
            self.layers[layer_name] = []
        self.layers[layer_name].append(ms)
    def print_summary(self, log_file):
        with open(log_file, 'w') as f:
            f.write("=== Layer Profiling (Average Latency) ===\n")
            total_time = 0
            # 按耗时从高到低排序
            sorted_layers = sorted(self.layers.items(), key=lambda x: np.mean(x[1]), reverse=True)
            for name, times in sorted_layers:
                avg_time = np.mean(times)
                total_time += avg_time
                f.write(f"{avg_time:8.4f} ms | {name}\n")
            f.write(f"-----------------------------------------\n")
            f.write(f"Total GPU Compute Time: {total_time:.4f} ms\n")
        print(f"[Profiler] Detailed layer timing saved to {log_file}")

class Sparse4DCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir, cache_file, node_key, shape, num_samples=100):
        super().__init__()
        self.data_dir = data_dir
        self.cache_file = cache_file
        self.shape = shape
        self.current_index = 0
        self.all_files = sorted([f for f in os.listdir(data_dir) 
                                if f.startswith("sample_") and f"_{node_key}_" in f 
                                and "_ori_imgs_" not in f and f.endswith(".bin")])
        self.all_files = self.all_files[:num_samples]
        self.device_input = torch.zeros(shape, dtype=torch.float32, device='cuda')
        print(f"[Calibrator] {node_key}: Found {len(self.all_files)} samples.")

    def get_batch_size(self): return 1
    def get_batch(self, names):
        if self.current_index >= len(self.all_files): return None
        file_path = os.path.join(self.data_dir, self.all_files[self.current_index])
        if self.current_index % 20 == 0:
            print(f"[Calibrator] Processing batch {self.current_index}/{len(self.all_files)}")
        data = np.fromfile(file_path, dtype=np.float32).reshape(self.shape)
        self.device_input.copy_(torch.from_numpy(data))
        self.current_index += 1
        return [int(self.device_input.data_ptr())]
    def read_calibration_cache(self): return None 
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f: f.write(cache)
        print(f"[Calibrator] Saved fresh cache to {self.cache_file}")

def run_profiling(engine_path, shape, profile_log):
    print(f"[Profiler] Starting performance analysis for {engine_path}...")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    profiler = SimpleProfiler()
    context.profiler = profiler
    
    # 准备输入
    d_input = torch.zeros(shape, dtype=torch.float32, device='cuda')
    bindings = [int(d_input.data_ptr())]
    # 添加输出绑定
    for i in range(1, engine.num_bindings):
        out_shape = engine.get_binding_shape(i)
        d_out = torch.zeros(tuple(out_shape), dtype=torch.float16, device='cuda') # 假设输出是fp16
        bindings.append(int(d_out.data_ptr()))

    # 预热
    for _ in range(5): context.execute_v2(bindings)
    
    # 统计 10 次推理
    for _ in range(10):
        context.execute_v2(bindings)
    
    profiler.print_summary(profile_log)

def build_engine(onnx_path, engine_path, cache_path, node_key, shape):
    log_file = engine_path.replace(".engine", "_build.log")
    profile_log = engine_path.replace(".engine", "_layer_timing.log")
    _tee = Tee(log_file) # 开始重定向

    if os.path.exists(cache_path): os.remove(cache_path)
    
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    print(f"\n[Build] Target: {engine_path}")
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    if "backbone" in onnx_path:
        profile = builder.create_optimization_profile()
        profile.set_shape("img", shape, shape, shape)
        config.add_optimization_profile(profile)

    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.int8_calibrator = Sparse4DCalibrator("deploy/val_data_e2e_fp32", cache_path, node_key, shape)
    
    engine_data = builder.build_serialized_network(network, config)
    if engine_data:
        with open(engine_path, "wb") as f: f.write(engine_data)
        print(f"🚀 SUCCESS: {engine_path}")
        # 构建成功后运行性能分析
        run_profiling(engine_path, shape, profile_log)
    
    del _tee # 恢复 stdout

if __name__ == "__main__":
    fixed_onnx = "deploy/onnx/sparse4dbackbone_fixed.onnx"
    if not os.path.exists(fixed_onnx):
        os.system(f"python3 deploy/fix_onnx.py")

    build_engine(
        onnx_path=fixed_onnx,
        engine_path="deploy/engine/sparse4dbackbone_int8_real.engine",
        cache_path="deploy/engine/backbone_int8.cache",
        node_key="imgs",
        shape=(1, 6, 3, 256, 704)
    )
