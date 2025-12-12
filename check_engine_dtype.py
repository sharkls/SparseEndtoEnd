import tensorrt as trt
import sys

logger = trt.Logger(trt.Logger.WARNING)
engine_path = "/share/Code/Sparse4dE2E/deploy/engine/sparse4dbackbone.engine"

try:
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if not engine:
            print("Failed to load engine")
            sys.exit(1)
        
        print(f"Engine: {engine_path}")
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            print(f"Binding {i}: {name} - {dtype}")
            
except Exception as e:
    print(f"Error: {e}")
