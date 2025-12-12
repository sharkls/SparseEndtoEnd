# Script to check TensorRT engine binding data types
import tensorrt as trt
import sys
import os

def check_engine(engine_path):
    if not os.path.exists(engine_path):
        print(f"Error: File not found: {engine_path}")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    try:
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if not engine:
                print(f"Failed to load engine: {engine_path}")
                return
            
            print(f"Engine: {engine_path}")
            print(f"Number of bindings: {engine.num_bindings}")
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                dtype = engine.get_binding_dtype(i)
                shape = engine.get_binding_shape(i)
                is_input = engine.binding_is_input(i)
                
                dtype_str = str(dtype)
                if dtype == trt.DataType.FLOAT: dtype_str = "FLOAT (FP32)"
                elif dtype == trt.DataType.HALF: dtype_str = "HALF (FP16)"
                elif dtype == trt.DataType.INT8: dtype_str = "INT8"
                elif dtype == trt.DataType.INT32: dtype_str = "INT32"
                elif dtype == trt.DataType.BOOL: dtype_str = "BOOL"
                
                type_str = "Input" if is_input else "Output"
                print(f"  Binding {i} [{type_str}]: {name} | Type: {dtype_str} | Shape: {shape}")
                
    except Exception as e:
        print(f"Error analyzing engine: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Check all engines
        base_path = "/share/Code/Sparse4dE2E/deploy/engine/"
        engines = ["sparse4dbackbone.engine", "sparse4dhead1st.engine", "sparse4dhead2nd.engine"]
        for eng in engines:
            path = os.path.join(base_path, eng)
            check_engine(path)
            print("-" * 50)
    else:
        check_engine(sys.argv[1])
