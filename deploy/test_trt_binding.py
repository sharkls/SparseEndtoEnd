#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    engine_path = "/share/Code/Sparse4dE2E/deploy/engine/sparse4dbackbone.engine"
    if len(sys.argv) > 1:
        engine_path = sys.argv[1]
    if not os.path.isfile(engine_path):
        print(f"[ERROR] engine 文件不存在: {engine_path}")
        sys.exit(1)

    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
        assert engine, "反序列化失败"

    print(f"Engine: {engine_path}")
    print(f"- TensorRT version: {trt.__version__}")
    print(f"- Explicit batch: {not engine.has_implicit_batch_dimension}")
    print(f"- Num bindings: {engine.num_bindings}")
    print("-" * 60)

    for b in range(engine.num_bindings):
        name = engine.get_binding_name(b)
        is_input = engine.binding_is_input(b)
        dtype = engine.get_binding_dtype(b)      # trt.DataType
        shape = engine.get_binding_shape(b)
        fmt = engine.get_binding_format(b) if hasattr(engine, "get_binding_format") else None
        prof = engine.binding_profile_index(b) if hasattr(engine, "binding_profile_index") else 0
        print(f"[{b}] name='{name}' "
              f"type={'INPUT' if is_input else 'OUTPUT'} "
              f"dtype={getattr(dtype,'name',dtype)} "
              f"shape={tuple(shape)} profile={prof} "
              f"format={getattr(fmt,'name',fmt)}")

    print("-" * 60)
    print("提示：若输入绑定 dtype=INT8，才需要按量化 scale/zero_point 提供 int8；"
          "否则按 FLOAT/HALF 直接喂浮点数据。")

if __name__ == "__main__":
    main()