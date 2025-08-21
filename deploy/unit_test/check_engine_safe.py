# check_engine_fixed.py
import tensorrt as trt
import os
import sys
import ctypes

def load_custom_plugins():
    """显式加载自定义插件"""
    print("=== Loading Custom Plugins ===")
    
    try:
        # 插件库路径
        plugin_path = "/share/Code/SparseEnd2End/deploy/dfa_plugin/lib/deformableAttentionAggr.so"
        
        if not os.path.exists(plugin_path):
            print(f"✗ Plugin library not found: {plugin_path}")
            return False
        
        # 显式加载插件库
        print(f"Loading plugin library: {plugin_path}")
        ctypes.cdll.LoadLibrary(plugin_path)
        print("✓ Plugin library loaded successfully")
        
        # 初始化TensorRT插件
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.INFO), "")
        print("✓ TensorRT plugins initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load plugins: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_engine_fixed(engine_path):
    """修复后的引擎检查"""
    try:
        print(f"\nChecking engine: {engine_path}")
        
        if not os.path.exists(engine_path):
            print(f"Error: Engine file not found: {engine_path}")
            return False
        
        # 检查文件大小
        file_size = os.path.getsize(engine_path)
        print(f"  - File size: {file_size / (1024*1024):.2f} MB")
        
        # 创建logger
        logger = trt.Logger(trt.Logger.INFO)
        print("✓ Logger created successfully")
        
        # 读取文件
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        print(f"✓ File read successfully, {len(engine_data)} bytes")
        
        # 反序列化引擎
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            print("✗ Error: Engine deserialization returned None")
            return False
            
        print("✓ Engine deserialized successfully!")
        
        # 检查基本信息 - 使用兼容的API
        try:
            num_layers = engine.num_layers
            print(f"  - Number of layers: {num_layers}")
        except Exception as e:
            print(f"✗ Failed to get num_layers: {e}")
            return False
        
        try:
            num_bindings = engine.num_bindings
            print(f"  - Number of bindings: {num_bindings}")
        except Exception as e:
            print(f"✗ Failed to get num_bindings: {e}")
            return False
        
        # 尝试获取workspace size（兼容不同版本）
        try:
            if hasattr(engine, 'max_workspace_size'):
                max_workspace = engine.max_workspace_size
                print(f"  - Max workspace size: {max_workspace}")
            elif hasattr(engine, 'get_max_workspace_size'):
                max_workspace = engine.get_max_workspace_size()
                print(f"  - Max workspace size: {max_workspace}")
            else:
                print("  - Max workspace size: Unknown (API not available)")
        except Exception as e:
            print(f"  - Max workspace size: Error getting value: {e}")
        
        # 检查绑定信息
        print("\n=== Bindings Information ===")
        try:
            for i in range(num_bindings):
                try:
                    name = engine.get_binding_name(i)
                    print(f"  Binding {i}: {name}")
                    
                    # 获取绑定类型
                    try:
                        dtype = engine.get_binding_dtype(i)
                        print(f"    Type: {dtype}")
                    except Exception as e:
                        print(f"    Type: Error - {e}")
                    
                    # 获取绑定形状
                    try:
                        shape = engine.get_binding_shape(i)
                        print(f"    Shape: {shape}")
                    except Exception as e:
                        print(f"    Shape: Error - {e}")
                    
                    # 检查是否为输入
                    try:
                        is_input = engine.binding_is_input(i)
                        print(f"    Is Input: {is_input}")
                    except Exception as e:
                        print(f"    Is Input: Error - {e}")
                    
                    print()
                    
                except Exception as e:
                    print(f"Error getting binding {i}: {e}")
                    break
                    
        except Exception as e:
            print(f"✗ Failed to check bindings: {e}")
            return False
        
        # 尝试创建执行上下文
        print("\n=== Execution Context Test ===")
        try:
            context = engine.create_execution_context()
            print("✓ Execution context created successfully!")
            
            # 检查上下文属性
            try:
                if hasattr(context, 'num_optimization_profiles'):
                    num_profiles = context.num_optimization_profiles
                    print(f"  - Optimization profiles: {num_profiles}")
                else:
                    print("  - Optimization profiles: API not available")
            except Exception as e:
                print(f"  - Optimization profiles: Error - {e}")
            
            # 清理上下文
            del context
            print("✓ Execution context cleaned up")
            
        except Exception as e:
            print(f"✗ Failed to create execution context: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error during engine check: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_engines():
    """对比两个引擎"""
    print("\n=== Engine Comparison ===")
    
    engine1_path = "/share/Code/SparseEnd2End/deploy/engine/sparse4dhead1st.engine"
    engine2_path = "/share/Code/SparseEnd2End/deploy/engine/sparse4dhead2nd.engine"
    
    # 获取文件信息
    if os.path.exists(engine1_path):
        size1 = os.path.getsize(engine1_path) / (1024*1024)
        print(f"Engine 1 size: {size1:.2f} MB")
    
    if os.path.exists(engine2_path):
        size2 = os.path.getsize(engine2_path) / (1024*1024)
        print(f"Engine 2 size: {size2:.2f} MB")
    
    if os.path.exists(engine1_path) and os.path.exists(engine2_path):
        size_diff = abs(size2 - size1)
        print(f"Size difference: {size_diff:.2f} MB")
        
        if size_diff > 1.0:  # 如果差异大于1MB
            print("⚠ Significant size difference detected!")
            print("This might indicate different model structures or optimizations")

if __name__ == "__main__":
    # 首先加载自定义插件
    if not load_custom_plugins():
        print("Failed to load plugins, exiting...")
        sys.exit(1)
    
    # 检查引擎
    default_engines = [
        "/share/Code/SparseEnd2End/deploy/engine/sparse4dhead1st.engine",
        "/share/Code/SparseEnd2End/deploy/engine/sparse4dhead2nd.engine"
    ]
    
    all_success = True
    for engine_path in default_engines:
        if os.path.exists(engine_path):
            print(f"\n{'='*60}")
            print(f"CHECKING: {engine_path}")
            print(f"{'='*60}")
            
            success = check_engine_fixed(engine_path)
            if not success:
                all_success = False
            
            print()
        else:
            print(f"Engine not found: {engine_path}")
            all_success = False
    
    # 对比引擎
    compare_engines()
    
    # 总结
    print(f"\n{'='*60}")
    if all_success:
        print("✓ ALL ENGINES CHECKED SUCCESSFULLY!")
        print("The engines are valid and can be loaded.")
        print("The previous segmentation faults were likely due to missing plugin libraries.")
    else:
        print("✗ SOME ENGINES FAILED TO CHECK!")
        print("Please review the errors above.")
    print(f"{'='*60}")