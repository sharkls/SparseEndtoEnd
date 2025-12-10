import onnx
import sys
"""
用来验证onnx模型的输入输出精度
使用方式：python check_onnx_precision.py <onnx_path>
示例：python check_onnx_precision.py deploy/onnx/sparse4dbackbone.onnx
"""

if len(sys.argv) != 2:
    print("Usage: python check_onnx_precision.py <onnx_path>")
    sys.exit(1)

onnx_path = sys.argv[1]
try:
    model = onnx.load(onnx_path)
    input_tensor = model.graph.input[0]
    elem_type = input_tensor.type.tensor_type.elem_type
    
    print(f"Checking ONNX: {onnx_path}")
    print(f"Input Name: {input_tensor.name}")
    
    # ONNX TensorProto.DataType: FLOAT=1, FLOAT16=10
    if elem_type == 1:
        print("Input Type: FLOAT (FP32)")
    elif elem_type == 10:
        print("Input Type: FLOAT16 (FP16)")
    else:
        print(f"Input Type: {elem_type} (Unknown)")
        
except Exception as e:
    print(f"Error loading ONNX: {e}")

