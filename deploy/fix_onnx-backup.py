import onnx
import argparse
import numpy as np

# 修复 ONNX 1.16+ 与 graphsurgeon 的兼容性补丁
if not hasattr(onnx, "mapping"):
    class OnnxMapping:
        from onnx import TensorProto
        TENSOR_TYPE_TO_NP_TYPE = {
            TensorProto.FLOAT: np.dtype('float32'),
            TensorProto.UINT8: np.dtype('uint8'),
            TensorProto.INT8: np.dtype('int8'),
            TensorProto.UINT16: np.dtype('uint16'),
            TensorProto.INT16: np.dtype('int16'),
            TensorProto.INT32: np.dtype('int32'),
            TensorProto.INT64: np.dtype('int64'),
            TensorProto.STRING: np.dtype('O'),
            TensorProto.BOOL: np.dtype('bool'),
            TensorProto.FLOAT16: np.dtype('float16'),
            TensorProto.DOUBLE: np.dtype('float64'),
            TensorProto.UINT32: np.dtype('uint32'),
            TensorProto.UINT64: np.dtype('uint64'),
            TensorProto.COMPLEX64: np.dtype('complex64'),
            TensorProto.COMPLEX128: np.dtype('complex128'),
            TensorProto.BFLOAT16: np.dtype('uint16'),
        }
        NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items()}
        for k, v in list(NP_TYPE_TO_TENSOR_TYPE.items()):
            NP_TYPE_TO_TENSOR_TYPE[np.dtype(k).type] = v
    onnx.mapping = OnnxMapping

import onnx_graphsurgeon as gs
from onnx import shape_inference

def fix_onnx_full(input_onnx_path, output_onnx_path):
    print(f"[Fix] Robust Surgery: {input_onnx_path}")
    model = onnx.load(input_onnx_path)
    
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[Warning] Shape inference failed: {e}")
        
    graph = gs.import_onnx(model)
    
    # 1. 基础类型转换 (移除可能导致 AttributeError 的直接赋值)
    # 对于 graphsurgeon，如果无法直接设置 dtype，我们跳过全局转换
    # 大多数情况下 TensorRT 8.6+ 可以处理 int64
    
    # 强制补全输入输出节点的 dtype
    for inp in graph.inputs:
        if inp.dtype is None:
            inp.dtype = np.float32
    for out in graph.outputs:
        if out.dtype is None:
            out.dtype = np.float32

    def find_actual_constant(tensor):
        if isinstance(tensor, gs.Constant): return tensor
        if isinstance(tensor, gs.Variable) and tensor.inputs:
            parent_node = tensor.inputs[0]
            if parent_node.op in ["Transpose", "Reshape"]:
                return find_actual_constant(parent_node.inputs[0])
        return None

    new_nodes = []
    nodes_to_remove = []

    for node in graph.nodes:
        # 修复 MatMul 相关的 6D 投影逻辑 (如果是从老版本导出的)
        if node.op == "MatMul":
            inputs = node.inputs
            if len(inputs) != 2: continue
            s0, s1 = inputs[0].shape, inputs[1].shape
            
            if s0 and len(s0) == 6 and s0[1] == 6:
                print(f"[Unroll] Unrolling 6D MatMul '{node.name}'")
                dtype = inputs[0].dtype if inputs[0].dtype else np.float32
                split_outputs = [gs.Variable(name=f"{node.name}_split_{i}", dtype=dtype) for i in range(6)]
                split_node = gs.Node(op="Split", name=f"{node.name}_split_node", inputs=[inputs[0]], outputs=split_outputs, attrs={"axis": 1})
                new_nodes.append(split_node)
                matmul_outputs = []
                for i in range(6):
                    cam_4d = gs.Variable(name=f"{node.name}_cam_{i}_4d", dtype=dtype)
                    rs_cam = gs.Node(op="Reshape", name=f"{node.name}_rs_cam_{i}", 
                                     inputs=[split_outputs[i], gs.Constant(name=f"{node.name}_sh_cam_{i}", values=np.array([-1, 1, 4, 4], dtype=np.int32))], outputs=[cam_4d])
                    pts_4d = gs.Variable(name=f"{node.name}_pts_{i}_4d", dtype=dtype)
                    rs_pts = gs.Node(op="Reshape", name=f"{node.name}_rs_pts_{i}", 
                                     inputs=[inputs[1], gs.Constant(name=f"{node.name}_sh_pts_{i}", values=np.array([-1, 900, 13, 4, 1], dtype=np.int32))], outputs=[pts_4d])
                    mm_out = gs.Variable(name=f"{node.name}_mm_out_{i}", dtype=dtype)
                    mm = gs.Node(op="MatMul", name=f"{node.name}_mm_{i}", inputs=[cam_4d, pts_4d], outputs=[mm_out])
                    new_nodes.extend([rs_cam, rs_pts, mm])
                    matmul_outputs.append(mm_out)
                concat_out = gs.Variable(name=f"{node.name}_concat", dtype=dtype)
                concat = gs.Node(op="Concat", name=f"{node.name}_concat_op", inputs=matmul_outputs, outputs=[concat_out], attrs={"axis": 1})
                rs_back = gs.Node(op="Reshape", name=f"{node.name}_rs_back", 
                                  inputs=[concat_out, gs.Constant(name=f"{node.name}_sb", values=np.array([x if isinstance(x, int) else -1 for x in node.outputs[0].shape], dtype=np.int32))], outputs=[node.outputs[0]])
                new_nodes.extend([concat, rs_back])
                nodes_to_remove.append(node)
                continue

    graph.nodes.extend(new_nodes)
    for n in nodes_to_remove: n.outputs = []
    graph.cleanup().toposort()
    
    # 导出并保存
    try:
        exported_model = gs.export_onnx(graph)
        onnx.save(exported_model, output_onnx_path)
        print(f"[Fix] Surgery successfully saved to {output_onnx_path}")
    except Exception as e:
        print(f"[Error] GS export failed: {e}")
        # 备选方案：如果导出失败，尝试用原始模型保存（至少完成了基础校验）
        # onnx.save(model, output_onnx_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_onnx", type=str, required=True)
    parser.add_argument("--output_onnx", type=str, required=True)
    args = parser.parse_args()
    fix_onnx_full(args.input_onnx, args.output_onnx)
