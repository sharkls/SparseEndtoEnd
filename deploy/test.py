import onnx
model = onnx.load("deploy/onnx/sparse4dhead2nd.onnx")
for node in model.graph.node:
    if "LayerNormalization" in node.op_type or "ReduceMean" in node.op_type:
        print(f"Found: {node.op_type} - {node.name}")