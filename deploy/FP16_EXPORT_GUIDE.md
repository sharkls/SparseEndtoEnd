# FP16 ONNX导出指南

## 问题背景

当使用`--fp16`构建TensorRT engine时，如果ONNX模型的数据类型是FP32，TensorRT在构建时调用`getWorkspaceSize`时会看到FP32输入，返回0。但运行时如果输入是FP16，需要workspace，导致workspace为nullptr。

## 长期解决方案

**导出FP16 ONNX模型**：确保ONNX模型本身使用FP16数据类型，这样TensorRT在构建时就能正确识别并分配workspace。

## 使用方法

### 1. 导出FP16 ONNX模型

使用`--fp16`参数导出ONNX模型：

```bash
python deploy/export_head_onnx.py \
    --cfg dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py \
    --ckpt ckpt/sparse4dv3_r50.pth \
    --save_onnx2 deploy/onnx/sparse4dhead2nd.onnx \
    --fp16
```

### 2. 构建TensorRT Engine

使用FP16模式构建engine：

```bash
./deploy/build_sparse4d_engine.sh fp16
```

### 3. 验证

构建日志中应该看到plugin的输入类型是`[HALF]`而不是`[FLOAT]`：

```
[V] [TRT] /DeformableAttentionAggrPlugin [DeformableAttentionAggrPlugin] inputs: 
  [feature -> (1, 89760, 256)[HALF]], 
  [/Transpose_output_0 -> (1, 900, 13, 6, 2)[HALF]], 
  [/Reshape_4_output_0 -> (1, 900, 13, 6, 4, 8)[HALF]]
```

## 实现细节

### 修改内容

1. **添加`--fp16`参数**：控制是否导出FP16 ONNX模型
2. **模型转换**：如果指定`--fp16`，将模型转换为FP16（`.half()`）
3. **输入数据类型**：所有浮点类型的输入都使用FP16（`torch.float16`）
4. **整数类型保持不变**：`spatial_shapes`、`level_start_index`、`mask`、`track_id`等保持为整数类型

### 代码修改

1. **`parse_args()`**：添加`--fp16`参数
2. **`dummpy_input()`**：根据`model._export_fp16`标志选择数据类型
3. **`main()`**：如果指定`--fp16`，将模型转换为FP16并设置标志

### 数据类型映射

| 输入 | FP32模式 | FP16模式 |
|------|----------|----------|
| feature | float32 | float16 |
| instance_feature | float32 | float16 |
| anchor | float32 | float16 |
| temp_instance_feature | float32 | float16 |
| temp_anchor | float32 | float16 |
| image_wh | float32 | float16 |
| lidar2img | float32 | float16 |
| spatial_shapes | int32 | int32 |
| level_start_index | int32 | int32 |
| time_interval | float32 | float16 |
| mask | int32 | int32 |
| track_id | int32 | int32 |

## 注意事项

1. **训练一致性**：导出FP16 ONNX不会影响训练，因为训练时仍然使用FP32
2. **精度影响**：FP16可能会有轻微的精度损失，但通常可以接受
3. **兼容性**：确保所有操作都支持FP16（大多数PyTorch操作都支持）

## 验证步骤

1. **检查ONNX模型数据类型**：
   ```python
   import onnx
   model = onnx.load('deploy/onnx/sparse4dhead2nd.onnx')
   for node in model.graph.node:
       if 'DeformableAttentionAggrPlugin' in node.name:
           for input_name in node.input:
               for vi in model.graph.value_info:
                   if vi.name == input_name:
                       print(f'{input_name}: {vi.type.tensor_type.elem_type}')
   ```

2. **检查构建日志**：确认plugin输入类型是`[HALF]`

3. **运行时验证**：不应该再出现workspace为null的错误

## 总结

通过导出FP16 ONNX模型，可以确保TensorRT在构建时正确识别FP16输入，从而正确分配workspace。这是长期解决方案，避免了运行时动态分配内存的开销。

