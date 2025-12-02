# 混合精度方案：ONNX导出指南

## 问题：导出ONNX时应该使用`--fp16`还是`--fp32`？

## 答案：**使用`--fp16`导出ONNX**

## 原因分析

### 1. 混合精度的目标
- **输入输出**：保持FP16（减少内存带宽）
- **关键点生成**：使用FP32（提高精度）
- **DFA模块**：接受FP16 value + FP32 keypoints

### 2. 使用`--fp16`导出ONNX的优势

#### ✅ 输入输出类型正确
- ONNX模型定义输入输出为FP16
- 符合混合精度方案的目标（输入输出保持FP16）

#### ✅ 插件自动处理类型转换
- **SparseBox插件**：`getOutputDataType`会返回FP32（覆盖ONNX定义）
  ```cpp
  if (inputTypes[0] == DataType::kHALF)
  {
      return DataType::kFLOAT;  // FP16输入时，输出FP32
  }
  ```
- **DFA插件**：自动检测混合精度（FP16 value + FP32 keypoints）并处理

#### ✅ TensorRT构建时的一致性
- 使用`--fp16`构建引擎时，TensorRT会：
  1. 识别ONNX输入输出为FP16
  2. 识别SparseBox插件输出为FP32（通过`getOutputDataType`）
  3. 自动插入类型转换层（FP32 → FP16，如果需要）
  4. DFA插件自动使用混合精度模式

### 3. 如果使用`--fp32`导出ONNX的问题

#### ❌ 输入输出类型不匹配
- ONNX模型定义输入输出为FP32
- 但目标是要FP16输入输出
- TensorRT构建时虽然可以转换，但可能不够理想

#### ❌ 可能影响workspace分配
- DFA插件需要workspace（FP32临时缓冲区）
- 如果ONNX是FP32，TensorRT可能不会正确分配workspace

## 使用步骤

### 1. 导出ONNX（使用`--fp16`）

```bash
# 导出head1st ONNX
python deploy/export/export_head_onnx.py \
    --cfg dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py \
    --ckpt ckpt/sparse4dv3_r50.pth \
    --save_onnx1 deploy/onnx/sparse4dhead1st.onnx \
    --fp16

# 导出head2nd ONNX（如果需要）
python deploy/export/export_head_onnx.py \
    --cfg dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py \
    --ckpt ckpt/sparse4dv3_r50.pth \
    --save_onnx2 deploy/onnx/sparse4dhead2nd.onnx \
    --fp16 \
    --o2
```

### 2. 构建TensorRT引擎（使用`--fp16`）

```bash
./deploy/build_sparse4d_engine.sh fp16
```

### 3. 验证混合精度是否生效

```bash
python deploy/val/validate_pytorch_vs_engine.py \
    --sample_idx 0 \
    --num_samples 1 \
    --use_val_dataset \
    --analyze_plugin_error \
    --capture_keypoints
```

## 数据流验证

### 预期数据流：
```
输入（FP16，来自ONNX定义）
  ↓
SparseBox插件
  ├─ 输入：FP16（来自ONNX）
  ├─ 内部计算：FP32
  └─ 输出：FP32（通过getOutputDataType覆盖）
  ↓
DFA插件
  ├─ value输入：FP16（来自ONNX）
  ├─ keypoints输入：FP32（来自SparseBox插件）
  ├─ 检测：混合精度模式
  ├─ 处理：使用混合精度kernel
  └─ 输出：FP16
  ↓
后续模块（FP16）
  ↓
最终输出（FP16，符合ONNX定义）
```

## 注意事项

### 1. ONNX导出时的keypoints类型
- 导出脚本可能会强制将keypoints转换为FP16（`export_head_onnx.py:169`）
- **这不影响运行时**，因为：
  - SparseBox插件会重新生成keypoints（FP32）
  - ONNX中的keypoints只是占位符，实际运行时不会被使用

### 2. TensorRT类型转换
- TensorRT会自动处理类型转换
- SparseBox输出FP32 → DFA输入FP32：无需转换
- DFA输出FP16 → 后续模块输入FP16：无需转换

### 3. Workspace分配
- DFA插件需要workspace（FP32临时缓冲区）
- 使用`--fp16`导出ONNX时，TensorRT会正确识别并分配workspace

## 总结

**推荐方案**：
1. ✅ 使用`--fp16`导出ONNX
2. ✅ 使用`--fp16`构建TensorRT引擎
3. ✅ 插件自动处理混合精度

**优势**：
- 输入输出类型正确（FP16）
- 插件自动处理关键点精度（FP32）
- TensorRT自动处理类型转换
- Workspace正确分配

