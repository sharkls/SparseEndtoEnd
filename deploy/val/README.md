# PyTorch vs TensorRT引擎验证脚本

## 概述

`validate_pytorch_vs_engine.py` 脚本用于验证PyTorch模型和TensorRT引擎在FP32精度下，使用相同输入时的输出差异。

## 功能

- 验证Backbone模块的输出差异
- 验证Head模块的输出差异（支持第一帧和第二帧）
- 计算多种差异指标：MSE、MAE、最大绝对差异、相对误差等
- 支持批量验证多个样本
- 支持两种验证模式：
  - **独立第一帧验证**（默认，推荐）：每个样本都作为独立的第一帧验证，确保结果一致
  - **连续帧验证**：第一个样本是第一帧，后续样本是第二帧，模拟实际推理流程
- 生成详细的验证报告（JSON格式）

## 使用方法

### 基本用法

```bash
python deploy/val/validate_pytorch_vs_engine.py \
    --config dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py \
    --checkpoint ckpt/sparse4dv3_r50.pth \
    --backbone_engine deploy/engine/sparse4dbackbone.engine \
    --head_engine deploy/engine/sparse4dhead1st.engine
```

### 参数说明

- `--config`: 配置文件路径（默认: `dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py`）
- `--checkpoint`: PyTorch模型检查点路径（默认: `ckpt/sparse4dv3_r50.pth`）
- `--backbone_engine`: Backbone TensorRT引擎路径（默认: `deploy/engine/sparse4dbackbone.engine`）
- `--head_engine`: Head第一帧TensorRT引擎路径（默认: `deploy/engine/sparse4dhead1st.engine`）
- `--head2nd_engine`: Head第二帧TensorRT引擎路径（可选，用于连续帧验证）
- `--validate_continuous_frames`: 启用连续帧验证模式（需要同时指定`--head2nd_engine`）
- `--sample_idx`: 验证的起始样本索引（默认: 0）
- `--num_samples`: 验证的样本数量（默认: 5）
- `--output_dir`: 验证结果输出目录（默认: `deploy/val/validation_results`）
- `--device`: 推理设备（默认: `cuda:0`）
- `--deterministic`: 是否使用确定性选项（用于可重复性）
- `--log`: 日志文件路径（默认: `deploy/val/validate_pytorch_vs_engine.log`）

### 示例

#### 验证单个样本

```bash
python deploy/val/validate_pytorch_vs_engine.py \
    --sample_idx 0 \
    --num_samples 1
```

#### 验证多个样本（独立第一帧模式，推荐）

```bash
python deploy/val/validate_pytorch_vs_engine.py \
    --sample_idx 0 \
    --num_samples 10
```

**注意**：默认情况下，每个样本都会作为独立的第一帧验证（会重置`instance_bank`状态），这样可以确保每个样本的验证结果一致，不会因为时序状态导致差异。

#### 验证连续帧（需要head2nd引擎）

```bash
python deploy/val/validate_pytorch_vs_engine.py \
    --sample_idx 0 \
    --num_samples 5 \
    --head2nd_engine deploy/engine/sparse4dhead2nd.engine \
    --validate_continuous_frames
```

**注意**：连续帧验证模式下，第一个样本使用head1st引擎，后续样本使用head2nd引擎。后续样本的验证结果可能会与独立第一帧验证有差异，这是正常的，因为第二帧会使用前一个样本的缓存信息。

#### 使用自定义引擎路径

```bash
python deploy/val/validate_pytorch_vs_engine.py \
    --backbone_engine /path/to/sparse4dbackbone.engine \
    --head_engine /path/to/sparse4dhead1st.engine
```

## 输出说明

### 控制台输出

脚本会在控制台输出：
- 每个样本的验证进度
- Backbone和Head的验证结果
- 各种差异指标（MSE、MAE、最大绝对差异等）

### 文件输出

1. **验证结果JSON文件** (`validation_results.json`)
   - 包含所有样本的详细验证结果
   - 每个输出张量的完整统计信息

2. **日志文件** (`validate_pytorch_vs_engine.log`)
   - 详细的执行日志
   - 错误和警告信息

## 验证指标说明

对于每个输出张量，脚本会计算以下指标：

- **MSE (Mean Squared Error)**: 均方误差
- **MAE (Mean Absolute Error)**: 平均绝对误差
- **Max Absolute Difference**: 最大绝对差异
- **Mean Absolute Difference**: 平均绝对差异
- **Std Absolute Difference**: 绝对差异的标准差
- **Relative Error**: 相对误差（平均绝对差异 / (|PyTorch输出| + ε)）

## 注意事项

1. **精度要求**: 脚本设计用于FP32精度验证。如果引擎是FP16精度，差异可能会更大。

2. **输入数据**: 确保使用相同的输入数据。脚本会自动从数据集中加载数据。

3. **验证模式选择**:
   - **独立第一帧验证（默认）**：每个样本都作为独立的第一帧验证，会重置`instance_bank`状态。这是推荐的验证方式，可以确保每个样本的验证结果一致。
   - **连续帧验证**：第一个样本是第一帧，后续样本是第二帧。这种模式下，后续样本的验证结果可能会与独立第一帧验证有差异，这是正常的，因为第二帧会使用前一个样本的缓存信息。

4. **引擎输出名称**: 如果引擎的输出名称与预期不同，脚本会尝试自动匹配。如果匹配失败，会在日志中显示警告。

5. **内存使用**: 验证过程会占用GPU内存，确保有足够的显存。

6. **确定性**: 使用 `--deterministic` 选项可以确保结果的可重复性。

7. **多样本验证**: 如果`num_samples`设置为5，在独立第一帧模式下，所有5个样本都会作为独立的第一帧验证，结果应该是一致的。如果使用连续帧模式，第一个样本是第一帧，后4个样本是第二帧，结果可能会有差异。

## 故障排除

### 问题：Segmentation fault (core dumped) 或引擎加载失败

**原因**: TensorRT引擎使用了自定义插件（如SparseBox3DKeyPointsPlugin），但在加载引擎前未加载插件库。

**解决**: 
- 脚本会自动尝试加载常用插件（sparsebox_plugin, dfa_plugin, ln_plugin）
- 如果插件不在默认位置，可以使用 `--plugin_paths` 参数指定插件路径：
  ```bash
  python deploy/val/validate_pytorch_vs_engine.py \
      --plugin_paths deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so \
      --plugin_paths deploy/dfa_plugin/lib/deformableAttentionAggr.so
  ```
- 确保插件库文件存在且可访问

### 问题：找不到引擎输出

**原因**: 引擎的输出名称可能与预期不同。

**解决**: 查看日志中的警告信息，了解引擎的实际输出名称，然后修改脚本中的输出名称映射。

### 问题：形状不匹配

**原因**: PyTorch模型和引擎的输出形状不一致。

**解决**: 检查模型配置和引擎构建过程，确保它们使用相同的配置。

### 问题：CUDA内存不足

**原因**: GPU显存不足。

**解决**: 减少 `--num_samples` 参数，或者使用更小的批次大小。

## 依赖项

- PyTorch
- TensorRT
- NumPy
- CUDA (用于TensorRT推理)

## 示例输出

```
================================================================================
验证 Backbone
================================================================================
执行PyTorch推理...
PyTorch输出形状: (1, 89760, 256)
执行TensorRT推理...
TensorRT输出名称: feature, 形状: (1, 89760, 256)
Backbone验证结果:
  MSE: 1.234567e-06
  MAE: 3.456789e-04
  最大绝对差异: 1.234567e-03
  相对误差: 1.234567e-05

================================================================================
验证 Head (第一帧)
================================================================================
...
```

