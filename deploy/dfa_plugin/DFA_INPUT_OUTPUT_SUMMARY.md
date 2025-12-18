# DFA (Deformable Attention Aggregation) 输入输出总结

## 概述

DFA Plugin 是 Sparse4D 的核心算子，用于从多尺度、多视角的特征图中聚合特征到每个 Anchor（Query）。

---

## 输入（5个）

### 1. **value** (inputs[0]) - 特征图
- **形状**: `[batch, spatial_size, num_embeds]`
- **数据类型**: 
  - FP16 (标准模式)
  - FP32 (标准模式)
  - INT8 (量化模式，暂未启用)
- **说明**: 
  - `spatial_size = num_cams × (H1×W1 + H2×W2 + H3×W3 + H4×W4)`
  - 典型值: `[1, 89760, 256]` (6相机 × 4尺度特征图展平)
  - 包含所有相机、所有尺度的特征图，按相机和尺度顺序展平

### 2. **spatial_shapes** (inputs[1]) - 空间形状
- **形状**: `[num_cams, num_levels, 2]`
- **数据类型**: INT32
- **说明**: 
  - 每个相机、每个尺度的特征图尺寸 `[H, W]`
  - 典型值: `[6, 4, 2]` → 6个相机，4个尺度，每个尺度有 `[H, W]`
  - 例如: `[[64, 176], [32, 88], [16, 44], [8, 22]]` (4个尺度的尺寸)

### 3. **level_start_index** (inputs[2]) - 层级起始索引
- **形状**: `[num_cams, num_levels]`
- **数据类型**: INT32
- **说明**: 
  - 每个相机、每个尺度在展平的 `value` 中的起始索引
  - 用于快速定位某个相机、某个尺度的特征在 `value` 中的位置
  - 典型值: `[6, 4]` → 6个相机，每个相机4个尺度的起始索引

### 4. **sampling_loc** (inputs[3]) - 采样位置
- **形状**: `[batch, num_query, num_point, num_cams, 2]`
- **数据类型**: 
  - FP32 (混合精度模式，推荐)
  - FP16 (标准模式)
- **说明**: 
  - 每个 Anchor 在每个相机上的采样点坐标 `[x, y]` (归一化坐标 [0, 1])
  - 典型值: `[1, 900, 13, 6, 2]`
    - `num_query = 900`: Anchor 数量
    - `num_point = 13`: 每个 Anchor 的采样点数（Scaling Points）
    - `num_cams = 6`: 相机数量
    - `2`: x, y 坐标
  - **关键**: 在混合精度模式下，必须使用 FP32 以保证精度

### 5. **attn_weight** (inputs[4]) - 注意力权重
- **形状**: `[batch, num_query, num_point, num_cams, num_levels, num_groups]`
- **数据类型**: 
  - FP32 (混合精度模式，推荐)
  - FP16 (标准模式)
- **说明**: 
  - 每个采样点的注意力权重（Softmax 输出）
  - 典型值: `[1, 900, 13, 6, 4, 8]`
    - `num_query = 900`: Anchor 数量
    - `num_point = 13`: 每个 Anchor 的采样点数
    - `num_cams = 6`: 相机数量
    - `num_levels = 4`: 特征尺度数量
    - `num_groups = 8`: 分组注意力组数
  - **关键**: 在混合精度模式下，必须使用 FP32 以避免 Softmax 下溢问题

---

## 输出（1个）

### **output** - 聚合后的特征
- **形状**: `[batch, num_query, num_embeds]`
- **数据类型**: 
  - FP16 (标准模式，混合精度模式)
  - FP32 (标准模式)
- **说明**: 
  - 每个 Anchor 聚合后的特征向量
  - 典型值: `[1, 900, 256]`
    - `num_query = 900`: Anchor 数量
    - `num_embeds = 256`: 特征维度
  - **计算过程**: 
    - 对每个 Anchor，从 `value` 中根据 `sampling_loc` 采样 312 个点 (13×6×4)
    - 使用 `attn_weight` 加权聚合
    - 输出该 Anchor 的聚合特征

---

## 计算流程

```
对于每个 Anchor (num_query = 900):
  对于每个采样点 (num_point = 13):
    对于每个相机 (num_cams = 6):
      对于每个尺度 (num_levels = 4):
        1. 根据 sampling_loc 从 value 中双线性采样特征
        2. 使用 attn_weight 加权
        3. 累加到该 Anchor 的输出特征中
```

**总采样点数**: `13 × 6 × 4 = 312` 个采样点/Anchor

---

## 数据类型组合

### 标准 FP16 模式
- `value`: FP16
- `sampling_loc`: FP16
- `attn_weight`: FP16
- `output`: FP16

### 混合精度模式（推荐）
- `value`: FP16
- `sampling_loc`: **FP32** (保证坐标精度)
- `attn_weight`: **FP32** (避免 Softmax 下溢)
- `output`: FP16
- **优势**: 精度恢复至 0.999+，性能损失可忽略

### 标准 FP32 模式
- `value`: FP32
- `sampling_loc`: FP32
- `attn_weight`: FP32
- `output`: FP32

### INT8 模式（暂未启用）
- `value`: INT8
- `sampling_loc`: FP32
- `attn_weight`: FP32
- `output`: FP32

---

## 关键参数说明

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `batch` | 1 | 批次大小 |
| `num_query` | 900 | Anchor/Query 数量 |
| `num_point` | 13 | 每个 Anchor 的采样点数（Scaling Points） |
| `num_cams` | 6 | 相机数量 |
| `num_levels` | 4 | 特征尺度数量 |
| `num_embeds` | 256 | 特征维度 |
| `num_groups` | 8 | 分组注意力组数 |
| `spatial_size` | 89760 | 展平后的特征图大小 (6×4尺度特征图) |

---

## 内存占用估算

### 输入内存
- `value`: `1 × 89760 × 256 × 2 bytes (FP16) ≈ 46 MB`
- `spatial_shapes`: `6 × 4 × 2 × 4 bytes (INT32) ≈ 0.2 KB`
- `level_start_index`: `6 × 4 × 4 bytes (INT32) ≈ 0.1 KB`
- `sampling_loc`: `1 × 900 × 13 × 6 × 2 × 4 bytes (FP32) ≈ 1.8 MB`
- `attn_weight`: `1 × 900 × 13 × 6 × 4 × 8 × 4 bytes (FP32) ≈ 9.4 MB`

**总输入**: 约 57.6 MB

### 输出内存
- `output`: `1 × 900 × 256 × 2 bytes (FP16) ≈ 0.5 MB`

---

## 注意事项

1. **混合精度模式**: `sampling_loc` 和 `attn_weight` 必须使用 FP32，否则精度会严重下降
2. **采样点总数**: 每个 Anchor 有 312 个采样点，这是 Scatter 模式下原子锁竞争的根本原因
3. **Gather 模式优化**: 当前实现采用 Gather 模式，避免了原子锁竞争，性能提升 9.6 倍
4. **Workspace**: 混合精度模式可能需要 workspace 缓冲区（当前实现中未使用）

