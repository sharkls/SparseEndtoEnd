# SparseBox3DKeyPointsPlugin 主要计算逻辑

## 一、功能概述

SparseBox插件用于从3D anchor生成关键点（keypoints），这些关键点用于后续的可变形注意力聚合（DFA）模块。

## 二、输入输出

### 2.1 输入
- **anchor**: `[B, N, 11]` - 3D边界框参数
  - `[0]`: centerX (中心点X坐标)
  - `[1]`: centerY (中心点Y坐标)
  - `[2]`: centerZ (中心点Z坐标)
  - `[3]`: log_size_x (宽度的对数)
  - `[4]`: log_size_y (长度的对数)
  - `[5]`: log_size_z (高度的对数)
  - `[6]`: sin_yaw (航向角的正弦值)
  - `[7]`: cos_yaw (航向角的余弦值)
  - `[8-10]`: 其他参数（速度等）

- **instance_feature**: `[B, N, embed_dims]` - 实例特征（可选）
  - 用于生成可学习的关键点
  - 如果`num_learnable_pts > 0`，则必须提供

### 2.2 输出
- **keypoints**: `[B, N, num_pts, 3]` - 3D关键点坐标
  - `num_pts = num_fix_pts + num_learnable_pts`
  - 每个关键点是3D空间中的`(x, y, z)`坐标

## 三、主要计算步骤

### 步骤1: 提取Anchor参数并计算尺寸

```cpp
// 从anchor中提取参数（转换为FP32）
float centerX = toFloat(anchorPtr[0]);
float centerY = toFloat(anchorPtr[1]);
float centerZ = toFloat(anchorPtr[2]);
float log_size_x = toFloat(anchorPtr[3]);
float log_size_y = toFloat(anchorPtr[4]);
float log_size_z = toFloat(anchorPtr[5]);
float sinYaw = toFloat(anchorPtr[6]);
float cosYaw = toFloat(anchorPtr[7]);

// 计算实际尺寸（通过exp）
float sizeX = expf(log_size_x);  // 宽度
float sizeY = expf(log_size_y);  // 长度
float sizeZ = expf(log_size_z);  // 高度
```

**说明**：
- Anchor中存储的是尺寸的对数，需要`exp`得到实际尺寸
- 所有计算都在FP32精度下进行，确保数值稳定性

### 步骤2: 生成关键点（固定点 + 可学习点）

#### 2.1 固定点（Fixed Points）

固定点基于预定义的`fix_scale`和anchor尺寸计算：

```cpp
if (i < fixedPts) {
    // 固定点计算
    localX = fixScale_x * sizeX;
    localY = fixScale_y * sizeY;
    localZ = fixScale_z * sizeZ;
}
```

**说明**：
- `fix_scale`: 预定义的缩放因子，通常表示相对于anchor尺寸的偏移
- 固定点数量：通常为7个（6个面的中心点 + 1个中心点）
- 固定点位置在anchor的局部坐标系中

#### 2.2 可学习点（Learnable Points）

可学习点通过全连接层从`instance_feature`生成：

```cpp
if (i >= fixedPts && instPtr != nullptr) {
    // 1. 初始化累加器（从bias开始）
    float accum[3];
    accum[0] = rowBias[0];
    accum[1] = rowBias[1];
    accum[2] = rowBias[2];
    
    // 2. 全连接层计算（使用fmaf进行融合乘加）
    for (int k = 0; k < params.embedDims; ++k) {
        const float val = toFloat(instPtr[k]);  // FP16转FP32
        const float w0 = rowWeight[0 * embedDims + k];
        const float w1 = rowWeight[1 * embedDims + k];
        const float w2 = rowWeight[2 * embedDims + k];
        
        // 使用fmaf进行高精度累加
        accum[0] = fmaf(w0, val, accum[0]);
        accum[1] = fmaf(w1, val, accum[1]);
        accum[2] = fmaf(w2, val, accum[2]);
    }
    
    // 3. 应用sigmoid并归一化到[-0.5, 0.5]
    float sig_x = sigmoid_centered(accum[0]);
    float sig_y = sigmoid_centered(accum[1]);
    float sig_z = sigmoid_centered(accum[2]);
    
    // 4. 乘以尺寸得到局部坐标
    localX = sig_x * sizeX;
    localY = sig_y * sizeY;
    localZ = sig_z * sizeZ;
}
```

**说明**：
- 全连接层：`FC(instance_feature) -> [num_learnable_pts * 3]`
- 使用`sigmoid_centered = sigmoid(x) - 0.5`将输出归一化到`[-0.5, 0.5]`
- 可学习点数量：通常为6个（在DFA模块中设置）
- 使用`fmaf`（融合乘加）进行高精度累加

### 步骤3: 应用旋转（从局部坐标系到全局坐标系）

关键点需要根据anchor的航向角进行旋转：

```cpp
// 构建旋转矩阵（绕Z轴旋转，即航向角）
// rotation_matrix = [
//   [cos_yaw, -sin_yaw, 0],
//   [sin_yaw,  cos_yaw, 0],
//   [0,        0,        1]
// ]

// 应用旋转（只影响X和Y，Z不变）
float rotX = cosYaw * localX - sinYaw * localY;
float rotY = sinYaw * localX + cosYaw * localY;
float rotZ = localZ;  // Z轴不受旋转影响
```

**说明**：
- 旋转矩阵是2D旋转（绕Z轴），因为anchor的航向角是水平方向的
- 旋转将关键点从anchor的局部坐标系转换到全局坐标系
- 所有计算使用FP32精度

### 步骤4: 加上中心点（平移到全局坐标系）

```cpp
// 加上anchor的中心点，得到全局坐标
float finalX = rotX + centerX;
float finalY = rotY + centerY;
float finalZ = rotZ + centerZ;
```

**说明**：
- 将旋转后的关键点平移到anchor的中心点位置
- 得到最终的关键点全局坐标

## 四、完整计算流程

```
输入: anchor [B, N, 11], instance_feature [B, N, embed_dims]
  ↓
步骤1: 提取参数并计算尺寸
  - centerX, centerY, centerZ
  - sizeX = exp(log_size_x)
  - sizeY = exp(log_size_y)
  - sizeZ = exp(log_size_z)
  - sinYaw, cosYaw
  ↓
步骤2: 生成关键点（对每个关键点）
  ├─ 固定点: local = fix_scale * size
  └─ 可学习点: 
      ├─ accum = FC(instance_feature)  // 全连接层
      ├─ scale = sigmoid_centered(accum)  // 归一化到[-0.5, 0.5]
      └─ local = scale * size
  ↓
步骤3: 应用旋转
  - rotX = cosYaw * localX - sinYaw * localY
  - rotY = sinYaw * localX + cosYaw * localY
  - rotZ = localZ
  ↓
步骤4: 加上中心点
  - finalX = rotX + centerX
  - finalY = rotY + centerY
  - finalZ = rotZ + centerZ
  ↓
输出: keypoints [B, N, num_pts, 3]
```

## 五、关键优化

### 5.1 数值稳定性
- **FP32中间计算**：所有计算都在FP32精度下进行
- **NaN/Inf检查**：每个计算步骤后都进行数值检查
- **范围限制**：对输入值进行clamp，防止溢出

### 5.2 精度优化
- **fmaf融合乘加**：使用`fmaf`进行全连接层计算，减少舍入误差
- **sigmoid_centered**：数值稳定的sigmoid实现
- **FP32输出**：在FP16输入模式下，输出FP32关键点（提高精度）

### 5.3 性能优化
- **并行化**：每个anchor独立处理，可以并行计算
- **内存访问优化**：使用stride访问，减少内存访问次数

## 六、与Python实现的对应关系

### Python实现（`SparseBox3DKeyPointsGenerator.forward`）

```python
# 步骤1: 提取尺寸并计算固定点
size = anchor[..., None, [W, L, H]].exp()  # [B, N, 1, 3]
key_points = self.fix_scale * size  # [B, N, 7, 3]

# 步骤2: 计算可学习点
if self.num_learnable_pts > 0:
    learnable_scale = (
        self.learnable_fc(instance_feature)  # [B, N, 6*3]
        .reshape(bs, num_anchor, self.num_learnable_pts, 3)
        .sigmoid() - 0.5  # 归一化到[-0.5, 0.5]
    )
    key_points = torch.cat([key_points, learnable_scale * size], dim=-2)

# 步骤3: 构建旋转矩阵并应用旋转
rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])
rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
rotation_mat[:, :, 2, 2] = 1
key_points = torch.matmul(rotation_mat[:, :, None], key_points[..., None])[..., 0]

# 步骤4: 加上中心点
key_points = key_points + anchor[..., None, :3]
```

### CUDA Kernel实现

CUDA kernel实现了相同的逻辑，但：
- 使用FP32精度进行所有计算
- 添加了数值稳定性检查
- 使用`fmaf`进行高精度累加
- 并行处理每个anchor

## 七、关键点用途

生成的关键点用于：
1. **可变形注意力聚合（DFA）**：在图像特征图上采样特征
2. **特征聚合**：将多视角图像特征聚合到anchor上
3. **Refinement**：基于聚合的特征对anchor进行精炼

## 八、参数说明

- **num_fix_pts**: 固定点数量（通常为7）
- **num_learnable_pts**: 可学习点数量（通常为6，在DFA模块中设置）
- **num_pts**: 总点数 = num_fix_pts + num_learnable_pts（通常为13）
- **embed_dims**: 实例特征维度（通常为256）

