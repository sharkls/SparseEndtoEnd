# TensorRT Profile 分析完整指南

## 一、Profile日志结构理解

### 1.1 Profile日志的基本格式

Profile日志通常包含以下几个部分：

```
=== Performance summary ===          # 总体性能指标
=== Profile (N iterations) ===      # 逐层性能分析
Layer   Time (ms)   Avg. Time (ms)   Median Time (ms)   Time %
```

### 1.2 关键指标含义

| 指标 | 含义 | 重要性 |
|------|------|--------|
| **Time (ms)** | 该层在所有迭代中的总时间 | 看总开销 |
| **Avg. Time (ms)** | 该层平均每次执行时间 | **最重要** - 单次延迟 |
| **Median Time (ms)** | 该层中位数执行时间 | 看稳定性 |
| **Time %** | 该层占总时间的百分比 | **最重要** - 识别瓶颈 |

## 二、如何识别性能瓶颈

### 2.1 第一步：找到总体性能指标

在profile日志中查找 `=== Performance summary ===` 部分：

```
GPU Compute Time: mean = 35.757 ms, median = 34.9946 ms
H2D Latency: mean = 14.6825 ms
Total: 8437.35 ms (38.8818 ms avg, 37.9575 ms median)
```

**关键问题**：
- GPU计算时间是多少？（目标：< 30ms）
- H2D传输时间是多少？（目标：< 10ms）
- 总延迟是多少？（目标：< 40ms）

### 2.2 第二步：按Time %排序，找出最耗时的节点

**方法1：直接看Time %列**
- 找出Time %最大的节点
- 通常Time % > 5%的节点需要重点关注

**方法2：计算总时间占比**
```bash
# 从profile日志中提取
grep "Time %" profile.log | sort -k5 -rn | head -20
```

### 2.3 第三步：识别节点类型

Profile中的节点类型：

#### 类型1: ForeignNode（最需要优化）
```
{ForeignNode[/kps_generator/Unsqueeze.../Transpose]}      740.33 ms (8.8%)
{ForeignNode[/layers.10/Slice_6.../Transpose_4]}          712.51 ms (8.4%)
```

**特征**：
- 名称包含 `ForeignNode`
- 通常包含 `Transpose`、`Gather`、`Slice` 等操作
- **为什么慢**：TensorRT无法优化这些节点，使用通用实现
- **优化优先级**：⭐⭐⭐⭐⭐（最高）

#### 类型2: Reformat节点（需要减少）
```
Reformatting CopyNode for Network Input feature      103.16 ms (1.2%)
Reformatting CopyNode for Input Tensor 3 to /DeformableAttentionAggrPlugin  2.26 ms
```

**特征**：
- 名称包含 `Reformatting CopyNode`
- **为什么慢**：数据类型或内存布局转换，纯内存拷贝
- **优化优先级**：⭐⭐⭐⭐（高）

#### 类型3: Plugin节点（核心计算）
```
/DeformableAttentionAggrPlugin      593.74 ms (7.0%)
/DeformableAttentionAggrPlugin_1   457.33 ms (5.4%)
```

**特征**：
- 名称是自定义Plugin
- **为什么慢**：这是核心计算，但可能还有优化空间
- **优化优先级**：⭐⭐⭐（中）

#### 类型4: 标准TensorRT节点（通常不需要优化）
```
/anchor_encoder/pos_fc/pos_fc.0/MatMul        3.85 ms
/Softmax       37.77 ms (0.4%)
```

**特征**：
- 标准ONNX操作（MatMul, Add, Relu等）
- TensorRT已经高度优化
- **优化优先级**：⭐（低，除非占比很大）

## 三、具体分析：你的模型中的瓶颈

### 3.1 瓶颈1: ForeignNode节点（42%时间，最严重）

#### 问题节点分析

**节点1: kps_generator相关的Transpose**
```
{ForeignNode[/kps_generator/Unsqueeze.../Transpose]}  740.33 ms (8.8%)
Avg. Time: 3.41 ms/次
```

**这个节点在做什么**：
- `kps_generator`：生成关键点（key points）
- `Unsqueeze`：增加维度
- `Transpose`：转置操作，改变内存布局

**为什么慢**：
1. **Transpose操作本身慢**：需要重新排列内存，不是连续访问
2. **TensorRT无法融合**：ForeignNode意味着TensorRT无法优化
3. **可能的内存访问模式差**：非连续内存访问

**如何优化**：
```python
# 在 export_head_onnx.py 中，kps_generator的输出
# 当前：可能输出 [B, C, H, W] 然后需要转置
# 优化：直接输出目标布局 [B, H, W, C] 或 [B, H*W, C]

# 方法1：修改kps_generator直接输出目标布局
# 方法2：使用view代替permute（如果内存布局允许）
# 方法3：合并多个Transpose操作
```

**节点2-5: layers.10/17/24/31的Transpose链**
```
{ForeignNode[/layers.10/Slice_6.../Transpose_4]}  712.51 ms (8.4%)
{ForeignNode[/layers.17/Slice_6.../Transpose_6]}  710.90 ms (8.4%)
{ForeignNode[/layers.24/Slice_6.../Transpose_8]}  709.35 ms (8.4%)
{ForeignNode[/layers.31/Slice_6.../Transpose_10]} 712.07 ms (8.4%)
```

**这个节点在做什么**：
- 这些是deformable attention中的操作
- `Slice_6`：切片操作
- `Transpose_X`：转置操作（X表示转置次数）

**为什么慢**：
1. **多个Transpose链**：每个deformable层都有类似的Transpose链
2. **Slice + Transpose组合**：先切片再转置，产生中间张量
3. **无法融合**：TensorRT无法将这些操作融合

**如何优化**：
```python
# 在 export_head_onnx.py 的 deformable 操作中
# 当前代码（339-342行）：
points_2d = points_2d.permute(0, 2, 3, 1, 4).contiguous()
points_2d = points_2d.reshape(...)

# 优化方案1：如果可能，直接输出目标布局
# 优化方案2：使用view代替permute（需要验证内存布局）
# 优化方案3：合并permute和reshape
points_2d = points_2d.permute(0, 2, 3, 1, 4).view(...)  # 合并操作
```

### 3.2 瓶颈2: Reformat节点（3%时间）

#### 问题节点分析

**节点1: 输入Reformat**
```
Reformatting CopyNode for Network Input feature  103.16 ms (1.2%)
Avg. Time: 0.48 ms/次
```

**这个节点在做什么**：
- 将输入数据从一种格式转换为另一种格式
- 可能是：FP32 → FP16，或 NCHW → NHWC

**为什么慢**：
1. **纯内存拷贝**：没有计算，只是数据移动
2. **可能不必要**：如果输入格式匹配，就不需要

**如何优化**：
```bash
# 在 build_sparse4d_engine.sh 中
# 确保IO格式参数正确
--inputIOFormats=fp16:chw --outputIOFormats=fp16:chw

# 确保输入数据已经是FP16格式
# 确保输入数据已经是CHW布局
```

**节点2: Plugin输入Reformat**
```
Reformatting CopyNode for Input Tensor 3 to /DeformableAttentionAggrPlugin  2.26 ms
Reformatting CopyNode for Input Tensor 4 to /DeformableAttentionAggrPlugin_1  8.14 ms
```

**这个节点在做什么**：
- 为Plugin准备输入数据
- 转换数据类型或布局以匹配Plugin要求

**为什么慢**：
- Plugin期望特定的数据类型/布局
- 如果输入不匹配，需要转换

**如何优化**：
```python
# 在 export_head_onnx.py 中，确保进入Plugin前的数据格式正确
# 在 deformable 操作中（354-360行）：
if getattr(self, "_export_head2_layout_opt", False):
    ref_dtype = feature.dtype
    if points_2d.dtype != ref_dtype:
        points_2d = points_2d.to(ref_dtype)
    if weights.dtype != ref_dtype:
        weights = weights.to(ref_dtype)
    
    # 确保内存布局连续
    points_2d = points_2d.contiguous()
    weights = weights.contiguous()
```

### 3.3 瓶颈3: DeformableAttentionAggrPlugin（38%时间）

#### 问题节点分析

```
/DeformableAttentionAggrPlugin      593.74 ms (7.0%)  Avg: 2.74 ms
/DeformableAttentionAggrPlugin_1    457.33 ms (5.4%)  Avg: 2.11 ms
/DeformableAttentionAggrPlugin_2    466.77 ms (5.5%)  Avg: 2.15 ms
/DeformableAttentionAggrPlugin_3    485.23 ms (5.8%)  Avg: 2.24 ms
/DeformableAttentionAggrPlugin_4    631.36 ms (7.5%)  Avg: 2.91 ms
/DeformableAttentionAggrPlugin_5    573.86 ms (6.8%)  Avg: 2.64 ms
```

**这个节点在做什么**：
- 这是deformable attention的核心计算
- 6个Plugin实例对应6个deformable层

**为什么慢**：
1. **这是核心计算**：必须执行，但可能还有优化空间
2. **Plugin实现可能不是最优**：需要检查CUDA kernel
3. **输入数据准备开销**：每个Plugin前都有Reformat

**如何优化**：
1. **检查Plugin实现**：
   ```bash
   # 查看Plugin的CUDA实现
   # 检查是否有优化空间（共享内存、warp shuffle等）
   ```

2. **减少输入准备开销**：
   - 确保输入数据格式正确，减少Reformat
   - 使用pinned memory加速数据传输

3. **考虑使用TensorRT内置实现**（如果支持）

### 3.4 其他节点分析

#### Softmax节点
```
/Softmax       37.77 ms (0.4%)  Avg: 0.17 ms
```

**分析**：
- 占比很小（0.4%），不需要优化
- TensorRT已经高度优化

#### MatMul节点
```
/anchor_encoder/pos_fc/pos_fc.0/MatMul  3.85 ms  Avg: 0.018 ms
/weights_fc/MatMul  12.03 ms  Avg: 0.055 ms
```

**分析**：
- 单个MatMul很快（< 0.1ms）
- 但有很多MatMul，累积起来也有开销
- 通常不需要优化，除非占比很大

## 四、优化优先级排序

### 优先级1: ForeignNode节点（42%时间）⭐⭐⭐⭐⭐

**为什么优先**：
- 占比最大（42%）
- 优化空间最大（可以减少到20%以下）
- 相对容易优化（修改Python代码）

**优化方法**：
1. **减少Transpose操作**：
   - 在导出ONNX前，尽量使用reshape代替permute
   - 直接输出目标布局

2. **合并Transpose链**：
   - 使用ONNX优化工具合并连续的Transpose
   - 如果A→B→C，可以合并为A→C

3. **优化kps_generator**：
   - 检查kps_generator的实现
   - 看能否直接输出目标布局

### 优先级2: Reformat节点（3%时间）⭐⭐⭐⭐

**为什么优先**：
- 虽然占比不大，但是纯开销（没有计算价值）
- 容易优化（统一数据类型和布局）

**优化方法**：
1. **统一数据类型**：
   - 确保所有中间张量使用相同dtype（FP16）
   - 在导出ONNX前进行显式转换

2. **统一内存布局**：
   - 确保所有张量是连续的（contiguous）
   - 使用相同的布局（NCHW或NHWC）

3. **使用IO格式参数**：
   - 在构建engine时指定IO格式
   - 确保输入输出格式匹配

### 优先级3: Plugin优化（38%时间）⭐⭐⭐

**为什么优先**：
- 占比很大，但这是核心计算
- 优化空间有限（需要修改CUDA代码）

**优化方法**：
1. **检查Plugin实现**：
   - 查看CUDA kernel是否有优化空间
   - 检查内存访问模式

2. **减少输入准备开销**：
   - 确保输入数据格式正确
   - 减少Reformat

### 优先级4: 其他节点（17%时间）⭐

**为什么低优先级**：
- 占比不大
- TensorRT已经高度优化
- 优化空间有限

## 五、实际操作：如何分析你的Profile

### 5.1 提取关键数据

```bash
# 1. 提取总体性能
grep -A 10 "Performance summary" profile.log

# 2. 提取最耗时的节点（按Time %排序）
grep "Time %" profile.log | awk '{print $NF, $0}' | sort -rn | head -20

# 3. 提取所有ForeignNode节点
grep "ForeignNode" profile.log | awk '{print $NF, $0}' | sort -rn

# 4. 提取所有Reformat节点
grep "Reformat" profile.log | awk '{print $NF, $0}' | sort -rn

# 5. 提取所有Plugin节点
grep "DeformableAttentionAggrPlugin" profile.log | awk '{print $NF, $0}' | sort -rn
```

### 5.2 计算总占比

```python
# 计算ForeignNode总时间占比
# 从profile中提取所有ForeignNode的Time %，求和

# 示例：
# ForeignNode节点：
# - kps_generator: 8.8%
# - layers.10: 8.4%
# - layers.17: 8.4%
# - layers.24: 8.4%
# - layers.31: 8.4%
# 总计: 42.4%
```

### 5.3 识别优化目标

**规则**：
1. **Time % > 5%**：必须优化
2. **Time % 2-5%**：应该优化
3. **Time % < 2%**：可选优化

**你的情况**：
- ForeignNode总计：~42% → **必须优化**
- Reformat总计：~3% → **应该优化**
- Plugin总计：~38% → **核心计算，优化空间有限**

## 六、具体优化步骤

### 6.1 优化ForeignNode（最重要）

#### 步骤1: 理解Transpose的来源

```python
# 在 export_head_onnx.py 中查找所有 permute 操作
# 这些操作会在ONNX中生成Transpose节点

# 关键位置：
# 1. kps_generator的输出（需要查看kps_generator的实现）
# 2. points_2d的permute（339行）
# 3. weights的permute（344行）
```

#### 步骤2: 尝试减少Transpose

```python
# 方法1：直接输出目标布局
# 如果kps_generator可以修改，让它直接输出目标布局

# 方法2：使用view代替permute（如果内存布局允许）
# 注意：view要求数据在内存中是连续的，且形状兼容
points_2d = points_2d.view(bs, num_anchor, num_pts, num_cams, 2)  # 如果可能

# 方法3：合并permute和reshape
points_2d = points_2d.permute(0, 2, 3, 1, 4).view(...)  # 合并操作
```

#### 步骤3: 使用ONNX优化工具

```python
# 在 simplify_onnx.py 中已经添加了优化pass
# 但可能需要更激进的优化

# 可以尝试：
# 1. 使用onnxruntime的优化器
# 2. 使用TensorRT的图优化
# 3. 手动合并Transpose链
```

### 6.2 优化Reformat

#### 步骤1: 统一数据类型

```python
# 在 export_head_onnx.py 中，确保所有关键路径使用相同dtype
if getattr(self, "_export_head2_layout_opt", False):
    # 统一dtype
    tgt_dtype = feature.dtype  # 通常是FP16
    instance_feature = instance_feature.to(tgt_dtype)
    anchor = anchor.to(tgt_dtype)
    points_2d = points_2d.to(tgt_dtype)
    weights = weights.to(tgt_dtype)
```

#### 步骤2: 确保内存连续

```python
# 在关键位置添加contiguous()
instance_feature = instance_feature.contiguous()
anchor = anchor.contiguous()
points_2d = points_2d.contiguous()
weights = weights.contiguous()
```

#### 步骤3: 使用IO格式参数

```bash
# 在 build_sparse4d_engine.sh 中
--inputIOFormats=fp16:chw --outputIOFormats=fp16:chw
```

### 6.3 优化Plugin

#### 步骤1: 检查Plugin实现

```bash
# 查看Plugin的CUDA实现
# 检查是否有优化空间：
# - 共享内存使用
# - Warp shuffle
# - 内存访问模式
# - 计算优化
```

#### 步骤2: 减少输入准备开销

```python
# 确保输入数据格式正确，减少Reformat
# 已经在layout优化中处理
```

## 七、验证优化效果

### 7.1 对比优化前后

```bash
# 1. 运行优化前的profile
bash deploy/tools/profile_head2.sh --fp16 > profile_before.log

# 2. 应用优化

# 3. 运行优化后的profile
bash deploy/tools/profile_head2.sh --fp16 > profile_after.log

# 4. 对比关键指标
# - GPU计算时间
# - ForeignNode时间占比
# - Reformat时间占比
# - 总延迟
```

### 7.2 检查优化是否生效

```bash
# 1. 检查ONNX文件大小（优化后应该更小或节点更少）
ls -lh deploy/onnx/sparse4dhead2nd.onnx
ls -lh deploy/engine/sparse4d_head2.simplified.onnx

# 2. 检查ONNX节点数
python3 -c "import onnx; m=onnx.load('deploy/onnx/sparse4dhead2nd.onnx'); print(f'Nodes: {len(m.graph.node)}')"

# 3. 检查构建日志，确认优化参数生效
grep -i "inputIOFormats\|timingCache\|workspace" deploy/engine/build_head2.log
```

## 八、常见问题

### Q1: 为什么ForeignNode这么慢？

**A**: ForeignNode是TensorRT无法优化的节点，使用通用实现：
- 没有kernel融合
- 没有特殊优化
- 内存访问模式可能不是最优

### Q2: 为什么Reformat需要时间？

**A**: Reformat是纯内存拷贝：
- 数据类型转换（FP32 ↔ FP16）
- 内存布局转换（NCHW ↔ NHWC）
- 虽然单次很快，但累积起来也有开销

### Q3: 如何判断优化是否成功？

**A**: 对比关键指标：
- ForeignNode时间占比：42% → 目标 < 25%
- Reformat时间占比：3% → 目标 < 1%
- GPU计算时间：35ms → 目标 < 30ms

### Q4: 优化后性能反而变差怎么办？

**A**: 
1. 检查优化是否真的生效
2. 逐步回退优化，找出问题
3. 验证精度，确保没有精度损失

## 九、实战练习

### 练习1: 分析你的Profile

```bash
# 1. 提取最耗时的10个节点
grep "Time %" deploy/profiles/trtexec_head2_*.log | \
  awk '{if ($NF+0 > 0) print $NF, $0}' | \
  sort -rn | head -10

# 2. 计算ForeignNode总占比
grep "ForeignNode" deploy/profiles/trtexec_head2_*.log | \
  awk '{sum+=$NF} END {print "ForeignNode总占比:", sum"%"}'

# 3. 计算Reformat总占比
grep "Reformat" deploy/profiles/trtexec_head2_*.log | \
  awk '{sum+=$NF} END {print "Reformat总占比:", sum"%"}'
```

### 练习2: 找出需要优化的节点

根据你的profile，需要优化的节点：

1. **ForeignNode[/kps_generator/.../Transpose]** - 8.8%
2. **ForeignNode[/layers.10/.../Transpose_4]** - 8.4%
3. **ForeignNode[/layers.17/.../Transpose_6]** - 8.4%
4. **ForeignNode[/layers.24/.../Transpose_8]** - 8.4%
5. **ForeignNode[/layers.31/.../Transpose_10]** - 8.4%
6. **Reformatting CopyNode for Network Input feature** - 1.2%

### 练习3: 制定优化计划

1. **短期（1-2天）**：
   - 优化points_2d和weights的permute操作
   - 统一数据类型和内存布局

2. **中期（1周）**：
   - 优化kps_generator的输出布局
   - 使用ONNX优化工具合并Transpose

3. **长期（2-4周）**：
   - 检查Plugin实现
   - 优化H2D传输

## 十、总结

### 关键要点

1. **看Time %，不是看绝对时间**：占比大的节点优先优化
2. **ForeignNode是最大瓶颈**：42%时间，优化空间最大
3. **Reformat是纯开销**：3%时间，容易优化
4. **Plugin是核心计算**：38%时间，优化空间有限

### 优化顺序

1. ✅ **ForeignNode**（42%）- 最高优先级
2. ✅ **Reformat**（3%）- 高优先级
3. ⚠️ **Plugin**（38%）- 中优先级（需要深入优化）
4. ⚠️ **其他**（17%）- 低优先级

### 下一步行动

1. 分析你的profile，找出最耗时的节点
2. 理解每个节点的作用
3. 制定优化计划
4. 逐步实施优化
5. 验证优化效果

