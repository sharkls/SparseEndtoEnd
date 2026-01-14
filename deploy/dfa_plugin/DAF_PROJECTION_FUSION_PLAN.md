# DAF 插件投影逻辑融合方案 (Plan A)

## 1. 背景
在 Orin AGX 平台上，Sparse4D 模型的推理瓶颈目前集中在 `project_points` 产生的 `ForeignNode`。这些节点包含复杂的 3D 到 2D 投影计算以及跨度极大的 `Transpose/Permute` 操作，导致单层延迟约 **3.15ms**，全模型累积延迟约 **20ms**。

## 2. 修改目标
通过将 3D 投影逻辑直接内置到 `DeformableAttentionAggrPlugin` CUDA Kernel 中，消除 ONNX 层面的所有布局转换算子，实现计算融合并降低 I/O 开销。

## 3. 修改计划

### 阶段一：C++ 插件层修改 (`deploy/dfa_plugin/`)

#### 3.1 Kernel 核心逻辑更新 (`deformableAttentionAggr.cu`)
- **坐标投影集成**：在 `deformable_aggregation_kernel` 内部增加矩阵乘法逻辑。
- **输入参数变更**：
    - 移除 `sampling_location` (2D [BS, Q, P, C, 2])。
    - 新增 `key_points` (3D [BS, Q, P, 3])。
    - 新增 `lidar2img` (矩阵 [BS, C, 4, 4])。
    - 新增 `image_wh` (尺寸 [BS, C, 2])。
- **内存优化**：利用 `shared memory` 存储常用的投影矩阵，减少全局显存访问。

#### 3.2 Plugin 类接口更新 (`deformableAttentionAggrPlugin.h/.cpp`)
- **描述符更新**：修改 `getOutputDimensions` 和 `supportsFormatCombination` 以适配新的输入 Tensor。
- **属性管理**：更新 `PluginField` 以处理新增的静态属性（如相机数量、采样点数等）。
- **FP16 支持**：确保投影矩阵运算支持 `half2` 或高精度 `float` 混合计算。

### 阶段二：Python 导出层修改 (`deploy/`)

#### 3.3 模型导出脚本更新 (`export_head_onnxv2.py`)
- **逻辑重构**：移除 `self.layers[i].project_points(...)` 调用及其后的 `.permute()` 和 `.reshape()`。
- **输入链接**：将 3D `key_points` 和 `lidar2img` 直接链接至 DAF 插件节点。

#### 3.4 算子映射更新 (`modules/ops/deformable_aggregation.py`)
- **Symbolic 注册**：更新 `DeformableAggregationFunction.symbolic` 方法，使其符合新的 Tensor 输入序列。

## 4. 验证计划

### 4.1 精度对齐验证 (Accuracy)
- **工具**：`deploy/dfa_plugin/unit_test/`。
- **方法**：
    1.  提取 PyTorch 原始投影输出作为 Ground Truth。
    2.  输入相同的 3D 点云给新插件，对比输出的特征聚合结果。
- **指标**：FP16 模式下相对误差 $< 10^{-3}$。

### 4.2 图结构验证 (Graph)
- **工具**：`netron`。
- **检查项**：
    - 确认原有的 `Transpose` 和 `Reshape` 节点被完全移除。
    - 确认 DAF 节点的输入直接来源于 `lidar2img` 矩阵。

### 4.3 性能收益验证 (Performance)
- **环境**：Orin AGX (Max-N mode)。
- **工具**：`trtexec --loadPlugins=... --fp16`。
- **目标**：
    - 总延迟预期下降 **15ms - 20ms**。
    - 彻底消除带有 `[Transpose]` 标记的 `ForeignNode`。

## 5. 预期风险
- **精度下降**：在 GPU 端直接进行投影计算，若使用 FP16 可能会因动态范围不足导致采样偏移。
- **策略**：若精度受损，将投影矩阵运算部分强制使用 `float32` 执行。

---

## 实施进度 (2026-01-14)

- [x] **Stage 1: C++ 插件修改**
    - [x] 在 `deformableAttentionAggr.cu` 中实现 3D->2D 投影逻辑（FP32/FP16/INT8）。
    - [x] 更新 `DeformableAttentionAggrPlugin` 接口以支持 7 个输入。
    - [x] 实现新增维度的序列化与反序列化。
    - [x] 在 Orin 平台上完成插件编译。
- [x] **Stage 2: Python 导出脚本修改**
    - [x] 更新 `DeformableAggregationFunction` 的 `symbolic` 和 `forward` 方法。
    - [x] 优化 `forward` 方法，增加导出模式下的 Dummy Tensor 返回逻辑，避免形状冲突。
    - [x] 修改 `deploy/export_head_onnxv2.py`，移除冗余投影逻辑，直接传递 3D 输入。
- [ ] **Stage 3: 验证**
    - [ ] 在工作站执行 ONNX 导出。
    - [ ] 在 Orin 上构建 Engine 并进行性能测试。
    - [ ] 验证检测精度。
