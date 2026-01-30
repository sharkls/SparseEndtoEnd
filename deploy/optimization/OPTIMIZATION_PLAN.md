# Sparse4D E2E 性能优化计划文档

## 1. 背景与现状

在 NVIDIA Orin AGX 平台上对 `head1` 模型进行 TensorRT 构建与推理分析。
- **初始状态 (V2)**: 单帧推理总耗时约为 **67.48 ms**。
- **阶段一优化后 (V3.1)**: 单帧推理总耗时降低至 **49.35 ms** (提升约 27%)。

为了满足实时性要求（通常目标 < 30ms），需要继续对模型进行深度优化。

## 2. 性能瓶颈分析 (基于 V3.1 - 49.35ms)

经过 Phase 1 的优化，DFA 数据重排的开销已基本消除。当前的主要瓶颈转移到了计算密集型层和内存受限型层：

### 2.1 FFN 部分的大矩阵乘法 (Top 1)
- **现象**: FFN (Feed Forward Network) 中的 MatMul 操作耗时显著。
- **日志特征**:
  ```
  [I] 132.79 2.1418 2.1413 4.3 {ForeignNode[.../fc_after_1/MatMul]}
  ```
  单层耗时约 **2.14 ms**，6 层总计约 **12.8 ms**，占总耗时的 **~26%**。
- **原因**: 模型维度较大 (Embed dims=256)，FP16 精度下计算密集。

### 2.2 Softmax 效率低下 (Top 2)
- **现象**: 注意力机制中的 Softmax 操作非常慢。
- **日志特征**:
  ```
  [I] 104.33 1.6827 1.6824 3.4 /Softmax
  ```
  单层耗时约 **1.68 ms**，6 层总计约 **10.1 ms**，占总耗时的 **~20%**。
- **原因**: Softmax 是带宽受限型操作 (Memory Bound)。

### 2.3 剩余的 Reformatting/Cast 节点 (Top 3)
- **现象**: 虽然消除了主要的 DFA Reformat，但仍存在一些细碎的转换。
- **日志特征**:
  ```
  [I] 7.23 0.1167 0.1163 0.2 /Cast_1
  ```
  这类 Cast 节点虽然单次不慢，但累积起来也有一定开销。

## 3. 优化解决方案

### 3.1 方案一：优化 DFA 输入数据生成 (Phase 1 - 已完成)
**状态**: ✅ 已完成 (V3.1)
**效果**: 成功消除了 3.15ms/层的 Reformat 开销。Orin 上总耗时从 67.48ms -> 49.35ms。

### 3.2 方案二：INT8 量化 (针对 Top 1 瓶颈)
**目标**: 利用 Orin AGX 强大的 INT8 Tensor Core 加速 MatMul。
**预期收益**: FFN 层耗时减半，预计总耗时减少 **~6 ms**。

**实施步骤**:
1.  **准备校准**: 解决 Calibrator 的 Segfault 问题，跑通校准流程。
2.  **PTQ (Post-Training Quantization)**: 对 FFN 的 Linear 层进行 INT8 量化。
3.  **混合精度**: 保持 DFA、LayerNorm、Softmax 为 FP16/FP32。

### 3.3 方案三：Softmax 优化与算子融合 (针对 Top 2 瓶颈)
**目标**: 将 Softmax 耗时降低至 < 0.2ms。
**预期收益**: 预计总耗时减少 **~8 ms**。

**实施步骤**:
1.  **插件化**: 将 Softmax 或整个 Attention 模块封装为插件。
2.  **算子融合**: 在 CUDA Kernel 中融合 Scale + Mask + Softmax。

## 4. 实施路线图

| 阶段 | 任务 | 状态 | 预计/实际收益 |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **消除 DFA 输入重排** (V3.1) | ✅ 完成 | **18.13 ms** (实测 Orin) |
| **Phase 2** | **INT8 量化** <br> 重点优化 FFN MatMul。 | 🚧 进行中 (A6000✅ Orin❌) | **RTX A6000**: **9.01 ms** (Kernel Time) <br> **Orin AGX**: **~50 ms** (性能回退，ForeignNode 导致) |
| **Phase 3** | **Softmax 优化** <br> 开发高效 Softmax/Attention 插件。 | 📅 待定 | **~8 ms** |

## 5. 下一步行动
1.  **修复 Orin INT8 性能**: 修改 `build_engine.py`，通过显式设置层精度 (Explicit Precision) 强制 Orin 上的 TensorRT 使用 INT8 Kernel，解决 `ForeignNode` 导致的性能回退。
2.  **完善误差分析**: 更新误差分析脚本，增加 Top-K 过滤逻辑，输出更有意义的精度指标。
3.  **端到端测试**: 完成优化后，测试 Backbone(INT8) + Head(INT8) 的全链路延迟。
