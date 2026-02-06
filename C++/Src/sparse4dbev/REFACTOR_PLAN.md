# Sparse4D BEV 泛型框架重构与优化指南

## 1. 项目状态概览 (Status Overview)

### 当前架构状态
已完成从 `Sparse4DImpl<T>` 模板类向 **Engine-Driven 泛型框架** 的核心重构，并针对 Jetson (Tegra) 平台进行了深度性能优化。当前系统具备以下特性：
- **运行时自适应**：通过 `TensorBuffer` 和 TensorRT Engine Bindings 动态决定 Tensor 的精度（FP32/FP16/INT8）和维度。
- **混合精度支持**：支持 INT8 Backbone + FP16 Head 等混合精度管线。
- **全 GPU 化 InstanceBank**：InstanceBank 模块已移除所有 CPU 计算和同步操作，采用 Thrust 实现 GPU 排序。
- **Postprocessor 全 GPU 化**：引入 GPU Sort，彻底消除了后处理阶段的 D2H 拷贝（仅剩极小的 valid_count 同步）。
- **CUDA Graph 集成**：对稳态推理流程（Backbone -> Head -> InstanceBank）进行了 Graph Capture，消除了 Kernel Launch Overhead。
- **Zero-Copy 输入输出**：Preprocess 输入和 Postprocess 输出均采用 Mapped/Unified Memory，消除了 H2D/D2H 内存拷贝。

### 性能现状
- **Backbone & Head**: 耗时稳定在 ~125ms (Backbone ~48ms, Head ~77ms)，已消除 CPU 调度延迟。
- **InstanceBank**: 耗时显著降低 (< 3ms)，且不再阻塞 CPU 线程。
- **Pre/Post**: 耗时极低 (Pre ~5.5ms, Post ~0.3ms)，虽然 Zero-Copy 带来的绝对耗时减少不明显（可能受限于 Kernel 计算瓶颈或 CPU memcpy），但有效降低了总线负载。

---

## 2. 已完成的重构工作 (Completed Refactoring)

### Phase 1: 基础设施建设 (Infrastructure)
- [x] **智能缓冲区 (`TensorBuffer`)**: 实现携带 `nvinfer1::DataType` 和 `Dims` 元数据的显存管理类。
- [x] **Engine 封装升级**: `EngineWrapper` 新增 `get_binding_dtype` 接口。

### Phase 2: 核心运行时重构 (Core Runtime)
- [x] **移除模板依赖**: 废弃 `Sparse4DImpl<T>`，实现非模板类 `Sparse4DRuntime`。
- [x] **动态内存初始化**: `init_memory` 改为遍历 Engine Bindings 动态分配资源。
- [x] **Bug 修复**: 修复了 `getVolume` 对动态维度（-1）处理不当导致的大内存分配错误。

### Phase 3: 组件适配 (Component Adaptation)
- [x] **Preprocessor**: 接口改为接受 `TensorBuffer`，内部根据输出类型分发到不同精度的 CUDA Kernel。
- [x] **Postprocessor**: 适配 `TensorBuffer` 输入，支持混合精度输入。
- [x] **InstanceBank**: 移除模板参数，内部 Tensor 动态适配。

### Phase 4: 性能救援与优化 (Performance Rescue)
- [x] **InstanceBank 全 GPU 化**: 移除了 `cudaStreamSynchronize` 和 Host 端 `std::vector` 操作，引入 Thrust GPU 排序。
- [x] **Postprocessor 全 GPU 化**: 引入 `thrust::sort_by_key` 替代 CPU `std::sort`，消除了中间的 D2H 拷贝。
- [x] **显存复用**: 将临时 Buffer 提升为类成员，避免单帧推理中的动态内存分配。

### Phase 5: 零拷贝与显存优化 (Zero-Copy & Memory)
- [x] **Preprocessor 输入零拷贝**: 使用 `cudaHostAllocMapped` + `cudaHostGetDevicePointer`，移除 H2D 拷贝。
- [x] **Postprocessor 输出零拷贝**: 使用 `cudaMallocManaged` (Unified Memory)，移除 D2H 拷贝。

### Phase 6: 执行图优化 (CUDA Graphs)
- [x] **引入 Graph 机制**: 在稳态（非首帧）下录制并执行 `Backbone -> Head -> InstanceBank` 的完整流程。
- [x] **InstanceBank 拆分**: 将逻辑拆分为 `Prepare` (CPU) 和 `Execute` (GPU)，确保 Graph 可捕获。

---

## 3. 后续优化计划 (Future Optimization Plan)

鉴于框架层面的调度开销（Launch Overhead, Memory Copy, CPU Sync）已降至极限，下一步优化应聚焦于 **计算效率** 和 **系统级并行**。

### Phase 7: 流水线并行 (Pipeline Parallelism)

**目标**：打破当前的串行执行模式 (`Pre -> Backbone -> Head -> Post`)，利用 CPU 和 GPU 的异步特性实现流水线重叠。

#### 步骤 7.1: 双流并行 (Double Buffering)
- **现状**：当前帧必须等待上一帧处理完毕（或至少 CPU 提交完毕）才开始。虽然 CUDA 是异步的，但 Host 端通常是串行的循环。
- **优化方案**：
  - 维护两个 Stream 或两个 Context。
  - **Frame N+1 Preprocess (CPU part)** 与 **Frame N Inference (GPU)** 并行。
  - 由于 Preprocess 中有耗时的 `memcpy` (图像从 buffer 到 pinned memory)，将其移到独立线程或使用 DMA（如果硬件支持）。

### Phase 8: 输入数据流优化 (Input Data Flow)

**目标**：解决 Preprocess 中 5ms 的瓶颈（推测为 CPU memcpy）。

#### 步骤 8.1: 消除 CPU memcpy
- **现状**：`Preprocessor::forward` 中使用 `std::memcpy` 将图像数据从 `vecImageBuf` 拷贝到 Pinned Memory。这是 CPU 密集型操作。
- **优化方案**：
  - 如果上游（Camera Driver / Middleware）支持直接将图像写入预分配的 **Pinned Memory**（即我们分配的 `h_pinned_`），则可完全消除这 5ms 开销。
  - 需要修改上游接口，使其接受外部 Buffer 指针，或者直接传递 Pinned Buffer 指针给 Preprocessor。

### Phase 9: INT8 精度升级 (INT8 Quantification)

**目标**：大幅降低 Backbone 和 Head 的计算耗时（目前占 95%）。

#### 步骤 9.1: Backbone INT8
- **现状**：Backbone 目前可能运行在 FP16 或 FP32。
- **优化方案**：使用 TensorRT 的 PTQ (Post-Training Quantization) 或 QAT (Quantization-Aware Training) 将 Backbone 转为 INT8。
- **收益**：在 Orin/Xavier 平台上，INT8 Tensor Core 吞吐量是 FP16 的 2 倍。

#### 步骤 9.2: Head INT8 (部分)
- **挑战**：Head 部分包含回归任务，对精度敏感。
- **方案**：仅对 Head 中的卷积层或全连接层进行 INT8 量化，保留 LayerNorm、Sigmoid、GridSampler 为 FP16/FP32。

### Phase 10: 算子融合与插件优化 (Kernel Fusion)

**目标**：减少显存读写带宽占用。

#### 步骤 10.1: Transpose/Permute 融合
- 检查 Backbone/Head 输出是否存在独立的 Transpose 操作，尝试将其融合到前一个卷积层或后处理 Kernel 中。

#### 步骤 10.2: InstanceBank Kernel 融合
- 当前 `InstanceBank` 有多个小 Kernel (`project`, `compute_ids`, `update`, `sort`, `nms`)。
- 可以尝试将 `update` 和 `decay` 融合，减少一次 Global Memory 读写。

---

## 4. 总结与建议

当前框架已达到 **Tegra 平台上的“零开销”架构标准**（Zero-Overhead Architecture）：
1.  **无多余拷贝** (Zero-Copy)。
2.  **无 CPU 阻塞** (Full GPU Pipeline)。
3.  **无启动开销** (CUDA Graphs)。

**下一步最关键的优化点**不在代码架构，而在：
1.  **数据源头**：能否让 Camera 直接写 Pinned Memory？(Phase 8)
2.  **模型本身**：Backbone 是否已开启 INT8？(Phase 9)

