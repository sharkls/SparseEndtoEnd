# Sparse4D E2E 异步流水线与双流优化计划

## 1. 优化目标
通过引入多级流水线和GPU双流并行机制，最大化硬件利用率，提升整体吞吐量（FPS）。

### 当前性能瓶颈
- **串行执行**: Preprocess (CPU) -> Backbone (GPU) -> Head (GPU) -> Postprocess (CPU) 严格串行。
- **GPU 空闲**: 在 CPU 执行预处理时，GPU 处于空闲状态。
- **单流阻塞**: Backbone 和 Head 位于同一 CUDA Stream，无法利用帧间并行性（即无法在计算第 N 帧 Head 时同时计算第 N+1 帧 Backbone）。

## 2. 架构设计

### 2.1 核心策略
1.  **显存资源池化 (N-Buffering)**: 将原本绑定在类上的 `PipelineContext` 解耦，创建包含 3 个 `FrameContext` 的资源池，支持多帧同时在不同阶段处理。
2.  **CPU/GPU 异步流水线**: 引入独立的 `InferenceThread`，主线程仅负责预处理提交，实现 CPU 预处理与 GPU 推理的重叠。
3.  **Backbone/Head 双流并行**:
    - **Stream 1 (Backbone Stream)**: 仅负责 Preprocess (H2D) 和 Backbone 计算。
    - **Stream 2 (Head Stream)**: 负责 InstanceBank, Head, Postprocess。
    - **并行逻辑**: Preprocess/Backbone (Stream 1) 与 InstanceBank::get (Stream 2) 并行；Head (Stream 2) 等待 Backbone (Stream 1)。

### 2.2 数据结构变更

#### 新增 `FrameContext`
将每帧独立的数据封装：
- `common::PipelineContext pipeline_context` (输入特征、中间变量)
- `common::HeadOutput head_output` (输出结果)
- `cudaEvent_t event_backbone_done` (同步信号：Backbone完成)
- `cudaEvent_t event_all_done` (同步信号：整帧完成，用于资源回收)

#### `CoreImplement` 成员变更
- 删除单例的 `pipeline_context_`, `head_output_`。
- 新增 `std::vector<std::shared_ptr<FrameContext>> context_pool_`。
- 新增 `std::queue<std::shared_ptr<FrameContext>> ready_queue_` (预处理 -> 推理)。
- 新增 `std::thread inference_thread_`。
- 新增 `cudaStream_t stream_backbone_`, `stream_head_`。

## 3. 执行流程 (Pipeline)

### 阶段 1: Main Thread (runAlgorithm)
1. 从 `FreePool` 获取空闲 `FrameContext`。
2. **预处理 (Stream Backbone)**: 执行 `Preprocessor::forward` (Host->Device 图像拷贝与处理)，使用 `stream_backbone_`。
3. **InstanceBank::get (Stream Head)**: 执行 `InstanceBank::get` (历史信息获取)，使用 `stream_head_`。
   - 注意：这与预处理在不同流上并行执行。
4. 将 Context 推入 `ReadyQueue` 并立即返回。

### 阶段 2: Inference Thread (Loop)
1. 从 `ReadyQueue` 取出 Context (Frame N)。
2. **Backbone 阶段 (Stream Backbone)**:
   - 发起 Backbone 推理 (使用 `stream_backbone_`)。
   - 记录 `event_backbone_done`。
3. **Head 阶段 (Stream Head)**:
   - **帧内同步**: 等待 `event_backbone_done` (确保当前帧 Backbone 完成)。
   - **Head 推理**: 执行 Head1/Head2 (使用 `stream_head_`)。
   - **InstanceBank::cache**: 更新历史信息。
   - **InstanceBank::getTrackId**: 获取 Track ID。
4. **Postprocess 阶段**:
   - 执行 `Postprocessor`。
   - 执行回调 `alg_cb_`。
   - 回收 Context 到 `FreePool`。

## 4. 实施步骤

### 步骤 1: 基础结构重构 (已完成)
- [x] 定义 `FrameContext` 结构体。
- [x] 修改 `CoreImplement::init`，分配 3 套 `FrameContext` 资源。
- [x] 实现资源池管理 (Acquire/Release)。

### 步骤 2: 预处理异步化 (已完成)
- [x] 实现 `InferenceThread` 线程循环。
- [x] 改造 `runAlgorithm` 为异步提交模式。
- [x] 将 GPU 推理逻辑移动到独立线程。
- [x] 修复 `InstanceBank::get` 导致的断错误 (移回主线程执行)。

### 步骤 3: 双流推理实现 (已完成)
- [x] **流分配**:
    - `Preprocessor` -> `stream_backbone_`
    - `InstanceBank::get` -> `stream_head_`
    - `Backbone` -> `stream_backbone_`
    - `Head`/`Postprocess` -> `stream_head_`
- [x] **同步机制**:
    - 在 Backbone 完成后记录 Event。
    - 在 Head 开始前等待 Event。
- [x] **验证**: 编译通过，测试程序运行正常 (Exit Code 0)。

## 5. 预期收益
- **预处理掩盖**: ~5ms 的预处理时间将被完全掩盖。
- **Backbone/Head 并行**: 理想情况下，吞吐量将由 `max(Backbone_Time, Head_Time)` 决定，而非两者的和。
    - 当前: 48ms + 76ms = 124ms
    - 优化后理论极限: max(48, 76) = 76ms (提升约 60%)
