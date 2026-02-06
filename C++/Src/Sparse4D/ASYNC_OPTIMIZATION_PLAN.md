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
2.  **CPU/GPU 异步流水线**: 引入独立的 `InputThread` 负责预处理，将准备好的数据推入队列。
3.  **Backbone/Head 双流并行**:
    - **Stream 1 (Backbone Stream)**: 仅负责 Backbone 计算。
    - **Stream 2 (Head Stream)**: 负责 InstanceBank, Head, Postprocess。
    - **并行逻辑**: 第 N+1 帧的 Backbone 可以与 第 N 帧的 Head 并行执行。

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
- 新增 `BlockingQueue<FrameContext*> ready_queue_` (预处理 -> 推理)。
- 新增 `std::thread preprocess_thread_`。
- 新增 `cudaStream_t stream_backbone_`, `stream_head_`。

## 3. 执行流程 (Pipeline)

### 阶段 1: Input Thread (CPU)
1. 从 `FreePool` 获取空闲 `FrameContext`。
2. 执行 `Preprocessor::forward` (CPU/GPU Copy)。
3. 将 Context 推入 `ReadyQueue`。

### 阶段 2: Inference Loop (Main Thread / GPU)
1. 从 `ReadyQueue` 取出 Context (Frame N)。
2. **Backbone 阶段**:
   - 在 `stream_backbone_` 上发起 Backbone 推理。
   - 记录 `event_backbone_done`。
3. **Head 阶段**:
   - 在 `stream_head_` 上等待 `event_prev_cache_done` (等待上一帧 N-1 的时序信息更新完毕)。
   - 执行 `InstanceBank::get` (Frame N)。
   - 在 `stream_head_` 上等待 `event_backbone_done` (等待当前帧 Backbone 完成)。
   - 执行 `Head1/Head2` (Frame N)。
   - 执行 `InstanceBank::cache` (Frame N)。
   - 记录 `event_prev_cache_done` (供下一帧 N+1 使用)。
4. **Postprocess 阶段**:
   - 执行 `Postprocessor`。
   - 回收 Context 到 `FreePool`。

## 4. 实施步骤

### 步骤 1: 基础结构重构
- [ ] 定义 `FrameContext` 结构体。
- [ ] 修改 `CoreImplement::init`，分配 3 套 `FrameContext` 资源。
- [ ] 实现资源池管理 (Acquire/Release)。

### 步骤 2: 预处理异步化
- [ ] 实现 `InputThread` 线程循环。
- [ ] 实现线程安全的队列。
- [ ] 将 `preprocess` 逻辑移动到独立线程。

### 步骤 3: 双流推理实现
- [ ] 创建双 CUDA Stream。
- [ ] 重构 `forward` 函数，分离 Backbone 和 Head 的执行流。
- [ ] 添加 CUDA Event 插入点，确保正确的依赖关系：
    - Head(N) 依赖 Backbone(N)。
    - InstanceBank::get(N) 依赖 InstanceBank::cache(N-1)。

## 5. 预期收益
- **预处理掩盖**: ~5ms 的预处理时间将被完全掩盖。
- **Backbone/Head 并行**: 理想情况下，吞吐量将由 `max(Backbone_Time, Head_Time)` 决定，而非两者的和。
    - 当前: 48ms + 76ms = 124ms
    - 优化后理论极限: max(48, 76) = 76ms (提升约 60%)
