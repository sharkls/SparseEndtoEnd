# Head2推理耗时波动分析

## 问题现象

从实际推理日志中观察到Head2推理耗时存在较大波动：

| Sample | Head2耗时 (ms) | 波动 |
|--------|---------------|------|
| 1      | 117.87        | -    |
| 2      | 101.99        | -15.88 |
| 3      | 137.77        | +35.78 |
| 4      | 113.13        | -24.64 |
| 5      | 113.45        | +0.32 |
| 6      | 140.10        | +26.65 |
| 7      | 153.72        | +13.62 |
| 8      | 102.07        | -51.65 |
| 9      | 98.08         | -3.99 |

**统计信息：**
- 最小值：98.08 ms
- 最大值：153.72 ms
- 平均值：~119 ms
- 标准差：~18 ms
- 波动范围：55.64 ms（约35-40%）

## 可能原因分析

### 1. GPU频率动态调整（DVFS） ⭐⭐⭐⭐⭐

**影响程度：高**

GPU的Dynamic Voltage and Frequency Scaling (DVFS)会根据负载自动调整频率，导致性能波动。

**验证方法：**
```bash
# 监控GPU频率
watch -n 0.1 nvidia-smi --query-gpu=clocks.current.sm,clocks.max.sm --format=csv
```

**解决方案：**
```bash
# 设置GPU为性能模式（需要root权限）
sudo nvidia-smi -pm 1  # 启用持久化模式
sudo nvidia-smi -pl 100  # 设置最大功耗限制

# 或者固定GPU频率（如果GPU支持）
sudo nvidia-smi -lgc <固定频率>
```

### 2. CUDA Graph未完全生效 ⭐⭐⭐⭐

**影响程度：中高**

虽然代码中有warmup逻辑，但Head2的CUDA Graph可能未完全捕获，导致每次推理都需要重新构建执行图。

**当前warmup设置：**
```cpp
const int HEAD2_WARMUP = 5;  // 可能不够
```

**验证方法：**
- 检查TensorRT日志中是否有 `Skip layer timing collection in CUDA graph capture mode.`
- 测量enqueue时间，如果>1ms说明CUDA Graph可能未生效

**解决方案：**
1. 增加warmup次数到10-15次
2. 确保所有输入缓冲区地址固定
3. 使用固定的CUDA stream（代码中已实现）

### 3. 动态Shape导致的Kernel选择 ⭐⭐⭐

**影响程度：中**

Head2的某些输入可能有动态shape（如`temp_instance_feature`、`temp_anchor`），导致TensorRT在不同样本间选择不同的kernel实现。

**检查点：**
- `temp_instance_feature`: [1, 600, 256] - 固定shape
- `temp_anchor`: [1, 600, 11] - 固定shape
- `mask`: [1] - 固定shape
- `track_id`: [1, 900] - 固定shape

虽然shape看起来是固定的，但实际数据内容可能影响某些操作的执行路径。

### 4. 内存分配和碎片化 ⭐⭐⭐

**影响程度：中**

每次推理可能有不同的内存分配模式，导致内存访问延迟不同。

**解决方案：**
- 使用CUDA内存池（`cudaMallocAsync`）
- 预分配所有需要的缓冲区
- 减少动态内存分配

### 5. 数据依赖导致的路径差异 ⭐⭐

**影响程度：低-中**

不同样本的数据可能导致模型内部的条件分支选择不同的计算路径。

**检查点：**
- `mask`值可能影响某些条件分支
- `track_id`的分布可能影响某些操作的性能

### 6. TensorRT优化策略选择 ⭐⭐

**影响程度：低-中**

TensorRT在构建engine时可能为某些操作选择了多个kernel实现，运行时根据数据特征选择不同的kernel。

## 优化建议

### 短期优化（立即实施）

1. **固定GPU频率**
   ```bash
   # 在推理前执行
   sudo nvidia-smi -pm 1
   sudo nvidia-smi -pl 100
   ```

2. **增加Head2 warmup次数**
   ```cpp
   const int HEAD2_WARMUP = 15;  // 从5增加到15
   ```

3. **添加性能监控**
   ```cpp
   // 在second_head.cpp中添加
   static int call_count = 0;
   call_count++;
   if (call_count <= 20) {
       auto start = std::chrono::high_resolution_clock::now();
       // ... inference ...
       auto end = std::chrono::high_resolution_clock::now();
       LOG(INFO) << "Head2 inference #" << call_count 
                << " time: " << duration.count() << " ms";
   }
   ```

### 中期优化（需要测试）

1. **使用CUDA Graph API显式管理**
   ```cpp
   // 显式创建和重用CUDA Graph
   cudaGraph_t graph;
   cudaGraphExec_t graphExec;
   // ... 捕获和执行逻辑 ...
   ```

2. **优化内存分配策略**
   - 使用`cudaMallocAsync`替代`cudaMalloc`
   - 预分配所有缓冲区，避免运行时分配

3. **分析逐层性能**
   ```bash
   # 使用trtexec分析每层耗时
   trtexec --loadEngine=sparse4dhead2nd.engine \
           --dumpProfile \
           --exportProfile=profile.json
   ```

### 长期优化（架构级）

1. **重新构建Engine时使用更严格的优化**
   ```bash
   # 在build_sparse4d_engine.sh中添加
   --best  # 使用最佳性能策略（构建时间更长）
   --noTF32  # 禁用TF32，确保FP16一致性
   ```

2. **考虑使用TensorRT的Profile功能**
   - 为不同的输入shape创建多个profile
   - 运行时根据实际shape选择对应的profile

3. **分析并优化模型结构**
   - 识别性能瓶颈层
   - 考虑算子融合优化

## 验证方法

### 1. 检查GPU频率稳定性
```bash
# 运行推理时监控GPU频率
nvidia-smi dmon -s u -c 1000 > gpu_freq.log
```

### 2. 检查CUDA Graph是否生效
```bash
# 查看TensorRT日志
grep -i "cuda graph" tensorrt.log
```

### 3. 统计分析
```python
import numpy as np
times = [117.87, 101.99, 137.77, 113.13, 113.45, 140.10, 153.72, 102.07, 98.08]
print(f"Mean: {np.mean(times):.2f} ms")
print(f"Std: {np.std(times):.2f} ms")
print(f"CV: {np.std(times)/np.mean(times)*100:.1f}%")  # 变异系数
```

## 预期效果

实施短期优化后，预期可以将波动范围从55ms降低到20-30ms（约15-20%的波动率）。

