# Deploy Sparse4D Pipeline In LocalWorkStation

## 概述

本部署流程包含3个优化的TensorRT自定义插件，用于加速Sparse4D模型的推理性能：

1. **DFA Plugin** (Deformable Attention Aggregation): 多尺度可变形注意力聚合，已优化FP16性能
2. **LN Plugin** (Custom LayerNorm): 自定义LayerNorm实现，支持FP32/FP16
3. **SparseBox Plugin** (SparseBox3DKeyPoints): 3D关键点生成，避免ForeignNode，性能提升10-50x

## STEP1. Export Onnx

```bash
cd /path/to/SparseEnd2End
python deploy/export_backbone_onnx.py --cfg /path/to/cfg --ckpt /path/to/ckpt
python deploy/export_head_onnx.py --cfg /path/to/cfg --ckpt /path/to/ckpt
```

onnx will save in deploy/onnxlog like below:  
>deploy/onnx  
>├── export_backbone_onnx.log  
>├── export_head_onnx.log  
>├── sparse4dbackbone.onnx  
>├── sparse4dhead1st.onnx  
>└── sparse4dhead2nd.onnx  

## STEP2. Compile Custom Plugins

### 2.1 设置环境变量

首先需要设置环境变量，在 `tools/set_env.sh` 中配置：

```bash
cd deploy
. tools/set_env.sh
```

环境变量配置示例：
```bash
====================================================================================================================
|||  Config Environment Below:
|||  TensorRT LIB        : /mnt/env/tensorrt/TensorRT-8.5.1.7/lib
|||  TensorRT INC        :  /mnt/env/tensorrt/TensorRT-8.5.1.7/include
|||  TensorRT BIN        : /mnt/env/tensorrt/TensorRT-8.5.1.7/bin
|||  CUDA_LIB    : /usr/local/cuda-11.6/lib64
|||  CUDA_ INC   : /usr/local/cuda-11.6/include
|||  CUDA_BIN    : /usr/local/cuda-11.6/bin
|||  CUDNN_LIB   : /mnt/env/tensorrt/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib
|||  CUDASM      : sm_86
|||  ENVBUILDDIR : build
|||  ENVTARGETPLUGIN     : dfa_plugin/lib/deformableAttentionAggr.so
|||  ENV_LAYER_NORM_PLUGIN: ln_plugin/lib/customLayerNorm.so
|||  ENV_SPARSEBOX_PLUGIN : sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so
====================================================================================================================
[INFO] Config Env Done, Please Check EnvPrintOut Above!
```

### 2.2 编译 DFA Plugin

**功能**: 多尺度可变形注意力聚合算子

**优化特性**:
- ✅ **FP16优化**: 使用FP32临时缓冲区进行累加，避免half的atomicCAS开销
- ✅ **性能提升**: FP16版本比FP32版本快 **1.5-2.0x**
- ✅ **内存优化**: 输入/输出使用FP16，内存带宽减半
- ✅ **精度保证**: 中间累加使用FP32，保证计算精度

**编译步骤**:
```bash
cd deploy/dfa_plugin
./build.sh
# 或使用 make
make -j8
```

**输出**: `lib/deformableAttentionAggr.so`

**详细文档**: 参见 `deploy/dfa_plugin/FP16_PERFORMANCE_OPTIMIZATION.md`

### 2.3 编译 LN Plugin (Custom LayerNorm)

**功能**: 自定义LayerNorm实现，替代TensorRT默认实现

**优化特性**:
- ✅ **FP32/FP16支持**: 同时支持FP32和FP16精度
- ✅ **Warp级优化**: 使用warp shuffle进行高效的reduction操作
- ✅ **内存高效**: 减少中间内存分配

**编译步骤**:
```bash
cd deploy/ln_plugin
./build.sh
# 或使用 make
make -j8
```

**输出**: `lib/customLayerNorm.so`

### 2.4 编译 SparseBox Plugin

**功能**: 3D关键点生成算子，替代PyTorch的SparseBox3DKeyPointsGenerator

**优化特性**:
- ✅ **消除ForeignNode**: 避免TensorRT无法优化的20+个基础操作
- ✅ **性能提升**: 相比原始实现性能提升 **10-50x**
- ✅ **内存优化**: 减少中间张量分配
- ✅ **内联计算**: 将Linear、Sigmoid等操作内联到CUDA kernel中

**编译步骤**:
```bash
cd deploy/sparsebox_plugin
./build.sh
# 或使用 make
make -j8
```

**输出**: `lib/SparseBox3DKeyPointsPlugin.so`

**详细文档**: 参见 `deploy/sparsebox_plugin/docs/QUICKSTART.md`

### 2.5 验证编译结果

编译完成后，检查所有插件是否生成：

```bash
ls -lh deploy/dfa_plugin/lib/deformableAttentionAggr.so
ls -lh deploy/ln_plugin/lib/customLayerNorm.so
ls -lh deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so
```

预期输出：
```
-rwxr-xr-x 1 user user 200K Dec 20 10:00 deploy/dfa_plugin/lib/deformableAttentionAggr.so
-rwxr-xr-x 1 user user 50K  Dec 20 10:00 deploy/ln_plugin/lib/customLayerNorm.so
-rwxr-xr-x 1 user user 150K Dec 20 10:00 deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so
```

## STEP3. BUILD Sparse4D Engine

### 3.1 设置环境并构建引擎

```bash
cd deploy
. tools/set_env.sh
bash build_sparse4d_engine.sh fp32/fp16/int8
```

**支持的精度**:
- `fp32`: 全精度FP32
- `fp16`: 半精度FP16（推荐，性能最佳）
- `int8`: INT8量化（实验性）
- `mixed`: 混合精度（Backbone FP16, Head FP32）

**构建脚本会自动**:
1. 检查所有3个插件是否存在
2. 加载插件库到TensorRT
3. 构建Backbone引擎
4. 构建Head1引擎（第一帧）
5. 构建Head2引擎（后续帧）

### 3.2 构建日志

构建过程中会显示插件加载信息：

```bash
[INFO] DeformableAttentionAggrPlugin enabled: /path/to/deformableAttentionAggr.so
[INFO] LayerNormPlugin enabled: /path/to/customLayerNorm.so
[INFO] SparseBox3DKeyPointsPlugin enabled: /path/to/SparseBox3DKeyPointsPlugin.so
```

### 3.3 输出文件

构建完成后，引擎文件保存在 `deploy/engine/`:

>deploy/engine  
>├── build_backbone.engine  
>├── build_head1st.engine  
>├── build_head2nd.engine  
>├── build_backbone.log  
>├── build_head1.log  
>├── build_head2.log  
>├── buildLayerInfo_backbone.json  
>├── buildLayerInfo_head1.json  
>├── buildLayerInfo_head2.json  
>├── buildOutput_backbone.json  
>├── buildOutput_head1.json  
>├── buildOutput_head2.json  
>├── buildProfile_backbone.json  
>├── buildProfile_head1.json  
>└── buildProfile_head2.json  

## 插件优化总结

| 插件 | 功能 | 核心优化点 | 性能提升 | 关键问题解决 |
|------|------|------------|----------|--------------|
| **DFA Plugin** | 可变形注意力聚合 | Scatter→Gather模式重构，消除atomicAdd；FP16混合精度优化 | **9.6x** (单层6.65ms→0.69ms) | 原子锁冲突严重，FP16精度丢失 |
| **LN Plugin** | 自定义LayerNorm | Warp shuffle优化，替代低效融合 | **2.6x** (Head1: 26.7ms→10.4ms) | Head1异常延迟，Myelin融合低效 |
| **SparseBox Plugin** | 3D关键点生成 | Per-Point细粒度并行，内联计算，消除ForeignNode | **12x** (单层0.59ms→0.05ms) | ForeignNode导致745ms延迟 |

## 核心优化点详解

### DFA Plugin 优化

#### 问题1: Scatter模式的原子锁冲突
**原始实现**:
- 采用Scatter（分散写）模式：每个线程负责一个采样点，通过`atomicAdd`累加到输出
- **瓶颈**: 每个输出位置对应约192个采样点，导致严重的原子锁冲突
- **耗时**: 单层约6.65ms，占总耗时74%

**优化方案**:
- 重构为**Gather（聚合读）模式**：每个线程负责一个输出位置，循环读取所有采样点并累加
- **优势**: 彻底消除`atomicAdd`，显著减少全局内存写入
- **结果**: 单层耗时从6.65ms降低至0.69ms，**提升9.6倍**

#### 问题2: FP16精度丢失
**问题**: 纯FP16模式下`sampling_loc`精度不足导致采样漂移，Cosine Similarity仅0.57

**优化方案**:
- **混合精度Gather策略**:
  - Input/Output: FP16（内存带宽减半）
  - Sampling Loc / Weights: FP32（保证采样精度）
  - Accumulation: FP32（保证累加精度）
- **结果**: 精度完全恢复（Cos Sim > 0.999），性能仅损失<0.2ms

#### 问题3: FP16 atomicAdd性能问题
**问题**: CUDA的`atomicAdd`不支持`__half`类型，使用`atomicCAS`实现非常慢（比FP32慢10-100倍）

**优化方案**:
- 使用FP32临时缓冲区进行累加（使用高效的`atomicAdd`）
- 最后批量转换为FP16输出
- **结果**: FP16版本比FP32版本快**1.5-2.0x**（内存带宽减半 + 高效atomic操作）

### SparseBox Plugin 优化

#### 问题1: ForeignNode导致严重性能损失
**原始问题**:
- PyTorch导出时分解为20+个基础操作（Gather、MatMul、Transpose等）
- TensorRT无法优化这些操作，回退到ForeignNode
- **性能影响**: 单个ForeignNode节点耗时高达745.50ms（11.9%）

**优化方案**:
- **单Kernel融合**: 所有操作在一个CUDA kernel中完成
- **直接内存访问**: 使用指针直接访问，避免Gather操作
- **内联计算**: Linear、旋转矩阵乘法等内联到kernel中
- **结果**: 消除ForeignNode，单层耗时从0.59ms降低至0.05ms，**提升12倍**

#### 问题2: Per-Anchor并行度不足
**原始实现**:
- 采用Per-Anchor并行，Grid Size仅900
- GPU利用率低，无法充分利用GPU并行能力

**优化方案**:
- 采用**Per-Point细粒度并行**（Grid Size = 11700）
- 每个线程处理一个关键点，大幅提升并行度
- **结果**: GPU利用率显著提升，性能提升12倍

### LN Plugin 优化

#### 问题: Head1异常延迟
**问题分析**:
- Head1推理耗时异常高（26.7ms），远高于Head2（8.7ms）
- Profiling显示大量时间消耗在Myelin融合节点（`ForeignNode...MatMul`），单个节点耗时高达5.67ms
- **原因**: TensorRT针对FP16的LayerNorm + MatMul融合策略在某些特定图结构下效率低下

**优化方案**:
- 使用**CustomLayerNormalizationPlugin**替换原生的`LayerNormalization`算子
- **实现**: 手写高效CUDA Kernel（FP16/FP32），利用Warp Shuffle进行快速规约
- **结果**: Head1推理耗时从26.7ms降低至10.4ms，**提升2.6倍**

## 部署优化关键点

### 1. 编译模式优化

**问题**: C++端集成Engine后，Head推理耗时显示为~50ms，与`trtexec`测得的~14ms严重不符

**原因**: 插件编译脚本默认开启`DEBUG=1`，导致：
- CUDA Kernel未经优化（`-O0`）
- 包含大量调试符号
- 严重拖慢GPU执行速度

**解决方案**: 修改编译脚本，强制**Release Mode** (`DEBUG=0`)

**结果**: C++端性能从47.32ms降低至9.85ms，与Engine理论性能对齐

### 2. 整体性能优化路径

| 优化阶段 | 关键改动 | 总耗时 (Mean) | 提升幅度 | 精度 (Cos Sim) |
|:--- |:--- |:--- |:--- |:--- |
| **基准 (Baseline)** | 原始FP16实现 | 54.03 ms | - | 0.9995 |
| **阶段一 (DFA Speed)** | Gather模式重构（消除原子锁） | 17.50 ms | **3.1x** | < 0.60 (严重失真) |
| **阶段二 (SparseBox)** | Per-Point细粒度并行 | 14.25 ms | **3.8x** | < 0.60 |
| **阶段三 (Accuracy)** | **混合精度策略** (FP16/32 Mixed) | **14.41 ms** | **3.7x** | **0.9995** (完全恢复) |
| **阶段四 (Head1)** | **Custom LayerNorm Plugin** | **10.40 ms (Head1)** | **2.6x (Head1)** | **0.999+** |
| **阶段五 (Deployment)** | **Release Mode Build** | **10.06 ms (C++ E2E)** | **5x (C++ vs Debug)** | - |

### 3. 最终性能指标

优化后的模型性能：
- **Head 1 (FP16 Optimized)**: ~10.4 ms
- **Head 2 (FP16 Optimized)**: ~8.8 ms  
- **Backbone (FP16)**: ~8.5 ms
- **端到端总耗时**: ~27.7 ms

**总体加速**: 从54.03ms优化至10.06ms，**整体加速5.4倍**

## 部署优化关键点

### 1. 编译模式优化（重要！）

**问题**: C++端集成Engine后，Head推理耗时显示为~50ms，与`trtexec`测得的~14ms严重不符

**原因分析**: 
- 插件编译脚本默认开启`DEBUG=1`
- 导致CUDA Kernel未经优化（`-O0`）
- 包含大量调试符号
- 严重拖慢GPU执行速度

**解决方案**: 
- 修改编译脚本，强制**Release Mode** (`DEBUG=0`)
- 确保所有插件都使用Release模式编译

**结果**: C++端性能从47.32ms降低至9.85ms，与Engine理论性能对齐

### 2. 整体性能优化路径

| 优化阶段 | 关键改动 | 总耗时 (Mean) | 提升幅度 | 精度 (Cos Sim) |
|:--- |:--- |:--- |:--- |:--- |
| **基准 (Baseline)** | 原始FP16实现 | 54.03 ms | - | 0.9995 |
| **阶段一 (DFA Speed)** | Gather模式重构（消除原子锁） | 17.50 ms | **3.1x** | < 0.60 (严重失真) |
| **阶段二 (SparseBox)** | Per-Point细粒度并行 | 14.25 ms | **3.8x** | < 0.60 |
| **阶段三 (Accuracy)** | **混合精度策略** (FP16/32 Mixed) | **14.41 ms** | **3.7x** | **0.9995** (完全恢复) |
| **阶段四 (Head1)** | **Custom LayerNorm Plugin** | **10.40 ms (Head1)** | **2.6x (Head1)** | **0.999+** |
| **阶段五 (Deployment)** | **Release Mode Build** | **10.06 ms (C++ E2E)** | **5x (C++ vs Debug)** | - |

### 3. 最终性能指标

优化后的模型性能：
- **Head 1 (FP16 Optimized)**: ~10.4 ms
- **Head 2 (FP16 Optimized)**: ~8.8 ms  
- **Backbone (FP16)**: ~8.5 ms
- **端到端总耗时**: ~27.7 ms

**总体加速**: 从54.03ms优化至10.06ms，**整体加速5.4倍**

### 4. 关键优化技术总结

#### DFA Plugin
- ✅ **Scatter→Gather重构**: 消除atomicAdd，单层从6.65ms→0.69ms（**9.6x**）
- ✅ **混合精度优化**: FP16 Value + FP32 Loc/Weights，精度完全恢复
- ✅ **FP16性能优化**: 使用FP32临时缓冲区，FP16版本比FP32快1.5-2.0x

#### SparseBox Plugin  
- ✅ **消除ForeignNode**: 单Kernel融合，避免20+个基础操作
- ✅ **Per-Point并行**: 从Per-Anchor改为细粒度并行，单层从0.59ms→0.05ms（**12x**）
- ✅ **内联计算**: Linear、旋转矩阵等内联到kernel，减少内存访问

#### LN Plugin
- ✅ **解决Head1延迟**: 替换低效的Myelin融合，Head1从26.7ms→10.4ms（**2.6x**）
- ✅ **Warp级优化**: 使用warp shuffle进行高效reduction

## 常见问题

### Q1: 编译时提示 "nvcc: Command not found"

**解决**: 确保已正确设置环境变量，或使用各插件的 `build.sh` 脚本（会自动检查环境）

### Q2: TensorRT构建时找不到插件

**解决**: 
1. 检查 `tools/set_env.sh` 中的插件路径是否正确
2. 确保插件已成功编译
3. 检查构建脚本中的 `--plugins=` 参数

### Q3: FP16精度问题

**解决**: 
- DFA Plugin使用FP32进行中间累加，保证精度
- 如仍有精度问题，可尝试使用FP32版本

### Q4: 性能未达到预期

**解决**:
1. 确保所有3个插件都已加载
2. 检查构建日志，确认没有ForeignNode警告
3. 使用FP16精度以获得最佳性能

## 参考文档

- **DFA Plugin优化**: `deploy/dfa_plugin/FP16_PERFORMANCE_OPTIMIZATION.md`
- **SparseBox Plugin**: `deploy/sparsebox_plugin/docs/QUICKSTART.md`
- **性能优化报告**: `deploy/docs/PERFORMANCE_OPTIMIZATION_REPORT.md`
- **混合精度指南**: `deploy/MIXED_PRECISION_OPTIMIZATION.md`
