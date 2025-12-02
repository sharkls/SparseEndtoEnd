# 混合精度优化方案

## 一、优化目标

在FP16精度下，保持backbone、head1st、head2nd的输入和输出为FP16，但head1st、head2nd内部关键部分（关键点生成和DFA）使用FP32精度，以提高最终anchor输出的精度。

## 二、实现方案

### 2.1 SparseBox插件（关键点生成）

**已实现**：
- ✅ 插件内部所有计算使用FP32精度
- ✅ FP16输入时，输出FP32关键点（`getOutputDataType`返回`kFLOAT`）
- ✅ Kernel内部使用FP32进行旋转矩阵计算和中心点累加

**关键代码**：
```cpp
// SparseBox3DKeyPointsPlugin.cpp:223
if (inputTypes[0] == DataType::kHALF)
{
    return DataType::kFLOAT;  // FP16 输入时，输出 FP32
}
```

### 2.2 DFA插件（可变形注意力聚合）

**新增混合精度支持**：
- ✅ 添加混合精度kernel：`thomas_deformable_aggregation_kernel_mixed`
- ✅ 添加混合精度前向传播函数：`thomas_deform_attn_cuda_forward_mixed`
- ✅ 修改`enqueue`函数，自动检测混合精度输入
- ✅ 修改`supportsFormatCombination`，支持FP16 value + FP32 keypoints

**关键特性**：
- **输入**：FP16 value（特征值）+ FP32 keypoints（关键点位置和注意力权重）
- **内部计算**：使用FP32临时缓冲区进行累加（atomicAdd更快）
- **输出**：FP16（批量转换，开销小）

**关键代码**：
```cpp
// deformableAttentionAggrPlugin.cpp:195-200
bool isMixedPrecision = (dataType == nvinfer1::DataType::kHALF) && 
                        (samplingLocType == nvinfer1::DataType::kFLOAT) && 
                        (attnWeightType == nvinfer1::DataType::kFLOAT);

if (isMixedPrecision)
{
    // 调用混合精度版本
    rc = thomas_deform_attn_cuda_forward_mixed(...);
}
```

## 三、数据流

```
输入（FP16）
  ↓
Backbone（FP16内部计算，FP16输出）
  ↓
Head1st/Head2nd
  ├─ 输入：FP16
  ├─ SparseBox插件：FP16输入 → FP32关键点输出
  ├─ DFA插件：FP16 value + FP32 keypoints → FP16输出
  ├─ Refinement模块：FP16内部计算
  └─ 输出：FP16
```

## 四、性能优势

1. **精度提升**：
   - 关键点使用FP32精度，减少精度损失
   - DFA使用FP32关键点，提高特征聚合精度
   - 最终anchor输出精度显著提升

2. **性能保持**：
   - 输入输出仍为FP16，内存带宽减半
   - 只有关键部分使用FP32，计算开销增加有限
   - DFA使用FP32临时缓冲区，atomicAdd更快

3. **平衡点**：
   - 相比全FP32：性能更好（内存带宽减半）
   - 相比全FP16：精度更高（关键点精度提升）

## 五、使用方法

### 5.1 构建引擎

使用FP16模式构建引擎（关键点会自动保持FP32）：
```bash
./deploy/build_sparse4d_engine.sh fp16
```

### 5.2 验证

运行验证脚本，检查混合精度是否生效：
```bash
python deploy/val/validate_pytorch_vs_engine.py \
    --sample_idx 0 \
    --num_samples 1 \
    --use_val_dataset \
    --analyze_plugin_error \
    --capture_keypoints \
    --analyze_error_patterns
```

## 六、技术细节

### 6.1 SparseBox插件

- **输入类型检测**：`inputDesc[0].type == DataType::kHALF`
- **输出类型**：`getOutputDataType`返回`DataType::kFLOAT`
- **Kernel实现**：所有计算使用FP32，输出FP32

### 6.2 DFA插件

- **混合精度检测**：检查value、samplingLoc、attnWeight的类型
- **Kernel实现**：
  - 关键点直接使用FP32（无需转换）
  - 特征值使用FP16（减少内存带宽）
  - 累加使用FP32临时缓冲区（atomicAdd更快）
- **输出转换**：批量转换为FP16（向量化，开销小）

### 6.3 TensorRT类型转换

TensorRT会自动处理类型转换：
- SparseBox输出FP32 → DFA输入FP32（无需转换）
- DFA输出FP16 → 后续模块输入FP16（无需转换）

## 七、预期效果

1. **精度提升**：
   - Anchor位置误差：从~2.0降低到~0.5-1.0
   - mAP差异：从~82%降低到~20-30%

2. **性能影响**：
   - 推理速度：相比全FP32提升~1.5-2x
   - 内存占用：相比全FP32减少~50%
   - 相比全FP16：性能损失<10%

## 八、注意事项

1. **引擎构建**：
   - 必须使用FP16模式构建引擎（`--fp16`）
   - TensorRT会自动识别插件的输出类型

2. **插件加载**：
   - 确保SparseBox和DFA插件都已正确编译和加载
   - 检查插件版本是否支持混合精度

3. **验证**：
   - 运行验证脚本，确认关键点输出为FP32
   - 检查DFA是否使用混合精度模式

## 九、段错误修复（2025-11-28）

### 9.1 问题描述
在构建TensorRT引擎时，DFA插件发生段错误（Segmentation fault），发生在TensorRT测试不同格式组合时。

### 9.2 根本原因
1. **`supportsFormatCombination`函数**：
   - 直接访问`inOut[pos]`和`inOut[0]`，没有检查指针有效性和边界
   - 当`pos >= nbInputs`时访问`inOut[0]`，可能越界

2. **`getOutputDimensions`函数**：
   - 访问`inputs[0].d[0]`和`inputs[3].d[1]`时，没有检查维度表达式是否有效

3. **`getWorkspaceSize`函数**：
   - 访问`inputs[0].dims.d[0]`和`inputs[3].dims.d[1]`时，没有检查dims结构是否有效

4. **`enqueue`函数**：
   - 访问`inputDesc`数组时，没有检查所有指针是否有效

### 9.3 修复方案
1. **添加指针有效性检查**：
   - 在所有函数中添加`nullptr`检查
   - 在访问数组元素前检查索引范围

2. **添加维度有效性检查**：
   - 检查`nbDims`是否足够
   - 检查`d`数组指针是否有效

3. **修复Makefile**：
   - 修复无限递归问题（`all`目标）
   - 修复`BUILD_PATH`环境变量问题

### 9.4 修复后的代码
- ✅ `supportsFormatCombination`：添加了`inOut`指针检查和`pos`边界检查
- ✅ `getOutputDimensions`：添加了`inputs`指针检查和维度表达式有效性检查
- ✅ `getWorkspaceSize`：添加了`inputs`和`outputs`指针检查，以及dims结构有效性检查
- ✅ `enqueue`：添加了所有输入输出指针的有效性检查

### 9.5 修复的函数列表
1. ✅ `supportsFormatCombination`：添加了`inOut`指针检查和`pos`边界检查
2. ✅ `getOutputDimensions`：添加了`inputs`指针检查和维度表达式有效性检查
3. ✅ `getWorkspaceSize`：添加了`inputs`和`outputs`指针检查，以及dims结构有效性检查
4. ✅ `enqueue`：添加了所有输入输出指针的有效性检查
5. ✅ `getOutputDataType`：添加了`inputTypes`指针和`nbInputs`检查

### 9.6 验证步骤
1. **重新编译插件**：
   ```bash
   cd /share/Code/Sparse4dE2E/deploy
   source tools/set_env.sh
   cd dfa_plugin
   make clean && make
   ```

2. **重新构建引擎**（重要：插件代码更新后必须重新构建引擎）：
   ```bash
   cd /share/Code/Sparse4dE2E/deploy
   ./build_sparse4d_engine.sh fp16
   ```

3. **验证引擎**：
   ```bash
   python deploy/val/validate_pytorch_vs_engine.py \
       --sample_idx 0 \
       --num_samples 1 \
       --use_val_dataset \
       --analyze_plugin_error \
       --capture_keypoints \
       --analyze_error_patterns
   ```

### 9.7 使用GDB调试段错误

如果构建引擎时发生段错误，可以使用GDB获取完整的堆栈跟踪信息：

#### 方法一：使用调试脚本（推荐）

```bash
cd /share/Code/Sparse4dE2E/deploy

# 调试head1引擎构建
./debug_build_engine.sh fp16 head1

# 调试head2引擎构建
./debug_build_engine.sh fp16 head2
```

脚本会自动：
- 使用GDB运行trtexec
- 捕获段错误
- 打印完整的堆栈跟踪
- 保存调试信息到日志文件

#### 方法二：手动使用GDB

```bash
cd /share/Code/Sparse4dE2E/deploy
source tools/set_env.sh

# 使用GDB运行trtexec
gdb --args ${ENV_TensorRT_BIN}/trtexec \
    --onnx=onnx/sparse4dhead1st.onnx \
    --plugins=dfa_plugin/lib/deformableAttentionAggr.so \
    --plugins=ln_plugin/lib/customLayerNorm.so \
    --plugins=sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so \
    --memPoolSize=workspace:2048 \
    --saveEngine=engine/sparse4dhead1st.engine \
    --fp16 \
    --verbose

# 在GDB中执行：
(gdb) run
# 等待段错误发生
(gdb) backtrace          # 打印堆栈跟踪
(gdb) backtrace full     # 打印详细堆栈信息（包括局部变量）
(gdb) info registers      # 打印寄存器信息
(gdb) thread apply all backtrace  # 打印所有线程的堆栈
```

#### 方法三：使用core dump

```bash
# 启用core dump
ulimit -c unlimited

# 运行调试脚本（会自动生成core dump）
./debug_with_coredump.sh fp16 head1

# 分析core dump文件
gdb ${ENV_TensorRT_BIN}/trtexec core.<pid>
(gdb) thread apply all backtrace full
```

#### 关键GDB命令

- `backtrace` 或 `bt`: 打印当前堆栈跟踪
- `backtrace full` 或 `bt full`: 打印详细堆栈信息（包括局部变量）
- `frame <n>`: 切换到第n帧
- `info locals`: 打印当前帧的局部变量
- `info args`: 打印当前帧的函数参数
- `print <variable>`: 打印变量值
- `info registers`: 打印所有寄存器
- `thread apply all backtrace`: 打印所有线程的堆栈

#### 设置断点（可选）

```bash
(gdb) break DeformableAttentionAggrPlugin::supportsFormatCombination
(gdb) break DeformableAttentionAggrPlugin::getOutputDimensions
(gdb) break DeformableAttentionAggrPlugin::getWorkspaceSize
(gdb) break DeformableAttentionAggrPlugin::getOutputDataType
(gdb) break DeformableAttentionAggrPlugin::enqueue
```

详细调试指南请参考：`deploy/GDB_DEBUG_GUIDE.md`

### 9.8 最新修复（2025-11-28）

**已完成的优化**：

1. **`supportsFormatCombination`函数增强**：
   - ✅ 添加了更严格的边界检查（`pos`、`nbInputs`、`nbOutputs`）
   - ✅ 改进了格式检查逻辑，提前返回无效格式
   - ✅ 优化了类型检查顺序，先检查位置0（value）的类型
   - ✅ 添加了更宽松的fallback策略，避免在异常情况下崩溃
   - ✅ 改进了代码结构，使逻辑更清晰

2. **关键改进点**：
   ```cpp
   // 1. 提前检查格式
   if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) {
       return false;
   }
   
   // 2. 分离类型检查逻辑
   nvinfer1::DataType valueType = inOut[0].type;
   if (valueType == nvinfer1::DataType::kFLOAT) {
       // FP32模式
   } else if (valueType == nvinfer1::DataType::kHALF) {
       // FP16/混合精度模式
   } else {
       // 更宽松的fallback策略
   }
   ```

3. **测试状态**：
   - ✅ 插件已重新编译
   - ✅ 构建过程不再出现段错误
   - ✅ 引擎构建可以正常进行（可能需要较长时间）

### 9.9 如果问题仍然存在
如果段错误仍然存在，建议：
1. 使用GDB调试获取完整的堆栈跟踪（见9.7节）
2. 检查构建日志中的`[DFA-PLUGIN-ERROR]`消息
3. 确认插件文件在正确位置：`dfa_plugin/lib/deformableAttentionAggr.so`
4. 检查引擎是否是用最新版本的插件构建的
5. 检查插件是否用调试信息编译（`DEBUG=1 make`）
6. 检查TensorRT版本兼容性（当前使用TensorRT 8.5.1.7）

## 十、后续优化

1. **进一步优化**：
   - 考虑Refinement模块也使用FP32关键点
   - 优化关键点范围，减少FP32表示范围

2. **性能调优**：
   - 优化FP32临时缓冲区的使用
   - 减少类型转换开销

3. **精度分析**：
   - 详细分析关键点精度对最终结果的影响
   - 找出精度瓶颈，进一步优化

