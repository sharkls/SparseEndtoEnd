# SparseBox3DKeyPointsPlugin 使用指南

## 一、编译 Plugin

首先需要编译 Plugin 库：

```bash
cd deploy/sparsebox_plugin
make
```

编译成功后，会在 `lib/` 目录下生成 `SparseBox3DKeyPointsPlugin.so`。

## 二、导出 ONNX（已集成）

导出脚本已经集成了 Plugin 替换功能，直接运行即可：

```bash
cd deploy
python export_head_onnx.py \
    --cfg dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py \
    --ckpt ckpt/sparse4dv3_r50.pth \
    --save_onnx1 onnx/sparse4dhead1st.onnx \
    --save_onnx2 onnx/sparse4dhead2nd.onnx \
    --fp16
```

**预期输出**：
```
[INFO] Replacing kps_generator with SparseBox3DKeyPointsPlugin...
Layer 0: Replaced kps_generator (num_pts=13, num_learnable_pts=6)
Layer 1: Replaced kps_generator (num_pts=13, num_learnable_pts=6)
...
Total replaced: 4 kps_generator(s)
[INFO] Replaced 4 kps_generator(s) with Plugin version
```

## 三、构建 TensorRT Engine（已集成）

构建脚本已经集成了 Plugin 加载，直接运行即可：

```bash
cd deploy
./build_sparse4d_engine.sh fp16
```

或者使用优化版本：

```bash
cd deploy
./build_sparse4d_engine_optimized.sh fp16
```

**预期输出**：
```
[INFO] SparseBox3DKeyPointsPlugin enabled: sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so
STEP1: build sparse4dbackbone fp16 engine...
STEP2: build 1st frame sparse4dhead fp16 engine...
STEP3: build frame > 2 sparse4dhead fp16 engine...
success build fp16 engines.
```

## 四、验证 Plugin 是否生效

### 4.1 检查 ONNX 文件

```python
import onnx

model = onnx.load("deploy/onnx/sparse4dhead2nd.onnx")
custom_nodes = [
    node for node in model.graph.node
    if "SparseBox3DKeyPointsPlugin" in node.op_type
]
print(f"Found {len(custom_nodes)} SparseBox3DKeyPointsPlugin nodes")
```

### 4.2 检查构建日志

查看构建日志，应该看到：

```bash
grep -i "SparseBox3DKeyPointsPlugin" deploy/engine/build_head2.log
```

应该看到类似：
```
[I] Found plugin: SparseBox3DKeyPointsPlugin
[I] Replacing node: /kps_generator_4/... with SparseBox3DKeyPointsPlugin
```

**不应该看到**：
```
[W] Unsupported ONNX node: Gather_8
[W] Falling back to ForeignNode
```

### 4.3 检查性能提升

查看构建日志中的性能分析：

```bash
grep -A 2 "ForeignNode.*kps_generator" deploy/engine/build_head2.log
```

如果 Plugin 生效，应该**不再有** `ForeignNode[/kps_generator_4/Gather_8.../Transpose_8]` 这样的节点。

## 五、故障排除

### Q1: 导出时提示 "SparseBox3DKeyPointsPlugin not available"

**原因**: 导入路径问题

**解决**:
1. 确保在项目根目录运行导出脚本
2. 检查 `deploy/sparsebox_plugin/replace_kps_generator.py` 是否存在
3. 检查 Python 路径设置

### Q2: 构建时提示 "SparseBox3DKeyPointsPlugin not found"

**原因**: Plugin 库未编译或路径错误

**解决**:
1. 检查 `sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so` 是否存在
2. 如果不存在，运行 `cd deploy/sparsebox_plugin && make`
3. 检查 `deploy/tools/set_env.sh` 中的 `ENV_SPARSEBOX_PLUGIN` 路径

### Q3: 构建时仍然出现 ForeignNode

**原因**: ONNX 中未包含自定义节点

**解决**:
1. 检查 ONNX 导出日志，确认是否成功替换
2. 验证 ONNX 文件是否包含 `custom::SparseBox3DKeyPointsPlugin` 节点
3. 检查 `symbolic()` 函数实现是否正确

### Q4: 运行时精度不一致

**原因**: CUDA kernel 实现与 Python 代码不一致

**解决**:
1. 检查 `SparseBox3DKeyPointsKernel.cu` 的实现
2. 验证权重是否正确提取和传递
3. 检查数据类型（FP16 vs FP32）

## 六、完整工作流程

```bash
# 1. 编译 Plugin
cd deploy/sparsebox_plugin
make

# 2. 导出 ONNX（自动替换 kps_generator）
cd ../..
python deploy/export_head_onnx.py \
    --cfg dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py \
    --ckpt ckpt/sparse4dv3_r50.pth \
    --save_onnx1 deploy/onnx/sparse4dhead1st.onnx \
    --save_onnx2 deploy/onnx/sparse4dhead2nd.onnx \
    --fp16

# 3. 构建 TensorRT Engine（自动加载 Plugin）
cd deploy
./build_sparse4d_engine_optimized.sh fp16

# 4. 验证结果
grep -i "SparseBox3DKeyPointsPlugin" engine/build_head2.log
grep -i "ForeignNode.*kps_generator" engine/build_head2.log
```

## 七、预期效果

使用 Plugin 后，应该看到：

1. ✅ **ForeignNode 消除**: 不再有 `ForeignNode[/kps_generator_4/Gather_8.../Transpose_8]`
2. ✅ **性能提升**: kps_generator 相关操作从 ~745ms 降低到 <10ms（10-50x 加速）
3. ✅ **内存优化**: 减少中间张量分配
4. ✅ **构建时间**: 可能略有增加（因为需要解析自定义节点）

## 八、注意事项

1. **首次使用**: 确保先编译 Plugin 库
2. **路径检查**: 确保所有路径都是相对于 `deploy/` 目录
3. **版本兼容**: 确保 TensorRT 版本支持自定义 Plugin
4. **精度验证**: 首次使用时建议对比 Python 和 TensorRT 的输出精度

