# SparseBox3DKeyPointsPlugin 快速开始指南

## 一、什么是"在 ONNX 导出时替换为 Plugin"？

### 问题

当直接导出 `SparseBox3DKeyPointsGenerator` 时，PyTorch 会将其分解为约 20+ 个基础 ONNX 操作（Gather、Transpose、MatMul 等），这些操作在 TensorRT 中无法优化，导致 **ForeignNode**，性能极差。

### 解决方案

**在 ONNX 导出时，使用自定义操作符替换原始实现**，这样：
1. ONNX 文件中会包含一个自定义节点 `custom::SparseBox3DKeyPointsPlugin`
2. TensorRT 构建引擎时，会识别这个节点并替换为高效的 Plugin
3. 避免了 ForeignNode，性能提升 10-50x

## 二、实现原理

```
┌─────────────────────────────────────────────────────────┐
│ 步骤 1: ONNX 导出                                        │
├─────────────────────────────────────────────────────────┤
│ Python 代码                                              │
│   ↓                                                      │
│ torch.onnx.export()                                      │
│   ↓                                                      │
│ 调用 symbolic() 函数                                     │
│   ↓                                                      │
│ 创建 ONNX 节点: custom::SparseBox3DKeyPointsPlugin      │
│   ↓                                                      │
│ 保存到 .onnx 文件                                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 步骤 2: TensorRT 构建                                    │
├─────────────────────────────────────────────────────────┤
│ 加载 .onnx 文件                                          │
│   ↓                                                      │
│ 解析 ONNX 图                                             │
│   ↓                                                      │
│ 发现 custom::SparseBox3DKeyPointsPlugin 节点           │
│   ↓                                                      │
│ 查找注册的 Plugin Creator                                │
│   ↓                                                      │
│ 创建 Plugin 实例                                         │
│   ↓                                                      │
│ 替换 ONNX 节点为 Plugin                                  │
│   ↓                                                      │
│ 优化并构建引擎                                           │
└─────────────────────────────────────────────────────────┘
```

## 三、如何使用

### 步骤 1: 确保文件已创建

已创建以下文件：
- ✅ `modules/ops/sparse_box3d_keypoints.py` - 自定义操作符
- ✅ `modules/head/sparse4d_blocks/sparse3d_keypoints_plugin.py` - 包装器
- ✅ `deploy/sparsebox_plugin/replace_kps_generator.py` - 替换函数

### 步骤 2: 修改导出脚本

在 `deploy/export/export_head_onnx.py` 中添加：

```python
# 在文件开头导入
from deploy.sparsebox_plugin.replace_kps_generator import (
    replace_kps_generator_with_plugin
)

# 在导出前调用（在 torch.onnx.export 之前）
if __name__ == "__main__":
    # ... 加载模型 ...
    
    # 第一帧
    if not args.o2:
        first_frame_head = Sparse4DHead1st(copy.deepcopy(model))
        # 🔥 添加这一行：替换 kps_generator
        replace_kps_generator_with_plugin(first_frame_head.model.head)
        
        torch.onnx.export(...)
    
    # 第二帧
    head = Sparse4DHead2nd(copy.deepcopy(model))
    # 🔥 添加这一行：替换 kps_generator
    replace_kps_generator_with_plugin(head.model.head)
    
    torch.onnx.export(...)
```

### 步骤 3: 导出 ONNX

运行导出脚本：

```bash
python deploy/export/export_head_onnx.py \
    --cfg dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py \
    --ckpt ckpt/sparse4dv3_r50.pth \
    --save_onnx1 deploy/onnx/sparse4dhead1st.onnx \
    --save_onnx2 deploy/onnx/sparse4dhead2nd.onnx \
    --fp16
```

**预期输出**：
```
Layer 0: Replaced kps_generator (num_pts=13, num_learnable_pts=6)
Layer 1: Replaced kps_generator (num_pts=13, num_learnable_pts=6)
...
Total replaced: 4 kps_generator(s)
```

### 步骤 4: 验证 ONNX

检查 ONNX 文件是否包含自定义节点：

```python
import onnx

model = onnx.load("deploy/onnx/sparse4dhead2nd.onnx")
custom_nodes = [
    node for node in model.graph.node
    if "SparseBox3DKeyPointsPlugin" in node.op_type
]
print(f"Found {len(custom_nodes)} SparseBox3DKeyPointsPlugin nodes")
```

### 步骤 5: 构建 TensorRT 引擎

使用 trtexec 构建引擎，**必须加载 Plugin 库**：

```bash
trtexec \
    --onnx=deploy/onnx/sparse4dhead2nd.onnx \
    --plugins=sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so \
    --plugins=dfa_plugin/lib/deformableAttentionAggr.so \
    --plugins=ln_plugin/lib/customLayerNorm.so \
    --saveEngine=deploy/engine/sparse4dhead2nd.engine \
    --fp16
```

**关键点**：
- ✅ 必须使用 `--plugins=` 参数加载 Plugin 库
- ✅ Plugin 库路径必须正确
- ✅ Plugin 名称必须匹配（`SparseBox3DKeyPointsPlugin`）

### 步骤 6: 验证构建结果

查看构建日志，应该看到：

```
[I] Found plugin: SparseBox3DKeyPointsPlugin
[I] Replacing node: /kps_generator_4/... with SparseBox3DKeyPointsPlugin
```

**不应该看到**：
```
[W] Unsupported ONNX node: Gather_8
[W] Falling back to ForeignNode
```

## 四、常见问题

### Q1: 导出时提示 "Unsupported operator"

**原因**: `symbolic()` 函数实现有问题

**解决**: 
1. 检查 `modules/ops/sparse_box3d_keypoints.py` 中的 `symbolic()` 函数
2. 确保所有输入都正确传递
3. 参考 `modules/ops/deformable_aggregation.py` 的实现

### Q2: TensorRT 构建时找不到 Plugin

**原因**: Plugin 库未加载或名称不匹配

**解决**:
1. 检查 Plugin 库路径是否正确
2. 确保使用 `--plugins=` 参数
3. 检查 Plugin 名称是否匹配（`SparseBox3DKeyPointsPlugin`）

### Q3: 运行时精度不一致

**原因**: CUDA kernel 实现与 Python 代码不一致

**解决**:
1. 检查 `SparseBox3DKeyPointsKernel.cu` 的实现
2. 验证权重是否正确提取和传递
3. 检查数据类型（FP16 vs FP32）

### Q4: 替换后导出失败

**原因**: 包装器实现有问题

**解决**:
1. 检查 `sparse3d_keypoints_plugin.py` 中的 `forward()` 函数
2. 确保所有参数都正确注册为 buffer
3. 验证 `fix_scale`、`fc_weight`、`fc_bias` 的形状

## 五、完整示例

参考 `deploy/dfa_plugin/` 目录，它展示了完整的自定义操作符和 Plugin 集成流程。

## 六、总结

**核心步骤**：
1. ✅ 创建自定义操作符（`sparse_box3d_keypoints.py`）
2. ✅ 创建包装器（`sparse3d_keypoints_plugin.py`）
3. ✅ 在导出前替换（`replace_kps_generator.py`）
4. ✅ 导出 ONNX
5. ✅ 构建 TensorRT 引擎（加载 Plugin 库）

**预期效果**：
- ✅ 消除 ForeignNode
- ✅ 性能提升 10-50x
- ✅ 减少内存使用

