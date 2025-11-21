# SparseBox3DKeyPointsPlugin ONNX 导出指南

## 一、为什么需要在 ONNX 导出时替换？

### 问题背景

当直接导出 `SparseBox3DKeyPointsGenerator` 时，PyTorch 会将其分解为多个基础 ONNX 操作：
- `Gather` (索引操作)
- `Unsqueeze` (维度扩展)
- `Exp` (指数运算)
- `MatMul` (矩阵乘法)
- `Reshape` (形状变换)
- `Sigmoid` (激活函数)
- `Concat` (拼接)
- `Transpose` (转置)
- ... 等约 20+ 个操作

这些操作在 TensorRT 中无法优化，导致 **ForeignNode**，性能极差（~745ms，占 11.9% 时间）。

### 解决方案

使用 `torch.autograd.Function` 创建自定义操作符，在 ONNX 导出时注册为自定义节点，然后在 TensorRT 构建时替换为 Plugin。

## 二、实现步骤

### 步骤 1: 创建自定义 Function

创建文件 `modules/ops/sparse_box3d_keypoints.py`:

```python
import torch
from torch.autograd.function import Function, once_differentiable
import numpy as np

class SparseBox3DKeyPointsFunction(Function):
    """
    自定义操作符：将 SparseBox3DKeyPointsGenerator 的操作融合为单个 ONNX 节点
    """
    
    @staticmethod
    def symbolic(
        g,
        anchor,                    # [B, N, 11]
        instance_feature,          # [B, N, embed_dims] 或 None
        embed_dims,                # int: 特征维度
        num_pts,                   # int: 总点数
        num_learnable_pts,         # int: 可学习点数
        fix_scale,                 # Tensor: [num_fix_pts * 3] 固定点缩放
        fc_weight=None,            # Tensor: [num_learnable_pts*3, embed_dims] 或 None
        fc_bias=None,              # Tensor: [num_learnable_pts*3] 或 None
    ):
        """
        symbolic 函数：定义 ONNX 导出时的操作符
        
        这个函数在 ONNX 导出时被调用，创建一个自定义 ONNX 节点
        TensorRT 在构建引擎时会识别这个节点并替换为 Plugin
        """
        # 将参数转换为 ONNX 常量
        embed_dims_attr = g.op("Constant", value_t=torch.tensor([embed_dims], dtype=torch.int64))
        num_pts_attr = g.op("Constant", value_t=torch.tensor([num_pts], dtype=torch.int64))
        num_learnable_pts_attr = g.op("Constant", value_t=torch.tensor([num_learnable_pts], dtype=torch.int64))
        fix_scale_attr = g.op("Constant", value_t=fix_scale)
        
        inputs = [anchor, embed_dims_attr, num_pts_attr, num_learnable_pts_attr, fix_scale_attr]
        
        # 如果有可学习点，添加权重和偏置
        if fc_weight is not None and fc_bias is not None:
            fc_weight_attr = g.op("Constant", value_t=fc_weight)
            fc_bias_attr = g.op("Constant", value_t=fc_bias)
            inputs.extend([fc_weight_attr, fc_bias_attr])
            if instance_feature is not None:
                inputs.insert(1, instance_feature)  # 插入到 anchor 之后
        elif instance_feature is not None:
            # 即使没有可学习点，也传递 instance_feature（虽然不会被使用）
            inputs.insert(1, instance_feature)
        
        # 创建自定义 ONNX 操作符
        # 注意：操作符名称必须与 TensorRT Plugin 的名称匹配
        return g.op(
            "custom::SparseBox3DKeyPointsPlugin",
            *inputs,
            embed_dims_i=embed_dims,
            num_pts_i=num_pts,
            num_learnable_pts_i=num_learnable_pts,
        )
    
    @staticmethod
    def forward(
        ctx,
        anchor,
        instance_feature,
        embed_dims,
        num_pts,
        num_learnable_pts,
        fix_scale,
        fc_weight=None,
        fc_bias=None,
    ):
        """
        forward 函数：Python 运行时执行（用于验证和测试）
        
        注意：在 ONNX 导出时，这个函数不会被调用
        但在 Python 推理时，会使用原始的 Python 实现作为 fallback
        """
        # 这里可以使用原始的 Python 实现作为 fallback
        # 或者调用 CUDA 扩展（如果有的话）
        from modules.head.sparse4d_blocks.sparse3d_embedding import SparseBox3DKeyPointsGenerator
        
        # 创建临时生成器（仅用于 forward）
        # 注意：这只是 fallback，实际部署时应该使用 Plugin
        bs, num_anchor = anchor.shape[:2]
        
        # 简化的 Python 实现（仅用于验证）
        # 实际应该使用完整的 SparseBox3DKeyPointsGenerator
        size = anchor[..., None, [3, 4, 5]].exp()  # W, L, H
        fix_scale_tensor = fix_scale.view(-1, 3)  # [num_fix_pts, 3]
        key_points = fix_scale_tensor[None, None] * size  # [B, N, num_fix_pts, 3]
        
        # 如果有可学习点
        if num_learnable_pts > 0 and instance_feature is not None and fc_weight is not None:
            learnable_scale = (
                torch.nn.functional.linear(instance_feature, fc_weight.t(), fc_bias)
                .reshape(bs, num_anchor, num_learnable_pts, 3)
                .sigmoid() - 0.5
            )
            key_points = torch.cat([key_points, learnable_scale * size], dim=-2)
        
        # 旋转和平移（简化实现）
        # ... 省略完整实现 ...
        
        return key_points
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        backward 函数：反向传播（推理时不需要）
        """
        raise NotImplementedError("SparseBox3DKeyPointsFunction does not support backward")


def sparse_box3d_keypoints(
    anchor,
    instance_feature,
    embed_dims,
    num_pts,
    num_learnable_pts,
    fix_scale,
    fc_weight=None,
    fc_bias=None,
):
    """
    便捷函数：调用自定义操作符
    """
    return SparseBox3DKeyPointsFunction.apply(
        anchor,
        instance_feature,
        embed_dims,
        num_pts,
        num_learnable_pts,
        fix_scale,
        fc_weight,
        fc_bias,
    )
```

### 步骤 2: 创建包装 Module

创建文件 `modules/head/sparse4d_blocks/sparse3d_keypoints_plugin.py`:

```python
import torch
import torch.nn as nn
from modules.ops.sparse_box3d_keypoints import sparse_box3d_keypoints

class SparseBox3DKeyPointsPluginWrapper(nn.Module):
    """
    包装类：将 SparseBox3DKeyPointsGenerator 替换为 Plugin 版本
    """
    
    def __init__(self, kps_generator):
        """
        Args:
            kps_generator: 原始的 SparseBox3DKeyPointsGenerator 实例
        """
        super().__init__()
        self.embed_dims = kps_generator.embed_dims
        self.num_pts = kps_generator.num_pts
        self.num_learnable_pts = kps_generator.num_learnable_pts
        
        # 提取固定点缩放参数
        self.register_buffer(
            'fix_scale',
            kps_generator.fix_scale.data.flatten()  # [num_fix_pts * 3]
        )
        
        # 提取可学习点的权重和偏置
        if self.num_learnable_pts > 0 and hasattr(kps_generator, 'learnable_fc'):
            self.register_buffer(
                'fc_weight',
                kps_generator.learnable_fc.weight.data  # [num_learnable_pts*3, embed_dims]
            )
            self.register_buffer(
                'fc_bias',
                kps_generator.learnable_fc.bias.data  # [num_learnable_pts*3]
            )
        else:
            self.register_buffer('fc_weight', None)
            self.register_buffer('fc_bias', None)
    
    def forward(self, anchor, instance_feature=None):
        """
        Args:
            anchor: [B, N, 11] 锚点
            instance_feature: [B, N, embed_dims] 实例特征（可选）
        
        Returns:
            key_points: [B, N, num_pts, 3] 关键点
        """
        return sparse_box3d_keypoints(
            anchor,
            instance_feature,
            self.embed_dims,
            self.num_pts,
            self.num_learnable_pts,
            self.fix_scale,
            self.fc_weight,
            self.fc_bias,
        )
```

### 步骤 3: 修改导出代码

在 `deploy/export/export_head_onnx.py` 中，替换 `kps_generator`:

**方法 1: 使用辅助函数（推荐）**

```python
from deploy.sparsebox_plugin.replace_kps_generator import (
    replace_kps_generator_with_plugin
)

# 在导出前调用
if __name__ == "__main__":
    # ... 加载模型 ...
    
    first_frame_head = Sparse4DHead1st(copy.deepcopy(model))
    # 替换 kps_generator
    replace_kps_generator_with_plugin(first_frame_head.model.head)
    
    # 导出 ONNX
    torch.onnx.export(...)
```

**方法 2: 手动替换**

```python
from modules.head.sparse4d_blocks.sparse3d_keypoints_plugin import (
    SparseBox3DKeyPointsPluginWrapper
)

# 在导出前，替换所有的 kps_generator
def replace_kps_generator_with_plugin(head):
    """
    将 head 中所有的 kps_generator 替换为 Plugin 版本
    """
    for i, layer in enumerate(head.layers):
        if hasattr(layer, 'kps_generator') and layer.kps_generator is not None:
            original_kps_gen = layer.kps_generator
            # 创建 Plugin 包装器
            plugin_kps_gen = SparseBox3DKeyPointsPluginWrapper(original_kps_gen)
            # 替换
            layer.kps_generator = plugin_kps_gen
            print(f"Replaced kps_generator in layer {i} with Plugin wrapper")

# 在导出前调用
if __name__ == "__main__":
    # ... 加载模型 ...
    
    first_frame_head = Sparse4DHead1st(copy.deepcopy(model))
    # 替换 kps_generator
    replace_kps_generator_with_plugin(first_frame_head.model.head)
    
    # 导出 ONNX
    torch.onnx.export(...)
```

### 步骤 4: 注册 ONNX 操作符到 TensorRT

在 TensorRT 构建时，需要确保 Plugin 能够识别自定义操作符。

创建文件 `deploy/sparsebox_plugin/register_plugin.py`:

```python
"""
TensorRT Plugin 注册脚本

在构建 TensorRT 引擎时，需要确保自定义操作符能够被正确识别并替换为 Plugin。
"""
import tensorrt as trt

def register_sparsebox_plugin():
    """
    注册 SparseBox3DKeyPointsPlugin
    
    注意：这个函数在构建引擎时调用，不是在运行时
    """
    # 加载 Plugin 库
    trt.init_libnvinfer_plugins(trt.Logger(), "")
    
    # 获取 Plugin 注册表
    plugin_registry = trt.get_plugin_registry()
    
    # 查找 Plugin Creator
    plugin_creator = plugin_registry.get_plugin_creator(
        "SparseBox3DKeyPointsPlugin",
        "1"
    )
    
    if plugin_creator is None:
        raise RuntimeError(
            "SparseBox3DKeyPointsPlugin not found. "
            "Make sure the plugin library is loaded."
        )
    
    return plugin_creator
```

### 步骤 5: 使用 trtexec 构建引擎

在构建 TensorRT 引擎时，需要加载 Plugin 库：

```bash
trtexec \
    --onnx=sparse4dhead2nd.onnx \
    --plugins=sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so \
    --plugins=dfa_plugin/lib/deformableAttentionAggr.so \
    --plugins=ln_plugin/lib/customLayerNorm.so \
    --saveEngine=sparse4dhead2nd.engine \
    --fp16
```

TensorRT 会自动识别 ONNX 中的 `custom::SparseBox3DKeyPointsPlugin` 节点，并使用注册的 Plugin 替换它。

## 三、工作原理

### 3.1 ONNX 导出流程

```
Python 代码
  ↓
torch.onnx.export()
  ↓
调用 symbolic() 函数
  ↓
创建 ONNX 节点: custom::SparseBox3DKeyPointsPlugin
  ↓
保存到 .onnx 文件
```

### 3.2 TensorRT 构建流程

```
加载 .onnx 文件
  ↓
解析 ONNX 图
  ↓
发现 custom::SparseBox3DKeyPointsPlugin 节点
  ↓
查找注册的 Plugin Creator
  ↓
创建 Plugin 实例
  ↓
替换 ONNX 节点为 Plugin
  ↓
优化并构建引擎
```

### 3.3 关键点

1. **操作符名称匹配**: ONNX 中的 `custom::SparseBox3DKeyPointsPlugin` 必须与 Plugin 的 `getPluginName()` 返回的名称匹配
2. **参数传递**: 通过 `symbolic()` 函数中的属性（如 `embed_dims_i`）传递标量参数
3. **权重提取**: 在包装器中提取并注册为 buffer，确保在 ONNX 导出时作为常量

## 四、验证步骤

### 4.1 验证 ONNX 导出

```python
import onnx

# 加载导出的 ONNX
model = onnx.load("sparse4dhead2nd.onnx")

# 检查是否包含自定义节点
custom_nodes = [
    node for node in model.graph.node
    if node.op_type == "SparseBox3DKeyPointsPlugin"
]

print(f"Found {len(custom_nodes)} SparseBox3DKeyPointsPlugin nodes")
```

### 4.2 验证 TensorRT 构建

查看构建日志，应该看到：
```
[I] Found plugin: SparseBox3DKeyPointsPlugin
[I] Replacing node: /kps_generator_4/... with SparseBox3DKeyPointsPlugin
```

而不是：
```
[W] Unsupported ONNX node: Gather_8
[W] Falling back to ForeignNode
```

## 五、常见问题

### Q1: ONNX 导出失败，提示 "Unsupported operator"

**A**: 检查 `symbolic()` 函数的实现，确保所有输入都正确传递。

### Q2: TensorRT 构建时找不到 Plugin

**A**: 
1. 确保 Plugin 库路径正确
2. 确保使用 `--plugins=` 参数加载库
3. 检查 Plugin 名称是否匹配

### Q3: 运行时精度不一致

**A**: 
1. 检查 CUDA kernel 实现是否与 Python 代码一致
2. 验证权重是否正确提取和传递
3. 检查数据类型（FP16 vs FP32）

## 六、完整示例

参考 `deploy/dfa_plugin/` 目录中的实现，它展示了完整的自定义操作符和 Plugin 集成流程。

