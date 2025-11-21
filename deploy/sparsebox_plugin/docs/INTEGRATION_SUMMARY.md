# SparseBox3DKeyPointsPlugin 集成总结

## 一、已完成的修改

### 1.1 ONNX 导出代码修改

**文件**: `deploy/export_head_onnx.py`

**修改内容**:
1. 添加了 `sparsebox_plugin` 的导入和替换逻辑
2. 在导出第一帧和第二帧 head 前，自动替换 `kps_generator` 为 Plugin 版本
3. 添加了错误处理，如果 Plugin 不可用会给出警告但继续导出

**关键代码**:
```python
# 导入替换函数
from deploy.sparsebox_plugin.replace_kps_generator import (
    replace_kps_generator_with_plugin
)

# 在导出前替换
if SPARSEBOX_PLUGIN_AVAILABLE:
    replace_kps_generator_with_plugin(first_frame_head.model.head, verbose=True)
```

### 1.2 环境变量设置修改

**文件**: `deploy/tools/set_env.sh`

**修改内容**:
1. 添加了 `ENV_SPARSEBOX_PLUGIN` 环境变量
2. 在环境信息输出中显示 Plugin 路径

**关键代码**:
```bash
export ENV_SPARSEBOX_PLUGIN=sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so
```

### 1.3 Engine 构建脚本修改

**文件**: 
- `deploy/build_sparse4d_engine.sh`
- `deploy/build_sparse4d_engine_optimized.sh`

**修改内容**:
1. 在插件参数中添加了 `SparseBox3DKeyPointsPlugin` 的加载
2. 添加了 Plugin 存在性检查和提示信息

**关键代码**:
```bash
if [[ -n "${ENV_SPARSEBOX_PLUGIN}" && -f "${ENV_SPARSEBOX_PLUGIN}" ]]; then
    PLUGIN_ARGS="${PLUGIN_ARGS} --plugins=${ENV_SPARSEBOX_PLUGIN}"
    echo "[INFO] SparseBox3DKeyPointsPlugin enabled"
fi
```

### 1.4 Makefile 修改

**文件**: `deploy/sparsebox_plugin/Makefile`

**修改内容**:
1. 修正了目标文件名，与环境变量保持一致

**关键代码**:
```makefile
TARGET ?= lib/SparseBox3DKeyPointsPlugin.so
```

## 二、使用流程

### 2.1 完整工作流程

```bash
# 步骤 1: 编译 Plugin
cd deploy/sparsebox_plugin
make

# 步骤 2: 导出 ONNX（自动替换 kps_generator）
cd ../..
python deploy/export_head_onnx.py \
    --cfg dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py \
    --ckpt ckpt/sparse4dv3_r50.pth \
    --save_onnx1 deploy/onnx/sparse4dhead1st.onnx \
    --save_onnx2 deploy/onnx/sparse4dhead2nd.onnx \
    --fp16

# 步骤 3: 构建 TensorRT Engine（自动加载 Plugin）
cd deploy
./build_sparse4d_engine_optimized.sh fp16
```

### 2.2 验证步骤

```bash
# 1. 检查 ONNX 是否包含自定义节点
python -c "
import onnx
model = onnx.load('deploy/onnx/sparse4dhead2nd.onnx')
nodes = [n for n in model.graph.node if 'SparseBox3DKeyPointsPlugin' in n.op_type]
print(f'Found {len(nodes)} SparseBox3DKeyPointsPlugin nodes')
"

# 2. 检查构建日志
grep -i "SparseBox3DKeyPointsPlugin" deploy/engine/build_head2.log

# 3. 检查是否还有 ForeignNode
grep -i "ForeignNode.*kps_generator" deploy/engine/build_head2.log
```

## 三、预期效果

### 3.1 性能提升

- **ForeignNode 消除**: 不再有 `ForeignNode[/kps_generator_4/Gather_8.../Transpose_8]`
- **执行时间**: 从 ~745ms 降低到 <10ms（10-50x 加速）
- **内存使用**: 减少中间张量分配

### 3.2 日志输出示例

**导出时**:
```
[INFO] Replacing kps_generator with SparseBox3DKeyPointsPlugin...
Layer 0: Replaced kps_generator (num_pts=13, num_learnable_pts=6)
Layer 1: Replaced kps_generator (num_pts=13, num_learnable_pts=6)
Total replaced: 4 kps_generator(s)
[INFO] Replaced 4 kps_generator(s) with Plugin version
```

**构建时**:
```
[INFO] SparseBox3DKeyPointsPlugin enabled: sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so
[I] Found plugin: SparseBox3DKeyPointsPlugin
[I] Replacing node: /kps_generator_4/... with SparseBox3DKeyPointsPlugin
```

## 四、故障排除

### 4.1 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| 导出时提示 "not available" | 导入路径错误 | 确保在项目根目录运行 |
| 构建时提示 "not found" | Plugin 未编译 | 运行 `cd deploy/sparsebox_plugin && make` |
| 仍然有 ForeignNode | ONNX 未包含自定义节点 | 检查导出日志，确认替换成功 |
| 精度不一致 | CUDA kernel 实现问题 | 检查 kernel 实现和权重传递 |

### 4.2 检查清单

- [ ] Plugin 已编译 (`lib/SparseBox3DKeyPointsPlugin.so` 存在)
- [ ] 环境变量已设置 (`ENV_SPARSEBOX_PLUGIN`)
- [ ] 导出日志显示替换成功
- [ ] ONNX 文件包含自定义节点
- [ ] 构建日志显示 Plugin 加载成功
- [ ] 构建日志中没有 ForeignNode

## 五、文件清单

### 5.1 新增文件

- `modules/ops/sparse_box3d_keypoints.py` - 自定义操作符
- `modules/head/sparse4d_blocks/sparse3d_keypoints_plugin.py` - Plugin 包装器
- `deploy/sparsebox_plugin/replace_kps_generator.py` - 替换函数
- `deploy/sparsebox_plugin/QUICKSTART.md` - 快速开始指南
- `deploy/sparsebox_plugin/ONNX_EXPORT_GUIDE.md` - 详细实现指南
- `deploy/sparsebox_plugin/USAGE.md` - 使用指南
- `deploy/sparsebox_plugin/INTEGRATION_SUMMARY.md` - 本文档

### 5.2 修改文件

- `deploy/export_head_onnx.py` - 添加 Plugin 替换逻辑
- `deploy/tools/set_env.sh` - 添加 Plugin 环境变量
- `deploy/build_sparse4d_engine.sh` - 添加 Plugin 加载
- `deploy/build_sparse4d_engine_optimized.sh` - 添加 Plugin 加载
- `deploy/sparsebox_plugin/Makefile` - 修正目标文件名

## 六、注意事项

1. **首次使用**: 必须先编译 Plugin 库
2. **路径要求**: 所有路径都是相对于 `deploy/` 目录
3. **版本兼容**: 确保 TensorRT 版本支持自定义 Plugin
4. **精度验证**: 首次使用时建议对比 Python 和 TensorRT 的输出精度
5. **回退机制**: 如果 Plugin 不可用，代码会自动回退到原始实现

## 七、后续优化建议

1. **性能调优**: 根据实际运行情况调整 CUDA kernel 的线程块大小
2. **精度优化**: 如果发现精度问题，检查 FP16 转换和数值稳定性
3. **内存优化**: 进一步减少中间张量分配
4. **错误处理**: 增强错误处理和日志记录

