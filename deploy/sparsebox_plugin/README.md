# SparseBox3DKeyPointsPlugin 编译和使用指南

## 问题说明

如果直接运行 `make` 出现 `nvcc: Command not found` 错误，这是因为环境变量未设置。

## 解决方案

### 方法 1: 使用编译脚本（推荐）

```bash
cd deploy/sparsebox_plugin
./build.sh
```

这个脚本会自动：
1. 加载环境变量（source set_env.sh）
2. 检查必要的环境变量
3. 检查 nvcc 是否存在
4. 运行 make 编译

### 方法 2: 手动加载环境变量

```bash
# 在 deploy 目录下
cd deploy
source tools/set_env.sh

# 然后编译
cd sparsebox_plugin
make
```

### 方法 3: 一行命令

```bash
cd deploy && source tools/set_env.sh && cd sparsebox_plugin && make
```

## 验证编译结果

编译成功后，应该看到：

```bash
ls -lh lib/SparseBox3DKeyPointsPlugin.so
```

输出应该类似：
```
-rwxr-xr-x 1 user user 123K Dec 20 10:00 lib/SparseBox3DKeyPointsPlugin.so
```

## 常见问题

### Q1: 仍然提示 "nvcc: Command not found"

**解决**:
1. 检查 `deploy/tools/set_env.sh` 中的 `ENV_CUDA_BIN` 路径是否正确
2. 确认该路径下确实有 `nvcc` 文件
3. 使用 `./build.sh` 脚本，它会自动检查

### Q2: 提示 "ENV_CUDA_BIN is not set"

**解决**:
- 使用 `./build.sh` 脚本（推荐）
- 或者手动运行 `source ../tools/set_env.sh`

### Q3: 编译失败，提示找不到头文件

**解决**:
1. 检查 `ENV_TensorRT_INC` 和 `ENV_CUDA_INC` 是否正确设置
2. 确认 TensorRT 和 CUDA 已正确安装

## 完整工作流程

```bash
# 1. 编译 Plugin（使用编译脚本）
cd deploy/sparsebox_plugin
./build.sh

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
```

