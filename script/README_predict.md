# Sparse4D 预测脚本使用指南

本文档介绍如何使用 `predict.py` 脚本进行 Sparse4D 模型的预测。

## 概述

`predict.py` 是一个简化的预测脚本，基于原始的 `test.py` 但移除了分布式训练和评估功能，专注于单张图片或少量图片的预测。该脚本提供了灵活的配置选项，支持不同的预测场景。

## 功能特性

- ✅ 单样本或批量样本预测
- ✅ 自定义置信度阈值
- ✅ GPU/CPU 推理支持
- ✅ 确定性预测选项
- ✅ 详细的预测结果统计
- ✅ JSON 格式结果输出
- ✅ 实时预测进度显示

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA (可选，用于GPU推理)
- 其他依赖项 (参考项目根目录的 requirements.txt)

## 文件结构

```
script/
├── predict.py              # 主预测脚本
├── predict_example.py      # 使用示例脚本
└── README_predict.md       # 本文档
```

## 快速开始

### 1. 基础预测

预测数据集中的前5个样本：

```bash
python script/predict.py \
    --config dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py \
    --checkpoint ckpt/sparse4dv3_r50.pth \
    --data_root data/nuscenes \
    --ann_file data/nuscenes/nuscenes_infos_val.pkl \
    --output_dir predictions/basic \
    --num_samples 5
```

### 2. 单样本预测

预测指定索引的单个样本：

```bash
python script/predict.py \
    --config dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py \
    --checkpoint ckpt/sparse4dv3_r50.pth \
    --data_root data/nuscenes \
    --ann_file data/nuscenes/nuscenes_infos_val.pkl \
    --output_dir predictions/single \
    --sample_idx 0
```

### 3. 使用示例脚本

运行所有示例：

```bash
python script/predict_example.py
```

## 参数说明

### 必需参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | `dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py` | 配置文件路径 |
| `--checkpoint` | str | `ckpt/sparse4dv3_r50.pth` | 模型检查点文件路径 |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | str | `data/nuscenes` | 数据根目录 |
| `--ann_file` | str | `data/nuscenes/nuscenes_infos_val.pkl` | 标注文件路径 |
| `--output_dir` | str | `predictions` | 预测结果输出目录 |
| `--sample_idx` | int | `None` | 指定预测的样本索引 |
| `--num_samples` | int | `5` | 预测的样本数量（当sample_idx未指定时） |
| `--score_threshold` | float | `0.3` | 检测置信度阈值 |
| `--deterministic` | bool | `False` | 是否设置确定性选项 |
| `--device` | str | `cuda:0` | 推理设备 |

## 输出结果

### 文件结构

```
output_dir/
├── predictions.json        # 详细预测结果
└── prediction_stats.json   # 统计信息
```

### predictions.json 格式

```json
[
  {
    "sample_idx": 0,
    "boxes_3d": [[x, y, z, w, l, h, yaw, vx, vy], ...],
    "scores_3d": [0.95, 0.87, ...],
    "labels_3d": [0, 1, ...],
    "track_ids": [1, 2, ...]  // 可选
  }
]
```

### prediction_stats.json 格式

```json
{
  "total_samples": 5,
  "class_names": ["car", "truck", ...],
  "detection_summary": {
    "car": {
      "total_detections": 15,
      "avg_confidence": 0.8234
    }
  },
  "overall": {
    "total_detections": 25,
    "avg_confidence": 0.7845
  }
}
```

## 使用场景

### 1. 开发调试

```bash
# 预测单个样本进行调试
python script/predict.py --sample_idx 0 --output_dir debug_output
```

### 2. 批量预测

```bash
# 预测多个样本
python script/predict.py --num_samples 100 --output_dir batch_predictions
```

### 3. 高精度预测

```bash
# 使用高置信度阈值
python script/predict.py --score_threshold 0.7 --output_dir high_precision
```

### 4. CPU 推理

```bash
# 在没有GPU的环境下使用CPU
python script/predict.py --device cpu --output_dir cpu_predictions
```

### 5. 可重现预测

```bash
# 设置确定性选项确保结果可重现
python script/predict.py --deterministic --output_dir reproducible
```

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```bash
   # 减少批处理大小或使用CPU
   python script/predict.py --device cpu
   ```

2. **文件路径错误**
   - 确保配置文件路径正确
   - 确保检查点文件存在
   - 确保数据文件路径正确

3. **依赖项缺失**
   ```bash
   # 安装依赖项
   pip install -r requirements.txt
   ```

### 调试模式

启用详细输出：

```bash
python script/predict.py --sample_idx 0 --output_dir debug 2>&1 | tee debug.log
```

## 性能优化

### GPU 优化

1. 使用混合精度训练：
   ```bash
   # 在配置文件中启用 fp16
   fp16 = dict(loss_scale=512.)
   ```

2. 调整批处理大小：
   ```bash
   # 根据GPU内存调整
   --num_samples 1  # 减少同时处理的样本数
   ```

### CPU 优化

1. 使用多进程：
   ```bash
   # 在配置文件中调整
   workers_per_gpu = 4
   ```

2. 减少模型复杂度：
   ```bash
   # 使用更小的模型配置
   --config dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py
   ```

## 扩展功能

### 自定义后处理

可以在 `predict.py` 中的 `visualize_predictions` 函数中添加自定义后处理逻辑：

```python
def visualize_predictions(predictions, output_dir, class_names):
    # 添加自定义后处理逻辑
    for pred in predictions:
        # 自定义处理
        pass
```

### 集成到其他系统

```python
from script.predict import predict_single_sample, build_module

# 构建模型和数据集
model = build_module(cfg["model"])
dataset = build_module(cfg["data"]["test"])

# 执行预测
result = predict_single_sample(model, dataset, sample_idx=0, device="cuda:0")
```

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../LICENSE) 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 参与讨论

---

**注意**: 使用前请确保已正确安装所有依赖项，并准备好相应的模型检查点和数据文件。 