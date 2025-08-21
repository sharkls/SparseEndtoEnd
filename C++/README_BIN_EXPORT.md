# SparseBEV8.6 Bin文件导出和时间戳处理功能

## 概述

本文档描述了为SparseBEV8.6添加的bin文件导出功能和时间戳处理机制，用于与Python推理结果进行对比测试。

## 功能特性

### 1. Bin文件导出功能

#### 1.1 预处理阶段
- **原始图像数据**: `sample_0_ori_imgs_1*6*3*H*W_float32.bin`
  - 保存原始多视角图像数据（归一化到[0,1]）
  - 形状: `[1, 6, 3, H, W]` - [batch_size, num_cameras, channels, height, width]

- **预处理后图像数据**: `sample_0_imgs_1*6*3*256*704_float32.bin`
  - 保存经过预处理（归一化、resize等）的图像数据
  - 形状: `[1, 6, 3, 256, 704]` - 标准化的输入尺寸

#### 1.2 推理阶段
- **特征提取结果**: `sample_0_feature_1*1536*256_float32.bin`
  - 保存ResNet+FPN提取的多尺度特征
  - 形状: `[1, 1536, 256]` - [batch_size, total_spatial_points, feature_dim]

#### 1.3 辅助数据
- **时间间隔**: `sample_0_time_interval_1_float32.bin`
  - 保存相邻帧之间的时间间隔
  - 形状: `[1]` - 单个浮点数值

- **图像宽高**: `sample_0_image_wh_2_float32.bin`
  - 保存图像宽高信息
  - 形状: `[2]` - [width, height]

- **Lidar2img变换矩阵**: `sample_0_lidar2img_6*4*4_float32.bin`
  - 保存6个相机的lidar到图像的变换矩阵
  - 形状: `[6, 4, 4]` - 6个相机的4x4变换矩阵

### 2. 时间戳处理机制

#### 2.1 时间戳差异分析
- **实际执行时间**: 从开始到结束的总耗时（毫秒）
- **时间间隔**: 相邻帧之间的时间差，用于时序推理（秒）
- **概念区分**: 两者概念不同，不应该直接比较

#### 2.2 时间戳信息保存
- 保存详细的时间戳分析到 `timestamp_info.txt`
- 包含开始时间戳、结束时间戳、执行时间等信息
- 提供时间间隔与实际执行时间的差异分析

## 文件结构

```
C++/
├── Src/SparseBEV8.6/
│   ├── Preprocess/
│   │   ├── img_preprocessor.h          # 添加saveOriginalDataToBin声明
│   │   └── img_preprocessor.cpp        # 实现bin文件保存功能
│   └── Inference/
│       └── SparseBEV.cpp               # 更新特征保存功能
├── TestSparseBEVAlgv2.cpp              # 添加时间戳处理和辅助数据保存
├── script/
│   ├── compare_cpp_vs_python.py        # C++与Python对比脚本
│   └── run_comparison_test.sh          # 完整测试流程脚本
└── Output/val_bin/                     # 输出目录
    ├── sample_0_ori_imgs_*.bin         # 原始图像数据
    ├── sample_0_imgs_*.bin             # 预处理后图像数据
    ├── sample_0_feature_*.bin          # 特征提取结果
    ├── sample_0_time_interval_*.bin    # 时间间隔
    ├── sample_0_image_wh_*.bin         # 图像宽高
    ├── sample_0_lidar2img_*.bin        # 变换矩阵
    └── timestamp_info.txt              # 时间戳分析信息
```

## 使用方法

### 1. 编译代码
```bash
cd /share/Code/SparseEnd2End/C++
./build.sh
```

### 2. 运行测试
```bash
# 运行C++测试（生成bin文件）
./TestSparseBEVAlgv2

# 或者运行完整对比测试
./script/run_comparison_test.sh
```

### 3. 手动对比
```bash
# 运行对比脚本
python3 script/compare_cpp_vs_python.py \
    --cpp_dir Output/val_bin/ \
    --python_dir /share/Code/SparseEnd2End/script/tutorial/asset/ \
    --save_dir script/compare/results \
    --tolerance 0.01
```

## 数据格式说明

### 1. 文件命名规范
- 格式: `sample_{sample_idx}_{data_name}_{shape}_{dtype}.bin`
- 示例: `sample_0_imgs_1*6*3*256*704_float32.bin`

### 2. 数据存储格式
- 所有数据都是小端序存储
- float32数据直接存储为4字节浮点数
- int32数据直接存储为4字节整数

### 3. 信息文件
- 每个bin文件都有对应的 `.info` 文件
- 包含数据统计信息、形状、大小等详细信息

## 测试要点

### 1. 数据格式一致性
- 确保C++和Python输出的数据格式完全一致
- 检查数据形状、类型、存储顺序

### 2. 数值精度
- 预处理阶段: 容忍度较高 (1e-2)
- 特征提取: 中等容忍度 (1e-3)
- 辅助数据: 较低容忍度 (1e-4)

### 3. 时间戳处理
- 区分实际执行时间和时间间隔的概念
- 时间间隔应该从数据的时间戳中获取
- 不要将执行时间误认为是时间间隔

## 注意事项

1. **数据归一化**: 原始图像数据需要归一化到[0,1]范围
2. **形状匹配**: 确保数据形状与Python脚本输出一致
3. **精度控制**: 不同阶段使用不同的误差容忍度
4. **时间戳概念**: 正确理解时间戳和时间间隔的区别
5. **文件路径**: 确保输出目录存在且有写入权限

## 故障排除

### 1. 编译错误
- 检查头文件包含是否正确
- 确认成员变量和方法名是否正确
- 验证数据结构访问方式

### 2. 运行时错误
- 检查输入数据是否完整
- 确认GPU内存分配是否成功
- 验证文件路径和权限

### 3. 对比失败
- 检查数据形状是否匹配
- 确认数值精度是否在容忍范围内
- 验证数据存储格式是否一致

## 更新日志

- **2024-01-XX**: 初始版本，添加基本的bin文件导出功能
- **2024-01-XX**: 添加时间戳处理和差异分析
- **2024-01-XX**: 完善对比脚本和测试流程 