# Sparse4D 感知部署工程 - C++ 实现

## 📋 目录

- [项目概述](#项目概述)
- [代码结构](#代码结构)
- [核心模块](#核心模块)
- [构建指南](#构建指南)
- [使用说明](#使用说明)
- [配置说明](#配置说明)
- [架构设计](#架构设计)
- [依赖项](#依赖项)

---

## 项目概述

Sparse4D 是一个基于稀疏表示的3D目标检测算法，本工程提供了完整的C++部署实现，支持多种精度和TensorRT版本。

### 主要特性

- ✅ **多精度支持**: FP32 和 FP16 (Half Precision)
- ✅ **多TensorRT版本**: 支持 TensorRT 8.6 和 TensorRT 10.2
- ✅ **模块化设计**: 预处理、特征提取、检测头、后处理等模块独立
- ✅ **GPU加速**: 完整的CUDA实现，包括自定义Kernel
- ✅ **时序建模**: Instance Bank机制支持多帧时序信息融合
- ✅ **高性能**: CUDA Graph优化，内存复用，异步流水线

---

## 代码结构

```
C++/
├── CMakeLists.txt              # 主构建文件
├── main.cpp                    # 主程序入口
├── build.sh                    # 快速构建脚本
├── Include/                    # 公共头文件
│   ├── Common/                 # 通用工具
│   │   ├── Core/              # 核心上下文
│   │   ├── Utils/             # CUDA包装器等工具
│   │   ├── ErrorHandler/      # 错误处理
│   │   └── Factory/           # 工厂模式
│   └── Interface/              # 对外接口
├── Src/                        # 源代码目录
│   ├── Sparse4D/              # FP32版本实现
│   ├── Sparse4DFP16/          # FP16版本实现
│   ├── SparseBEV8.6/          # TensorRT 8.6版本
│   ├── sparse4dbev/            # 新泛化框架（模板化待完善）
│   └── Common/                 # 公共模块
├── Submodules/                 # 子模块
│   ├── Protoser/              # Protobuf序列化
│   ├── Fastddsser/            # FastDDS序列化
│   └── ThirdParty/            # 第三方库
├── Output/                     # 输出目录
│   ├── Lib/                   # 编译生成的库文件
│   └── Configs/               # 配置文件
└── TestSparse4D.cpp           # 测试程序
```

---

## 核心模块

### 1. 预处理模块 (Preprocessor)

**位置**: `Src/*/preprocessor/`

**功能**:
- 图像尺寸调整（双线性插值）
- 图像裁剪
- 归一化（ImageNet标准）
- GPU加速的CUDA Kernel实现

**关键文件**:
- `img_preprocessor.cpp/hpp`: 预处理主类
- `img_aug_with_bilinearinterpolation_kernel.cu`: CUDA Kernel实现

**特性**:
- 支持多相机输入
- CHW格式（Channel-Height-Width planar）
- 支持FP32和FP16输出

### 2. 特征提取模块 (Backbone)

**位置**: `Src/*/sparse4d/backbone.cpp/hpp`

**功能**:
- 多尺度特征提取
- 使用TensorRT引擎进行推理
- 输出多层级特征图

**输入**: 预处理后的图像 `[N, C, H, W]`
**输出**: 多尺度特征 `[N, C, H_i, W_i]` (i=0,1,2,3)

### 3. 检测头模块 (Head)

**位置**: `Src/*/sparse4d/first_head.cpp` 和 `second_head.cpp`

**功能**:
- **Head1 (First Frame)**: 处理第一帧，使用初始anchor
- **Head2 (Subsequent Frames)**: 处理后续帧，使用历史信息

**关键组件**:
- Instance Bank: 管理时序信息
- Graph Neural Network: 时序建模
- Deformable Attention: 空间注意力
- 分类和回归分支

**输出**:
- `pred_instance_feature`: 实例特征
- `pred_anchor`: 3D边界框参数
- `pred_class_score`: 类别分数
- `pred_quality_score`: 质量分数
- `pred_track_id`: 跟踪ID（仅Head2）

### 4. 实例库模块 (Instance Bank)

**位置**: `Src/*/sparse4d/instance_bank.cpp/hpp`

**功能**:
- 管理多帧时序信息
- Anchor投影和更新
- Track ID生成和管理
- 特征缓存和复用

**关键操作**:
- `get()`: 获取历史信息
- `cache()`: 缓存当前帧结果
- `update()`: 更新实例库状态
- `project_anchors()`: 投影anchor到新坐标系

### 5. 后处理模块 (Postprocessor)

**位置**: `Src/*/postprocessor/`

**功能**:
- GPU NMS (Non-Maximum Suppression)
- 结果格式转换
- 置信度阈值过滤
- 输出最终检测结果

**关键文件**:
- `postprocessor.cpp/hpp`: 后处理主类
- `gpu_nms.cu`: GPU加速的NMS实现

---

## 构建指南

### 前置要求

- **CMake**: >= 3.10
- **C++标准**: C++17
- **CUDA**: >= 11.0
- **TensorRT**: 8.6 或 10.2
- **OpenCV**: >= 4.0
- **Protobuf**: >= 3.0
- **Eigen3**: 线性代数库

### 快速构建

```bash
# 使用构建脚本
cd C++
./build.sh

# 或手动构建
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 构建选项

在 `CMakeLists.txt` 中可以配置：

```cmake
option(BUILD_SPARSEBEV "Build SparseBEV with TensorRT 10.2" OFF)
option(BUILD_SPARSEBEV8_6 "Build SparseBEV8.6 with TensorRT 8.6" ON)
```

### 输出文件

编译完成后，生成的文件位于：
- **库文件**: `Output/Lib/`
  - `libSparse4DAlg.so`: Sparse4D算法库
  - `libSparseBEVAlg8.6.so`: SparseBEV 8.6版本库
- **可执行文件**: `Output/testAlgLib`

---

## 使用说明

### 基本使用流程

```cpp
#include "ExportSparse4D.h"

// 1. 创建算法实例
ICore* alg = CreateCoreObj();

// 2. 初始化（需要配置文件路径）
std::string config_path = "path/to/config.conf";
alg->initAlgorithm(config_path, callback, handle);

// 3. 更新参数（lidar2img矩阵）
float lidar2img[96]; // 6 cameras * 4x4 matrix
// ... 填充矩阵数据
alg->update(lidar2img);

// 4. 运行算法
CTimeMatchSrcData* input_data = ...;
alg->runAlgorithm(input_data);

// 5. 清理
delete alg;
```

### 配置文件

配置文件使用Protobuf文本格式，示例见 `Output/Configs/Alg/Sparse4d.conf`

主要配置项：
- **preprocessor_params**: 预处理参数（图像尺寸、归一化等）
- **backbone_engine**: Backbone引擎路径和插件
- **head1st_engine**: 第一帧Head引擎配置
- **head2nd_engine**: 后续帧Head引擎配置
- **instance_bank_params**: 实例库参数（query数量、anchor路径等）
- **postprocessor_params**: 后处理参数（NMS阈值等）

---

## 配置说明

### 预处理配置

```protobuf
preprocessor_params {
    num_cams: 6                    # 相机数量
    raw_img_c: 3                   # 原始图像通道数
    raw_img_h: 1080                # 原始图像高度
    raw_img_w: 1920                # 原始图像宽度
    model_input_img_h: 256         # 模型输入高度
    model_input_img_w: 704         # 模型输入宽度
    resize_ratio: 0.5              # 缩放比例
    crop_height: 0                 # 裁剪高度偏移
    crop_width: 0                  # 裁剪宽度偏移
}
```

### 引擎配置

```protobuf
backbone_engine {
    engine_path: "/path/to/backbone.engine"
    plugin_paths: "/path/to/plugin1.so"  # 可配置多个插件
    input_names: ["img"]
    output_names: ["feature"]
}

head1st_engine {
    engine_path: "/path/to/head1st.engine"
    plugin_paths: "/path/to/deformableAttention.so"
    plugin_paths: "/path/to/customLayerNorm.so"
    input_names: ["feature", "spatial_shapes", ...]
    output_names: ["pred_instance_feature", ...]
}
```

### 实例库配置

```protobuf
instance_bank_params {
    num_querys: 900                # Query数量
    query_dims: 11                 # Query维度
    instance_bank_anchor_path: "/path/to/anchors.bin"  # Anchor文件路径
    default_time_interval: 0.1     # 默认时间间隔（秒）
}
```

---

## 架构设计

### 数据流

```
输入图像 (CTimeMatchSrcData)
    ↓
[Preprocessor] 图像预处理
    ↓
[Backbone] 特征提取
    ↓
[Instance Bank] 获取历史信息
    ↓
[Head1/Head2] 检测推理
    ↓
[Instance Bank] 缓存结果
    ↓
[Postprocessor] 后处理
    ↓
输出结果 (CAlgResult)
```

### 版本对比

| 版本 | 精度 | TensorRT | 特点 |
|------|------|----------|------|
| **Sparse4D** | FP32 | 8.6 | 原始版本，稳定可靠 |
| **Sparse4DFP16** | FP16 | 8.6 | 半精度优化，性能提升 |
| **SparseBEV8.6** | FP32 | 8.6 | 独立实现，易于维护 |
| **sparse4dbev** | FP32/FP16 | 8.6+ | 模板化设计，统一接口 |

### 模板化设计 (sparse4dbev)

新的 `sparse4dbev` 框架采用模板化设计：

```cpp
template <typename T>  // T可以是float或half
class Sparse4DImpl {
    // 统一的实现，支持FP32和FP16
};
```

**优势**:
- 代码复用：FP32和FP16共享同一套实现
- 类型安全：编译时类型检查
- 易于扩展：支持新的数据类型

---

## 依赖项

### 核心依赖

- **CUDA**: GPU计算
- **TensorRT**: 推理引擎
- **cuDNN**: 深度学习加速库
- **Protobuf**: 配置序列化
- **OpenCV**: 图像处理

### 第三方库（Submodules）

- **FastDDS**: 数据分发服务
- **Eigen3**: 线性代数
- **glog**: 日志库
- **gflags**: 命令行参数解析

### 编译依赖

- **CMake**: 构建系统
- **GCC/G++**: C++编译器（支持C++17）
- **NVCC**: CUDA编译器

---

## 关键实现细节

### 1. 预处理坐标映射

使用双线性插值进行图像缩放，坐标映射公式：

```cpp
resize_ratio_x = w / floor(w * resize_ratio)
resize_ratio_y = h / floor(h * resize_ratio)
src_x = (dst_x + crop_w + 0.5) * resize_ratio_x - 0.5
src_y = (dst_y + crop_h + 0.5) * resize_ratio_y - 0.5
```

### 2. 归一化

ImageNet标准归一化：
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

### 3. CUDA Kernel优化

- **Block大小**: 16x16 或 32x32
- **Grid布局**: `(num_cams, H_blocks, W_blocks)`
- **内存访问**: 合并访问，减少bank conflicts

### 4. 内存管理

- **CudaWrapper**: 自动内存管理类
- **Pinned Memory**: 加速Host-Device传输
- **内存复用**: 避免频繁分配释放

### 5. 时序建模

- **Instance Bank**: 维护历史帧的anchor和特征
- **Anchor投影**: 使用lidar2global矩阵投影到新坐标系
- **Track ID**: 基于IoU和特征相似度匹配

---

## 测试

### 运行测试

```bash
cd Output
./testAlgLib
```

测试程序会根据 `Output/Configs/Alg/TestConfig.conf` 中的配置选择运行哪个测试单元。

### 测试单元

- **SparseBEV_8_6**: TensorRT 8.6版本测试
- **Sparse4D**: Sparse4D算法测试

---

## 性能优化

### CUDA Graph

使用CUDA Graph优化推理性能，需要：
- 固定的CUDA Stream
- 固定的内存地址
- 预热推理（2-3次）

### 混合精度

FP16版本相比FP32版本：
- 内存占用减少50%
- 推理速度提升30-50%
- 精度损失<1%

### 异步流水线

- 预处理和推理异步执行
- 使用CUDA Stream实现流水线并行
- 减少CPU-GPU同步开销

---

## 常见问题

### Q: 如何选择使用哪个版本？

**A**: 
- 需要最高精度：使用 `Sparse4D` (FP32)
- 需要最佳性能：使用 `Sparse4DFP16` (FP16)
- 需要统一接口：使用 `sparse4dbev` (模板化)

### Q: 如何添加新的TensorRT插件？

**A**: 在配置文件的 `engine` 配置中添加 `plugin_paths` 字段。

### Q: 内存不足怎么办？

**A**: 
- 使用FP16版本减少内存占用
- 减少batch size
- 优化图像输入尺寸

### Q: 如何调试CUDA Kernel？

**A**: 
- 使用 `cuda-gdb` 或 `cuda-memcheck`
- 添加 `cudaDeviceSynchronize()` 确保kernel执行完成
- 检查CUDA错误码

---

## 贡献指南

1. 遵循现有的代码风格
2. 添加必要的注释和文档
3. 确保所有测试通过
4. 提交前运行代码格式化工具

---

## 许可证

[根据项目许可证填写]

---

## 联系方式

[根据项目信息填写]

---

**最后更新**: 2025-01-XX
