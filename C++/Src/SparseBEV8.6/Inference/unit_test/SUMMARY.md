# SparseBEV8.6 特征提取模块单元测试总结

## 概述

本文档总结了在SparseBEV8.6路径下的Inference目录中创建的unit_test，用于测试特征提取模块的输入数据。该测试参考了`sparse4d_extract_feat_trt_infer_unit_test`的实现。

## 创建的文件

### 1. 主要测试文件
- **`sparse4d_extract_feat_unit_test.cpp`** (15KB, 396行)
  - 特征提取模块的主要单元测试文件
  - 包含4个测试用例：输入数据处理、TensorRT推理一致性、内存管理、数据格式转换
  - 参考了`sparse4d_extract_feat_trt_infer_unit_test`的实现

### 2. 构建配置文件
- **`CMakeLists.txt`** (2.9KB, 104行)
  - 配置GoogleTest、CUDA、TensorRT等依赖
  - 设置编译选项和链接库
  - 配置测试环境变量

### 3. 构建和运行脚本
- **`build_and_run_test.sh`** (6.2KB, 260行)
  - 完整的构建和运行脚本
  - 支持环境检查、构建、测试、清理等功能
  - 提供彩色输出和错误处理

- **`test_compile.sh`** (1.5KB, 77行)
  - 简单的编译测试脚本
  - 用于快速验证编译是否成功

### 4. 文档文件
- **`README.md`** (4.8KB, 189行)
  - 详细的使用说明文档
  - 包含构建步骤、运行方法、故障排除等

## 测试内容

### 1. InputDataProcessingTest
- 验证模块初始化
- 测试输入数据读取和处理
- 检查数据格式和大小
- 验证数据范围

### 2. TrtInferConsistencyVerification
- 测试TensorRT引擎推理
- 验证推理结果与期望结果的一致性
- 测量推理性能
- 计算误差统计

### 3. MemoryManagementTest
- 测试GPU内存分配和释放
- 验证数据在CPU和GPU之间的传输
- 检查内存泄漏

### 4. DataFormatConversionTest
- 测试不同精度格式的转换
- 验证数据完整性
- 检查精度损失

## 技术特点

### 1. 环境检查
- 自动检查CUDA环境
- 验证TensorRT安装
- 确认GoogleTest可用性

### 2. 错误处理
- 完善的错误检查和报告
- 彩色输出便于识别
- 详细的错误信息

### 3. 灵活性
- 支持多种运行模式
- 可配置的构建选项
- 模块化的测试结构

## 使用方法

### 快速开始
```bash
# 进入测试目录
cd C++/Src/SparseBEV8.6/Inference/unit_test

# 检查环境
./build_and_run_test.sh -e

# 完整流程（检查环境 -> 构建 -> 测试）
./build_and_run_test.sh -a

# 仅构建
./build_and_run_test.sh -b

# 仅运行测试
./build_and_run_test.sh -t

# 清理构建文件
./build_and_run_test.sh -c
```

### 手动构建
```bash
# 创建构建目录
mkdir -p build && cd build

# 配置CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON

# 构建
make sparse4d_extract_feat_unit_test

# 运行测试
./Output/unit_test/sparse4d_extract_feat_unit_test
```

## 配置要求

### 硬件要求
- NVIDIA GPU (支持CUDA 11.x)
- 至少4GB GPU内存

### 软件要求
- CUDA 11.x
- TensorRT 8.6
- CMake 3.16+
- GoogleTest (项目中已包含)

### 测试数据
- 输入图像：`sample_0_imgs_1*6*3*256*704_float32.bin`
- 期望特征：`sample_0_feature_1*89760*256_float32.bin`
- 位置：`script/tutorial/asset/`

## 集成到主项目

### 1. CMakeLists.txt更新
在主CMakeLists.txt中添加了条件编译支持：
```cmake
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(Inference/unit_test)
endif()
```

### 2. 构建选项
使用`-DBUILD_TESTING=ON`启用测试构建：
```bash
cmake .. -DBUILD_TESTING=ON
```

## 扩展性

### 添加新测试
1. 在`sparse4d_extract_feat_unit_test.cpp`中添加新的测试方法
2. 更新`CMakeLists.txt`以包含新的源文件
3. 添加相应的测试数据文件

### 自定义配置
- 修改`CMakeLists.txt`中的路径和选项
- 更新`build_and_run_test.sh`中的环境检查逻辑
- 调整测试参数和阈值

## 故障排除

### 常见问题
1. **GoogleTest未找到**：确保项目中的GoogleTest已构建
2. **CUDA内存不足**：减少批处理大小或释放其他GPU进程
3. **TensorRT引擎文件不存在**：检查引擎文件路径和权限
4. **测试数据文件缺失**：确保测试数据文件存在

### 调试技巧
- 启用详细日志：`export GLOG_v=2`
- 使用GDB调试：`gdb --args ./Output/unit_test/sparse4d_extract_feat_unit_test`
- 检查CUDA设备：`nvidia-smi`

## 总结

成功创建了一个完整的特征提取模块单元测试框架，包括：

1. **完整的测试覆盖**：涵盖输入处理、推理一致性、内存管理、数据转换等关键功能
2. **自动化构建**：提供便捷的构建和运行脚本
3. **完善的文档**：详细的使用说明和故障排除指南
4. **良好的集成性**：与主项目无缝集成
5. **高可扩展性**：易于添加新的测试用例和功能

该unit_test为SparseBEV8.6特征提取模块提供了可靠的测试保障，确保模块的正确性和稳定性。 