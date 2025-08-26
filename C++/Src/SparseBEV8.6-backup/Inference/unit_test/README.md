# SparseBEV8.6 特征提取模块单元测试

## 概述

本单元测试用于验证SparseBEV8.6特征提取模块的输入数据处理和推理一致性。测试参考了`sparse4d_extract_feat_trt_infer_unit_test`的实现，确保特征提取模块能够正确处理输入数据并产生一致的输出结果。

## 测试内容

### 1. 输入数据处理测试 (InputDataProcessingTest)
- 验证模块初始化
- 测试输入数据读取
- 检查数据格式和大小
- 验证数据范围

### 2. TensorRT推理一致性测试 (TrtInferConsistencyVerification)
- 测试TensorRT引擎推理
- 验证推理结果与期望结果的一致性
- 测量推理性能
- 计算误差统计

### 3. 内存管理测试 (MemoryManagementTest)
- 测试GPU内存分配和释放
- 验证数据在CPU和GPU之间的传输
- 检查内存泄漏

### 4. 数据格式转换测试 (DataFormatConversionTest)
- 测试不同精度格式的转换
- 验证数据完整性
- 检查精度损失

## 构建和运行

### 前置条件
- CUDA 11.x
- TensorRT 8.6
- GoogleTest
- CMake 3.16+

### 构建步骤

1. 进入项目根目录：
```bash
cd /share/Code/SparseEnd2End/C++
```

2. 创建构建目录：
```bash
mkdir -p build
cd build
```

3. 配置CMake：
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

4. 构建项目：
```bash
make -j$(nproc)
```

### 运行测试

1. 直接运行测试可执行文件：
```bash
./Output/unit_test/sparse4d_extract_feat_unit_test
```

2. 使用CTest运行：
```bash
ctest -R sparse4d_extract_feat_unit_test -V
```

## 测试数据

测试使用以下数据文件：
- 输入图像：`sample_0_imgs_1*6*3*256*704_float32.bin`
- 期望特征：`sample_0_feature_1*89760*256_float32.bin`

这些文件位于：`script/tutorial/asset/`

## 配置要求

### TensorRT引擎配置
测试需要配置正确的TensorRT引擎文件路径：
```cpp
std::string engine_path = "/path/to/sparse4d_extract_feat_engine.trt";
```

### 模型参数配置
测试中使用的模型参数：
- 输入图像尺寸：1×6×3×256×704
- 输出特征尺寸：1×89760×256
- 相机数量：6
- 图像通道数：3
- 图像高度：256
- 图像宽度：704

## 输出说明

### 成功输出示例
```
[==========] Running 4 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 4 tests from Sparse4dExtractFeatUnitTest
[ RUN      ] Sparse4dExtractFeatUnitTest.InputDataProcessingTest
[INFO] 开始测试特征提取模块输入数据处理...
[INFO] 模块初始化成功
[INFO] 成功读取测试样本: sample_0_imgs_1*6*3*256*704_float32.bin
[INFO] 输入图像数据大小: 3244032
[INFO] 期望特征数据大小: 22978560
[       OK ] Sparse4dExtractFeatUnitTest.InputDataProcessingTest (123 ms)
[ RUN      ] Sparse4dExtractFeatUnitTest.TrtInferConsistencyVerification
[INFO] 开始TensorRT推理一致性验证...
[TensorRT Test] Sparse4d特征提取模块TensorRT推理(FP32)耗时 = 13.271 [ms]
Error >0.1 percentage is: 0.0107776
MaxError = 1.3429
[       OK ] Sparse4dExtractFeatUnitTest.TrtInferConsistencyVerification (9575 ms)
[ RUN      ] Sparse4dExtractFeatUnitTest.MemoryManagementTest
[INFO] 开始测试内存管理...
[INFO] GPU内存管理测试通过
[       OK ] Sparse4dExtractFeatUnitTest.MemoryManagementTest (45 ms)
[ RUN      ] Sparse4dExtractFeatUnitTest.DataFormatConversionTest
[INFO] 开始测试数据格式转换...
[INFO] 数据格式转换测试通过
[       OK ] Sparse4dExtractFeatUnitTest.DataFormatConversionTest (23 ms)
[----------] 4 tests from Sparse4dExtractFeatUnitTest (9766 ms total)
[----------] Global test environment tear-down
[==========] 1 test suite ran. (9766 ms total)
[  PASSED  ] 4 tests.
```

### 错误输出说明
- `[ERROR]`：严重错误，测试失败
- `[WARNING]`：警告信息，可能影响测试结果
- `[INFO]`：一般信息，用于调试

## 故障排除

### 常见问题

1. **TensorRT引擎文件不存在**
   - 确保引擎文件路径正确
   - 检查文件权限

2. **CUDA内存不足**
   - 减少批处理大小
   - 释放其他GPU进程

3. **测试数据文件缺失**
   - 确保测试数据文件存在
   - 检查文件路径配置

4. **依赖库缺失**
   - 安装GoogleTest：`sudo apt-get install libgtest-dev`
   - 确保CUDA和TensorRT正确安装

### 调试技巧

1. 启用详细日志：
```bash
export GLOG_v=2
./Output/unit_test/sparse4d_extract_feat_unit_test
```

2. 使用GDB调试：
```bash
gdb --args ./Output/unit_test/sparse4d_extract_feat_unit_test
```

3. 检查CUDA设备：
```bash
nvidia-smi
```

## 扩展测试

如需添加新的测试用例，可以：

1. 在测试类中添加新的测试方法
2. 更新CMakeLists.txt以包含新的源文件
3. 添加相应的测试数据文件

## 贡献

欢迎提交问题报告和改进建议。请确保：
- 测试代码遵循项目的编码规范
- 新测试用例有充分的文档说明
- 测试结果可重现 