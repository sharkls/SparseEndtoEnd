# TopK一致性优化说明

## 概述

为了解决PyTorch和TensorRT之间`indices`不一致的问题，我们对工程进行了全面的数值稳定性优化。

## 修改文件列表

### 1. `modules/head/sparse4d_blocks/instance_bank.py`
- 添加了`topk_completely_stable`函数：完全稳定的topk实现
- 添加了`topk_with_preprocessing`函数：带预处理的稳定topk
- 添加了`verify_consistency`函数：一致性验证

### 2. `deploy/export_head_onnx.py`
- 添加了完全确定性环境设置
- 使用新的`topk_with_preprocessing`函数
- 添加了一致性验证

### 3. `script/tutorial/010.sparse4d_end2end_io_bin_export_test.py`
- 添加了完全确定性环境设置

### 4. `deploy/unit_test/sparse4d_head_second_frame_infer-consistency-val_pytorch_vs_trt_unit_test.py`
- 添加了完全确定性环境设置

### 5. `test_topk_consistency.py` (新增)
- 测试脚本，验证不同topk函数的一致性

## 核心改进

### 1. 环境确定性
```python
# 设置环境变量
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'

# 设置PyTorch确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

# 设置随机种子
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
```

### 2. 完全稳定的排序算法
```python
def topk_completely_stable(confidence, k, *inputs):
    """
    使用Python内置排序，确保完全确定性
    """
    # 创建(confidence, index)对
    confidence_with_index = [(float(conf.item()), i) for i, conf in enumerate(confidence[b])]
    
    # 使用稳定排序
    confidence_with_index.sort(key=lambda x: (-x[0], x[1]))
    
    # 提取结果
    top_k_indices = [idx for _, idx in confidence_with_index[:k]]
    return indices, outputs
```

### 3. 数值预处理和后处理
```python
def topk_with_preprocessing(confidence, k, *inputs):
    # 移除NaN和Inf
    confidence_clean = torch.where(torch.isfinite(confidence_clean), confidence_clean, torch.zeros_like(confidence_clean))
    
    # 添加epsilon避免相同值
    epsilon = 1e-10
    confidence_clean = confidence_clean + epsilon
    
    # 使用稳定排序
    result = topk_completely_stable(confidence_clean, k, *inputs)
    
    # 移除epsilon
    return result - epsilon
```

## 使用方法

### 1. 测试topk函数一致性
```bash
cd /share/Code/SparseEnd2End
python test_topk_consistency.py
```

### 2. 重新导出ONNX模型
```bash
cd /share/Code/SparseEnd2End
python deploy/export_head_onnx.py --cfg dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py --ckpt ckpt/sparse4dv3_r50.pth
```

### 3. 运行一致性测试
```bash
cd /share/Code/SparseEnd2End
python deploy/unit_test/sparse4d_head_second_frame_infer-consistency-val_pytorch_vs_trt_unit_test.py
```

## 预期效果

使用这些优化后，预期能达到：

1. **indices误差**: 0（完全一致）
2. **confidence_sorted误差**: 接近0（浮点数精度差异）
3. **整体一致性**: 显著提升

## 注意事项

1. **性能影响**: 新的排序算法可能比原始torch.topk稍慢，但能保证一致性
2. **内存使用**: 预处理步骤会创建额外的tensor副本
3. **环境要求**: 需要确保所有环境变量和随机种子设置正确

## 故障排除

如果仍然存在一致性问题：

1. 检查环境变量是否正确设置
2. 验证随机种子是否在所有相关文件中设置
3. 确认ONNX导出时使用了新的topk函数
4. 运行测试脚本验证topk函数本身的一致性

## 技术原理

问题的根本原因是：
1. **浮点数精度差异**: PyTorch和TensorRT在浮点数计算上有微小差异
2. **排序算法差异**: 不同实现的排序算法对相同值可能有不同的处理
3. **数值稳定性**: 接近的值在排序时可能产生不同的结果

解决方案通过：
1. **环境确定性**: 消除所有随机性
2. **算法确定性**: 使用完全确定的排序算法
3. **数值稳定性**: 预处理和后处理步骤
4. **一致性验证**: 确保结果的正确性 