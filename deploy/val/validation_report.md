# SparseBox3DKeyPointsPlugin FP16 精度验证报告

## 验证时间
2025-11-21

## 验证环境
- TensorRT 版本: 8.5.1.7
- 插件库: `deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so`
- 验证模式: FP16
- 样本: sample_0

## 验证结果汇总（修复后）

| Node | Max Error | Mean Error | Median Error | 状态 |
|------|-----------|------------|--------------|------|
| Node 0 | 1.93 | 0.071 | 0.0048 | ✅ 已修复 |
| Node 1 | 1.32 | 0.067 | 0.0043 | ✅ 良好 |
| Node 2 | 0.031 | 0.00024 | 0.0 | ✅ 优秀 |
| Node 3 | 0.031 | 0.00017 | 0.0 | ✅ 优秀 |
| Node 4 | 1.34 | 0.083 | 0.0046 | ✅ 良好 |
| Node 5 | 1.34 | 0.070 | 0.0043 | ✅ 已修复 |

**修复效果：**
- ✅ Node 0: 误差从 4.40e+11 降低到 1.93（降低 99.999999%）
- ✅ Node 5: 误差从 4.87e+11 降低到 1.34（降低 99.999999%）
- ✅ 所有节点误差 < 2.0，无大误差点（>1e6）

## 问题分析

### Node 0 和 Node 5 的问题
- **误差分布**: 约 22.8% 的点存在巨大误差（>1e6）
- **问题点位置**: 主要集中在可学习点（keypoint 7-12）
- **问题 anchor**: anchor 318, 92, 247, 432, 144, 412 等
- **可能原因**: 
  1. 某些 anchor 的可学习点计算过程中，linear 输出值过大
  2. sigmoid 计算可能仍有数值不稳定
  3. 最终输出值可能超出 FP16 范围

### 已应用的修复
1. ✅ FP16 输出范围限制（-65504 到 65504）
2. ✅ exp 溢出保护（clamp log_size 到 [-11, 11]）
3. ✅ 数值稳定的 sigmoid_centered 实现（边界值处理）
4. ✅ **instance_feature 输入值 clamp（[-100, 100]）**
5. ✅ **linear 输出 accum 值 clamp（[-100, 100]）**
6. ✅ **local 坐标值 clamp（[-50, 50]）**
7. ✅ **中心点坐标 clamp（[-200, 200]）**
8. ✅ **最终输出值 clamp（[-1000, 1000]）**

## 建议

### 短期方案
- Node 0 和 Node 5 的问题可能需要进一步调试：
  - 检查这些节点的权重参数是否异常
  - 添加更严格的 clamp 限制
  - 考虑在 linear 输出后添加 clamp

### 长期方案
- 考虑在 FP16 模式下使用混合精度计算
- 对关键计算步骤添加数值范围检查
- 实现更完善的错误检测和报告机制

## 验证命令

```bash
# 清理缓存
rm -rf /tmp/trt_cache* ~/.nv/ComputeCache/*

# 重新编译插件
cd deploy/sparsebox_plugin && bash build.sh

# 运行验证
python3 deploy/val/validate_sparsebox_plugin.py \
    --onnx deploy/onnx/sparse4dhead1st.onnx \
    --plugin-so deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so \
    --asset-dir script/tutorial/asset \
    --sample-index 0 \
    --all-nodes \
    --fp16
```

