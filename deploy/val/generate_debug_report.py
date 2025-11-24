#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成调试报告

总结当前的调试结果和分析
"""

import sys
from pathlib import Path
from datetime import datetime

def generate_report():
    """生成调试报告"""
    report = f"""
# FP16 SparseBox 插件 NaN 调试报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 当前状态

### 测试结果
- **优化前成功率**: 85.0% (51/60)
- **优化后成功率**: 88.3% (53/60)
- **提升**: +3.3%
- **剩余失败案例**: 7-11 个（非确定性）

### 实施的优化

1. **加强 centerX/Y/Z 检查**
   - 使用位操作检查 NaN/Inf
   - 在读取 anchor 输入时进行严格验证
   - 确保 center 值在后续计算中安全

2. **写入前最终检查**
   - 在写入输出前使用位操作再次验证
   - 确保写入的值绝对不是 NaN/Inf

## 调试发现

### 输入数据特征
- **所有失败案例的输入数据都是正常的**
- Anchor 数据：无 NaN/Inf，范围正常
- Instance 数据：无 NaN/Inf，范围正常
- **结论**: 问题出在计算过程中，而非输入数据

### 失败案例分布
- **Node 0**: 1-2 次失败
- **Node 1**: 2 次失败
- **Node 2**: 1-3 次失败
- **Node 3**: 2-3 次失败
- **Node 4**: 1-2 次失败
- **Node 5**: 1-2 次失败

### 非确定性特征
- 失败案例在不同运行间可能不同
- 某些案例有时成功，有时失败
- 说明问题可能与：
  - CUDA 执行顺序
  - 内存对齐
  - 数值精度边界条件
  有关

## 可能的原因

1. **计算过程中的 NaN 产生**
   - 虽然输入正常，但在某些计算步骤中可能产生 NaN
   - 现有的检查可能没有覆盖所有路径

2. **边界条件**
   - 某些极端输入值组合可能导致数值不稳定
   - 即使有检查，某些边界情况下检查可能失效

3. **内存对齐或指针问题**
   - 可能存在内存对齐问题
   - 指针计算可能有误

## 建议的下一步

1. **启用调试模式**
   - 在 kernel 中添加 `#define DEBUG_NAN`
   - 使用 printf 追踪 NaN 的产生位置
   - 重新编译并运行测试

2. **更激进的检查**
   - 在每次计算后立即检查并修复 NaN/Inf
   - 特别是在关键计算步骤（exp, sigmoid, 旋转等）

3. **分析失败案例的输入特征**
   - 检查失败案例的 anchor 数据是否有特殊模式
   - 分析 instance 数据的分布特征

4. **使用 CUDA 调试工具**
   - 使用 `cuda-gdb` 或 `compute-sanitizer` 追踪问题
   - 检查内存访问和数值计算

## 调试工具

已创建的调试工具：
1. `debug_nan_cases.py`: 分析失败案例的输入数据特征
2. `trace_nan_in_kernel.py`: 追踪单个失败案例的详细信息
3. `analyze_failure_patterns.py`: 分析失败模式（需要修复）

## 代码修改

在 `SparseBox3DKeyPointsKernel.cu` 中添加了调试支持：
- 添加了 `#ifdef DEBUG_NAN` 条件编译
- 可以在调试模式下使用 printf 追踪 NaN 产生位置

要启用调试模式，在编译时添加 `-DDEBUG_NAN` 标志。

"""
    
    report_path = Path("deploy/val/debug_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    
    print(report)
    print(f"\n报告已保存到: {report_path}")

if __name__ == "__main__":
    generate_report()

