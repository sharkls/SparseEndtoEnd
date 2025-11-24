#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试sparsebox插件的确定性行为
运行多次测试，检查成功率是否一致
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import batch_validate_all_samples

def main():
    """运行多次测试，检查确定性"""
    import subprocess
    import numpy as np
    
    results = []
    num_runs = 10
    
    print(f"运行 {num_runs} 次测试，检查确定性...\n")
    
    for i in range(num_runs):
        print(f"运行 {i+1}/{num_runs}...", end=" ", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "batch_validate_all_samples.py"),
                "--onnx", "deploy/onnx/sparse4dhead1st.onnx",
                "--plugin-so", "deploy/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so",
                "--asset-dir", "script/tutorial/asset",
                "--fp16",
                "--max-samples", "10"
            ],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True
        )
        
        # 解析输出
        output = result.stdout + result.stderr
        if "成功次数:" in output:
            for line in output.split('\n'):
                if "成功次数:" in line:
                    parts = line.split()
                    success = int(parts[1])
                    total = int(parts[3].strip('()').split('/')[1])
                    success_rate = float(parts[4].strip('()%'))
                    results.append({
                        'success': success,
                        'total': total,
                        'rate': success_rate
                    })
                    print(f"成功率: {success_rate:.1f}%")
                    break
        else:
            print("解析失败")
            print(output[-500:])
    
    if results:
        print(f"\n{'='*60}")
        print("确定性测试结果")
        print(f"{'='*60}")
        rates = [r['rate'] for r in results]
        print(f"平均成功率: {np.mean(rates):.2f}%")
        print(f"成功率标准差: {np.std(rates):.2f}%")
        print(f"最小成功率: {np.min(rates):.2f}%")
        print(f"最大成功率: {np.max(rates):.2f}%")
        print(f"\n成功率列表: {[f'{r:.1f}%' for r in rates]}")
        
        if np.std(rates) < 1.0:
            print("\n✅ 测试通过：成功率稳定（标准差 < 1%）")
        else:
            print(f"\n⚠️  测试失败：成功率不稳定（标准差 = {np.std(rates):.2f}%）")
            print("   说明存在非确定性行为")

if __name__ == "__main__":
    main()

