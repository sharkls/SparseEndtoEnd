#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NaN 失败案例调试工具

功能：
1. 运行验证并捕获失败案例
2. 启用 DEBUG_NAN 模式重新运行失败案例
3. 分析 NaN 产生的具体位置和原因
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import logging

LOGGER = logging.getLogger("NaNDebugger")


def setup_logger(verbose: bool):
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)


def run_validation(onnx_path, plugin_so, asset_dir, fp16=True, num_samples=10, all_nodes=True):
    """运行验证并返回失败案例列表"""
    cmd = [
        sys.executable,
        "deploy/val/validate_sparsebox_plugin.py",
        "--onnx", str(onnx_path),
        "--plugin-so", str(plugin_so),
        "--asset-dir", str(asset_dir),
        "--num-samples", str(num_samples),
    ]
    if fp16:
        cmd.append("--fp16")
    if all_nodes:
        cmd.append("--all-nodes")
    
    LOGGER.info("运行验证...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 解析输出，找出失败案例
    failures = []
    for line in result.stdout.split('\n'):
        if "NaN" in line or "Inf" in line:
            # 尝试解析节点和样本信息
            if "node" in line.lower() and "sample" in line.lower():
                # 提取节点和样本索引
                parts = line.split()
                node_idx = None
                sample_idx = None
                for i, part in enumerate(parts):
                    if part == "node" and i + 1 < len(parts):
                        try:
                            node_idx = int(parts[i + 1])
                        except:
                            pass
                    if part == "sample" and i + 1 < len(parts):
                        try:
                            sample_idx = int(parts[i + 1])
                        except:
                            pass
                if node_idx is not None and sample_idx is not None:
                    failures.append((node_idx, sample_idx))
    
    return failures, result.stdout


def rebuild_plugin_with_debug(plugin_dir, debug_nan=True):
    """使用 DEBUG_NAN 选项重新编译插件"""
    LOGGER.info("使用 DEBUG_NAN=%d 重新编译插件...", 1 if debug_nan else 0)
    
    env = dict(os.environ)
    env['DEBUG_NAN'] = '1' if debug_nan else '0'
    
    result = subprocess.run(
        ["make", "clean"],
        cwd=plugin_dir,
        capture_output=True,
        text=True
    )
    
    result = subprocess.run(
        ["make"],
        cwd=plugin_dir,
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        LOGGER.error("编译失败: %s", result.stderr)
        return False
    
    LOGGER.info("编译成功")
    return True


def debug_single_failure(onnx_path, plugin_so, asset_dir, node_idx, sample_idx, fp16=True):
    """调试单个失败案例"""
    LOGGER.info("调试失败案例: Node %d, Sample %d", node_idx, sample_idx)
    
    cmd = [
        sys.executable,
        "deploy/val/validate_sparsebox_plugin.py",
        "--onnx", str(onnx_path),
        "--plugin-so", str(plugin_so),
        "--asset-dir", str(asset_dir),
        "--node-index", str(node_idx),
        "--sample-index", str(sample_idx),
        "--verbose",
    ]
    if fp16:
        cmd.append("--fp16")
    
    # 运行并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 提取 DEBUG_NAN 输出
    debug_lines = []
    for line in result.stdout.split('\n'):
        if "[DEBUG_NAN]" in line:
            debug_lines.append(line)
    
    return debug_lines, result.stdout


def main():
    parser = argparse.ArgumentParser(description="调试 NaN 失败案例")
    parser.add_argument("--onnx", required=True, help="Head ONNX path")
    parser.add_argument("--plugin-so", required=True, help="Plugin shared library")
    parser.add_argument("--plugin-dir", default="deploy/sparsebox_plugin", help="Plugin source directory")
    parser.add_argument("--asset-dir", default="script/tutorial/asset", help="Asset directory")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to validate")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mode")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild plugin with DEBUG_NAN")
    parser.add_argument("--node-index", type=int, default=None, help="Specific node index to debug")
    parser.add_argument("--sample-index", type=int, default=None, help="Specific sample index to debug")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    setup_logger(args.verbose)
    
    onnx_path = Path(args.onnx)
    plugin_so = Path(args.plugin_so)
    plugin_dir = Path(args.plugin_dir)
    asset_dir = Path(args.asset_dir)
    
    # 如果需要，重新编译插件
    if args.rebuild:
        if not rebuild_plugin_with_debug(plugin_dir, debug_nan=True):
            LOGGER.error("重新编译失败")
            return 1
    
    # 如果指定了特定的节点和样本，直接调试
    if args.node_index is not None and args.sample_index is not None:
        debug_lines, full_output = debug_single_failure(
            onnx_path, plugin_so, asset_dir, args.node_index, args.sample_index, args.fp16
        )
        print("\n=== DEBUG_NAN 输出 ===")
        for line in debug_lines:
            print(line)
        if not debug_lines:
            print("未发现 DEBUG_NAN 输出。请确保使用 --rebuild 重新编译插件。")
        return 0
    
    # 否则，运行完整验证并找出失败案例
    failures, validation_output = run_validation(
        onnx_path, plugin_so, asset_dir, args.fp16, args.num_samples, all_nodes=True
    )
    
    if not failures:
        LOGGER.info("未发现失败案例！")
        return 0
    
    LOGGER.info("发现 %d 个失败案例", len(failures))
    for node_idx, sample_idx in failures:
        LOGGER.info("  - Node %d, Sample %d", node_idx, sample_idx)
    
    # 调试每个失败案例
    print("\n=== 开始调试失败案例 ===\n")
    for node_idx, sample_idx in failures:
        debug_lines, full_output = debug_single_failure(
            onnx_path, plugin_so, asset_dir, node_idx, sample_idx, args.fp16
        )
        print(f"\n--- Node {node_idx}, Sample {sample_idx} ---")
        if debug_lines:
            for line in debug_lines:
                print(line)
        else:
            print("未发现 DEBUG_NAN 输出。请使用 --rebuild 重新编译插件。")
    
    return 0


if __name__ == "__main__":
    import os
    sys.exit(main())

