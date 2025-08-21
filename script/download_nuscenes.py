#!/usr/bin/env python3
# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

import os
import argparse
import subprocess
from tqdm import tqdm

def download_nuscenes(version, output_dir, api_key):
    """下载nuScenes数据集
    
    Args:
        version (str): 数据集版本，如 'v1.0-mini', 'v1.0-trainval', 'v1.0-test'
        output_dir (str): 输出目录
        api_key (str): nuScenes API密钥
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置环境变量
    os.environ['NUSCENES'] = api_key
    
    # 下载命令
    cmd = [
        'python', '-m', 'nuscenes.scripts.download',
        '--version', version,
        '--output_dir', output_dir,
        '--all'
    ]
    
    # 执行下载命令
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # 实时显示下载进度
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # 检查是否下载成功
    if process.returncode == 0:
        print(f"成功下载 {version} 数据集到 {output_dir}")
    else:
        print(f"下载失败，错误码: {process.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载nuScenes数据集")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                      help="数据集版本: v1.0-mini, v1.0-trainval, v1.0-test")
    parser.add_argument("--output_dir", type=str, default="./data/nuscenes",
                      help="输出目录")
    parser.add_argument("--api_key", type=str, required=True,
                      help="nuScenes API密钥")
    
    args = parser.parse_args()
    
    # 开始下载
    download_nuscenes(args.version, args.output_dir, args.api_key) 