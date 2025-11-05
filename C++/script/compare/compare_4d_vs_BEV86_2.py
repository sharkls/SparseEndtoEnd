#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比Sparse4D生成的bin文件与参考文件（val_bin_gpu）
检查数据格式和内容的一致性
"""

import os
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re


def parse_filename(filename: str) -> Dict[str, Any]:
    """
    从文件名解析数据类型和形状信息
    
    文件名格式示例:
    - sample_0_input_features_1*89760*256_float32.bin
    - sample_0_pred_track_ids_1*900_int32.bin
    
    Returns:
        包含 dtype, shape 的字典
    """
    info = {
        'dtype': None,
        'shape': None,
        'base_name': None
    }
    
    # 提取数据类型 (float32, int32, float64, int64等)
    dtype_match = re.search(r'_(float\d+|int\d+|uint\d+)\.bin$', filename)
    if dtype_match:
        dtype_str = dtype_match.group(1)
        if dtype_str.startswith('float'):
            if dtype_str == 'float32':
                info['dtype'] = np.float32
            elif dtype_str == 'float64':
                info['dtype'] = np.float64
            else:
                info['dtype'] = np.float32
        elif dtype_str.startswith('int'):
            if dtype_str == 'int32':
                info['dtype'] = np.int32
            elif dtype_str == 'int64':
                info['dtype'] = np.int64
            else:
                info['dtype'] = np.int32
        elif dtype_str.startswith('uint'):
            if dtype_str == 'uint32':
                info['dtype'] = np.uint32
            elif dtype_str == 'uint64':
                info['dtype'] = np.uint64
            else:
                info['dtype'] = np.uint32
    
    # 提取形状信息 (1*89760*256, 6*4*2等)
    shape_match = re.search(r'_(\d+(?:\*\d+)*)_(?:float|int|uint)', filename)
    if shape_match:
        shape_str = shape_match.group(1)
        shape = tuple(int(x) for x in shape_str.split('*'))
        info['shape'] = shape
    
    # 提取基础名称（去掉扩展名和形状信息）
    # 尝试匹配到形状信息之前的部分
    # 例如: sample_0_input_features_1*89760*256_float32.bin -> sample_0_input_features
    base_match = re.search(r'(sample_\d+_[a-zA-Z_]+)', filename)
    if base_match:
        info['base_name'] = base_match.group(1)
    else:
        # 如果上面的正则不匹配，尝试更通用的方式：提取到第一个数字（形状信息）之前
        base_match = re.search(r'^(.+?)_\d+\*', filename)
        if base_match:
            info['base_name'] = base_match.group(1)
        else:
            # 最后尝试：去掉扩展名，去掉类型后缀，提取主要部分
            name_without_ext = filename.replace('.bin', '')
            # 去掉类型后缀（如 _float32, _int32）
            name_without_type = re.sub(r'_(float|int|uint)\d+$', '', name_without_ext)
            # 去掉形状信息（最后一个 _数字*数字 模式）
            name_clean = re.sub(r'_\d+(?:\*\d+)+$', '', name_without_type)
            if name_clean:
                info['base_name'] = name_clean
    
    return info


def load_bin_file(file_path: str, dtype: Optional[np.dtype] = None, 
                  expected_shape: Optional[Tuple] = None) -> Optional[np.ndarray]:
    """
    加载二进制文件
    
    Args:
        file_path: 文件路径
        dtype: 数据类型，如果为None则从文件名推断
        expected_shape: 期望的形状，如果为None则从文件名推断
    
    Returns:
        numpy数组，失败返回None
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    try:
        # 如果未指定dtype，尝试从文件名推断
        if dtype is None:
            file_info = parse_filename(os.path.basename(file_path))
            dtype = file_info.get('dtype', np.float32)
        
        # 读取二进制文件
        data = np.fromfile(file_path, dtype=dtype)
        
        # 如果指定了期望形状，进行reshape
        if expected_shape is None:
            file_info = parse_filename(os.path.basename(file_path))
            expected_shape = file_info.get('shape')
        
        if expected_shape is not None:
            expected_size = np.prod(expected_shape)
            if len(data) != expected_size:
                print(f"⚠️  警告: 文件大小不匹配. 期望: {expected_size}, 实际: {len(data)}")
                print(f"   文件: {file_path}")
                # 尝试reshape，如果失败则返回一维数组
                if len(data) % expected_size == 0:
                    # 可能是多batch数据
                    print(f"   检测到可能的多batch数据，使用前{expected_size}个元素")
                    data = data[:expected_size]
                else:
                    print(f"   无法reshape，返回一维数组")
                    return data
            data = data.reshape(expected_shape)
        
        return data
    except Exception as e:
        print(f"❌ 加载失败: {file_path}, 错误: {e}")
        return None


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str, 
                   tolerance: float = 1e-6, relative_tolerance: float = 1e-5) -> bool:
    """
    对比两个numpy数组
    
    Args:
        arr1: 第一个数组
        arr2: 第二个数组
        name: 数据名称
        tolerance: 绝对容差
        relative_tolerance: 相对容差
    
    Returns:
        是否匹配
    """
    if arr1 is None or arr2 is None:
        print(f"❌ {name}: 其中一个数组为空")
        return False
    
    # 检查形状
    if arr1.shape != arr2.shape:
        print(f"❌ {name}: 形状不匹配")
        print(f"   形状1: {arr1.shape}")
        print(f"   形状2: {arr2.shape}")
        return False
    
    # 检查数据类型
    if arr1.dtype != arr2.dtype:
        print(f"⚠️  {name}: 数据类型不匹配 - {arr1.dtype} vs {arr2.dtype}")
        print(f"   尝试转换为相同类型...")
        if arr1.dtype.kind == 'f' and arr2.dtype.kind == 'f':
            # 都是浮点型，转换为float32
            arr1 = arr1.astype(np.float32)
            arr2 = arr2.astype(np.float32)
        else:
            # 整数类型，转换为int32
            arr1 = arr1.astype(np.int32)
            arr2 = arr2.astype(np.int32)
    
    # 计算差异
    if arr1.dtype.kind == 'f':  # 浮点类型
        diff = np.abs(arr1 - arr2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        # 计算相对误差（避免除零）
        mask = np.abs(arr2) > 1e-10
        if np.any(mask):
            relative_diff = np.abs((arr1[mask] - arr2[mask]) / arr2[mask])
            max_relative_diff = np.max(relative_diff)
            mean_relative_diff = np.mean(relative_diff)
        else:
            max_relative_diff = 0.0
            mean_relative_diff = 0.0
        
        # 统计信息
        print(f"📊 {name}:")
        print(f"   形状: {arr1.shape}")
        print(f"   数据类型: {arr1.dtype}")
        print(f"   数值范围1: [{arr1.min():.6f}, {arr1.max():.6f}]")
        print(f"   数值范围2: [{arr2.min():.6f}, {arr2.max():.6f}]")
        print(f"   最大绝对差异: {max_diff:.6e}")
        print(f"   平均绝对差异: {mean_diff:.6e}")
        print(f"   标准差差异: {std_diff:.6e}")
        print(f"   最大相对差异: {max_relative_diff:.6e}")
        print(f"   平均相对差异: {mean_relative_diff:.6e}")
        
        # 检查匹配
        absolute_match = max_diff <= tolerance
        relative_match = max_relative_diff <= relative_tolerance if np.any(mask) else True
        
        if absolute_match and relative_match:
            print(f"   状态: ✅ 数据一致（容差: abs={tolerance}, rel={relative_tolerance}）")
            return True
        else:
            if not absolute_match:
                print(f"   状态: ❌ 绝对差异超过容差 {tolerance}")
            if not relative_match and np.any(mask):
                print(f"   状态: ❌ 相对差异超过容差 {relative_tolerance}")
            
            # 找出差异最大的位置
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"   最大差异位置: {max_idx}")
            print(f"   值1: {arr1[max_idx]:.6e}")
            print(f"   值2: {arr2[max_idx]:.6e}")
            print(f"   差异: {diff[max_idx]:.6e}")
            
            return False
    else:  # 整数类型
        diff = np.abs(arr1.astype(np.int64) - arr2.astype(np.int64))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff.astype(np.float64))
        num_mismatches = np.sum(diff > 0)
        total_elements = arr1.size
        
        print(f"📊 {name}:")
        print(f"   形状: {arr1.shape}")
        print(f"   数据类型: {arr1.dtype}")
        print(f"   数值范围1: [{arr1.min()}, {arr1.max()}]")
        print(f"   数值范围2: [{arr2.min()}, {arr2.max()}]")
        print(f"   最大差异: {max_diff}")
        print(f"   平均差异: {mean_diff:.2f}")
        print(f"   不匹配元素数: {num_mismatches} / {total_elements} ({100*num_mismatches/total_elements:.2f}%)")
        
        if max_diff == 0:
            print(f"   状态: ✅ 数据完全一致")
            return True
        else:
            print(f"   状态: ❌ 数据不一致")
            # 找出差异最大的位置
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"   最大差异位置: {max_idx}")
            print(f"   值1: {arr1[max_idx]}")
            print(f"   值2: {arr2[max_idx]}")
            print(f"   差异: {diff[max_idx]}")
            return False


def find_matching_files(dir1: str, dir2: str) -> List[Tuple[str, str, str]]:
    """
    在两个目录中查找匹配的文件
    
    Args:
        dir1: 第一个目录（生成的文件）
        dir2: 第二个目录（参考文件）
    
    Returns:
        匹配的文件对列表: [(base_name, file1_path, file2_path), ...]
    """
    matches = []
    
    if not os.path.exists(dir1):
        print(f"❌ 目录不存在: {dir1}")
        return matches
    
    if not os.path.exists(dir2):
        print(f"❌ 目录不存在: {dir2}")
        return matches
    
    # 获取第一个目录中的所有bin文件
    files1 = {}
    for file in os.listdir(dir1):
        if file.endswith('.bin'):
            file_info = parse_filename(file)
            base_name = file_info.get('base_name') or file  # 如果base_name是None，使用文件名
            if base_name:  # 确保base_name不为空
                files1[base_name] = os.path.join(dir1, file)
    
    # 在第二个目录中查找匹配的文件
    for file in os.listdir(dir2):
        if file.endswith('.bin'):
            file_info = parse_filename(file)
            base_name = file_info.get('base_name') or file  # 如果base_name是None，使用文件名
            
            if not base_name:  # 跳过无效的base_name
                continue
            
            if base_name in files1:
                file2_path = os.path.join(dir2, file)
                matches.append((base_name, files1[base_name], file2_path))
            else:
                # 尝试模糊匹配（文件名可能略有不同）
                for base1, path1 in files1.items():
                    if base1 and base_name:  # 确保两个都不为None
                        if base_name in base1 or base1 in base_name:
                            file2_path = os.path.join(dir2, file)
                            matches.append((base_name, path1, file2_path))
                            break
    
    return matches


def filter_files_by_type(matches: List[Tuple[str, str, str]], 
                        file_type: str = 'all') -> List[Tuple[str, str, str]]:
    """
    根据文件类型过滤匹配的文件
    
    Args:
        matches: 匹配的文件对列表
        file_type: 'all'(全部), 'input'(输入), 'output'(输出), 'pred'(预测输出)
    
    Returns:
        过滤后的文件对列表
    """
    if file_type == 'all':
        return matches
    
    filtered = []
    for base_name, file1_path, file2_path in matches:
        filename = os.path.basename(file1_path)
        
        if file_type == 'input':
            # 输入文件：包含 input_ 且不包含 pred_
            if 'input_' in filename and 'pred_' not in filename:
                filtered.append((base_name, file1_path, file2_path))
        elif file_type == 'output' or file_type == 'pred':
            # 输出文件：包含 pred_
            if 'pred_' in filename:
                filtered.append((base_name, file1_path, file2_path))
    
    return filtered


def compare_arrays_with_details(arr1: np.ndarray, arr2: np.ndarray, name: str, 
                                tolerance: float = 1e-6, relative_tolerance: float = 1e-5,
                                is_output: bool = False) -> Tuple[bool, Dict]:
    """
    对比两个numpy数组，返回详细统计信息
    
    Returns:
        (是否匹配, 统计信息字典)
    """
    stats = {
        'max_diff': 0.0,
        'mean_diff': 0.0,
        'std_diff': 0.0,
        'max_relative_diff': 0.0,
        'mean_relative_diff': 0.0,
        'num_mismatches': 0,
        'total_elements': 0,
        'match': False
    }
    
    if arr1 is None or arr2 is None:
        return False, stats
    
    if arr1.shape != arr2.shape:
        return False, stats
    
    stats['total_elements'] = arr1.size
    
    # 检查数据类型并转换
    if arr1.dtype != arr2.dtype:
        if arr1.dtype.kind == 'f' and arr2.dtype.kind == 'f':
            arr1 = arr1.astype(np.float32)
            arr2 = arr2.astype(np.float32)
        else:
            arr1 = arr1.astype(np.int32)
            arr2 = arr2.astype(np.int32)
    
    # 计算差异
    if arr1.dtype.kind == 'f':  # 浮点类型
        diff = np.abs(arr1 - arr2)
        stats['max_diff'] = float(np.max(diff))
        stats['mean_diff'] = float(np.mean(diff))
        stats['std_diff'] = float(np.std(diff))
        
        # 计算相对误差
        mask = np.abs(arr2) > 1e-10
        if np.any(mask):
            relative_diff = np.abs((arr1[mask] - arr2[mask]) / arr2[mask])
            stats['max_relative_diff'] = float(np.max(relative_diff))
            stats['mean_relative_diff'] = float(np.mean(relative_diff))
        
        # 统计不匹配元素
        if is_output:
            # 对于输出，使用更严格的阈值
            mismatch_mask = diff > tolerance
            stats['num_mismatches'] = int(np.sum(mismatch_mask))
        else:
            stats['num_mismatches'] = int(np.sum(diff > tolerance))
        
        absolute_match = stats['max_diff'] <= tolerance
        relative_match = stats['max_relative_diff'] <= relative_tolerance if np.any(mask) else True
        stats['match'] = absolute_match and relative_match
    else:  # 整数类型
        diff = np.abs(arr1.astype(np.int64) - arr2.astype(np.int64))
        stats['max_diff'] = float(np.max(diff))
        stats['mean_diff'] = float(np.mean(diff.astype(np.float64)))
        stats['num_mismatches'] = int(np.sum(diff > 0))
        stats['match'] = (stats['max_diff'] == 0)
    
    return stats['match'], stats


def main():
    parser = argparse.ArgumentParser(description='对比Sparse4D生成的bin文件与参考文件')
    parser.add_argument('--gen_dir', type=str, 
                       default='/share/Code/Sparse4dE2E/C++/Output/1104/',
                       help='生成文件的目录（默认: /share/Code/Sparse4dE2E/C++/Output/1104/）')
    parser.add_argument('--ref_dir', type=str,
                       default='/share/Code/Sparse4dE2E/C++/Output/val_bin_gpu/',
                       help='参考文件目录（默认: /share/Code/Sparse4dE2E/C++/Output/val_bin_gpu/）')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='浮点数绝对容差（默认: 1e-6）')
    parser.add_argument('--relative_tolerance', type=float, default=1e-5,
                       help='浮点数相对容差（默认: 1e-5）')
    parser.add_argument('--file_pattern', type=str, default=None,
                       help='文件匹配模式（可选，用于过滤特定文件）')
    parser.add_argument('--file_type', type=str, default='all',
                       choices=['all', 'input', 'output', 'pred'],
                       help='文件类型过滤：all(全部), input(输入), output/pred(输出)（默认: all）')
    parser.add_argument('--output_tolerance', type=float, default=None,
                       help='输出文件的特殊容差（如果未指定，使用--tolerance）')
    parser.add_argument('--show_statistics', action='store_true',
                       help='显示详细的统计信息')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Sparse4D vs BEV86 数据对比工具")
    print("=" * 80)
    print(f"生成文件目录: {args.gen_dir}")
    print(f"参考文件目录: {args.ref_dir}")
    print(f"绝对容差: {args.tolerance}")
    print(f"相对容差: {args.relative_tolerance}")
    if args.output_tolerance is not None:
        print(f"输出文件容差: {args.output_tolerance}")
    if args.file_type != 'all':
        print(f"文件类型过滤: {args.file_type}")
    if args.show_statistics:
        print("详细统计: 启用")
    print("=" * 80)
    print()
    
    # 查找匹配的文件
    print("🔍 查找匹配的文件...")
    matches = find_matching_files(args.gen_dir, args.ref_dir)
    
    if not matches:
        print("❌ 未找到匹配的文件")
        return
    
    print(f"✅ 找到 {len(matches)} 对匹配的文件")
    
    # 根据文件类型过滤
    if args.file_type != 'all':
        matches = filter_files_by_type(matches, args.file_type)
        print(f"📁 过滤后（类型: {args.file_type}）: {len(matches)} 对文件")
    
    if not matches:
        print("❌ 过滤后没有匹配的文件")
        return
    
    print()
    
    # 对比每一对文件
    results = []
    output_results = []  # 单独统计输出结果
    input_results = []   # 单独统计输入结果
    
    for base_name, file1_path, file2_path in matches:
        # 判断是否为输出文件
        is_output = 'pred_' in os.path.basename(file1_path)
        file_type_label = "📤 输出" if is_output else "📥 输入"
        
        print(f"\n{'='*80}")
        print(f"{file_type_label} 对比文件: {base_name}")
        print(f"  文件1: {file1_path}")
        print(f"  文件2: {file2_path}")
        print(f"{'='*80}")
        
        # 解析文件信息
        file_info = parse_filename(os.path.basename(file1_path))
        dtype = file_info.get('dtype')
        shape = file_info.get('shape')
        
        # 加载文件
        data1 = load_bin_file(file1_path, dtype=dtype, expected_shape=shape)
        data2 = load_bin_file(file2_path, dtype=dtype, expected_shape=shape)
        
        if data1 is None or data2 is None:
            print(f"❌ 无法加载文件，跳过")
            results.append((base_name, False))
            if is_output:
                output_results.append((base_name, False))
            else:
                input_results.append((base_name, False))
            continue
        
        # 根据文件类型选择容差
        tolerance = args.output_tolerance if (is_output and args.output_tolerance is not None) else args.tolerance
        
        if is_output and args.output_tolerance is not None:
            print(f"   使用输出文件容差: {tolerance}")
        
        # 对比数据
        is_match = compare_arrays(data1, data2, base_name, 
                                 tolerance=tolerance,
                                 relative_tolerance=args.relative_tolerance)
        results.append((base_name, is_match))
        
        # 分类统计
        if is_output:
            output_results.append((base_name, is_match))
        else:
            input_results.append((base_name, is_match))
        
        # 如果启用详细统计，显示额外信息
        if args.show_statistics and is_output:
            match_result, stats = compare_arrays_with_details(
                data1, data2, base_name, 
                tolerance=tolerance,
                relative_tolerance=args.relative_tolerance,
                is_output=True
            )
            print(f"   详细统计:")
            print(f"     不匹配元素: {stats['num_mismatches']} / {stats['total_elements']} "
                  f"({100*stats['num_mismatches']/stats['total_elements']:.2f}%)")
            if stats['max_relative_diff'] > 0:
                print(f"     最大相对差异: {stats['max_relative_diff']:.6e}")
    
    # 输出总结
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    
    total = len(results)
    passed = sum(1 for _, match in results if match)
    failed = total - passed
    
    print(f"总文件数: {total}")
    print(f"✅ 匹配: {passed}")
    print(f"❌ 不匹配: {failed}")
    print()
    
    # 分类统计
    if input_results:
        input_passed = sum(1 for _, match in input_results if match)
        input_total = len(input_results)
        print(f"📥 输入文件: {input_passed}/{input_total} 匹配")
    
    if output_results:
        output_passed = sum(1 for _, match in output_results if match)
        output_total = len(output_results)
        print(f"📤 输出文件: {output_passed}/{output_total} 匹配")
        if output_total > 0:
            print(f"   输出匹配率: {100*output_passed/output_total:.1f}%")
    
    print()
    
    if failed > 0:
        print("不匹配的文件:")
        
        # 分类显示
        failed_inputs = [name for name, match in input_results if not match] if input_results else []
        failed_outputs = [name for name, match in output_results if not match] if output_results else []
        
        if failed_inputs:
            print("  📥 输入文件:")
            for name in failed_inputs:
                print(f"    ❌ {name}")
        
        if failed_outputs:
            print("  📤 输出文件:")
            for name in failed_outputs:
                print(f"    ❌ {name}")
        
        # 如果还有其他类型的不匹配文件
        other_failed = [name for name, match in results 
                       if not match and name not in failed_inputs and name not in failed_outputs]
        if other_failed:
            print("  🔍 其他文件:")
            for name in other_failed:
                print(f"    ❌ {name}")
    
    print("=" * 80)
    
    # 返回退出码
    exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()

