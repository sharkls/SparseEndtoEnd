"""
替换 kps_generator 为 Plugin 版本的辅助函数

在 ONNX 导出前调用此函数，将所有的 SparseBox3DKeyPointsGenerator 
替换为 SparseBox3DKeyPointsPluginWrapper。
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from modules.head.sparse4d_blocks.sparse3d_keypoints_plugin import (
    SparseBox3DKeyPointsPluginWrapper
)


def replace_kps_generator_with_plugin(head, verbose=True):
    """
    将 head 中所有的 kps_generator 替换为 Plugin 版本
    
    Args:
        head: Sparse4DHead 实例
        verbose: 是否打印替换信息
    
    Returns:
        int: 替换的数量
    """
    replaced_count = 0
    
    # 遍历所有层
    if hasattr(head, 'layers') and head.layers is not None:
        for i, layer in enumerate(head.layers):
            if layer is None:
                continue
            
            # 检查是否有 kps_generator
            if hasattr(layer, 'kps_generator') and layer.kps_generator is not None:
                original_kps_gen = layer.kps_generator
                
                # 检查是否已经是 Plugin 版本
                if isinstance(original_kps_gen, SparseBox3DKeyPointsPluginWrapper):
                    if verbose:
                        print(f"Layer {i}: kps_generator is already a Plugin wrapper")
                    continue
                
                # 创建 Plugin 包装器
                plugin_kps_gen = SparseBox3DKeyPointsPluginWrapper(original_kps_gen)
                
                # 替换
                layer.kps_generator = plugin_kps_gen
                replaced_count += 1
                
                if verbose:
                    print(
                        f"Layer {i}: Replaced kps_generator "
                        f"(num_pts={plugin_kps_gen.num_pts}, "
                        f"num_learnable_pts={plugin_kps_gen.num_learnable_pts})"
                    )
    
    if verbose:
        print(f"\nTotal replaced: {replaced_count} kps_generator(s)")
    
    return replaced_count


def replace_kps_generator_in_model(model, verbose=True):
    """
    在整个模型中替换所有的 kps_generator
    
    Args:
        model: Sparse4D 模型实例
        verbose: 是否打印替换信息
    
    Returns:
        int: 替换的数量
    """
    total_replaced = 0
    
    # 检查 head
    if hasattr(model, 'head'):
        total_replaced += replace_kps_generator_with_plugin(model.head, verbose)
    
    return total_replaced


if __name__ == "__main__":
    """
    测试脚本：验证替换功能
    """
    import torch
    from tool.utils.config import read_cfg
    from typing import Optional, Dict, Any
    
    # 定义 build_module 函数（与导出脚本中相同）
    def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
        cfg2 = cfg.copy()
        if default_args is not None:
            for name, value in default_args.items():
                cfg2.setdefault(name, value)
        type_name = cfg2.pop("type")
        return eval(type_name)(**cfg2)
    
    # 加载配置
    cfg = read_cfg("dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py")
    model = build_module(cfg["model"])
    
    # 替换
    count = replace_kps_generator_in_model(model, verbose=True)
    print(f"\n✓ Successfully replaced {count} kps_generator(s)")

