"""
SparseBox3DKeyPointsPlugin 包装器

用于在 ONNX 导出时将 SparseBox3DKeyPointsGenerator 替换为 Plugin 版本。
"""
import torch
import torch.nn as nn
from modules.ops.sparse_box3d_keypoints import sparse_box3d_keypoints


class SparseBox3DKeyPointsPluginWrapper(nn.Module):
    """
    包装类：将 SparseBox3DKeyPointsGenerator 替换为 Plugin 版本
    
    这个类在 ONNX 导出时会使用自定义操作符，在 TensorRT 构建时会被替换为 Plugin。
    
    使用方法:
        # 原始代码
        kps_generator = SparseBox3DKeyPointsGenerator(...)
        
        # 替换为 Plugin 版本
        plugin_kps_gen = SparseBox3DKeyPointsPluginWrapper(kps_generator)
        
        # 使用方式相同
        key_points = plugin_kps_gen(anchor, instance_feature)
    """
    
    def __init__(self, kps_generator):
        """
        初始化包装器
        
        Args:
            kps_generator: 原始的 SparseBox3DKeyPointsGenerator 实例
        """
        super().__init__()
        self.embed_dims = kps_generator.embed_dims
        self.num_pts = kps_generator.num_pts
        self.num_learnable_pts = kps_generator.num_learnable_pts
        
        # 提取固定点缩放参数
        # fix_scale 的形状: [num_fix_pts, 3] -> 展平为 [num_fix_pts * 3]
        # 存储为 Python 列表，以便在 ONNX 导出时使用
        fix_scale_data = kps_generator.fix_scale.data.flatten().cpu().tolist()
        self.fix_scale_list = fix_scale_data
        # 同时注册为 buffer，用于 forward 时使用
        self.register_buffer('fix_scale', kps_generator.fix_scale.data.flatten())
        
        # 提取可学习点的权重和偏置
        if self.num_learnable_pts > 0 and hasattr(kps_generator, 'learnable_fc'):
            # learnable_fc 是 nn.Linear(embed_dims, num_learnable_pts * 3)
            # weight: [num_learnable_pts * 3, embed_dims]
            # bias: [num_learnable_pts * 3]
            # 存储为 Python 列表，以便在 ONNX 导出时使用
            self.fc_weight_list = kps_generator.learnable_fc.weight.data.cpu().flatten().tolist()
            self.fc_bias_list = kps_generator.learnable_fc.bias.data.cpu().tolist()
            # 同时注册为 buffer，用于 forward 时使用
            self.register_buffer('fc_weight', kps_generator.learnable_fc.weight.data)
            self.register_buffer('fc_bias', kps_generator.learnable_fc.bias.data)
        else:
            # 如果没有可学习点，设置为 None
            self.fc_weight_list = None
            self.fc_bias_list = None
            self.register_buffer('fc_weight', None)
            self.register_buffer('fc_bias', None)
    
    def forward(self, anchor, instance_feature=None):
        """
        前向传播
        
        Args:
            anchor: [B, N, 11] 锚点
            instance_feature: [B, N, embed_dims] 实例特征（可选）
        
        Returns:
            key_points: [B, N, num_pts, 3] 关键点
        """
        return sparse_box3d_keypoints(
            anchor,
            instance_feature,
            self.embed_dims,
            self.num_pts,
            self.num_learnable_pts,
            self.fix_scale,  # 使用 buffer，forward 时需要 Tensor
            self.fc_weight,  # 使用 buffer，forward 时需要 Tensor
            self.fc_bias,   # 使用 buffer，forward 时需要 Tensor
            self.fix_scale_list,  # Python 列表，用于 ONNX 导出
            self.fc_weight_list,  # Python 列表，用于 ONNX 导出
            self.fc_bias_list,    # Python 列表，用于 ONNX 导出
        )

