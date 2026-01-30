"""
SparseBox3DKeyPoints 自定义操作符

用于在 ONNX 导出时将 SparseBox3DKeyPointsGenerator 替换为自定义节点，
在 TensorRT 构建时会被替换为 SparseBox3DKeyPointsPlugin。
"""
import torch
from torch.autograd.function import Function, once_differentiable


class SparseBox3DKeyPointsFunction(Function):
    """
    自定义操作符：将 SparseBox3DKeyPointsGenerator 的操作融合为单个 ONNX 节点
    
    使用方法:
        key_points = SparseBox3DKeyPointsFunction.apply(
            anchor, instance_feature, embed_dims, num_pts, 
            num_learnable_pts, fix_scale, fc_weight, fc_bias
        )
    """
    
    @staticmethod
    def symbolic(
        g,
        anchor,                    # [B, N, 11]
        instance_feature,          # [B, N, embed_dims] 或 None
        embed_dims,                # int: 特征维度
        num_pts,                   # int: 总点数
        num_learnable_pts,         # int: 可学习点数
        fix_scale,                 # Tensor: [num_fix_pts * 3] 固定点缩放
        fc_weight=None,            # Tensor: [num_learnable_pts*3, embed_dims] 或 None
        fc_bias=None,              # Tensor: [num_learnable_pts*3] 或 None
        fix_scale_list=None,       # List[float]: fix_scale 的 Python 列表（用于 ONNX 导出）
        fc_weight_list=None,       # List[float]: fc_weight 的 Python 列表（用于 ONNX 导出）
        fc_bias_list=None,         # List[float]: fc_bias 的 Python 列表（用于 ONNX 导出）
    ):
        """
        symbolic 函数：定义 ONNX 导出时的操作符
        
        这个函数在 ONNX 导出时被调用，创建一个自定义 ONNX 节点。
        TensorRT 在构建引擎时会识别这个节点并替换为 Plugin。
        
        Args:
            g: ONNX 图构建器
            其他参数: 与 forward 相同
        
        Returns:
            ONNX 节点输出
        """
        # 准备输入列表
        inputs = [anchor]
        
        # 如果有 instance_feature，添加到输入
        if instance_feature is not None:
            inputs.append(instance_feature)
        
        # 创建属性（标量参数）
        # 注意：TensorRT Plugin 通过 PluginField 接收这些参数
        # 在 ONNX 中，我们通过节点属性传递
        
        # 将 Tensor 参数转换为常量值
        # 在 ONNX 导出时，模型参数需要转换为常量
        # 注意：在 symbolic 函数中，参数可能是 ONNX 图节点，无法直接访问值
        # 优先使用传入的列表参数（如果提供），否则尝试从 Tensor 转换
        
        # 获取 fix_scale 的值
        if fix_scale_list is None:
            try:
                if isinstance(fix_scale, torch.Tensor):
                    fix_scale_list = fix_scale.detach().cpu().tolist()
                elif isinstance(fix_scale, (list, tuple)):
                    fix_scale_list = list(fix_scale)
            except:
                fix_scale_list = None
        
        # 获取权重和偏置的值
        if fc_weight_list is None and fc_weight is not None:
            try:
                if isinstance(fc_weight, torch.Tensor):
                    fc_weight_list = fc_weight.detach().cpu().flatten().tolist()
                elif isinstance(fc_weight, (list, tuple)):
                    import numpy as np
                    fc_weight_list = np.array(fc_weight).flatten().tolist()
            except:
                fc_weight_list = None
        
        if fc_bias_list is None and fc_bias is not None:
            try:
                if isinstance(fc_bias, torch.Tensor):
                    fc_bias_list = fc_bias.detach().cpu().tolist()
                elif isinstance(fc_bias, (list, tuple)):
                    fc_bias_list = list(fc_bias)
            except:
                fc_bias_list = None
        
        # 创建自定义 ONNX 操作符
        # 操作符名称必须与 TensorRT Plugin 的 getPluginName() 返回的名称匹配
        # 格式: custom::<PluginName>
        # 注意：权重和偏置通过节点属性传递，TensorRT Plugin 会从属性中读取
        # ONNX 属性类型后缀：_i=整数, _f=浮点数, _s=字符串, _t=张量（列表会自动追加 's'）
        # TensorRT Plugin 期望的属性名称：embed_dims, num_pts, num_learnable_pts, fix_scale, fc_weight, fc_bias
        # 在 PyTorch ONNX 导出中，属性名称需要包含类型后缀，但 TensorRT parser 会匹配基础名称
        node_kwargs = {
            "embed_dims_i": embed_dims,  # INT32
            "num_pts_i": num_pts,  # INT32
            "num_learnable_pts_i": num_learnable_pts,  # INT32
        }
        
        # 只有当 fix_scale_list 是有效的 Python 列表时才添加
        if fix_scale_list is not None and isinstance(fix_scale_list, list) and len(fix_scale_list) > 0:
            node_kwargs["fix_scale_f"] = fix_scale_list  # FLOAT32 array
        
        # 如果有可学习点，且权重和偏置是有效的 Python 列表，添加作为属性
        if (fc_weight_list is not None and isinstance(fc_weight_list, list) and len(fc_weight_list) > 0 and
            fc_bias_list is not None and isinstance(fc_bias_list, list) and len(fc_bias_list) > 0):
            node_kwargs["fc_weight_f"] = fc_weight_list  # FLOAT32 array
            node_kwargs["fc_bias_f"] = fc_bias_list  # FLOAT32 array
        
        node = g.op("custom::SparseBox3DKeyPointsPlugin", *inputs, outputs=1, **node_kwargs)
        return node
    
    @staticmethod
    def forward(
        ctx,
        anchor,
        instance_feature,
        embed_dims,
        num_pts,
        num_learnable_pts,
        fix_scale,
        fc_weight=None,
        fc_bias=None,
        fix_scale_list=None,  # 用于 ONNX 导出，forward 中不使用
        fc_weight_list=None,  # 用于 ONNX 导出，forward 中不使用
        fc_bias_list=None,    # 用于 ONNX 导出，forward 中不使用
    ):
        """
        forward 函数：Python 运行时执行（用于验证和测试）
        
        注意：
        1. 在 ONNX 导出时，这个函数不会被调用
        2. 在 Python 推理时，可以使用原始的 Python 实现作为 fallback
        3. 实际部署时应该使用 TensorRT Plugin，而不是这个 fallback
        
        Args:
            ctx: 上下文（用于反向传播）
            anchor: [B, N, 11] 锚点
            instance_feature: [B, N, embed_dims] 实例特征（可选）
            embed_dims: 特征维度
            num_pts: 总点数
            num_learnable_pts: 可学习点数
            fix_scale: [num_fix_pts * 3] 固定点缩放
            fc_weight: [num_learnable_pts*3, embed_dims] 权重（可选）
            fc_bias: [num_learnable_pts*3] 偏置（可选）
        
        Returns:
            key_points: [B, N, num_pts, 3] 关键点
        """
        # 使用原始的 Python 实现作为 fallback
        # 注意：这只是用于验证，实际部署应该使用 Plugin
        from modules.head.sparse4d_blocks.sparse3d_embedding import (
            SparseBox3DKeyPointsGenerator
        )
        from dataset.config.nusc_std_bbox3d import W, L, H, SIN_YAW, COS_YAW
        
        bs, num_anchor = anchor.shape[:2]
        
        # 1. 提取尺寸并计算固定点
        size = anchor[..., None, [W, L, H]].exp()  # [B, N, 1, 3]
        fix_scale_tensor = fix_scale.view(-1, 3)  # [num_fix_pts, 3]
        key_points = fix_scale_tensor[None, None] * size  # [B, N, num_fix_pts, 3]
        
        # 2. 计算可学习点（如果存在）
        if num_learnable_pts > 0 and instance_feature is not None:
            if fc_weight is not None and fc_bias is not None:
                # 使用提供的权重和偏置
                # fc_weight 的形状: [num_learnable_pts * 3, embed_dims]
                # torch.nn.functional.linear 期望 weight: [out_features, in_features]
                # 所以直接使用 fc_weight，不需要转置
                learnable_scale = (
                    torch.nn.functional.linear(instance_feature, fc_weight, fc_bias)
                    .reshape(bs, num_anchor, num_learnable_pts, 3)
                    .sigmoid() - 0.5
                )
                key_points = torch.cat([key_points, learnable_scale * size], dim=-2)
        
        # 3. 构建旋转矩阵并应用旋转
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1
        
        # 应用旋转
        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        )[..., 0]
        
        # 4. 加上中心点
        key_points = key_points + anchor[..., None, :3]
        
        return key_points
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        backward 函数：反向传播
        
        注意：推理时不需要反向传播，这里仅为了完整性
        """
        raise NotImplementedError(
            "SparseBox3DKeyPointsFunction does not support backward. "
            "This function is only for inference."
        )


def sparse_box3d_keypoints(
    anchor,
    instance_feature,
    embed_dims,
    num_pts,
    num_learnable_pts,
    fix_scale,
    fc_weight=None,
    fc_bias=None,
    fix_scale_list=None,
    fc_weight_list=None,
    fc_bias_list=None,
):
    """
    便捷函数：调用自定义操作符
    
    Args:
        anchor: [B, N, 11] 锚点
        instance_feature: [B, N, embed_dims] 实例特征（可选）
        embed_dims: 特征维度
        num_pts: 总点数
        num_learnable_pts: 可学习点数
        fix_scale: [num_fix_pts * 3] 固定点缩放
        fc_weight: [num_learnable_pts*3, embed_dims] 权重（可选）
        fc_bias: [num_learnable_pts*3] 偏置（可选）
    
    Returns:
        key_points: [B, N, num_pts, 3] 关键点
    """
    return SparseBox3DKeyPointsFunction.apply(
        anchor,
        instance_feature,
        embed_dims,
        num_pts,
        num_learnable_pts,
        fix_scale,
        fc_weight,
        fc_bias,
        fix_scale_list,
        fc_weight_list,
        fc_bias_list,
    )

