import torch
from torch.autograd.function import Function, once_differentiable

from . import e2e_deformable_aggregation_ext


class DeformableAggregationFunction(Function):
    @staticmethod
    def symbolic(   # 符号函数: 构建DFA自定义算子PyTorch 到 ONNX 映射规则
        g, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights
    ):
        return g.op(
            "custom::DeformableAttentionAggrPlugin",
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )

    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        # output: [bs, num_pts, num_embeds]
        # 确保输入张量是连续的且类型正确
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        # 调用CUDA实现的前向传播
        output = e2e_deformable_aggregation_ext.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )

        # 保存用于反向传播的张量
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # 获取保存的张量
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors

        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        # 初始化梯度
        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)

        #调用CUDA实现的反向传播
        e2e_deformable_aggregation_ext.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )
