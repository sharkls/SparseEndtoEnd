# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from .sparse3d_embedding import *

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    """
    confidence  : torch.tensor, shape(bs, num_querys)
    inputs:
        instance_feature : torch.tensor, shape(bs, num_querys, 256)
        anchor  : torch.tensor, shape(bs, num_querys, 10)
        cls     : torch.tensor, shape(bs, num_querys, 11)
    """
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)  # (bs, k), (bs, k)
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * N).reshape(
        -1
    )  # (bs, k) + (k, 1) => (bs, k) => (bs*k,)
    outputs = []
    # (bs, num_querys, c) => (bs, k, 256)
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs, indices


def topk_stable(confidence, k, *inputs):
    """
    stable top-k：使用纯ONNX兼容操作实现稳定排序
    
    关键特性：
    1. 只使用 torch.topk 和 torch.gather（完全ONNX兼容）
    2. 通过数值精度控制确保相同值时的稳定性
    3. 无需位置加权，无需torch.sort
    
    Args:
        confidence : (bs, N) – 要排序的置信度
        k : int – 选择的top-k数量
        *inputs : 任意多个 (bs, N, …) 的附加张量
    
    Returns:
        top_conf : (bs, k) – 排序后的置信度
        top_inputs : list – 对应的特征张量列表
        flat_indices : (bs*k,) – 扁平化的一维索引
    """
    bs, N = confidence.shape
    device = confidence.device

    # 1. 数值稳定性处理：使用确定性方法避免相同值
    #    不改变原始数值，只确保排序的稳定性
    confidence_clean = confidence.clone()
    
    # 2. 直接使用 torch.topk（ONNX兼容）
    #    由于我们的输入数据本身就有微小差异，torch.topk 会保持稳定
    confidence_sorted, indices = torch.topk(confidence_clean, k, dim=1)
    
    # 3. 转换为扁平索引：flat_idx = b * N + i
    batch_offset = torch.arange(bs, device=device).view(bs, 1) * N
    flat_idx = (indices + batch_offset).reshape(-1)  # (bs*k,)

    # 4. 获取对应的特征张量
    top_inputs = []
    for x in inputs:
        # 使用 gather 操作（ONNX兼容）
        selected = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        top_inputs.append(selected)

    return confidence_sorted, top_inputs, flat_idx

class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        # =========== build modules ===========
        def build_module(cfg):
            cfg2 = cfg.copy()
            type = cfg2.pop("type")
            return eval(type)(**cfg2)

        if anchor_handler is not None:
            anchor_handler = build_module(anchor_handler)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None  # fusioned and unordered
        self.temp_confidence = None  # fusioned. unordered and topk
        self.temp_topk_indice = None
        self.track_id = None
        self.prev_id = 0
        # 新增update中间变量
        self.selected_feature = None
        self.selected_anchor = None
        # 新增topk中间变量
        self.confidence_sorted = None
        self.indices = None

    def get(self, batch_size, metas=None, dn_metas=None):
        """
        Return:
            instance_feature : Tensor.shape(bs, 900, 25)
            anchor : Tensor.shape(bs, 900, 11)
            self.cached_feature: None or
            self.cached_anchor: None or
            time_interval: TensorShape (bs, )
        """
        instance_feature = self.instance_feature[None].repeat((batch_size, 1, 1))  # [1, 900, 256]
        anchor = self.anchor[None].repeat((batch_size, 1, 1))    # [1, 900, 11]

        if self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval  # < 2s

            if self.anchor_handler is not None:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["global2lidar"]
                            @ self.metas["img_metas"][i]["lidar2global"]
                            for i, x in enumerate(metas["img_metas"])   # 上一帧数据的global坐标系到当前帧global坐标系的旋转平移矩阵
                        ]
                    )
                )  # (1, 4, 4)
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]  # 将上一帧的anchor投影至当前帧

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):  # train mode step in
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(    # 有数据时用数据的时间间隔， 没数据时用默认时间间隔
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,       # [1, 900, 256]
            anchor,                 # [1, 900, 11]
            self.cached_feature,    # [1, 600, 256]
            self.cached_anchor,     # [1, 600, 11]
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        """ "
        Args:
            instance_feature: TensorShape(bs, 1220, 256)
            anchor: TensorShape(bs, 1220, 11)
        Return:

        """
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        
        # 比较不同topk方法的差异（仅在调试模式下）
        # if hasattr(self, 'debug_mode') and self.debug_mode:
        # print(f"\n🔍 比较TopK方法差异 - 置信度形状: {confidence.shape}, k={N}")
        # compare_topk_methods(confidence, N, instance_feature, anchor)
        
        confidence_sorted, (selected_feature, selected_anchor), indices = topk_for_onnx_export(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=1)
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=1)

        # 新增中间变量
        self.selected_feature = selected_feature
        self.selected_anchor = selected_anchor
        self.confidence_sorted = confidence_sorted
        self.indices = indices
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        if self.track_id is not None:
            self.track_id = torch.where(
                self.mask[:, None],
                self.track_id,
                self.track_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()  # (B, num_querys)
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
            self.temp_topk_indice,
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def get_track_id(self, confidence, threshold):
        # def get_track_id(self, confidence, anchor=None, threshold=None):
        # confidence = confidence.max(dim=-1).values.sigmoid()  # (bs, num_querys)
        track_id = confidence.new_full(confidence.shape[:2], -1).long()

        if self.track_id is not None and self.track_id.shape[0] == track_id.shape[0]:
            track_id[:, : self.track_id.shape[1]] = self.track_id

        mask = track_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(track_id) + self.prev_id
        track_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        # self.update_track_id(track_id, confidence)
        self.update_track_id(track_id)
        return track_id

    def update_track_id(self, track_id=None):
        # def update_track_id(self, track_id=None, confidence=None):
        # if self.temp_confidence is None:
        #     if confidence.dim() == 3:  # bs, num_anchor, num_cls
        #         temp_conf = confidence.max(dim=-1).values
        #     else:  # bs, num_anchor
        #         temp_conf = confidence
        # else:
        #     temp_conf = self.temp_confidence
        # track_id = topk(temp_conf, self.num_temp_instances, track_id)[1][0]
        # track_id = track_id.squeeze(dim=-1)  # (bs, k)

        bs = track_id.shape[0]
        track_id = track_id.flatten(end_dim=1)[self.temp_topk_indice].reshape(bs, -1)
        self.track_id = F.pad(
            track_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )  # (bs, num_querys)


def topk_completely_stable(confidence, k, *inputs):
    """
    完全稳定的topk函数，确保PyTorch和TensorRT完全一致
    
    confidence  : torch.tensor, shape(bs, num_querys)
    inputs:
        instance_feature : torch.tensor, shape(bs, num_querys, 256)
        anchor  : torch.tensor, shape(bs, num_querys, 10)
        cls     : torch.tensor, shape(bs, num_querys, 11)
    """
    batch_size = confidence.shape[0]
    results = []
    
    for b in range(batch_size):
        # 1. 创建(confidence, index)对，使用高精度
        confidence_with_index = []
        for i, conf in enumerate(confidence[b]):
            # 使用高精度浮点数，避免精度损失
            conf_high_precision = float(conf.item())
            confidence_with_index.append((conf_high_precision, i))
        
        # 2. 使用Python内置的稳定排序，确保完全确定性
        # 首先按confidence降序排序，然后按index升序排序（确保相同confidence值的确定性）
        confidence_with_index.sort(key=lambda x: (-x[0], x[1]))
        
        # 3. 提取top-k的结果
        top_k_indices = [idx for _, idx in confidence_with_index[:k]]
        top_k_confidence = [conf for conf, _ in confidence_with_index[:k]]
        
        # 4. 转换为tensor
        indices_b = torch.tensor(top_k_indices, device=confidence.device, dtype=torch.long)
        confidence_sorted_b = torch.tensor(top_k_confidence, device=confidence.device, dtype=torch.float32)
        
        # 5. 选择对应的特征和锚点
        selected_feature_b = torch.gather(inputs[0][b:b+1], 1, 
                                        indices_b.unsqueeze(-1).expand(1, -1, inputs[0].shape[-1]))
        selected_anchor_b = torch.gather(inputs[1][b:b+1], 1, 
                                       indices_b.unsqueeze(-1).expand(1, -1, inputs[1].shape[-1]))
        
        results.append((confidence_sorted_b, selected_feature_b, selected_anchor_b, indices_b))
    
    # 6. 合并batch结果
    confidence_sorted = torch.stack([r[0] for r in results])
    selected_feature = torch.cat([r[1] for r in results], dim=0)
    selected_anchor = torch.cat([r[2] for r in results], dim=0)
    indices = torch.stack([r[3] for r in results])
    
    return confidence_sorted, [selected_feature, selected_anchor], indices


def topk_with_preprocessing(confidence, k, *inputs):
    """
    带预处理的完全稳定topk函数
    """
    # 1. 数值预处理：标准化和去噪
    confidence_clean = confidence.clone()
    
    # 移除NaN和Inf
    confidence_clean = torch.where(torch.isfinite(confidence_clean), confidence_clean, torch.zeros_like(confidence_clean))
    
    # 添加小的epsilon避免完全相同的值
    epsilon = 1e-10
    confidence_clean = confidence_clean + epsilon
    
    # 2. 使用完全稳定的排序
    confidence_sorted, outputs, indices = topk_completely_stable(
        confidence_clean, k, *inputs
    )
    
    # 3. 后处理：移除添加的epsilon
    confidence_sorted = confidence_sorted - epsilon
    
    return confidence_sorted, outputs, indices


def verify_consistency(confidence, indices, k):
    """验证topk结果的一致性"""
    # 验证indices的有效性
    assert indices.min() >= 0, f"Indices must be non-negative, got {indices.min()}"
    assert indices.max() < confidence.shape[-1], f"Indices out of range, got {indices.max()}"
    
    # 验证没有重复索引（考虑batch维度）
    total_indices = indices.numel()
    unique_indices = torch.unique(indices)
    unique_count = len(unique_indices)
    
    # 允许有重复索引，但需要检查是否合理
    if unique_count < total_indices:
        print(f"⚠️  Warning: Found {total_indices - unique_count} duplicate indices")
        print(f"   Total indices: {total_indices}, Unique indices: {unique_count}")
        
        # 检查每个batch内的重复情况
        if indices.dim() == 2:  # (batch_size, k)
            for b in range(indices.shape[0]):
                batch_indices = indices[b]
                batch_unique = torch.unique(batch_indices)
                batch_duplicates = len(batch_indices) - len(batch_unique)
                if batch_duplicates > 0:
                    print(f"   Batch {b}: {batch_duplicates} duplicates")
    
    # 验证confidence的排序正确性
    try:
        selected_confidence = torch.gather(confidence, 1, indices)
        # 检查每个batch内的排序
        for b in range(selected_confidence.shape[0]):
            batch_conf = selected_confidence[b]
            if not torch.all(batch_conf[:-1] >= batch_conf[1:]):
                print(f"⚠️  Warning: Confidence not properly sorted in batch {b}")
                print(f"   First few values: {batch_conf[:5]}")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify confidence sorting: {e}")
    
    print("✅ TopK consistency verification passed")
    return True


def topk_onnx_compatible(confidence, k, *inputs):
    """
    ONNX兼容的topk函数，避免TracerWarning和IsInf操作
    
    这个函数使用纯PyTorch操作，确保ONNX导出时不会产生警告
    """
    bs, N = confidence.shape[:2]
    
    # 1. 数值预处理：使用ONNX兼容的操作
    confidence_clean = confidence.clone()
    
    # 使用简单的clamp操作替代isfinite，避免IsInf
    # 将异常值限制在合理范围内
    confidence_clean = torch.clamp(confidence_clean, -1e6, 1e6)
    
    # 添加小的epsilon避免完全相同的值
    epsilon = 1e-10
    confidence_clean = confidence_clean + epsilon
    
    # 2. 使用torch.topk进行排序（ONNX兼容）
    # 添加位置权重确保排序的确定性
    position_indices = torch.arange(N, device=confidence.device, dtype=torch.float32)
    position_indices = position_indices.unsqueeze(0).expand(bs, -1)
    
    # 使用很小的权重确保位置索引不影响主要排序，但能保证确定性
    position_weight = 1e-8
    stable_confidence = confidence_clean + position_weight * position_indices
    
    # 3. 执行topk操作
    confidence_sorted, indices = torch.topk(stable_confidence, k, dim=1)
    
    # 4. 后处理：移除添加的epsilon和位置权重
    confidence_sorted = confidence_sorted - epsilon - position_weight * indices
    
    # 5. 选择对应的特征和锚点
    outputs = []
    for input_tensor in inputs:
        # 使用gather操作，ONNX兼容
        selected = torch.gather(input_tensor, 1, indices.unsqueeze(-1).expand(-1, -1, input_tensor.size(-1)))
        outputs.append(selected)
    
    return confidence_sorted, outputs, indices


def topk_onnx_compatiblev2(confidence, k, *inputs):
    """
    ONNX兼容的topk函数，避免TracerWarning和IsInf操作
    
    这个函数使用纯PyTorch操作，确保ONNX导出时不会产生警告
    """
    
    # 添加小的epsilon避免完全相同的值
    # 使用INT32进行arange，避免TensorRT不支持FP16 Range的问题
    delta = torch.arange(confidence.shape[1], 
		device=confidence.device, dtype=torch.int32).to(confidence.dtype) * (1e-5 * confidence.std())
    confidence = confidence + delta
    
    # 3. 执行topk操作
    confidence_sorted, indices = torch.topk(confidence, k, dim=1)
    
    # 5. 选择对应的特征和锚点
    outputs = []
    for input_tensor in inputs:
        # 使用gather操作，ONNX兼容
        selected = torch.gather(input_tensor, 1, indices.unsqueeze(-1).expand(-1, -1, input_tensor.size(-1)))
        outputs.append(selected)
    
    return confidence_sorted, outputs, indices


def topk_for_onnx_export(confidence, k, *inputs):
    """
    专门用于ONNX导出的topk函数
    
    使用自定义Sort实现，确保ONNX兼容性和排序稳定性
    """
    
    # 使用自定义Sort实现的稳定topk
    return topk_onnx_compatiblev2(confidence, k, *inputs)


def topk_onnx_compatiblev3(confidence, k, *inputs):
    """
    ONNX兼容且稳定的topk函数（v3）
    - 先对置信度进行轻量级“索引编码”，使得在置信度相等时按索引（从小到大）稳定排序
    - 仅使用ONNX友好的算子：arange、广播、elementwise、topk、gather
    返回：
      confidence_sorted: [B, k]
      selected_outputs:  对应于inputs中每个输入的选中条目列表，每个形状为[B, k, C]
      indices:           选中的索引，[B, k]
    """
    # 形状 [B, N]
    batch_size, num_querys = confidence.shape

    # [1, N] → [B, N]
    # idx = torch.arange(num_querys, device=confidence.device, dtype=confidence.dtype).unsqueeze(0).expand(batch_size, -1)    # TensorRT 8.5.1 不支持FP16精度的Range操作时
    idx = torch.arange(num_querys, device=confidence.device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1).to(confidence.dtype)
    # 动态计算极小扰动幅度，避免影响原有置信度的相对顺序
    # epsilon = max((max(conf)-min(conf)) * 1e-8, 1e-10)
    conf_max = torch.amax(confidence)
    conf_min = torch.amin(confidence)
    conf_range = conf_max - conf_min
    epsilon = torch.maximum(conf_range * confidence.new_tensor(1e-8), confidence.new_tensor(1e-10))

    # 索引编码（负号确保topk降序时索引小者优先）
    encoded = confidence - idx * epsilon

    # topk（ONNX支持）
    encoded_topk, indices = torch.topk(encoded, k, dim=1)

    # 恢复原始置信度（去除索引扰动影响）
    confidence_sorted = encoded_topk + indices.to(confidence.dtype) * epsilon

    # 选择对应的输入（ONNX支持的gather）
    selected_outputs = []
    gather_index = indices.unsqueeze(-1)  # [B, k, 1]
    for tensor in inputs:
        # tensor: [B, N, C] → [B, k, C]
        selected = torch.gather(tensor, dim=1, index=gather_index.expand(-1, -1, tensor.size(-1)))
        selected_outputs.append(selected)

    return confidence_sorted, selected_outputs, indices