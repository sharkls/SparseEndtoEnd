# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch

from typing import Optional
from dataset.config.nusc_std_bbox3d import *


from dataset.config.nusc_std_bbox3d import *


class SparseBox3DDecoder(object):
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sorted: bool = True,
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sorted = sorted

    def decode_box(self, box):
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        box = torch.cat(
            [
                box[:, [X, Y, Z]],
                box[:, [W, L, H]].exp(),
                yaw[:, None],
                box[:, VX:],
            ],
            dim=-1,
        )
        return box

    def decode(
        self,
        cls_scores,         # [1, 900, 10]
        box_preds,          # [1, 900, 11]
        track_id=None,      # [1, 900]
        qulity=None,        # [1,900,2]
        output_idx=-1,
    ):
        squeeze_cls = track_id is not None

        # 调试：在sigmoid之前打印最大的前10个cls_scores值
        print(f"[DEBUG] Top-10 largest cls_scores[output_idx] before sigmoid:")
        cls_scores_flat = cls_scores[output_idx].flatten()
        top_values, top_indices = torch.topk(cls_scores_flat, min(10, cls_scores_flat.numel()))
        for i, (value, idx) in enumerate(zip(top_values, top_indices)):
            print(f"[DEBUG] {i}: value={value:.6f}, flat_idx={idx}")
        print()

        cls_scores = cls_scores[output_idx].sigmoid()   # 1， 900， 10

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape        # 1 ，900，  1
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(     # [1, 300] , [1, 300]
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:                 # 跳过
            cls_ids = indices % num_cls
        if self.score_threshold is not None:    # 跳过
            mask = cls_scores >= self.score_threshold

        if qulity is not None:      # 
            centerness = qulity[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls: # 跳过
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:    # 跳过
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:        # 跳过
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if qulity is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = self.decode_box(box)
            output.append(
                {
                    "boxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids.cpu(),
                }
            )
            if qulity is not None:
                output[-1]["cls_scores"] = scores_origin.cpu()
            if track_id is not None:
                ids = track_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["track_ids"] = ids
        return output
