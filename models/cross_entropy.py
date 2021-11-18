__all__ = ["CrossEntropy"]

import torch
import torch.nn as nn


class CrossEntropy(nn.Module):

    def __init__(self,
                 weight=None,
                 reduction='mean',
                 smoothing=None,
                 ignore_index=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

        if self.smoothing is not None and not (0 <= self.smoothing < 1):
            raise ValueError(f"smoothing value must be "
                             f"between 0 and 1: {self.smoothing}")

    def _smoothen(self, target: torch.FloatTensor):
        if not self.smoothing:
            return target
        alpha = self.smoothing
        num_classes = target.size(-1)
        new_targets = target * (1 - alpha) + alpha / num_classes
        return new_targets

    def _prepare_targets_and_mask(self, target: torch.Tensor, num_classes):
        # ensure that the target is in the distributed form
        if target.type().endswith("LongTensor"):
            if self.ignore_index is not None:
                ignore_mask = target == self.ignore_index
                target = target.masked_fill(ignore_mask, 0)
            else:
                ignore_mask = target.clone().bool().zero_()
            new_target = (target.new(*target.size(), num_classes).float()
                          .zero_().scatter(-1, target.unsqueeze(-1), 1.0))
            ignore_mask = ignore_mask.unsqueeze(-1).expand_as(new_target)
            new_target = new_target.masked_fill(ignore_mask, 0)

        elif target.type().endswith("FloatTensor"):
            ignore_mask = target.clone().bool().zero_()
            if (self.ignore_index is not None and
                    0 <= self.ignore_index < target.size()[-1]):
                ignore_mask[..., self.ignore_index] = 1
            new_target = target.masked_fill(ignore_mask, 0)
            # normalize
            new_target = new_target / new_target.sum(-1).unsqueeze(-1)

        else:
            raise TypeError(f"unexpected tensor type: {target.type()}")

        return new_target, ignore_mask

    def _forward_dist(self, logit, target: torch.FloatTensor):
        assert logit.size(-1) == target.size(-1)

    def forward(self, logit, target: torch.Tensor):
        num_classes = logit.size(-1)

        with torch.no_grad():
            new_target, ignore_mask = \
                self._prepare_targets_and_mask(target, num_classes)

            # apply probability smoothing (generalized form of label smoothing)
            new_target = self._smoothen(new_target)

        log_prob = torch.log_softmax(logit, -1)

        # apply class weighting
        if self.weight is not None:
            log_prob = log_prob * self.weight

        loss = -(new_target * log_prob)
        loss = loss.masked_fill(ignore_mask, 0).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"unexpected reduction type: {self.reduction}")

        return loss
