import torch

class BaseRegistry:
    _registry = {}
    ITEM_TYPE = "item_type"
    
    @classmethod
    def register(
        cls,
        name: str | None = None,
        *,
        overwrite: bool = False,
    ):

        def decorator(obj):
            key = (name or obj.__name__).lower()
            if not overwrite and key in cls._registry:
                raise KeyError(f"{cls.ITEM_TYPE} '{key}' already registered.")
            cls._registry[key] = obj
            return obj

        return decorator

    @classmethod
    def get(cls, name: str, *args, **kwargs):
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"{cls.ITEM_TYPE} '{name}' not in registry.")
        target = cls._registry[key]
        return target(*args, **kwargs)

    @classmethod
    def has(cls, name: str) -> bool:
        return name.lower() in cls._registry

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())

class ModelRegistry(BaseRegistry):
    ITEM_TYPE = "Model"

class LossRegistry(BaseRegistry):
    ITEM_TYPE = "Loss"

class OptimRegistry(BaseRegistry):
    ITEM_TYPE = "Optim"

class DatasetRegistry(BaseRegistry):
    ITEM_TYPE = "Dataset"

class TrainerRegistry(BaseRegistry):
    ITEM_TYPE = "Trainer"

LossRegistry._registry = {
    "bce": torch.nn.BCEWithLogitsLoss,
}

@LossRegistry.register("bce_pos_weight")
class BCEWithLogitsLossPosWeight(torch.nn.BCEWithLogitsLoss):
    def __init__(self, pos_weight: float = 1.0, **kwargs):
        super().__init__(pos_weight=torch.tensor(pos_weight), **kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F

LossRegistry._registry = {
    "bce": nn.BCEWithLogitsLoss,
}

@LossRegistry.register("bce_pos_weight")
class BCEWithLogitsLossPosWeight(nn.BCEWithLogitsLoss):
    """
    Standard BCEWithLogitsLoss with a scalar positive class weight.
    """
    def __init__(self, pos_weight: float = 1.0, **kwargs):
        # turn scalar into tensor for torch
        pos_weight_tensor = torch.tensor(pos_weight)
        super().__init__(pos_weight=pos_weight_tensor, **kwargs)


@LossRegistry.register("focal_bce")
class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal loss for binary classification/segmentation with logits input.

    Args:
        alpha (float): weight for the positive class (similar to pos_weight but softer).
        gamma (float): focusing parameter; gamma=0 -> BCE.
        reduction (str): 'none' | 'mean' | 'sum'.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input:  logits, any shape
        target: same shape, values in {0,1} (or probabilities in [0,1])
        """
        # per-element BCE with logits (no reduction)
        bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")

        # pt = exp(-bce) because bce = -log(pt)
        pt = torch.exp(-bce)

        # focal term: alpha * (1 - pt)^gamma * BCE
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


OptimRegistry._registry = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

