from vuad.utils.registry import LossRegistry
from torch import nn

LossRegistry.register(name="mse")(nn.MSELoss)
LossRegistry.register(name="l1")(nn.L1Loss)

