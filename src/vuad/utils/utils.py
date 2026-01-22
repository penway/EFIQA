import random
import numpy as np
import torch
from torchvision import transforms as T

def set_seeds(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def on_epoch_start_transform(transform, epoch: int):
    if hasattr(transform, "on_epoch_start"):
        transform.on_epoch_start(epoch)
    if isinstance(transform, T.Compose):
        for t in transform.transforms:
            on_epoch_start_transform(t, epoch)
