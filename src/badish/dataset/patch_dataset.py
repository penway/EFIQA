import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from badish.dataset.sif_dataset import SIFDataset
from badish.utils.registry import DatasetRegistry


@DatasetRegistry.register("PatchDataset")
class PatchDataset(Dataset):
    """
    Patch-level dataset from SIFDataset
    Each patch is a sample, with label from mask.
    """
    def __init__(self, patch_dataset_path: Path, balance: bool = False):
        z = np.load(patch_dataset_path)
        self.X = z["X"]
        self.y = z["y"]

        if balance:
            bad = self.X[self.y == 1]
            good = self.X[self.y == 0]
            n = min(len(bad), len(good))
            import random
            bad = bad[random.sample(range(len(bad)), n)]
            good = good[random.sample(range(len(good)), n)]
            self.X = np.vstack([bad, good])
            self.y = np.array([1]*n + [0]*n, dtype=np.uint8)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y


@DatasetRegistry.register("patchdataset_pu")
class PatchDatasetPU(Dataset):
    """
    Dataset for Positive-Unlabeled learning, can designate one class.
    """
    def __init__(
        self,
        patch_dataset_path: Path,
        positive_class: int,
        split: str,
        dtype: str = "float32",
    ):
        assert split in ["P", "U"], "split must be 'P' or 'U'"
        z = np.load(patch_dataset_path)
        X = z["X"]
        y = z["y"]
        
        pos_mask = (y == positive_class)
        if split == "P":
            idx = np.where(pos_mask)[0]
        else:
            idx = np.where(~pos_mask)[0]

        np_dtype = np.float32 if dtype == "float32" else np.float64
        self.X = np.asarray(X[idx], dtype=np_dtype, order="C")


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i])
