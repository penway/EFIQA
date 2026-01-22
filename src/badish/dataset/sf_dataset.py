import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

from badish.utils.registry import DatasetRegistry


@DatasetRegistry.register("SFDataset")
class SFDataset(Dataset):
    FEAT_EXTS = (".npy", ".npz")
    MASK_EXTS = (".png", ".jpg", ".jpeg", ".tif")

    def __init__(self, feat_dir: Path, seg_dir: Path):
        self.feat_dir = Path(feat_dir)
        self.seg_dir  = Path(seg_dir)

        feat_files = self.get_files(self.feat_dir, self.FEAT_EXTS)
        seg_files  = self.get_files(self.seg_dir,  self.MASK_EXTS)
        keys = sorted(set(feat_files.keys()) & set(seg_files.keys()))

        self.items = [(feat_files[k], seg_files[k]) for k in keys]
        print(f"Found {len(self.items)} samples.")

    @staticmethod
    def get_files(dir: Path, exts: tuple):
        name_dict = {}
        for f in dir.glob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                name_dict[f.stem] = f
        return name_dict

    def __len__(self):
        return len(self.items)

    def _load_feat(self, p: Path):
        x = np.load(p)
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float()  # (D,H,W)

    def _load_mask(self, p: Path):
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 0).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(m)).unsqueeze(0)  # (1,H,W)

    def __getitem__(self, idx):
        f_path, s_path = self.items[idx]
        X = self._load_feat(f_path)
        M = self._load_mask(s_path)
        return X, M
