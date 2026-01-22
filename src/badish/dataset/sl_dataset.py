import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from omegaconf.listconfig import ListConfig

from badish.utils.registry import DatasetRegistry


@DatasetRegistry.register("SLDataset")
class SLDataset(Dataset):
    FEAT_EXTS = (".npy", ".npz")
    MASK_EXTS = (".npy", ".npz", ".png", ".jpg", ".jpeg")

    def __init__(
        self,
        feat_dir: list | ListConfig | Path | str,
        seg_dir: list | ListConfig | Path | str,
    ):
        if not isinstance(feat_dir, list | ListConfig):
            assert not isinstance(seg_dir, list | ListConfig), "If feat_dir is a path, seg_dir must also be a path."
            self.feat_dir = Path(feat_dir)
            self.seg_dir  = Path(seg_dir)

            feat_files = self.get_files(self.feat_dir, self.FEAT_EXTS)
            seg_files  = self.get_files(self.seg_dir,  self.MASK_EXTS)
        
        else:
            assert isinstance(seg_dir, list | ListConfig), "If feat_dir is a list, seg_dir must also be a list."
            # turn dir from ConfigList[str] to list[Path]
            feat_dir = [Path(d) for d in feat_dir]
            seg_dir  = [Path(d) for d in seg_dir]

            feat_files = {}
            for d in feat_dir:
                feat_files.update(self.get_files(d, self.FEAT_EXTS))
            seg_files = {}
            for d in seg_dir:
                seg_files.update(self.get_files(d, self.MASK_EXTS))

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
        if p.suffix in (".png", ".jpg", ".jpeg"):
            from PIL import Image
            m = np.array(Image.open(p))
            if m.ndim == 3:
                m = m[..., 0]
            m = torch.from_numpy(np.ascontiguousarray(m)).unsqueeze(0).float()  # (1,H,W)
            m = (m > 0).float()
            return m
        elif p.suffix not in (".npy", ".npz"):
            m = np.load(p)
            m = torch.from_numpy(np.ascontiguousarray(m)).unsqueeze(0).float()  # (1,H,W)
            m = torch.sigmoid(m)
            thres = 0.7
            m = (m > thres).float()  # binarize
            return m
        else:
            raise ValueError(f"Unsupported mask file format: {p.suffix}")

    def __getitem__(self, idx):
        f_path, s_path = self.items[idx]
        X = self._load_feat(f_path)
        M = self._load_mask(s_path)
        return X, M
