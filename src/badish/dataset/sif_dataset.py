import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

from badish.utils.registry import DatasetRegistry

@DatasetRegistry.register("SIFDataset")
class SIFDataset(Dataset):

    """
    SIF: Segmentation-Image-Feature Dataset
    Expect same naming and count in three paths, nothing else.
    """
    IMG_EXTS = (".jpg", ".jpeg", ".png")
    FEAT_EXTS = (".npy", ".npz")

    def __init__(self, feat_dir: Path, seg_dir: Path, img_dir: Path):
        self.feat_dir = Path(feat_dir)
        self.seg_dir  = Path(seg_dir)
        self.img_dir  = Path(img_dir)

        feat_files = self.get_files(self.feat_dir)
        seg_files = self.get_files(self.seg_dir)
        img_files = self.get_files(self.img_dir)
    
        keys = set(feat_files.keys()) & set(seg_files.keys()) & set(img_files.keys())

        # merge to stem: (feat_path, seg_path, img_path)
        self.items = [(feat_files[k], seg_files[k], img_files[k]) for k in sorted(keys)]
        print(f"Found {len(self.items)} samples.")
        
    @staticmethod
    def get_files(dir: Path, exts: tuple = None):
        p = dir.glob('*')
        name_dict = {}
        for f in p:
            if f.is_file() and (
                exts is None or f.suffix.lower() in exts
            ):
                name_dict[f.stem] = f
        return name_dict

    def __len__(self):
        return len(self.items)

    def _load_feat(self, p: Path):
        if p.suffix == ".npy":
            x = np.load(p)
        else:
            raise ValueError(f"Unsupported feature file format: {p.suffix}")
        return x.astype(np.float32)

    def _load_mask(self, p: Path):
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[..., 0]
        return (m > 0).astype(np.uint8)

    def _load_image(self, p: Path):
        img = Image.open(p).convert("RGB")
        return np.array(img)

    def __getitem__(self, idx):
        f_path, s_path, i_path = self.items[idx]
        X = self._load_feat(f_path)  # (H,W,D)
        M = self._load_mask(s_path)  # (H,W)
        I = self._load_image(i_path)  # (H,W,3)
        return I, X, M
