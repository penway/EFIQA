import numpy as np
from tqdm import tqdm
from pathlib import Path

from badish.dataset.sif_dataset import SIFDataset


def prepare_patch_dataset(sif_dataset: SIFDataset, output_path: Path):
    X_all = []
    M_all = []
    for _, X, M in tqdm(sif_dataset, desc="Preparing patch dataset"):
        H, W, D = X.shape
        Xf = X.reshape(-1, D)
        Mf = M.reshape(-1)
        for i in range(len(Mf)):
            X_all.append(Xf[i])
            M_all.append(Mf[i])

    X_all = np.array(X_all, dtype=np.float32)
    M_all = np.array(M_all, dtype=np.uint8)

    return X_all, M_all


if __name__ == "__main__":

    feature_dir = Path("/home/penway/Projects/FIQA/IQX/data/MSHF/emb_dino/")
    seg_dir = Path("/home/penway/Projects/FIQA/IQX/data/MSHF/seg/vuad/patch")
    img_dir = Path("/home/penway/Projects/datasets/IQX/MSHF/images")

    sif_dataset = SIFDataset(feature_dir, seg_dir, img_dir)
    output_path = Path("/home/penway/Projects/FIQA/IQX/data/MSHF/patch/mshf_patch_dataset.npz")
    X_all, M_all = prepare_patch_dataset(sif_dataset, output_path)

    print(f"Saving dataset with {len(M_all)} patches, feature dim: {X_all.shape[1]}")
    np.savez_compressed(output_path, X=X_all, y=M_all)
    print(f"Saved patch dataset to {output_path}, total patches: {len(M_all)}")

