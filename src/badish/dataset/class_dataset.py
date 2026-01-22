# class dataset, a dataset for loading classification data

import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

from badish.utils.registry import DatasetRegistry

@DatasetRegistry.register("ClassDataset")
class ClassDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        label_path: str | Path,
        expected_label: dict[int, tuple[int, str]] | dict[str, tuple[int, str]] = {0:(0, 'bad'), 1:(1, 'good')},
        file_col: str = 'file',
        label_col: str = 'quality',
    ):
        """
        Dataset for Badish Logit model.
        Args:
            data_path (str | Path): Path to the directory containing .npy feature files.
            label_path (str | Path): Path to the CSV file containing labels.
            expected_label (dict): Mapping of label values to (int, str) tuples.
        """
        self.data_path = Path(data_path)
        self.label_path = Path(label_path)
        self.expected_label = expected_label
        labels = pd.read_csv(self.label_path)[[file_col, label_col]]
        labels = labels.rename(columns={file_col: 'file', label_col: 'quality'})
        self.data = []

        for row in labels.iterrows():
            file_stem = Path(row[1]['file']).stem
            quality = row[1]['quality']
            data_file = self.data_path / f"{file_stem}.npy"
            if data_file.exists() and quality in expected_label:
                # lazy loading, load data when accessed
                self.data.append((data_file, expected_label[quality][0]))
            elif not data_file.exists():
                print(f"Warning: Data file {data_file} does not exist.")
            elif quality not in expected_label:
                print(f"Warning: Quality label {quality} not in expected labels.")
            else:
                print(f"Warning: Unexpected issue with file {data_file} and quality {quality}. Check the code.")


        print(f"Loaded {len(self.data)} samples. Expected samples: {len(labels)}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_file, label = self.data[idx]
        data = np.load(data_file)
        data_tensor = torch.from_numpy(data).permute(2, 0, 1).float()
        label_tensor = torch.tensor(label).long()
        return data_tensor, label_tensor
    
    def get_with_image(self, idx, image_folder: str | Path):
        data_file, label = self.data[idx]
        data = np.load(data_file)
        data_tensor = torch.from_numpy(data).permute(2, 0, 1).float()
        label_tensor = torch.tensor(label).long()
        image_file = Path(image_folder) / f"{data_file.stem}.png"
        return data_tensor, label_tensor, image_file
 

@DatasetRegistry.register("SimpleFeatureDataset")
class SimpleFeatureDataset(Dataset):
    def __init__(self, feat_dir):
        self.feat_dir = Path(feat_dir)
        self.feat_paths = list(self.feat_dir.glob('*.npy'))

    def __len__(self):
        return len(self.feat_paths)

    def __getitem__(self, idx):
        feat_path = self.feat_paths[idx]
        feat = np.load(feat_path)
        feat_tensor = torch.from_numpy(feat).permute(2, 0, 1).float()
        return feat_tensor
