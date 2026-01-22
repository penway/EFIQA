# seg_dataset: a dataset for the manual score label

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path


from badish.utils.registry import DatasetRegistry

@DatasetRegistry.register("SegDataset")
class SegDataset(Dataset):
    IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    def __init__(self, image_dir, mask_dir, feat_dir, mask_size=16):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.feat_dir = Path(feat_dir)
        self.mask_size = mask_size

        self.mask_paths = [
            p for p in self.mask_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTS
        ]
        self.image_paths = [
            self.image_dir / mask_path.name
            for mask_path in self.mask_paths
        ]
        self.feat_paths = [
            self.feat_dir / (mask_path.stem + '.npy')
            for mask_path in self.mask_paths
        ]
        # confirm existence
        for img_path, feat_path in zip(self.image_paths, self.feat_paths):
            if not img_path.exists():
                print(f'Warning: image {img_path} does not exist!')
            if not feat_path.exists():
                print(f'Warning: feature {feat_path} does not exist!')

    def __len__(self):
        return len(self.image_paths)

    def get_mask(self, mask_path):
        # 4 labels: red: 3, blue: 2, green: 1, black: 0
        mask = Image.open(mask_path).convert('RGB')
        # resize mask to (mask_size, mask_size)
        mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)
        mask = np.array(mask)
        # just get the red channel
        red_channel = mask[:, :, 0]
        blue_channel = mask[:, :, 2]
        green_channel = mask[:, :, 1]
        label_mask = np.zeros((self.mask_size, self.mask_size), dtype=np.uint8)
        label_mask[red_channel > 200] = 3
        label_mask[blue_channel > 200] = 2
        label_mask[green_channel > 200] = 1
        return label_mask

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        feat_path = self.feat_paths[idx]

        image = Image.open(image_path).convert('RGB')
        mask = self.get_mask(mask_path)
        feat = np.load(feat_path)

        transform = transforms.ToTensor()
        image = transform(image)
        mask = torch.from_numpy(mask).long()
        feat = torch.from_numpy(np.ascontiguousarray(feat)).permute(2, 0, 1).float()  # (D,H,W)

        return image, mask, feat


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    # only add one must be specified choose 'mode': 'prepare' or 'test'
    parser.add_argument('mode', type=str,
                        choices=['prepare', 'test'],
                        help='Mode: prepare dataset or test dataset class')
    args = parser.parse_args()
    
    if args.mode == 'test':
        image_dir = Path('/home/penway/Projects/datasets/IQX/PWLQ/EyeQ_sub_proj/Images/')
        mask_dir = Path('/home/penway/Projects/datasets/IQX/PWLQ/EyeQ_sub_proj/Labels/')
        feat_dir = Path('/home/penway/Projects/datasets/IQX/PWLQ/EyeQ_sub_proj/Features/')

        dataset = SegDataset(image_dir, mask_dir, feat_dir, mask_size=16)
        print(f'Dataset size: {len(dataset)}')

        for i in range(5):
            image, mask, feat = dataset[i]
            print(
                f'Image shape: {image.shape}, Mask shape: {mask.shape}, Feat shape: {feat.shape}'
            )
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(image.permute(1, 2, 0).numpy())
            plt.title('Image')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(mask.numpy(), cmap='jet', vmin=0, vmax=3)
            plt.title('Mask')
            plt.axis('off')
            plt.show()

    elif args.mode == 'prepare':
        IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        in_image_dir = Path('/home/penway/Projects/datasets/EyeQ/test/')
        mask_dir = Path('/home/penway/Projects/datasets/IQX/PWLQ/EyeQ_sub_proj/Labels/')
        out_image_dir = Path('/home/penway/Projects/datasets/IQX/PWLQ/EyeQ_sub_proj/Images/')
        feat_dir = Path('/home/penway/Projects/datasets/IQX/PWLQ/EyeQ_sub_proj/Features/')
        feat_16_dir = Path('/home/penway/Projects/datasets/IQX/PWLQ/EyeQ_sub_proj/Features_16/')
        feat_ret_dir = Path('/home/penway/Projects/datasets/IQX/PWLQ/EyeQ_sub_proj/Features_ret/')
        feat_vfm_dir = Path('/home/penway/Projects/datasets/IQX/PWLQ/EyeQ_sub_proj/Features_vfm/')

        # create dataset, copy images to out_image_dir
        mask_paths = [p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        image_paths = [in_image_dir / mask_path.name for mask_path in mask_paths]
        # confirm existence
        for img_path in image_paths:
            if not img_path.exists():
                print(f'Warning: image {img_path} does not exist!')

        # copy images to out_image_dir
        out_image_dir.mkdir(parents=True, exist_ok=True)
        for img_path in image_paths:
            out_path = out_image_dir / img_path.name
            if not out_path.exists():
                os.system(f'cp {img_path} {out_path}')

        print(f'Copied {len(image_paths)} images to {out_image_dir}')

        # excecute feature extraction
        print("Existsing feature DINO 1024 features...")
        feat_dir.mkdir(parents=True, exist_ok=True)
        os.system(f'python -m src.badish.scripts.extract_dino \
            -i {out_image_dir} \
            -o {feat_dir} \
            --size 1024 \
            --model_id facebook/dinov3-vitl16-pretrain-lvd1689m \
            --revision ea8dc2863c51be0a264bab82070e3e8836b02d51')

        print("Extracting feature DINO 16 features...")
        feat_16_dir.mkdir(parents=True, exist_ok=True)
        os.system(f'python -m src.badish.scripts.extract_dino \
            -i {out_image_dir} \
            -o {feat_16_dir} \
            --size 256 \
            --model_id facebook/dinov3-vitl16-pretrain-lvd1689m \
            --revision ea8dc2863c51be0a264bab82070e3e8836b02d51')

        print("Extracting feature RETFound features...")
        feat_ret_dir.mkdir(parents=True, exist_ok=True)
        os.system(f'python -m src.badish.scripts.extract_ret \
                  -i {out_image_dir} \
                  -o {feat_ret_dir} \
                  --size 224 \
                  --model_path /home/penway/Projects/FM/RETFound_MAE/Checkpoints/RETFound_mae_natureCFP.pth')

        print("Extracting feature VFM features...")
        feat_vfm_dir.mkdir(parents=True, exist_ok=True)
        os.system(f'python -m src.badish.scripts.extract_vfm \
                    -i {out_image_dir} \
                    -o {feat_vfm_dir} \
                    --size 224 \
                    --model_path /home/penway/Projects/FM/VisionFM/pretrained_weights/VFM_Fundus_weights.pth')

    else:
        raise NotImplementedError
