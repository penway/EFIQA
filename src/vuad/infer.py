"""
Inference script to process a folder of images using a pre-trained
VUAD model and save the output images to a specified directory.
"""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import cv2 as cv
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser

import torch
from torchvision import transforms

from vuad import model
from vuad.utils.registry import ModelRegistry



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input_dir",
        type=str, required=True,
        help="Path to the input image directory"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Path to the output image directory"
    )
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained model"
    )
    parser.add_argument(
        "-d", "--device",
        type=str, default="cuda",
        help="Device to run the model on (e.g., 'cuda' or 'cpu')"
    )
    parser.add_argument(
        "--model_config",
        type=str, required=True,
        help="Path to the model config file"
    )
    parser.add_argument(
        "--mode",
        type=str, default="hard",
        help="mode: 'hard' or 'soft'"
    )
    parser.add_argument(
        "--vit_size",
        type=int, default=16,
        help="ViT input size"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    args = parser.parse_args()
    
    # get model from cfg
    model_cfg = OmegaConf.load(args.model_config)
    # model_key is in model.key and params are in model.params
    args.model_key = model_cfg.model.key
    args.model_params = model_cfg.model.params

    return args


def load_model(model_key, model_params, model_path, device):
    model = ModelRegistry.get(
        model_key,
        **model_params
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def process(image: np.ndarray, thres: int, size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, thres, 255, cv.THRESH_BINARY)
    image[mask == 0] = 0
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    x = transform(image).unsqueeze(0)
    return x
    

def main():
    args = parse_args()

    fig, axs = None, None
    if args.debug:
        import matplotlib.pyplot as plt
    model = load_model(args.model_key, args.model_params, args.model_path, args.device)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full_res").mkdir(parents=True, exist_ok=True)
    (output_dir / "patch").mkdir(parents=True, exist_ok=True)
    (output_dir / "diff_map").mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.glob("*.*"))
    image_paths.sort()
    print(f"Found {len(image_paths)} images in {input_dir}")

    for image_path in image_paths:
        if args.debug:
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        print(f"Processing {image_path.name}...")
        image = cv.imread(str(image_path))
        if args.debug:
            axs[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            axs[0].set_title("Original Image")
            axs[0].axis('off')
        x = process(image, thres=50, size=256).to(args.device)
        if args.debug:
            axs[1].imshow(x.squeeze(0).permute(1, 2, 0).cpu().numpy())
            axs[1].set_title("Processed Input")
            axs[1].axis('off')
            
        with torch.no_grad():
            y = model(x)

        diff_map = (x - y).abs().mean(1, keepdim=True)
        # save diff map simply to .npy
        diff_map_path = output_dir / "diff_map" / f"{image_path.stem}.npy"
        np.save(str(diff_map_path), diff_map.squeeze(0).squeeze(0).cpu().numpy())
        print(f"Saved diff map to {diff_map_path}")

        if args.debug:
            axs[2].imshow(y.squeeze(0).permute(1, 2, 0).cpu().numpy())
            axs[2].set_title("Model Output")
            axs[2].axis('off')
        thres = 0.02
        if args.mode == "hard":
            binary_map = (diff_map > thres).float()
        else:
            # make 0-0.02 scale to 0-1, >0.02 to 1
            binary_map = torch.clamp((diff_map - thres) / thres, 0, 1)
        binary_map = binary_map.squeeze(0).squeeze(0).cpu().numpy()
        output_image = binary_map * 255
        if args.debug:
            axs[3].imshow(output_image, cmap='gray')
            axs[3].set_title("Anomaly Map")
            axs[3].axis('off')
            plt.show()

        output_path = output_dir / "full_res" / image_path.name
        cv.imwrite(str(output_path), output_image)
        print(f"Saved enhanced image to {output_path}")
        # down sample to for ViT input
        # only turn 255 if all patch is 255, or 0, can be relaxed by 0.9
        vit_size = args.vit_size  # default 16
        relax_ratio = 0.9
        patch_h = output_image.shape[0] // vit_size
        patch_w = output_image.shape[1] // vit_size
        vit_image = np.zeros((vit_size, vit_size), dtype=np.uint8)
        for i in range(vit_size):
            for j in range(vit_size):
                patch = output_image[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                if args.mode == "hard":
                    if np.mean(patch) / 255 >= relax_ratio:
                        vit_image[i, j] = 255
                    else:
                        vit_image[i, j] = 0
                else:
                    vit_image[i, j] = int(np.mean(patch))
        vit_output_path = output_dir / "patch" / f"{image_path.name}"
        cv.imwrite(str(vit_output_path), vit_image)
        print(f"Saved ViT input image to {vit_output_path}")

    print("Processing completed.")


if __name__ == "__main__":
    main()
