"""
Input: A folder with DINO features. name.npy
Output: A folder with the results:
    - render/ plt plots to visualize the results
    - logits/ npy files with the raw logits
"""

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from badish import model
from badish.utils.registry import ModelRegistry


def parse_args():
    parser = ArgumentParser(description="Run inference using the Badish model.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="A folder with DINO features. name.npy")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output folder to save the results.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the pretrained Badish model.")
    parser.add_argument("-c", "--model_config", type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    args = parser.parse_args()

    model_cfg = OmegaConf.load(args.model_config)
    args.model_key = model_cfg.model.key
    args.model_params = model_cfg.model.params

    return args


def load_model(model_key, model_params, model_path, device):
    model = ModelRegistry.get(model_key, **model_params).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    model = load_model(args.model_key, args.model_params, args.model_path, args.device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "render").mkdir(parents=True, exist_ok=True)
    (output_dir / "logits").mkdir(parents=True, exist_ok=True)

    feature_files = list(input_dir.glob("*.npy"))
    print(f"Found {len(feature_files)} feature files in {input_dir}")

    for feature_file in tqdm(feature_files, desc="Processing feature files"):
        features = np.load(feature_file)  # (H, W, C)
        H, W, C = features.shape

        if args.model_key.startswith("sc"):
            features = features.reshape(-1, C)  # (H*W, C)
            features_tensor = torch.from_numpy(features).float().to(args.device)
            with torch.inference_mode():
                # print(f"features_tensor shape: {features_tensor.shape}")
                logits = model(features_tensor).squeeze().cpu().numpy()  # (H*W,)
            logits = logits.reshape(H, W)  # (H, W)
        else:
            features = features.transpose(2, 0, 1)  # (C, H, W)
            features_tensor = torch.from_numpy(features).unsqueeze(0).float().to(args.device)  # (1, C, H, W)
            with torch.inference_mode():
                logits = model(features_tensor).squeeze().cpu().numpy()  # (H, W)

        np.save(output_dir / "logits" / f"{feature_file.stem}.npy", logits)
        # Save visualization
        
        # cmap = plt.get_cmap("viridis")
        # colored_logits = (cmap(norm(logits))[:, :, :3] * 255).astype(np.uint8)
        # Image.fromarray(colored_logits).save(output_dir / "render" / f"{feature_file.stem}_logits.png")

        cmap = plt.get_cmap("viridis")
        # pass logits through a sigmoid
        probs = 1 / (1 + np.exp(-logits))
        # no normalization, but fix colormap to be between 0 and 1 to see the real probabilities
        colored_probs = (cmap(probs)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(colored_probs).save(output_dir / "render" / f"{feature_file.stem}.png")


if __name__ == "__main__":
    main()
