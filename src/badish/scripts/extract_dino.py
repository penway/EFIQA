import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import transformers as tr
from transformers import AutoImageProcessor, AutoModel


def list_images(root: Path) -> list[Path]:
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    images = []
    for ext in exts:
        images.extend(root.rglob(f"*{ext}"))
    images = sorted(images)
    return images


def main():
    parser = argparse.ArgumentParser(description="Extract DINO features from images")
    parser.add_argument("-i", "--input_dir", required=True, type=Path, help="Path to input image directory")
    parser.add_argument("-o", "--output_dir", required=True, type=Path, help="Path to output feature directory")
    parser.add_argument("-m", "--model_id", required=True, type=str, help="Model ID for DINO")
    parser.add_argument("-v", "--revision", default=None, type=str, help="Revision for the model")
    parser.add_argument('-s', "--size", type=int, required=True, help="Size to which images will be resized")
    parser.add_argument("--layer", default="last", help="Layer from which to extract features")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 precision, or fp16 if not set")
    parser.add_argument("--save_format", default="npy", choices=["npy", "pt"], help="Format to save features")
    parser.add_argument("--save_cls", action="store_true", help="Also save CLS token features")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use for computation")
    args = parser.parse_args()

    # Load mean and std from DINO
    processor = AutoImageProcessor.from_pretrained(args.model_id, revision=args.revision)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])

    # Load model
    need_hidden = (args.layer != "last")
    model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        output_hidden_states=need_hidden,
    )
    model.eval(); model.requires_grad_(False); model.to(args.device)

    # get patch size from model config, fallback to 16 if not found
    ps = getattr(
        getattr(model.config, "vision_config", model.config),
        "patch_size",
        16, 
    )
    if args.size > 0:
        if args.size % ps != 0:
            raise ValueError(f"Image size {args.size} is not divisible by patch size {ps}")
        H = W = args.size // ps
    else:
        H = W = -1  # variable size, will check later

    # get register token number of patches
    num_reg = getattr(
        getattr(model.config, "vision_config", model.config),
        "num_register_tokens",
        0,
    )

    save_dtype = np.float32 if args.fp32 else np.float16

    # save a metadata file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_file = args.output_dir / "metadata.json"
    meta = {
        "model_id": args.model_id,
        "revision": args.revision,
        "transformers_version": tr.__version__,
        "torch_version": torch.__version__,
        "image_size": args.size,
        "dtype": "fp32" if args.fp32 else "fp16",
        "layer": args.layer,
        "patch_size": ps,
    }
    meta_file.write_text(json.dumps(meta, indent=2))

    # collect images
    images = list_images(args.input_dir)
    if len(images) == 0:
        print(f"No images found in {args.input_dir}")
        return

    if args.size > 0:
        transform = transforms.Compose([
            transforms.Resize((args.size, args.size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    for img_path in tqdm(images, desc="Extracting features", unit="image"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue

        img = transform(img)
        img = img.unsqueeze(0).to(args.device)

        with torch.no_grad():
            out = model(img)
        
        if args.layer == "last":
            tokens = out.last_hidden_state
        else:
            try:
                layer_idx = int(args.layer)
                tokens = out.hidden_states[layer_idx]
            except:
                raise ValueError(f"Invalid layer: {args.layer}")
        
        cls = tokens[:, 0, :]
        patches = tokens[:, 1+num_reg:, :]
        N = patches.shape[1]
        D = patches.shape[-1]
        H = img.shape[2] // ps
        W = img.shape[3] // ps

        expected_N = H * W
        if N != expected_N:
            print(f"Image {img_path} has {N} patches, expected {expected_N}.")
        
        # print(f"input image shape is {img.shape}, extracted feature shape is {tokens.shape} current patch shape is {patches.shape}, expected ({H*W}, {D})")
        patches = patches.reshape(1, H, W, D).squeeze(0).cpu().numpy().astype(save_dtype)

        out_path = args.output_dir / img_path.with_suffix(f".{args.save_format}").name
        # if no cls, save as single array, else save as dict
        if args.save_cls:
            array_to_save = {"cls": cls.squeeze(0).cpu().numpy().astype(save_dtype), "patches": patches}
        else:
            array_to_save = patches

        if args.save_format == "npy":
            np.save(out_path, array_to_save)
        else:  # pt
            torch.save(array_to_save, out_path)

    print(f"Features saved to {args.output_dir}")

if __name__ == "__main__":
    main()
