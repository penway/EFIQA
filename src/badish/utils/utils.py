import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seeds(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_val_grid(writer, epoch, Is, Ms, preds, vmax_abs=5.0, tag="Val/Grid"):
    """
    Is: list of HxWx3 images (np or torch)
    Ms: (N, H, W) numpy array (or list) of masks
    preds: (N, H, W) numpy array of predicted maps
    """
    # ensure numpy
    Is_np = [i.detach().cpu().numpy() if hasattr(i, "detach") else np.asarray(i) for i in Is]
    Ms_np = Ms.detach().cpu().numpy() if hasattr(Ms, "detach") else np.asarray(Ms)
    preds_np = preds.detach().cpu().numpy() if hasattr(preds, "detach") else np.asarray(preds)

    n = len(Is_np)
    cols = 3
    rows = n

    # Consistent color scale for preds
    if vmax_abs is None:
        vmax_abs = float(np.nanpercentile(np.abs(preds_np), 99))
        if vmax_abs == 0: vmax_abs = 1.0
    vmin, vmax = -vmax_abs, vmax_abs

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=100,
                             squeeze=False)
    for r in range(rows):
        # Image
        ax = axes[r, 0]
        ax.imshow(Is_np[r])
        ax.set_title(f"Image #{r}")
        ax.axis("off")

        # Mask
        ax = axes[r, 1]
        ax.imshow(Ms_np[r], cmap="gray", vmin=0, vmax=1 if Ms_np[r].max() <= 1 else Ms_np[r].max())
        ax.set_title("Mask")
        ax.axis("off")

        # Prediction
        ax = axes[r, 2]
        im = ax.imshow(preds_np[r], cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title("Prediction")
        ax.axis("off")

    # One shared colorbar for predictions
    cbar = fig.colorbar(im, ax=axes[:, 2].ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Pred score")

    plt.tight_layout()
    writer.add_figure(tag, fig, global_step=epoch, close=True)
