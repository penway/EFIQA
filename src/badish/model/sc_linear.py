import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from badish.utils.registry import ModelRegistry
from badish.model.space_cadet import SpaceCadet


@ModelRegistry.register(name="sc_linear")
class SpaceCadetLinear(SpaceCadet):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = torch.nn.Linear(input_dim, output_dim)

    def _image_apply(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        W = self.model.weight.to(device=x.device, dtype=x.dtype).view(-1, c, 1, 1)
        b = (
            None if self.model.bias is None
            else self.model.bias.to(device=x.device, dtype=x.dtype)
        )
        return F.conv2d(x, W, b)

    @torch.inference_mode()
    def visualize(
        self,
        Is: list[np.ndarray],
        Xs: torch.Tensor,
        Ms: torch.Tensor,
        img_cols: int = 3,
    ) -> plt.Figure:
        """
        Is: list of HxWx3 images (np or torch), can be heterogeneous
        Xs: (N, input_dim) torch tensor of input embeddings
        Ms: (N, H, W) torch tensor of masks
        len(Is) == N == Xs.shape[0] == Ms.shape[0]
        img_cols: number of image columns in the grid, 
                  default 3 means 3 set of (image, mask, pred) making a 9-column grid
        """
        N = len(Is)
        assert N == Xs.shape[0] == Ms.shape[0], "Mismatched input lengths."

        self.model.eval()
        preds = self.model(Xs).squeeze().cpu().numpy()
        # take pred to sigmoid
        preds = 1 / (1 + np.exp(-preds))
        Ms = Ms.cpu().numpy()

        # ensure numpy
        Is_np = [
            i.detach().cpu().numpy() if hasattr(i, "detach") else np.asarray(i)
            for i in Is
        ]

        cols = img_cols * 3
        rows = N // img_cols + (1 if N % img_cols != 0 else 0)

        # vim = preds.min()
        # vam = preds.max()
        # vmax_abs = max(abs(vim), abs(vam))
        vim, vam = 0, 1

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=100,
                                 squeeze=False)

        for r in range(rows):
            for c in range(img_cols):
                i = r * img_cols + c
                if i >= N:  # out of range, fill with empty
                    for subc in range(3):
                        axes[r, c * 3 + subc].axis("off")
                    continue  

                # Image
                ax = axes[r, c * 3 + 0]
                ax.imshow(Is_np[i])
                ax.set_title(f"Image #{i}")
                ax.axis("off")

                # Mask
                ax = axes[r, c * 3 + 1]
                ax.imshow(Ms[i], cmap="gray", vmin=0, vmax=1 if Ms[i].max() <= 1 else Ms[i].max())
                ax.set_title("Mask")
                ax.axis("off")

                # Prediction
                ax = axes[r, c * 3 + 2]
                ax.imshow(preds[i], cmap="coolwarm", vmin=vim, vmax=vam)
                ax.set_title("Prediction")
                ax.axis("off")

        # sm = ScalarMappable(cmap="coolwarm", norm=Normalize(vmin=-vmax_abs, vmax=vmax_abs)) 
        sm = ScalarMappable(cmap="coolwarm", norm=Normalize(vmin=vim, vmax=vam))
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, location="right")

        return fig


