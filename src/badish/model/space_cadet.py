import torch
from matplotlib import pyplot as plt


class SpaceCadet(torch.nn.Module):
    """
    Base class for space cadet models.
    These models are designed to explore the latent space.
    They only get input of one patch of embedding, such as 1024-dim vector, no image.
    """
    def __init__(self):
        super().__init__()
        self.model: torch.nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expect correct usage by me, no need to check
        # if self.model is None:
        #     raise ValueError("Model not defined.")
        # if x.dim() not in (1, 2):
        #     raise ValueError(f"Input tensor must be 1D or 2D. Got {x.dim()}D.")
        return self.model(x)

    def _image_apply(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not defined.")

        if x.dim() in (1, 2):
            return self.model(x)

        if x.dim() == 4:
            y = self._image_apply(x)
            if y is not None:
                return y
            # fallback
            b, c, h, w = x.shape
            x2 = x.permute(0, 2, 3, 1).contiguous().view(-1, c)
            y2 = self.model(x2)
            o = y2.shape[-1]
            return y2.view(b, h, w, o).permute(0, 3, 1, 2).contiguous()

        raise ValueError(f"Input tensor must be 1D, 2D or 4D. Got {x.dim()}D.")

    @torch.inference_mode()
    def visualize(self, Is, Xs, Ms) -> plt.Figure | None:
        raise NotImplementedError("Visualization not implemented for base class.")


