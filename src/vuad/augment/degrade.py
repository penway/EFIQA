import torch
from torch import nn, Tensor
from torch.nn import functional as F
from collections.abc import Sequence

from vuad.utils.registry import DegradeRegistry 


@DegradeRegistry.register(name="degrade")
class Degrade(nn.Module):
    """
    Degregate module for vessel segmentation map.
    Patch range: (min_patch_num, max_patch_num) example: (4, 8) means 4x4 to 8x8.
    Ratio range: (min_ratio, max_ratio) example: (0.1, 0.3) means 10% to 30%.
    p: probability of applying degregate.
    generator: torch.Generator for reproducibility.
    """
    def __init__(self, 
                 patch_range: Sequence[int],
                 ratio_range: Sequence[float],
                 p:           float = 0.5,
                 generator:   torch.Generator | None = None,
                 return_mask: bool = False,
                 ) -> None:
        super().__init__()
        assert len(patch_range) == 2, "patch_range should be a tuple/list of (min_patch_num, max_patch_num)"
        assert len(ratio_range) == 2, "ratio_range should be a tuple/list of (min_ratio, max_ratio)"
        self.min_patch_num, self.max_patch_num = patch_range
        self.min_ratio, self.max_ratio = ratio_range
        self.p = p
        self.generator = generator
        self.return_mask = return_mask

    @torch.no_grad()
    def forward(self, x:Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass to degregate the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) or (C, H, W).
        
        Returns:
            torch.Tensor: Degregated tensor of shape (B, C, H, W) or (C, H, W).
        """
        assert x.dim() in [3, 4], "Input tensor must be 3D or 4D"
        batch_mode = (x.dim() == 4)
        if not batch_mode:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape
        device = x.device
        g = self.generator
        apply = torch.rand((B,), device=device, generator=g) < self.p

        patch = torch.randint(self.min_patch_num, self.max_patch_num + 1, (B,),
                              generator=g, device=device)
        ratio = torch.empty((B,), device=device).uniform_(
                              self.min_ratio, self.max_ratio, generator=g)

        out = x.clone()

        if self.return_mask:
            masks = torch.ones_like(x)
        
        for psize in patch.unique().tolist():
            # find the batch indices with the same patch size
            idx = (patch == psize) & apply
            if not idx.any():
                continue
            n = int(idx.sum().item())
            r = ratio[idx].view(n, 1, 1, 1)

            # random generate what to keep by Bernoulli distribution
            keep = (torch.rand((n, 1, psize, psize), device=device, generator=g) > r).float()
            mask = F.interpolate(keep, size=(H, W), mode='nearest')
            mask = mask.expand(-1, C, -1, -1)

            # apply mask
            bx = out[idx]
            bx = bx * mask
            out[idx] = bx
            if self.return_mask:
                masks[idx] = mask

        if not batch_mode:
            out = out.squeeze(0)

        if self.return_mask:
            if not batch_mode:
                masks = masks.squeeze(0)
            return out, masks
        return out


@DegradeRegistry.register(name="progressive")
class ProgressiveDegrade(Degrade):
    """
    Degrade module with progressive ratio decrease to encourage learning.
    Usually, the model will fail to learn, and learn identity, if the ratio
    is too low. As the model can get low enough loss.
    So we have to start with strong masking. But with only strong masking,
    the model will overfit, and try to fill any black region, not recognizing
    the region that covered with sparse vessels.
    So we progressively decrease the ratio to encourage learning.

    Default setting: start with [0.8, 1], and progressively decrease to [0, 1].
    """
    def __init__(self, 
                 patch_range: Sequence[int],
                 init_ratio_range: Sequence[float] = (0.8, 1.0),
                 final_ratio_range: Sequence[float] = (0.0, 1.0),
                 transition_start: int = 10,
                 transition_end:   int = 100,
                 p:           float = 0.5,
                 generator:   torch.Generator | None = None,
                 return_mask: bool = False,
                 ) -> None:
        super().__init__(patch_range, init_ratio_range, p, generator, return_mask)
        assert len(final_ratio_range) == 2, "final_ratio_range should be a tuple/list of (min_ratio, max_ratio)"
        self.init_min_ratio, self.init_max_ratio = init_ratio_range
        self.final_min_ratio, self.final_max_ratio = final_ratio_range
        self.transition_start = transition_start
        self.transition_end = transition_end

    def on_epoch_start(self, epoch:int):
        """
        Call this function at the start of each epoch to update the ratio range.
        """
        if epoch < self.transition_start:
            self.min_ratio, self.max_ratio = self.init_min_ratio, self.init_max_ratio
        elif epoch >= self.transition_end:
            self.min_ratio, self.max_ratio = self.final_min_ratio, self.final_max_ratio
        else:
            alpha = (epoch - self.transition_start) / (self.transition_end - self.transition_start)
            self.min_ratio = self.init_min_ratio * (1 - alpha) + self.final_min_ratio * alpha
            self.max_ratio = self.init_max_ratio * (1 - alpha) + self.final_max_ratio * alpha


if __name__ == "__main__":
    from matplotlib import pyplot as plt 
    generator = torch.Generator().manual_seed(36)

    degregate = Degrade([4, 16], [0.5, 0.9], 1, generator, return_mask=True)
    batch_size = 16
    x = torch.rand(batch_size, 3, 256, 256)
    fig, axs = plt.subplots(4, 8, figsize=(8, 8))
    y, m = degregate(x)
    for i in range(batch_size):
        axs[i//4, (i%4)*2].imshow(y[i].permute(1, 2, 0).cpu())
        axs[i//4, (i%4)*2+1].imshow(m[i].permute(1, 2, 0).cpu(), cmap='gray')
        axs[i//4, (i%4)*2].axis('off')
        axs[i//4, (i%4)*2+1].axis('off')
    plt.show()

    # speed test
    import time
    batch_size = 64
    generator = torch.Generator(device='cuda').manual_seed(36)
    x = torch.rand(batch_size, 3, 512, 512).cuda()
    degregate = Degrade((4, 16), (0.1, 0.8), 0.5, generator).cuda()
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        y = degregate(x)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"Speed: {1000/(t2-t1):.2f} FPS")
