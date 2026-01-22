import torch
import math
from PIL import Image, ImageDraw


def random_fov(
    image: Image.Image,
    p: float,
    ratio_range: tuple[float, float], 
    pixel_limits: int,
) -> Image.Image:
    """
    Randomly crop the image to simulate different fields of view (FOV).
    args:
        image: input PIL image to be cropped
        p: probability of applying the random FOV crop
        ratio_range: tuple (min_ratio, max_ratio) to determine the scale ratio of the crop
        pixel_limits: minimum pixel size for the cropped image, will be useful if minimum ratio is small
    """
    if torch.rand(1).item() > p:
        return image

    W, H = image.size # width, height
    R_O = W / 2       # original radius
    min_ratio, max_ratio = ratio_range
    
    # sanity checks
    assert 0 < min_ratio <= max_ratio <= 1, \
                    "ratio_range should be in (0, 1] and min_ratio <= max_ratio"
    assert pixel_limits > 0, "pixel_limits should be positive"
    assert abs(H - W) < 1, "image should be square"
    if min_ratio * W < pixel_limits:
        min_ratio = pixel_limits / W
    
    ratio = min_ratio + (max_ratio - min_ratio) * torch.rand(1).item()
    R_N = R_O * ratio
    
    # sample C_N, ||C_N - C_O|| <= ||R_O - R_N||
    r_s = R_O - R_N # sample radius
    cox, coy = W / 2, H / 2 # original center
    # sample center using polar coordinates
    d = r_s * math.sqrt(torch.rand(1).item())
    theta = 2 * math.pi * torch.rand(1).item()
    cnx = cox + d * math.cos(theta)
    cny = coy + d * math.sin(theta)

    # crop the image
    left = max(0, int(cnx - R_N))
    right = min(W, int(cnx + R_N))
    top = max(0, int(cny - R_N))
    bottom = min(H, int(cny + R_N))
    image_N = image.crop((left, top, right, bottom))

    # crop with a new circle mask
    W_N, H_N = image_N.size
    mask = Image.new("L", (W_N, H_N), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, W_N, H_N), fill=255)
    image_N = Image.composite(image_N, Image.new("RGB", (W_N, H_N), (0, 0, 0)), mask)

    return image_N


class RandomFOV:
    def __init__(self,
                 p: float = 0.5,
                 ratio_range: tuple[float, float] = (0.2, 1.0),
                 pixel_limits: int = 64,
                 schedule: str | None = None,
                 start_epoch: int = 0,
                 end_epoch: int = 100,
                 start_p: float = 1.0,
                 end_p: float = 0.0,
                 ) -> None:
        self.p = p
        self.ratio_range = ratio_range
        self.pixel_limits = pixel_limits
        self.schedule = schedule
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_p = start_p
        self.end_p = end_p

    def __call__(self, img: Image.Image) -> Image.Image:
        return random_fov(
            img,
            p=self.p,
            ratio_range=self.ratio_range,
            pixel_limits=self.pixel_limits,
        )

    def on_epoch_start(self, epoch: int) -> None:
        if self.schedule is None:
            return
        elif self.schedule.lower() == "linear":
            if epoch < self.start_epoch:
                p = self.start_p
            elif epoch > self.end_epoch:
                p = self.end_p
            else:
                p = self.start_p + (self.end_p - self.start_p) * (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            self.p = p
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(p={self.p}, "
                f"ratio_range={self.ratio_range}, "
                f"pixel_limits={self.pixel_limits})")


if __name__ == "__main__":
    # Demo: generate a 1000x1000 synthetic image with
    # - random noise inside the circle (fundus)
    # - pure red outside the circle
    # - a 3px green outline at the circle boundary
    # Then apply random_fov to a batch of 16 and show in a 4x4 grid.
    import numpy as np
    from matplotlib import pyplot as plt

    torch.manual_seed(36)

    W = H = 1000
    R = W // 2

    # Generate a polar color wheel image:
    # - Hue varies with angle (polar)
    # - Saturation increases with radius (0 at center -> 1 at edge)
    # - Value fixed at 1 (bright)
    # - Outside circle is pure red

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    cx, cy = W / 2.0, H / 2.0
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)  # [-pi, pi]
    h = (theta + np.pi) / (2 * np.pi)  # [0, 1)
    s = np.clip(r / R, 0.0, 1.0)
    v = np.ones_like(s, dtype=np.float32)
    inside = r <= R

    # HSV -> RGB conversion (vectorized)
    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = np.mod(i, 6)

    rch = np.zeros_like(h)
    gch = np.zeros_like(h)
    bch = np.zeros_like(h)

    m0 = i_mod == 0
    m1 = i_mod == 1
    m2 = i_mod == 2
    m3 = i_mod == 3
    m4 = i_mod == 4
    m5 = i_mod == 5

    rch[m0] = v[m0]; gch[m0] = t[m0]; bch[m0] = p[m0]
    rch[m1] = q[m1]; gch[m1] = v[m1]; bch[m1] = p[m1]
    rch[m2] = p[m2]; gch[m2] = v[m2]; bch[m2] = t[m2]
    rch[m3] = p[m3]; gch[m3] = q[m3]; bch[m3] = v[m3]
    rch[m4] = t[m4]; gch[m4] = p[m4]; bch[m4] = v[m4]
    rch[m5] = v[m5]; gch[m5] = p[m5]; bch[m5] = q[m5]

    rgb = np.stack([rch, gch, bch], axis=-1)
    # Outside the circle -> pure red
    rgb[~inside] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    base = Image.fromarray(rgb8, mode="RGB")

    # Draw green circle outline (width=3)
    draw = ImageDraw.Draw(base)
    draw.ellipse((1, 1, W - 2, H - 2), outline=(0, 255, 0), width=3)

    # Apply transform to a batch of 16 images
    batch_size = 16
    imgs = [
        random_fov(
            base,
            p=1.0,
            ratio_range=(0.2, 1.0),
            pixel_limits=128,
        )
        for _ in range(batch_size)
    ]

    # Visualize in a 4x4 grid. If output has alpha, composite on red for display.
    rows, cols = 4, 4
    fig, axs = plt.subplots(rows, cols, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        img = imgs[i]
        if "A" in img.getbands():
            bg = Image.new("RGB", img.size, (255, 0, 0))
            img = Image.alpha_composite(bg.convert("RGBA"), img).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
