from torch import nn
from vuad.model.unet import UNetModule
from vuad.utils.registry import ModelRegistry

@ModelRegistry.register('unet_res')
class UNetRes(UNetModule):
    """
    U-Net architecture with skip connection between input and output.
    Making the model predicting the missing part instead of the whole image.
    """
    def __init__(self, activf=nn.SiLU, *args, **kwargs):
        super(UNetRes, self).__init__(*args, **kwargs)

    def forward(self, x):
        return x + super(UNetRes, self).forward(x)

    def forward_res(self, x):
        return super(UNetRes, self).forward(x)
