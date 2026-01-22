import torch
import torch.nn as nn
import torch.nn.functional as F

from vuad.utils.registry import ModelRegistry

class ConvBlock(nn.Module):
    def __init__(self, input_ch=3, output_ch=64, activf=nn.ReLU, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(output_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv_block = nn.Sequential(
            self.conv1,
            activf(inplace=True),
            self.conv2,
            activf(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):
    def __init__(self, input_ch=64, output_ch=32, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(input_ch, output_ch, 2, 2, bias=bias)
        self.conv_block = nn.Sequential(self.conv)

    def forward(self, x):
        return self.conv_block(x)


@ModelRegistry.register(name="unet")
class UNetModule(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch, activf=nn.ReLU):
        super().__init__()

        # Encoder
        self.conv1 = ConvBlock(input_ch, base_ch, activf)
        self.conv2 = ConvBlock(base_ch, 2* base_ch, activf)
        self.conv3 = ConvBlock(2 * base_ch, 4 * base_ch, activf)
        self.conv4 = ConvBlock(4 * base_ch, 8 * base_ch, activf)
        self.conv5 = ConvBlock(8 * base_ch, 16 * base_ch, activf)

        # Decoder
        self.upconv1 = UpConv(16 * base_ch, 8 * base_ch)
        self.conv6 = ConvBlock(16 * base_ch, 8 * base_ch, activf)
        self.upconv2 = UpConv(8 * base_ch, 4 * base_ch)
        self.conv7 = ConvBlock(8 * base_ch, 4 * base_ch, activf)
        self.upconv3 = UpConv(4 * base_ch, 2 * base_ch)
        self.conv8 = ConvBlock(4 * base_ch, 2 * base_ch, activf)
        self.upconv4 = UpConv(2 * base_ch, base_ch)
        self.conv9 = ConvBlock(2 * base_ch, base_ch, activf)

        self.outconv = nn.Conv2d(base_ch, output_ch, 1, bias=True)

    def forward(self, x):

        x1 = self.conv1(x)
        x = F.max_pool2d(x1, 2, 2)

        x2 = self.conv2(x)
        x = F.max_pool2d(x2, 2, 2)

        x3 = self.conv3(x)
        x = F.max_pool2d(x3, 2, 2)

        x4 = self.conv4(x)
        x = F.max_pool2d(x4, 2, 2)

        x = self.conv5(x)
        x = self.upconv1(x)
        x = torch.cat((x4, x), dim=1)

        x = self.conv6(x)
        x = self.upconv2(x)
        x = torch.cat((x3, x), dim=1)

        x = self.conv7(x)
        x = self.upconv3(x)
        x = torch.cat((x2, x), dim=1)

        x = self.conv8(x)
        x = self.upconv4(x)
        x = torch.cat((x1, x), dim=1)

        x = self.conv9(x)
        x = self.outconv(x)

        return x
