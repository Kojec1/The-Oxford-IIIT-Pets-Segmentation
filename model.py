import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.transforms import CenterCrop


class UNet(nn.Module):
    """UNet model architecture"""

    def __init__(self, enc_channels: tuple, dec_channels: tuple, n_classes: int, output_size: tuple) -> None:
        super(UNet, self).__init__()
        # Store the output size
        self.output_size = output_size
        # Initialise the encoder, decoder and output layers
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        self.out = nn.Conv2d(dec_channels[-1], n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # Feed the input through each layer
        x = self.encoder(x)
        x = self.decoder(x[-1], x[::-1][1:])
        x = self.out(x)
        # Resize the output to the given size
        x = F.interpolate(x, self.output_size)

        # Return the segmentation mask
        return x


class Encoder(nn.Module):
    """Encoder module"""

    def __init__(self, channels: tuple) -> None:
        super(Encoder, self).__init__()
        # Initialise a list of convolutional blocks
        self.blocks = nn.ModuleList([ConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        # Initialise a max pooling layer
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> list:
        # Initialise an empty list for skip connection values
        outputs = list()

        # Loop through each convolutional block
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
            x = self.maxpool(x)

        # Return a list with the outputs of each block
        return outputs


class Decoder(nn.Module):
    """Decoder module"""

    def __init__(self, channels: tuple) -> None:
        super(Decoder, self).__init__()
        # Store the number of channels
        self.n_channels = len(channels)
        # Initialise a list of convolutional blocks
        self.blocks = nn.ModuleList([ConvBlock(channels[i], channels[i + 1])
                                     for i in range(len(channels) - 1)])
        # Initialise a list of up-scaling layers
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1)
             for i in range(len(channels) - 1)])

    def forward(self, x: Tensor, enc_outputs: list) -> Tensor:
        # Loop through each convolutional block
        for i in range(self.n_channels - 1):
            x = self.upconvs[i](x)
            # Resize the output from the skip connection to match a size of the up-scaling layer's output
            skip = CenterCrop(x.shape[2:])(enc_outputs[i])
            # Concatenate the outputs from the skip connection and up-scaling layer
            x = torch.concat([x, skip], dim=1)
            x = self.blocks[i](x)

        # Return the input with applied transformations
        return x


class ConvBlock(nn.Module):
    """ Convolutional Block module"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            # 1st convolutional layer: (in_channels, H, W) => (out_channels, H, W)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 2nd convolutional layer: (out_channels, H, W) => (out_channels, H, W)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Return the input with applied transformations
        return self.block(x)
