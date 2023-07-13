import torch
from torch import nn, Tensor
from torchvision.models.densenet import _DenseBlock

from utilities.misc import center_crop


class TransitionUp(nn.Module):
    """
    Scale the resolution up by transposed convolution
    """

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        if scale == 2:
            self.convTrans = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=2, padding=0, bias=True)
        elif scale == 4:
            self.convTrans = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ConvTranspose2d(
                    in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=2, padding=0, bias=True)
            )

    def forward(self, x: Tensor, skip: Tensor):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class DoubleConv(nn.Module):
    """
    Two conv2d-bn-relu modules
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Tokenizer(nn.Module):
    """
    Expanding path of feature descriptor using DenseBlocks
    """

    # ([4,4,4,4], [64,128,128], 128, 4)
    def __init__(self, block_config: list, backbone_feat_channel: list, hidden_dim: int, growth_rate: int):
        super(Tokenizer, self).__init__()

        backbone_feat_channel.reverse()  # [128,128,64]
        block_config.reverse()           # [4,4,4,4]

        self.num_resolution = len(backbone_feat_channel)  # 3
        self.block_config = block_config  # [4,4,4,4]
        self.growth_rate = growth_rate    # 4

        self.bottle_neck = _DenseBlock(block_config[0], backbone_feat_channel[0], 4, drop_rate=0.0,
                                       growth_rate=growth_rate)
        up = []
        dense_block = []
        prev_block_channels = growth_rate * block_config[0]  # 16
        for i in range(self.num_resolution):  # [0,1,2]
            if i == self.num_resolution - 1:  # 2
                up.append(TransitionUp(prev_block_channels, hidden_dim, 4))
                dense_block.append(DoubleConv(hidden_dim + 3, hidden_dim))
            else:
                up.append(TransitionUp(prev_block_channels, prev_block_channels))
                cur_channels_count = prev_block_channels + backbone_feat_channel[i + 1]
                dense_block.append(
                    _DenseBlock(block_config[i + 1], cur_channels_count, 4, drop_rate=0.0, growth_rate=growth_rate))
                prev_block_channels = growth_rate * block_config[i + 1]

        self.up = nn.ModuleList(up)
        self.dense_block = nn.ModuleList(dense_block)

    def forward(self, features: list):
        """
        :param features:
            list containing feature descriptors at different spatial resolution
                0: [2N, 3, H, W]
                1: [2N, 64, H//4, W//4]
                2: [2N, 128, H//8, W//8]
                3: [2N, 128, H//16, W//16]
        :return: feature descriptor at full resolution [2N,C,H,W]
        """

        features.reverse()
        # (2N,128,H/16,W/16)
        output = self.bottle_neck(features[0])
        # (2N,128+16,H/16,W/16)
        # 前 128 维是 features[0] concate 来的，只取后 16 维
        output = output[:, -(self.block_config[0] * self.growth_rate):]
        # (2N,16,H/16,W/16)

        for i in range(self.num_resolution):  # [0,1,2]
            hs = self.up[i](output, features[i + 1])  # scale up and concat
            # (2N,128+16,H/8,W/8) -> (2N,64+16,H/4,W/4) -> (2N,3+128,H,W)
            output = self.dense_block[i](hs)  # denseblock
            # (2N,128+16+16,H/8,W/8) -> (2N,64+16+16,H/4,W/4) -> (2N,128,H,W)

            if i < self.num_resolution - 1:  # <2
                # take only the new features
                output = output[:, -(self.block_config[i + 1] * self.growth_rate):]
                # (2N,16,H/8,W/8) -> (2N,16,H/4,W/4)

        return output


def build_tokenizer(args, layer_channel):
    growth_rate = 4
    block_config = [4, 4, 4, 4]
    return Tokenizer(block_config, layer_channel, args.channel_dim, growth_rate)
