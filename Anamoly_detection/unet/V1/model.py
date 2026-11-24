"""ResNet18 U-Net with a compact fixed decoder."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


DECODER_CHANNELS = (128, 64, 32, 16, 8)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, value, skip=None):
        value = F.interpolate(self.project(value), scale_factor=2, mode="nearest")
        if skip is not None:
            value = torch.cat((value, skip), dim=1)
        return self.refine(value)


class ResNet18UNet(nn.Module):
    def __init__(self, imagenet_state=None):
        super().__init__()
        self.encoder = resnet18(weights=None)
        if imagenet_state is not None:
            self.encoder.load_state_dict(imagenet_state, strict=True)
        self.encoder.fc = nn.Identity()
        self.decoder = nn.ModuleList(UpBlock(incoming, skip, outgoing) for incoming, skip, outgoing in zip(
            (512, *DECODER_CHANNELS[:-1]), (256, 128, 64, 64, 0), DECODER_CHANNELS, strict=True))
        self.head = nn.Conv2d(DECODER_CHANNELS[-1], 1, 1)

    def forward(self, rgb):
        e = self.encoder
        stem = e.relu(e.bn1(e.conv1(rgb)))
        first = e.layer1(e.maxpool(stem))
        second = e.layer2(first)
        third = e.layer3(second)
        value = e.layer4(third)
        for block, skip in zip(self.decoder, (third, second, first, stem, None), strict=True):
            value = block(value, skip)
        return self.head(value)


def bce_dice(logits, targets):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probability = logits.sigmoid().flatten(start_dim=1)
    target = targets.flatten(start_dim=1)
    overlap = (probability * target).sum(dim=1)
    dice = 1 - ((2 * overlap + 1) / (probability.sum(dim=1) + target.sum(dim=1) + 1)).mean()
    return bce, dice
