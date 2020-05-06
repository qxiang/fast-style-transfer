import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """
    This is a residual block net, which is
    used to construct the transforming network.
    """
    def __init__(self):
        super(_ResBlock, self).__init__()
        self.model = nn.Sequential(
           nn.Conv2d(128, 128, 3, 1, 1, padding_mode='reflect'),
           nn.InstanceNorm2d(128),
           nn.ReLU(),
           nn.Conv2d(128, 128, 3, 1, 1, padding_mode='reflect'),
           nn.InstanceNorm2d(128)
        )

    def forward(self, input):
        return self.model(input) + input


class NormalNet(nn.Module):
    def __init__(self, mean, std):
        super(NormalNet, self).__init__()
        self.mean = mean
        self.std = std
 
    def forward(self, input):
        return (input - self.mean) / self.std 



class TransNet(nn.Module):
    """
    This is the network used to transform images.
    We need to train this net on a relatively large dataset.
    The style would be encoded in this network.
    """
    def __init__(self):
        super(TransNet, self).__init__()
        self.model = nn.Sequential(
            # Downsampling subnet.
            nn.Conv2d(3, 32, 9, 1, 4, padding_mode='reflect'),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            # Residual blocks.
            _ResBlock(),
            _ResBlock(),
            _ResBlock(),
            _ResBlock(),
            _ResBlock(),

            # Upsampling subnet.
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, padding_mode="reflect"),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 9, 1, 4, padding_mode='reflect'),
            nn.Tanh()
        )
    
    def forward(self, input):
        return (self.model(input) + 1) / 2
