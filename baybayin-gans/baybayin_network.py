import torch.nn as nn


class LinearGeneratorBlock(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.fc_layer = nn.Linear(inp, oup)
        self.bnorm1d_layer = nn.BatchNorm1d(oup)
        self.gelu_layer = nn.GELU()

    def forward(self, x):
        x = self.fc_layer(x)
        x = self.bnorm1d_layer(x)
        x = self.gelu_layer(x)
        return x

class LinearGenerator(nn.Module):
    def __init__(self, z_dim=784, i_dim=784, h_dim=32):
        super().__init__()
        self.gen = nn.Sequential(
            LinearGeneratorBlock(z_dim, h_dim),
            LinearGeneratorBlock(h_dim, h_dim*2),
            LinearGeneratorBlock(h_dim*2, h_dim*4),
            LinearGeneratorBlock(h_dim*4, h_dim*8),
            nn.Linear(h_dim*8, i_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.gen(x)
        return x

class LinearDiscriminatorBlock(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.fc_layer = nn.Linear(inp, oup)
        self.gelu_layer = nn.GELU()

    def forward(self, x):
        x = self.fc_layer(x)
        x = self.gelu_layer(x)
        return x

class LinearDiscriminator(nn.Module):
    def __init__(self, i_dim=784, h_dim=32):
        super().__init__()
        self.disc = nn.Sequential(
            LinearDiscriminatorBlock(i_dim, h_dim*8),
            LinearDiscriminatorBlock(h_dim*8, h_dim*4),
            LinearDiscriminatorBlock(h_dim*4, h_dim*2),
            LinearDiscriminatorBlock(h_dim*2, h_dim),
            nn.Linear(h_dim, 1),
        )

    def forward(self, x):
        x = self.disc(x)
        return x