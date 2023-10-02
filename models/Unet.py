import torch
import torch.nn as nn
import math

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up = False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up:
            # upsampling
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, padding = 1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            # maxpool
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x, timestep):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)
        # time embeddings
        t = self.time_mlp(timestep)
        t = self.relu(t)
        t = t[(..., ) + (None, ) * 2]
        # add time channel
        x = x + t
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.transform(x)
        return x

class time_embdding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        half = self.dim // 2
        embeddings = math.log(10000) / (half - 1)
        embeddings = torch.exp(torch.arange(half) * -embeddings).to(t.device)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, down_channels = [64, 128, 256, 512, 1024], up_channels = [1024, 512, 256, 128, 64], time_emb_dim = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            time_embdding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding = 1)
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels) - 1)])
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up = True) for i in range(len(up_channels) - 1)])
        self.out = nn.Conv2d(up_channels[-1], out_channels, 1)
    
    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual = []
        for down in self.downs:
            x = down(x, t)
            residual.append(x)
        for up in self.ups:
            x = torch.cat((x, residual.pop()), dim = 1)
            x = up(x, t)
        x = self.out(x)
        return x