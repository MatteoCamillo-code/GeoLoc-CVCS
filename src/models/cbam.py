import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        # Channel attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        # Spatial attention
        self.spatial = nn.Conv2d(
            in_channels=2, out_channels=1,
            kernel_size=7, padding=3, bias=False
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # ----- Channel attention -----
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))

        channel_att = torch.sigmoid(
            self.mlp(avg_pool) + self.mlp(max_pool)
        ).view(b, c, 1, 1)

        x = x * channel_att

        # ----- Spatial attention -----
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.amax(x, dim=1, keepdim=True)

        spatial_att = torch.sigmoid(
            self.spatial(torch.cat([avg_pool, max_pool], dim=1))
        )

        x = x * spatial_att
        return x
