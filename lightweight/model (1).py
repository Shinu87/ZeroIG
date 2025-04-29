import torch
import torch.nn as nn
from torch.nn import functional as F
from loss import LossFunction, TextureDifference
from utils import blur, pair_downsampler

class LiteHybridConv(nn.Module):
    """Optimized hybrid conv with split channels"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, 
                                 padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels//2, 1)
        self.regular = nn.Conv2d(in_channels, out_channels//2, 3, padding=1)
        
    def forward(self, x):
        return torch.cat([
            self.pointwise(self.depthwise(x)),
            self.regular(x)
        ], dim=1)

class Denoise_1(nn.Module):
    def __init__(self, chan_embed=32):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = LiteHybridConv(3, chan_embed)
        self.conv2 = LiteHybridConv(chan_embed, chan_embed)
        self.conv3 = nn.Conv2d(chan_embed, 3, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)

class Denoise_2(nn.Module):
    def __init__(self, chan_embed=48):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = LiteHybridConv(6, chan_embed)
        self.conv2 = LiteHybridConv(chan_embed, chan_embed)
        self.conv3 = nn.Conv2d(chan_embed, 6, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)

class Enhancer(nn.Module):
    def __init__(self, layers=3, channels=32):
        super().__init__()
        self.in_conv = nn.Sequential(
            LiteHybridConv(3, channels),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([  
            nn.Sequential(
                LiteHybridConv(channels, channels),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ) for _ in range(layers)
        ])
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        fea = self.in_conv(x)
        for block in self.blocks:
            fea = fea + block(fea)
        return torch.clamp(self.out_conv(fea), 1e-4, 1)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.enhance = Enhancer(layers=3, channels=32)
        self.denoise_1 = Denoise_1(chan_embed=32)
        self.denoise_2 = Denoise_2(chan_embed=48)
        
        self._l2_loss = nn.MSELoss()
        self._l1_loss = nn.L1Loss()
        self._criterion = LossFunction()
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.TextureDifference = TextureDifference()

    def enhance_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def denoise_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, input):
        eps = 1e-4
        input = input + eps

        # First denoising stage
        L11, L12 = pair_downsampler(input)
        L_pred1 = L11 - self.denoise_1(L11)
        L_pred2 = L12 - self.denoise_1(L12)
        L2 = input - self.denoise_1(input)
        L2 = torch.clamp(L2, eps, 1)

        # Enhancement stage
        s2 = self.enhance(L2.detach())
        s21, s22 = pair_downsampler(s2)
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)

        # Second denoising stage
        H11 = L11 / s21
        H11 = torch.clamp(H11, eps, 1)
        H12 = L12 / s22
        H12 = torch.clamp(H12, eps, 1)

        H3_pred = torch.cat([H11, s21], 1).detach() - self.denoise_2(torch.cat([H11, s21], 1))
        H3_pred = torch.clamp(H3_pred, eps, 1)
        H13 = H3_pred[:, :3, :, :]
        s13 = H3_pred[:, 3:, :, :]

        H4_pred = torch.cat([H12, s22], 1).detach() - self.denoise_2(torch.cat([H12, s22], 1))
        H4_pred = torch.clamp(H4_pred, eps, 1)
        H14 = H4_pred[:, :3, :, :]
        s14 = H4_pred[:, 3:, :, :]

        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]
        s3 = H5_pred[:, 3:, :, :]

        # Texture analysis
        L_pred1_L_pred2_diff = self.TextureDifference(L_pred1, L_pred2)
        H3_denoised1, H3_denoised2 = pair_downsampler(H3)
        H3_denoised1_H3_denoised2_diff = self.TextureDifference(H3_denoised1, H3_denoised2)

        # Post-processing
        H1 = L2 / s2
        H1 = torch.clamp(H1, 0, 1)
        H2_blur = blur(H1)
        H3_blur = blur(H3)

        return (L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, 
                H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff,
                H2_blur, H3_blur)

    def _loss(self, input):
        outputs = self(input)
        loss = self._criterion(input, *outputs)
        return loss

class Finetunemodel(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.enhance = Enhancer(layers=3, channels=32)
        self.denoise_1 = Denoise_1(chan_embed=32)
        self.denoise_2 = Denoise_2(chan_embed=48)

        base_weights = torch.load(weights, map_location='cuda:0')
        pretrained_dict = {k: v for k, v in base_weights.items() if k in self.state_dict()}
        self.load_state_dict(pretrained_dict, strict=False)

    def forward(self, input):
        eps = 1e-4
        input = input + eps
        L2 = input - self.denoise_1(input)
        L2 = torch.clamp(L2, eps, 1)
        s2 = self.enhance(L2)
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)
        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]
        return H2, H3

# Parameter verification
if __name__ == '__main__':
    model = Network()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")  # Should output ~0.7M parameters
