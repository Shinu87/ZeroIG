import torch
import torch.nn as nn
from loss import LossFunction, TextureDifference
from utils import blur, pair_downsampler


class Denoise_1(nn.Module):
    def __init__(self, chan_embed=48):
        super(Denoise_1, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(3, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 3, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Enhancer(nn.Module):
    def __init__(self, layers, channels):
        super(Enhancer, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        fea = torch.clamp(fea, 0.0001, 1)

        return fea


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.enhance = Enhancer(layers=3, channels=64)
        self.denoise_1 = Denoise_1(chan_embed=48)
        self._l2_loss = nn.MSELoss()
        self._l1_loss = nn.L1Loss()
        self._criterion = LossFunction()
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.TextureDifference = TextureDifference()

    def enhance_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def denoise_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        eps = 1e-4
        input = input + eps

        # First denoising step
        L11, L12 = pair_downsampler(input)
        L_pred1 = L11 - self.denoise_1(L11)
        L_pred2 = L12 - self.denoise_1(L12)
        L2 = input - self.denoise_1(input)
        L2 = torch.clamp(L2, eps, 1)

        # Enhancement step
        s2 = self.enhance(L2.detach())
        s21, s22 = pair_downsampler(s2)
        
        # Calculate H components (illumination)
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)
        
        H11 = L11 / s21
        H11 = torch.clamp(H11, eps, 1)
        
        H12 = L12 / s22
        H12 = torch.clamp(H12, eps, 1)
        
        # Final output is now H2 (without denoise_2)
        H3 = H2  # Direct output instead of going through denoise_2
        s3 = s2  # Use s2 directly as s3
        
        # Create dummy values for H13, s13, H14, s14 to maintain API compatibility
        H13 = H11  # Use H11 as substitute for H13
        s13 = s21  # Use s21 as substitute for s13
        H14 = H12  # Use H12 as substitute for H14
        s14 = s22  # Use s22 as substitute for s14
        
        # Create dummy H3_pred and H4_pred to maintain API compatibility
        H3_pred = torch.cat([H11, s21], 1)
        H4_pred = torch.cat([H12, s22], 1)
        
        # Calculate texture differences for loss computation
        L_pred1_L_pred2_diff = self.TextureDifference(L_pred1, L_pred2)
        H3_denoised1, H3_denoised2 = pair_downsampler(H3)
        H3_denoised1_H3_denoised2_diff = self.TextureDifference(H3_denoised1, H3_denoised2)
        
        # Blur operations
        H1 = L2 / s2
        H1 = torch.clamp(H1, 0, 1)
        H2_blur = blur(H1)
        H3_blur = blur(H3)
        
        # Return all the outputs needed by the loss function
        return L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur

    def _loss(self, input):
        L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur = self(input)
        
        loss = self._criterion(input, L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3,
                            H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur)
        return loss


class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()

        self.enhance = Enhancer(layers=3, channels=64)
        self.denoise_1 = Denoise_1(chan_embed=48)
        
        # Load weights but skip denoise_2 related parameters
        base_weights = torch.load(weights, map_location='cuda:0')
        pretrained_dict = {k: v for k, v in base_weights.items() if not k.startswith('denoise_2')}
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        eps = 1e-4
        input = input + eps
        
        # First denoising step
        L2 = input - self.denoise_1(input)
        L2 = torch.clamp(L2, eps, 1)
        
        # Enhancement step
        s2 = self.enhance(L2)
        
        # Calculate H component (illumination)
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)
        
        # Final output is now H2 (without denoise_2)
        return H2, H2  # Return same output twice to maintain API compatibility