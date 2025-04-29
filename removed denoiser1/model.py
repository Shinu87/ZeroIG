import torch
import torch.nn as nn
from loss import LossFunction, TextureDifference
from utils import blur, pair_downsampler


class Denoise_2(nn.Module):
    def __init__(self, chan_embed=96):
        super(Denoise_2, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(6, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 6, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Enhancer(nn.Module):
    def __init__(self, layers, channels):
        super(Enhancer, self).__init__()
        kernel_size = 3
        padding = kernel_size // 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList([self.conv for _ in range(layers)])

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 3, 3, stride=1, padding=1),
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
        self.denoise_2 = Denoise_2(chan_embed=48)
        self._criterion = LossFunction()
        self.TextureDifference = TextureDifference()

    def enhance_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def denoise_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input):
        eps = 1e-4
        input = input + eps

        # Skip denoise_1
        L2 = torch.clamp(input, eps, 1)

        s2 = self.enhance(L2.detach())
        s21, s22 = pair_downsampler(s2)

        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)

        H11, H12 = pair_downsampler(H2)
        H11, H12 = torch.clamp(H11, eps, 1), torch.clamp(H12, eps, 1)

        H3_pred = torch.cat([H11, s21], 1).detach() - self.denoise_2(torch.cat([H11, s21], 1))
        H3_pred = torch.clamp(H3_pred, eps, 1)
        H13, s13 = H3_pred[:, :3, :, :], H3_pred[:, 3:, :, :]

        H4_pred = torch.cat([H12, s22], 1).detach() - self.denoise_2(torch.cat([H12, s22], 1))
        H4_pred = torch.clamp(H4_pred, eps, 1)
        H14, s14 = H4_pred[:, :3, :, :], H4_pred[:, 3:, :, :]

        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3, s3 = H5_pred[:, :3, :, :], H5_pred[:, 3:, :, :]

        # Dummy placeholders
        L_pred1_L_pred2_diff = torch.tensor(0.0, device=input.device)
        H3_denoised1, H3_denoised2 = pair_downsampler(H3)
        H3_denoised1_H3_denoised2_diff = self.TextureDifference(H3_denoised1, H3_denoised2)

        H1 = torch.clamp(L2 / s2, 0, 1)
        H2_blur = blur(H1)
        H3_blur = blur(H3)

        L_pred1, L_pred2 = pair_downsampler(L2)  # Add this line
        H13_H14_diff = self.TextureDifference(H13, H14)  # Add this line

        return L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, \
              H3_pred, H4_pred, L_pred1_L_pred2_diff, H13_H14_diff, H2_blur, H3_blur


    def _loss(self, input):
        # Get the outputs from the forward pass
        outputs = self(input)

        # Unpack everything (ensure you unpack 21 values here)
        L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, \
        H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, \
        H2_blur, H3_blur = outputs

        # Generate L_pred1 and L_pred2 via pair_downsampler
        # This is probably redundant now since L_pred1 and L_pred2 are already part of the outputs
        # You can either remove this or leave it in if it's required elsewhere
        # L_pred1, L_pred2 = pair_downsampler(input)
        L2 = input  # This is most likely L2 as passed into the loss

        # Compute loss using all required arguments
        loss = self._criterion(
            input, L_pred1, L_pred2, L2,
            s2, s21, s22,
            H2, H11, H12, H13, s13, H14, s14,
            H3, s3, H3_pred, H4_pred,
            L_pred1_L_pred2_diff,
            H3_denoised1_H3_denoised2_diff,
            H2_blur, H3_blur
        )

        return loss



class Finetunemodel(nn.Module):
    def __init__(self, weights):
        super(Finetunemodel, self).__init__()

        self.enhance = Enhancer(layers=3, channels=64)
        self.denoise_2 = Denoise_2(chan_embed=48)

        pretrained_dict = torch.load(weights, map_location='cuda:0')
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, input):
        eps = 1e-4
        input = input + eps
        L2 = torch.clamp(input, eps, 1)
        s2 = self.enhance(L2)
        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)
        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]
        return H2, H3
