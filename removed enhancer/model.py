import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.denoise_1 = Denoise_1(chan_embed=48)
        self.denoise_2 = Denoise_2(chan_embed=48)
        self._l2_loss = nn.MSELoss()
        self._l1_loss = nn.L1Loss()
        self._criterion = LossFunction()
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
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
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        eps = 1e-4
        input = input + eps

        L11, L12 = pair_downsampler(input)

        # Initialize s2 and ensure proper resizing
        s2 = torch.ones_like(input)

        # Resize s2 to match input size for H2
        s2_input = F.interpolate(s2, size=input.shape[2:], mode='bilinear', align_corners=False)
        s21, s22 = pair_downsampler(s2_input)

        # Resize s2 to match L11/L12 size for H11, H12
        s2_L = F.interpolate(s2, size=L11.shape[2:], mode='bilinear', align_corners=False)

        L_pred1 = L11 - self.denoise_1(L11)
        L_pred2 = L12 - self.denoise_1(L12)
        L2 = input - self.denoise_1(input)
        L2 = torch.clamp(L2, eps, 1)

        # Compute H2, H11, H12
        H2 = input / s2_input
        H2 = torch.clamp(H2, eps, 1)

        H11 = L11 / s2_L
        H11 = torch.clamp(H11, eps, 1)

        H12 = L12 / s2_L
        H12 = torch.clamp(H12, eps, 1)

        # Compute H13, s13
        H3_pred = torch.cat([H11, s2_L], 1).detach() - self.denoise_2(torch.cat([H11, s2_L], 1))
        H3_pred = torch.clamp(H3_pred, eps, 1)
        H13 = H3_pred[:, :3, :, :]
        s13 = H3_pred[:, 3:, :, :]

        # Compute H14, s14
        H4_pred = torch.cat([H12, s2_L], 1).detach() - self.denoise_2(torch.cat([H12, s2_L], 1))
        H4_pred = torch.clamp(H4_pred, eps, 1)
        H14 = H4_pred[:, :3, :, :]
        s14 = H4_pred[:, 3:, :, :]

        # Compute H3, s3
        H5_pred = torch.cat([H2, s2_input], 1).detach() - self.denoise_2(torch.cat([H2, s2_input], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]
        s3 = H5_pred[:, 3:, :, :]

        # Texture differences
        L_pred1_L_pred2_diff = self.TextureDifference(L_pred1, L_pred2)
        H3_denoised1, H3_denoised2 = pair_downsampler(H3)
        H3_denoised1_H3_denoised2_diff = self.TextureDifference(H3_denoised1, H3_denoised2)

        # Compute H1 and blur maps
        H1 = L2 / s2_input
        H1 = torch.clamp(H1, 0, 1)
        H2_blur = blur(H1)
        H3_blur = blur(H3)

        return (
    L_pred1, L_pred2, L2, s2_input,
    s21, s22,                    # ✅ Newly added
    H2, H11, H12, H13, s13,
    H14, s14, H3, s3,
    H3_pred, H4_pred,
    L_pred1_L_pred2_diff,
    H3_denoised1_H3_denoised2_diff,
    H2_blur, H3_blur
)


    def _loss(self, input):
        L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, \
        H3_pred, H4_pred, L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur = self(input)

        loss = self._criterion(
            input, L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13,
            H14, s14, H3, s3, H3_pred, H4_pred,
            L_pred1_L_pred2_diff, H3_denoised1_H3_denoised2_diff,
            H2_blur, H3_blur
        )

        return loss




class Finetunemodel(nn.Module):
    def __init__(self, weights):
        super(Finetunemodel, self).__init__()

        # Remove the Enhancer module as per the experiment
        self.denoise_1 = Denoise_1(chan_embed=48)
        self.denoise_2 = Denoise_2(chan_embed=48)

        base_weights = torch.load(weights, map_location='cuda:0')
        pretrained_dict = base_weights
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
        L2 = input - self.denoise_1(input)
        L2 = torch.clamp(L2, eps, 1)

        # Skip Enhancer, set s2 = torch.ones_like(L2)
        s2 = torch.ones_like(L2)
        
        # Ensure s2 matches input dimensions
        s2 = s2.expand(-1, input.size(1), -1, -1)

        H2 = input / s2
        H2 = torch.clamp(H2, eps, 1)

        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]

        return H2, H3
