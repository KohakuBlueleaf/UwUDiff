import torch
import torch.nn as nn
from torch import fft


class HPF_MSE(nn.Module):

    def __init__(self, freq_radius=0.1, hpf_lmda=1000):
        super().__init__()
        self.freq_radius = freq_radius
        self.hpf_lmda = hpf_lmda

    def __fft__(self, x):
        return fft.fftshift(fft.fft2(x))

    def __ifft__(self, x):
        return fft.ifft2(fft.ifftshift(x))

    def __get_hpf_mask__(self, size, freq_radius):
        cx, cy = size[0] / 2, size[1] / 2
        center = torch.tensor([cx, cy]).reshape(1, -1).to(freq_radius.device)
        xy = (
            torch.cartesian_prod(torch.arange(size[0]), torch.arange(size[1])).to(
                freq_radius.device
            )
            + 0.5
        )
        xy = xy[None].repeat(freq_radius.shape[0], 1, 1)
        mask = (xy - center[None]).float().norm(
            dim=2
        ) >= center.float().norm() * freq_radius.reshape(-1, 1)
        return mask.reshape([freq_radius.shape[0], size[0], size[1]])

    def forward(self, x0, x0_pred):
        r = self.freq_radius * x0.new_ones(x0.shape[0])
        mask = self.__get_hpf_mask__(size=[x0.shape[2], x0.shape[3]], freq_radius=r)

        hpf_p = self.hpf_lmda / (self.hpf_lmda + 1)
        mask = hpf_p * mask + (1 - hpf_p) * torch.ones_like(mask)
        return (
            self.__ifft__(mask[:, None] * self.__fft__(x0 - x0_pred))
            .abs()
            .flatten(start_dim=1)
            .square()
            .mean(dim=1)
        )
