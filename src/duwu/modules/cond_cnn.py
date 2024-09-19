import numpy as np
import torch
from torch.nn.functional import leaky_relu


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    if mode == "zero":
        return torch.zeros(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Fully-connected layer.


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class CondCNN(torch.nn.Module):
    def __init__(
        self,
        nc=4,
        ngf=32,
        c_dim=1280,
    ):
        super().__init__()

        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="zero")

        self.pe = PositionalEmbedding(num_channels=128)
        self.conv1 = Conv2d(nc, ngf, 3, down=True, **init)  # 64x64 --> 32x32
        self.conv2 = Conv2d(ngf, ngf * 2, 3, down=True, **init)  # 32x32 --> 16x16
        self.conv3 = Conv2d(ngf * 2, ngf * 4, 3, down=True, **init)  # 16x16 --> 8x8
        self.conv4 = Conv2d(ngf * 4, ngf * 8, 3, down=True, **init)  # 8x8 --> 4x4
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.lin0 = Linear(in_features=ngf * 8, out_features=128, **init)
        self.lin1 = Linear(in_features=128, out_features=1, **init_zero)

        self.cemb = (
            None if c_dim == 0 else Linear(in_features=c_dim, out_features=128, **init)
        )
        self.temb1 = Linear(in_features=128, out_features=ngf, **init)
        self.temb2 = Linear(in_features=128, out_features=ngf * 2, **init)
        self.temb3 = Linear(in_features=128, out_features=ngf * 4, **init)
        self.temb4 = Linear(in_features=128, out_features=ngf * 8, **init)

    def forward(
        self,
        x,
        sigmas,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        added_cond_kwargs=None,
    ):
        # Embedding
        # temb = time.clamp(min=1e-3, max=1 - 1e-3)
        # temb = temb / (1 - temb)  # This gives sigmas
        temb = self.pe(sigmas.log() / 4)
        if self.cemb is not None:
            temb += self.cemb(added_cond_kwargs["text_embeds"].reshape(x.shape[0], -1))

        # Forward
        x = leaky_relu(self.conv1(x) + self.temb1(temb)[..., None, None])
        x = leaky_relu(self.conv2(x) + self.temb2(temb)[..., None, None])
        x = leaky_relu(self.conv3(x) + self.temb3(temb)[..., None, None])
        x = leaky_relu(self.conv4(x) + self.temb4(temb)[..., None, None])
        x = self.pool(x).flatten(start_dim=1)
        x = leaky_relu(self.lin0(x))
        x = self.lin1(x)
        return x
