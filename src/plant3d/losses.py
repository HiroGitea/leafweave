from __future__ import annotations

import torch
import torch.nn.functional as F


def _gauss_window(size: int, sigma: float, channels: int, device) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(channels, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM of two [B, C, H, W] images in [0, 1]."""
    # callers pass permuted [B, H, W, C] renders; a non-contiguous input sends the depthwise conv down a ~8x slower path
    a, b = a.contiguous(), b.contiguous()
    c = a.shape[1]
    w = _gauss_window(size, sigma, c, a.device)
    pad = size // 2
    mu_a = F.conv2d(a, w, padding=pad, groups=c)
    mu_b = F.conv2d(b, w, padding=pad, groups=c)
    s_aa = F.conv2d(a * a, w, padding=pad, groups=c) - mu_a**2
    s_bb = F.conv2d(b * b, w, padding=pad, groups=c) - mu_b**2
    s_ab = F.conv2d(a * b, w, padding=pad, groups=c) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2))
    return m.mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return -10 * torch.log10(F.mse_loss(a, b))
