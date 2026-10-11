"""Spectral helpers: vegetation indices and colormaps. All inputs are linear (relative) reflectance."""

from __future__ import annotations

import numpy as np
import torch


def ndvi(nir, red, eps: float = 1e-6):
    return (nir - red) / (nir + red + eps)


def ndre(nir, red_edge, eps: float = 1e-6):
    """Normalized difference red-edge index; more sensitive to chlorophyll in dense canopies than NDVI."""
    return (nir - red_edge) / (nir + red_edge + eps)


# brown -> yellow -> light green -> dark green
_STOPS = np.array([0.0, 0.35, 0.7, 1.0])
_COLS = np.array([[0.55, 0.32, 0.18], [0.95, 0.88, 0.35], [0.35, 0.75, 0.25], [0.02, 0.38, 0.12]])


def index_colormap(x, vmin: float = -0.2, vmax: float = 1.0):
    """Map an index (NDVI etc.) to RGB in [0,1]. Accepts numpy or torch; returns the same type, shape [..., 3]."""
    is_t = isinstance(x, torch.Tensor)
    a = x.detach().cpu().numpy() if is_t else np.asarray(x)
    t = np.clip((a - vmin) / (vmax - vmin), 0, 1)
    rgb = np.stack([np.interp(t, _STOPS, _COLS[:, c]) for c in range(3)], -1).astype(np.float32)
    return torch.from_numpy(rgb).to(x.device) if is_t else rgb
