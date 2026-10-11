"""Gaussian parameters, rendering, and I/O.

Parameter layout (gsplat convention, keys used by gsplat.strategy):
  means [N,3], scales [N,3] (log), quats [N,4], opacities [N] (logit),
  sh0 [N,1,3], shN [N,K-1,3]  -- RGB appearance as spherical harmonics
  extra [N,C] (optional)      -- additional spectral channels (e.g. NIR), view-independent
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization

SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def _knn_mean_dist(points: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    """Mean distance to k nearest neighbours (brute force on GPU, chunked)."""
    out = torch.empty(points.shape[0], device=points.device)
    for i in range(0, points.shape[0], chunk):
        d = torch.cdist(points[i : i + chunk], points)
        out[i : i + chunk] = d.topk(k + 1, largest=False).values[:, 1:].mean(-1)
    return out


def init_params(
    n_random: int,
    extent: float,
    sh_degree: int,
    device: str = "cuda",
    points: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    init_opacity: float = 0.1,
) -> torch.nn.ParameterDict:
    if points is None:
        pts = (torch.rand(n_random, 3, device=device) * 2 - 1) * extent
        rgb = torch.rand(n_random, 3, device=device)
    else:
        pts = torch.as_tensor(points, dtype=torch.float32, device=device)
        rgb = torch.as_tensor(colors, dtype=torch.float32, device=device)
    n = pts.shape[0]
    dist = _knn_mean_dist(pts).clamp_min(1e-7)
    k = (sh_degree + 1) ** 2
    sh = torch.zeros(n, k, 3, device=device)
    sh[:, 0] = rgb_to_sh0(rgb)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(pts),
            "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(torch.rand(n, 4, device=device)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), init_opacity, device=device))),
            "sh0": torch.nn.Parameter(sh[:, :1].contiguous()),
            "shN": torch.nn.Parameter(sh[:, 1:].contiguous()),
        }
    )


def make_optimizers(params: torch.nn.ParameterDict, scene_scale: float, lrs: dict[str, float]) -> dict:
    """One Adam per parameter (required by gsplat strategies). `means` LR is scaled by scene size."""
    opts = {}
    for name, p in params.items():
        lr = lrs[name] * (scene_scale if name == "means" else 1.0)
        opts[name] = torch.optim.Adam([{"params": p, "lr": lr, "name": name}], eps=1e-15, fused=True)
    return opts


def render(
    params: torch.nn.ParameterDict,
    w2c: torch.Tensor,  # [B,4,4]
    K: torch.Tensor,  # [B,3,3]
    width: int,
    height: int,
    sh_degree: int | None,
    bg: torch.Tensor | None,
    colors: torch.Tensor | None = None,
    **kw,
):
    """Render RGB from SH (default) or arbitrary per-Gaussian features `colors` [N,D] (sh_degree=None)."""
    if colors is None:
        colors = torch.cat([params["sh0"], params["shN"]], 1)
    img, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=w2c,
        Ks=K,
        width=width,
        height=height,
        sh_degree=sh_degree,
        backgrounds=None if bg is None else bg.expand(w2c.shape[0], -1),
        packed=False,
        **kw,
    )
    return img, alpha, info


def save_checkpoint(path: str | Path, params: torch.nn.ParameterDict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"params": {k: v.detach().cpu() for k, v in params.items()}, "meta": meta}, path)


def load_checkpoint(path: str | Path, device: str = "cuda") -> tuple[torch.nn.ParameterDict, dict]:
    ck = torch.load(path, map_location=device, weights_only=True)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in ck["params"].items()})
    return params, ck["meta"]


def export_ply(path: str | Path, params: torch.nn.ParameterDict, extra_fields: dict | None = None) -> None:
    """Standard 3DGS .ply (readable by SuperSplat, gsplat viewers, etc.) + optional per-Gaussian scalar fields."""
    from plyfile import PlyData, PlyElement

    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    n = p["means"].shape[0]
    f_dc = p["sh0"].reshape(n, 3)
    f_rest = p["shN"].transpose(0, 2, 1).reshape(n, -1)  # [N, 3*(K-1)], channel-major like INRIA
    cols = (
        [("x", p["means"][:, 0]), ("y", p["means"][:, 1]), ("z", p["means"][:, 2])]
        + [("nx", np.zeros(n)), ("ny", np.zeros(n)), ("nz", np.zeros(n))]
        + [(f"f_dc_{i}", f_dc[:, i]) for i in range(3)]
        + [(f"f_rest_{i}", f_rest[:, i]) for i in range(f_rest.shape[1])]
        + [("opacity", p["opacities"])]
        + [(f"scale_{i}", p["scales"][:, i]) for i in range(3)]
        + [(f"rot_{i}", p["quats"][:, i]) for i in range(4)]
    )
    if "extra" in p:
        cols += [(f"extra_{i}", p["extra"][:, i]) for i in range(p["extra"].shape[1])]
    for name, v in (extra_fields or {}).items():
        cols.append((name, np.asarray(v).reshape(n)))
    arr = np.empty(n, dtype=[(name, "f4") for name, _ in cols])
    for name, v in cols:
        arr[name] = v
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def exp_lr_factor(step: int, max_steps: int, final_ratio: float = 0.01) -> float:
    return math.exp(math.log(final_ratio) * min(step, max_steps) / max_steps)
