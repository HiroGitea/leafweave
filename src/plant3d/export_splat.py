"""Export a trained spectral model as a compact .splat file plus colour layers for the web viewer.

    uv run python -m plant3d.export_splat --spectral outputs/v3/houseplant_spec_s0/spectral.pt \
        --weights outputs/v3/houseplant_spec_s0/weights.pt --leaves outputs/v3/leaves --out outputs/v3/viewer

Writes <out>/plant.splat (antimatter15 / gsplat.js format: 32 bytes per Gaussian = xyz f32, scale f32, rgba u8, quat
u8 (w, x, y, z)), one RGBA u8 layer per colouring (ndvi.rgba, cir.rgba, leaf.rgba, N*4 bytes each), plus meta.json
with the per-Gaussian NDVI quantised to u8 (for the threshold slider) and per-leaf statistics.

The world is re-centred on the plant and rotated so the turntable axis points to -y (y down, the INRIA convention
most web splat viewers assume), so the plant stands upright on first load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro

from plant3d.data import load_colmap
from plant3d.gaussians import SH_C0, load_checkpoint
from plant3d.spectral import index_colormap, ndvi


@dataclass
class Args:
    spectral: Path = Path("outputs/v3/houseplant_spec_s0/spectral.pt")
    weights: Path = Path("outputs/v3/houseplant_spec_s0/weights.pt")
    leaves: Path | None = Path("outputs/v3/leaves")
    data: Path = Path("data/greenhouse/houseplant_prep/undistorted_v3")
    out: Path = Path("outputs/v3/viewer")
    min_weight: float = 1.0
    min_opacity: float = 0.02  # drop Gaussians that never show
    scale_clip: float = 0.2  # clip huge floaters (world units) so they do not wash out the view


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (w, x, y, z) quaternions, broadcasting."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def upright_frame(data: Path) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R (world -> viewer) and the pivot, from the training cameras: turntable axis -> -y, plant centre -> 0."""
    sc = load_colmap(data, 0)
    c2w = np.stack([np.linalg.inv(f.w2c.numpy()) for f in sc.train])
    C, up = c2w[:, :3, 3], -c2w[:, :3, 1]  # camera centres and "up" vectors (OpenCV y is down)
    axis = np.linalg.svd(C - C.mean(0))[2][2]
    axis = axis * np.sign(axis @ up.mean(0))
    D = c2w[:, :3, 2]
    P = np.eye(3)[None] - D[:, :, None] * D[:, None, :]
    pivot = np.linalg.solve(P.sum(0), (P @ C[:, :, None]).sum(0)[:, 0])  # closest point to all optical axes
    # map `axis` to -y; x stays as close to the first camera's right vector as possible
    y = -axis
    x = c2w[0, :3, 0]
    x = x - (x @ y) * y
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    R = np.stack([x, y, z])  # rows = new basis
    return R, pivot


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    raise ValueError("degenerate rotation")  # not reachable for the frames built above


def main() -> None:
    a = tyro.cli(Args)
    a.out.mkdir(parents=True, exist_ok=True)
    sp, meta = load_checkpoint(a.spectral, device="cpu")
    ch = meta["channels"]
    p = {k: v.detach().numpy() for k, v in sp.items()}
    n = p["means"].shape[0]
    opa = 1 / (1 + np.exp(-p["opacities"]))
    weights = torch.load(a.weights, weights_only=True)["weights"].numpy()
    extra = p["extra"]
    nd = ndvi(extra[:, ch.index("nir")], extra[:, ch.index("red")])
    conf = weights >= a.min_weight

    R, pivot = upright_frame(a.data)
    xyz = (p["means"] - pivot) @ R.T
    q = p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)
    q = quat_mul(rot_to_quat(R)[None], q)
    scale = np.minimum(np.exp(p["scales"]), a.scale_clip)

    keep = opa >= a.min_opacity
    print(f"{n} Gaussians, keeping {keep.sum()} with opacity >= {a.min_opacity}; {conf.sum()} confident")
    xyz, q, scale, opa, nd, conf, extra = (x[keep] for x in (xyz, q, scale, opa, nd, conf, extra))
    m = int(keep.sum())
    alpha = (opa * 255).astype(np.uint8)

    def rgba(rgb: np.ndarray) -> np.ndarray:
        return np.concatenate([(np.clip(rgb, 0, 1) * 255).astype(np.uint8), alpha[:, None]], 1)

    # colour layers
    col_ndvi = index_colormap(nd)
    col_ndvi[~conf] = 0.5
    cir = p["sh0"][keep, 0] * SH_C0 + 0.5  # DC colour of the geometry model (CIR: R = NIR, G = red, B = green)
    layers = {"ndvi": rgba(col_ndvi), "cir": rgba(cir)}

    leaf_table = []
    if a.leaves is not None and (a.leaves / "gaussian_leaves.pt").exists():
        gl = torch.load(a.leaves / "gaussian_leaves.pt", weights_only=False)
        label = gl["label"].numpy()[keep]
        summ = pd.read_csv(a.leaves / "per_leaf_summary.csv")
        rng = np.random.default_rng(3)
        pal = rng.random((int(label.max()) + 1, 3)) * 0.75 + 0.2
        col_leaf = np.full((m, 3), 0.18)
        has = label >= 0
        col_leaf[has] = pal[label[has]]
        layers["leaf"] = rgba(col_leaf)
        for _, r in summ.iterrows():
            k = int(r.leaf)
            if (label == k).sum() == 0:
                continue
            leaf_table.append({"leaf": k, "color": [int(c * 255) for c in pal[k]], "views": int(r.views),
                               "ndvi_mean": round(float(r.ndvi_measured_mean), 4), "ndvi_se": round(float(r.ndvi_measured_se), 4),
                               "ndvi_gaussians": round(float(r.ndvi_gaussians), 4), "n_gaussians": int((label == k).sum())})

    # .splat
    buf = np.zeros(m, dtype=[("xyz", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    buf["xyz"], buf["scale"], buf["rgba"] = xyz, scale, layers["ndvi"]
    buf["rot"] = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
    (a.out / "plant.splat").write_bytes(buf.tobytes())
    for k, v in layers.items():
        (a.out / f"{k}.rgba").write_bytes(np.ascontiguousarray(v).tobytes())
    (a.out / "ndvi.u8").write_bytes((np.clip((nd + 0.2) / 1.2, 0, 1) * 255).astype(np.uint8).tobytes())
    w = weights[keep]
    veg = conf & (nd > 0.3) & (extra[:, ch.index("nir")] > 0.2)
    meta_out = {
        "n_gaussians": m, "n_confident": int(conf.sum()), "ndvi_range": [-0.2, 1.0],
        "plant_ndvi_weighted": float((nd[veg] * w[veg]).sum() / w[veg].sum()),
        "plant_ndvi_p10_p50_p90": np.quantile(nd[veg], [0.1, 0.5, 0.9]).round(4).tolist(),
        "channels": ch, "leaves": leaf_table, "source": str(a.spectral),
    }
    (a.out / "meta.json").write_text(json.dumps(meta_out, indent=1))
    print({k: f"{(a.out / f).stat().st_size / 1e6:.2f} MB" for k, f in [("splat", "plant.splat"), ("ndvi", "ndvi.rgba")]},
          f"| {len(leaf_table)} leaves")


if __name__ == "__main__":
    main()
