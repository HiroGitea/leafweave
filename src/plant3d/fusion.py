"""Two-camera fusion: an RGB camera and a separate NIR camera, fused through the 3D model, versus 2D warping.

Simulated on the Greenhouse houseplant, whose two physical cameras sit at different positions:
  cam1 = "RGB camera"  -> only its green (580 nm) and red (660 nm) bands are used
  cam2 = "NIR camera"  -> only its NIR (820 nm) band is used
Both cameras actually recorded all four bands, so every cam1 view has a perfectly co-registered true NDVI, and cam1's
NIR is never used for anything except scoring. Geometry and spectral channels are trained on all 64 frames (this is the
deployment setting: the question is how well NIR can be carried over to the RGB camera's own views).

    uv run python -m plant3d.fusion prepare     # fusion image set (per-camera bands only)
    uv run python -m plant3d.train_rgb --format colmap --data data/greenhouse/houseplant_prep/fusion --test-every 0 \
        --camera-color --max-steps 30000 --out outputs/v3/fusion_rgb_s0
    uv run python -m plant3d.fusion run         # spectral fusion + evaluation against 2D plane-homography warping

Methods scored on every cam1 frame (NDVI always uses the RGB camera's own measured red):
  3d          NIR rendered from the 3D model at the cam1 view
  3d_both_rendered   red rendered as well (shows why the RGB camera's own red should be used)
  2d          NIR of the cam2 frame closest in viewpoint, warped by the homography of a fronto-parallel plane through
              the plant centre (the usual calibrated two-camera registration at a working distance)
  2d_best     same, but the plane depth chosen per frame to minimise the error against the truth (an oracle upper
              bound for any single-plane 2D registration)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro

from plant3d.data import load_colmap
from plant3d.gaussians import load_checkpoint, render, save_checkpoint
from plant3d.prep_greenhouse import BANDS, to_u8
from plant3d.spectral import ndvi
from plant3d.train_channels import Config as SpecConfig
from plant3d.train_channels import bands_source_from_colmap, train_channels, view_colors


@dataclass
class Args:
    data: Path = Path("data/greenhouse/houseplant_prep/undistorted_v3")  # full 4-band data: truth + poses
    fusion_data: Path = Path("data/greenhouse/houseplant_prep/fusion")
    rgb_cam: str = "cam1"
    nir_cam: str = "cam2"
    geometry: Path = Path("outputs/v3/fusion_rgb_s0/ckpt_30000.pt")
    out: Path = Path("outputs/v3/fusion_s0")
    steps: int = 5000
    batch_views: int = 1
    init_value: float = 0.2
    view_sh: int = 0
    frame_gain: bool = False
    seed: int = 0
    depth_factors: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35)  # plane depths tried by the 2d_best oracle


def prepare(a: Args) -> None:
    """Fusion image set: RGB camera images from (red, green, green), NIR camera images from (nir, nir, nir).
    Masks, bands and poses are linked from the full data set (masks come from the prep NIR threshold; a real RGB camera
    would segment its own images, which for a plant on black velvet is just as easy)."""
    dst = a.fusion_data
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for sub in ("masks", "bands", "sparse"):
        os.symlink(os.path.relpath(a.data / sub, dst), dst / sub)
    for cam in (a.rgb_cam, a.nir_cam):
        (dst / "images" / cam).mkdir(parents=True)
        for f in sorted((a.data / "bands" / cam).glob("*.npy")):
            r = np.load(f).astype(np.float32)
            g, red, nir = r[BANDS.index("green")], r[BANDS.index("red")], r[BANDS.index("nir")]
            img = np.stack([red, g, g], -1) if cam == a.rgb_cam else np.stack([nir, nir, nir], -1)
            iio.imwrite(dst / "images" / cam / f"{f.stem}.png", to_u8(img))
    print(f"fusion data set -> {dst}")


def plane_homography(fa, fb, centre: np.ndarray, depth_factor: float = 1.0) -> np.ndarray:
    """Homography mapping pixels of frame a to frame b for the plane facing camera a through `centre` (world)."""
    Ta, Tb = fa.w2c.numpy().astype(np.float64), fb.w2c.numpy().astype(np.float64)
    Ra, ta, Rb, tb = Ta[:3, :3], Ta[:3, 3], Tb[:3, :3], Tb[:3, 3]
    R = Rb @ Ra.T
    t = tb - R @ ta
    d = (Ra @ centre + ta)[2] * depth_factor
    n = np.array([0.0, 0.0, 1.0])
    return fb.K.numpy().astype(np.float64) @ (R + np.outer(t, n) / d) @ np.linalg.inv(fa.K.numpy().astype(np.float64))


def warp(img: np.ndarray, H: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)


@torch.no_grad()
def evaluate(a: Args, geom, extra, ch, extra_sh=None) -> list[dict]:
    full = load_colmap(a.data, 0)
    centre = np.median(full.init_points, 0)
    frames = {f.name: f for f in full.train}
    nir_frames = [f for n, f in frames.items() if n.startswith(a.nir_cam)]
    cen = lambda f: np.linalg.inv(f.w2c.numpy())[:3, 3]
    unit = lambda v: v / np.linalg.norm(v)
    src = bands_source_from_colmap(a.data, BANDS, 0, folders=[a.rgb_cam])
    nir_vals = {}
    for f in nir_frames:
        nir_vals[f.name] = np.load(a.data / "bands" / f"{f.name}.npy").astype(np.float32)[BANDS.index("nir")]
    jn, jr = ch.index("nir"), ch.index("red")
    rows = []
    for fr in src.train:
        fa = frames[fr.name]
        v, alpha_t = fr.values.cuda(), fr.alpha.cuda()[..., 0]
        red, nir_t = v[..., BANDS.index("red")], v[..., BANDS.index("nir")]
        nd_t = ndvi(nir_t, red)
        da = unit(cen(fa) - centre)
        fb = max(nir_frames, key=lambda f: unit(cen(f) - centre) @ da)
        parallax = float(np.degrees(np.arccos(np.clip(unit(cen(fb) - centre) @ da, -1, 1))))
        # 3D: NIR rendered at the RGB camera's view
        cols = view_colors(extra, extra_sh, geom["means"], fr.w2c[None].cuda(), a.view_sh)
        img, al, _ = render(geom, fr.w2c[None].cuda(), fr.K[None].cuda(), fr.width, fr.height, None, None, colors=cols)
        nir3, red3, al3 = img[0][..., jn], img[0][..., jr], al[0, ..., 0]
        # 2D: plane homography from the nearest NIR view, at several plane depths
        mb = (iio.imread(a.data / "masks" / f"{fb.name}.png") > 127).astype(np.float32)
        est = {}
        for k in a.depth_factors:
            H = plane_homography(fa, fb, centre, k)
            est[k] = (torch.from_numpy(warp(nir_vals[fb.name], H, fr.width, fr.height)).cuda(),
                      torch.from_numpy(warp(mb, H, fr.width, fr.height)).cuda() > 0.5)
        veg_t = (alpha_t > 0.5) & (nd_t > 0.3) & (nir_t > 0.2)
        support = veg_t & (al3 > 0.5)
        for k, (_, m) in est.items():
            support &= m  # one common pixel set for every method and plane depth
        rec = {"frame": fr.name, "nir_frame": fb.name, "parallax_deg": parallax, "n_px": int(support.sum())}
        if rec["n_px"] < 200:
            continue

        def score(nir_e, red_e=red):
            nd_e = ndvi(nir_e, red_e)
            e = (nd_e - nd_t)[support]
            Lf = support.float()
            pool = lambda x: F.avg_pool2d((x * Lf)[None, None], 32, 32)[0, 0]
            ok = pool(torch.ones_like(Lf)) >= 0.5
            blk = (ndvi(pool(nir_e), pool(red_e)) - ndvi(pool(nir_t), pool(red)))[ok]
            whole = ndvi((nir_e * Lf).sum(), (red_e * Lf).sum()) - ndvi((nir_t * Lf).sum(), (red * Lf).sum())
            return {"pixel_mae": e.abs().mean().item(), "pixel_bias": e.mean().item(), "block32_mae": blk.abs().mean().item(),
                    "whole_err": whole.item(), "nir_rel_mae": ((nir_e - nir_t).abs()[support].mean() / nir_t[support].mean()).item()}

        rec["3d"] = score(nir3)
        rec["3d_both_rendered"] = score(nir3, red3)
        rec["2d"] = score(est[1.0][0])
        by_k = {k: score(e[0]) for k, e in est.items()}
        kbest = min(by_k, key=lambda k: by_k[k]["pixel_mae"])
        rec["2d_best"] = {**by_k[kbest], "depth_factor": kbest}
        rows.append(rec)
    return rows


@torch.no_grad()
def nir_fit(geom, extra, ch, src, extra_sh=None, view_sh=0) -> float:
    """Rendered / measured NIR (sums over vegetation) on the NIR camera's own training views: fit quality."""
    jn, num, den = ch.index("nir"), 0.0, 0.0
    for fr in src.train:
        v, m = fr.values.cuda()[..., 0], fr.alpha.cuda()[..., 0] > 0.5
        cols = view_colors(extra[:, jn:jn + 1], None if extra_sh is None else extra_sh[:, :, jn:jn + 1], geom["means"],
                           fr.w2c[None].cuda(), view_sh)
        img = render(geom, fr.w2c[None].cuda(), fr.K[None].cuda(), fr.width, fr.height, None, None, colors=cols)[0][0, ..., 0]
        L = m & (v > 0.2)
        num, den = num + img[L].sum().item(), den + v[L].sum().item()
    return num / den


def run(a: Args) -> None:
    torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    geom, meta = load_checkpoint(a.geometry)
    rgb = bands_source_from_colmap(a.fusion_data, ["green", "red"], 0, folders=[a.rgb_cam], file_bands=BANDS, name="rgb")
    nirc = bands_source_from_colmap(a.fusion_data, ["nir"], 0, folders=[a.nir_cam], file_bands=BANDS, name="nir")
    print(f"RGB camera: {len(rgb.train)} frames {rgb.channels} | NIR camera: {len(nirc.train)} frames {nirc.channels}")
    cfg = SpecConfig(steps=a.steps, seed=a.seed, batch_views=a.batch_views, init_value=a.init_value, view_sh=a.view_sh,
                     frame_gain=a.frame_gain)
    extra, ch, aux, log = train_channels(geom, [rgb, nirc], cfg)
    save_checkpoint(a.out / "spectral.pt", torch.nn.ParameterDict({**geom, "extra": torch.nn.Parameter(extra)}),
                    {**meta, "channels": ch})
    rows = evaluate(a, geom, extra, ch, aux["extra_sh"])
    summary = {}
    bins = [(0, 2), (2, 6), (6, 10), (10, 15), (15, 25)]
    for m in ("3d", "3d_both_rendered", "2d", "2d_best"):
        summary[m] = {k: float(np.nanmean([r[m][k] for r in rows])) for k in ("pixel_mae", "pixel_bias", "block32_mae", "nir_rel_mae")}
        summary[m]["whole_mae"] = float(np.mean([abs(r[m]["whole_err"]) for r in rows]))
        summary[m]["by_parallax"] = {f"{lo}-{hi} deg": float(np.mean([r[m]["pixel_mae"] for r in rows if lo <= r["parallax_deg"] < hi] or [np.nan]))
                                     for lo, hi in bins}
    res = {"channels": ch, "frames": len(rows), "config": {k: str(v) for k, v in vars(a).items()}, "summary": summary,
           "nir_fit_on_nir_camera": nir_fit(geom, extra, ch, nirc, aux["extra_sh"], a.view_sh), "per_frame": rows}
    (a.out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"frames": len(rows), "summary": summary}, indent=1))


def main() -> None:
    cmd = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1] in ("prepare", "run") else "run"
    a = tyro.cli(Args)
    prepare(a) if cmd == "prepare" else run(a)


if __name__ == "__main__":
    main()
