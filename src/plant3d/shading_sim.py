"""Synthetic check of the shading model on the real houseplant geometry: known per-leaf NIR albedo, known light.

Observations are rendered as albedo x shading (ambient 0.4, transmission 0.3, normals smoothed over 16 neighbours as
the "true" smooth leaf surface) plus noise, at the real 64 camera poses; the spectral stage is then fitted with no
shading model, with the shading model on raw Gaussian normals, and with smoothed normals.

    uv run python -m plant3d.shading_sim --light at_camera --out outputs/v3/shading_sim_at_camera.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from plant3d.gaussians import load_checkpoint, render
from plant3d.shading import Shading
from plant3d.train_channels import ChannelFrame, ChannelSource, Config, bands_source_from_colmap, train_channels

LIGHTS = {"at_camera": [0.8, 0.0, 0.6], "offset_70deg": [float(np.cos(np.radians(70))), float(np.sin(np.radians(70))), 0.45]}


@dataclass
class Args:
    light: str = "at_camera"
    geometry: Path = Path("outputs/v3/houseplant_rgb_s0/ckpt_30000.pt")
    data: Path = Path("data/greenhouse/houseplant_prep/undistorted_v3")
    leaves: Path = Path("outputs/v3/leaves/gaussian_leaves.pt")
    out: Path = Path("outputs/v3/shading_sim_at_camera.json")
    steps: int = 8000
    noise: float = 0.02
    seed: int = 1


def main() -> None:
    a = tyro.cli(Args)
    torch.manual_seed(0)
    geom, _ = load_checkpoint(a.geometry)
    for p in geom.values():
        p.requires_grad_(False)
    lab = torch.load(a.leaves, weights_only=False)["label"].cuda()
    src = bands_source_from_colmap(a.data, ["nir"], 4, test_group=4, file_bands=["green", "red", "red_edge", "nir"])
    allf = src.train + src.test
    rng = np.random.default_rng(a.seed)
    albedo = torch.ones(geom["means"].shape[0], 1, device="cuda")
    leaf_ids = torch.unique(lab[lab >= 0]).tolist()
    true_leaf = {j: float(rng.uniform(0.6, 1.6)) for j in leaf_ids}
    for j, v in true_leaf.items():
        albedo[lab == j] = v
    truth = Shading(geom, torch.stack([f.w2c for f in allf]).cuda(), ["cam1", "cam2"], 1, normal_k=16).cuda()
    with torch.no_grad():
        truth.light[:] = torch.tensor([LIGHTS[a.light]] * 2, device="cuda")
        truth.log_ambient[:] = float(np.log(0.4))
        truth.log_trans[:] = float(np.log(0.3))

    @torch.no_grad()
    def synth(fr):
        w2c = fr.w2c[None].cuda()
        cols = albedo[None] * truth.factor(w2c, truth.cam_index([fr.name]), geom["means"])
        img, al, _ = render(geom, w2c, fr.K[None].cuda(), fr.width, fr.height, None, None, colors=cols)
        return ChannelFrame(fr.name, fr.K, fr.w2c, fr.width, fr.height, (img[0] + a.noise * torch.randn_like(img[0])).clamp_min(0).cpu(), al[0].cpu())

    syn = ChannelSource("syn", ["nir"], [synth(f) for f in src.train], [synth(f) for f in src.test])
    out = {"light": a.light, "true": {"ambient": 0.4, "transmission": 0.3, "light_in_camera_frame": LIGHTS[a.light]},
           "true_leaf_albedo": true_leaf, "fits": {}}
    for name, shading, nk in [("no shading model", False, 0), ("shading, raw normals", True, 0), ("shading, smoothed normals", True, 16)]:
        torch.manual_seed(0)
        ex, _, aux, _ = train_channels(geom, [syn], Config(steps=a.steps, batch_views=4, shading=shading, shading_normal_k=nk))
        rec = {j: ex[lab == j, 0].median().item() for j in leaf_ids if (lab == j).sum() >= 200}
        t = np.array([true_leaf[j] for j in rec])
        r = np.array(list(rec.values()))
        fit = {"recovered_leaf_albedo": rec, "corr_with_true": float(np.corrcoef(t, r)[0, 1]),
               "rel_spread_error": float(np.std(r / r.mean() - t / t.mean()))}
        if shading:
            fit["learned"] = aux["shading"].summary()
        out["fits"][name] = fit
        print(name, round(fit["corr_with_true"], 4))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
