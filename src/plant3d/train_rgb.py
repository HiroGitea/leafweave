"""Stage 1: train RGB 3D Gaussian Splatting (geometry + appearance).

    uv run python -m plant3d.train_rgb --data data/nerf_synthetic/ficus --out outputs/ficus_rgb
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from gsplat.strategy import DefaultStrategy
from tqdm import trange

from plant3d.data import SceneData, composite, load_blender, load_colmap
from plant3d.gaussians import exp_lr_factor, export_ply, init_params, make_optimizers, render, save_checkpoint
from plant3d.losses import psnr, ssim


@dataclass
class Config:
    data: Path = Path("data/nerf_synthetic/ficus")
    out: Path = Path("outputs/ficus_rgb")
    format: str = "blender"  # "blender" (NeRF-synthetic) or "colmap" (PINHOLE model + images/ + masks/)
    downscale: int = 1
    test_every: int = 8  # evaluate on every k-th test view (blender) / hold out every k-th group (colmap)
    test_group: int = 1  # colmap: hold out groups of this many consecutive frames (Greenhouse: 4 = a whole azimuth stop)
    max_steps: int = 7_000
    eval_steps: list[int] = field(default_factory=lambda: [7_000, 30_000])
    sh_degree: int = 3
    sh_degree_interval: int = 1_000
    ssim_lambda: float = 0.2
    alpha_lambda: float = 0.0  # >0: also supervise rendered opacity with the GT alpha / foreground mask (L1)
    camera_color: bool = False  # learn a 3x3 colour matrix per camera folder (cameras with different spectral bands)
    white_bg: bool = True  # background used for evaluation (and training if random_bg is off)
    random_bg: bool = True  # train on a random solid background each step; stops "white fog" on RGBA data
    init_random: int = 100_000
    init_extent: float = 1.5  # random init: points uniform in [-extent, extent]^3 (world units)
    seed: int = 0
    lrs: dict[str, float] = field(
        default_factory=lambda: {
            "means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20,
        }
    )
    strategy: DefaultStrategy = field(default_factory=DefaultStrategy)


def load_scene(cfg: Config) -> SceneData:
    if cfg.format == "blender":
        return load_blender(cfg.data, cfg.downscale, cfg.test_every)
    if cfg.format == "colmap":
        return load_colmap(cfg.data, cfg.test_every, test_group=cfg.test_group)
    raise NotImplementedError(cfg.format)


def camera_of(name: str) -> str:
    return name.split("/")[0] if "/" in name else ""


def render_cam(params, fr, sh_degree, bg, cam_mats=None):
    """Render one frame; with per-camera colour matrices the matrix acts on the scene colour, not the background."""
    w2c, K = fr.w2c[None].cuda(), fr.K[None].cuda()
    if cam_mats is None:
        return render(params, w2c, K, fr.width, fr.height, sh_degree, bg)
    img, alpha, info = render(params, w2c, K, fr.width, fr.height, sh_degree, None)
    return img @ cam_mats[camera_of(fr.name)].T + (1 - alpha) * bg, alpha, info


@torch.no_grad()
def evaluate(params, scene: SceneData, sh_degree: int, bg: torch.Tensor, out_dir: Path, save_n: int = 4,
             cam_mats: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not scene.test:
        return {"n_views": 0, "n_gaussians": int(params["means"].shape[0])}
    ps, ss, ts, fog, iou = [], [], [], [], []
    for i, fr in enumerate(scene.test):
        gt, gt_a = composite(fr.image.cuda(), bg)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        img, alpha, _ = render_cam(params, fr, sh_degree, bg, cam_mats)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        img = img[0].clamp(0, 1)
        if gt_a is not None:  # share of true-background pixels that Gaussians still cover (floaters / fog)
            bg_mask = gt_a[..., 0] < 0.01
            fog.append((alpha[0, ..., 0][bg_mask] > 0.05).float().mean().item())
            # silhouette IoU: a direct geometry metric, independent of colour
            pa, ga = alpha[0, ..., 0] > 0.5, gt_a[..., 0] > 0.5
            iou.append(((pa & ga).sum() / (pa | ga).sum().clamp_min(1)).item())
        ps.append(psnr(img, gt).item())
        ss.append(ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).item())
        if i < save_n:
            both = torch.cat([gt, img], 1)
            iio.imwrite(out_dir / f"{fr.name.replace('/', '_')}.png", (both.cpu().numpy() * 255).astype(np.uint8))
    r = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "fps": float(1 / np.mean(ts[1:] or ts)),
         "n_views": len(ps), "n_gaussians": int(params["means"].shape[0])}
    if fog:
        r["bg_fog_frac"] = float(np.mean(fog))
        r["alpha_iou"] = float(np.mean(iou))
    return r


def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    cfg.out.mkdir(parents=True, exist_ok=True)
    scene = load_scene(cfg)
    bg = torch.ones(3, device="cuda") if cfg.white_bg else torch.zeros(3, device="cuda")
    print(f"train {len(scene.train)} / test {len(scene.test)} views, scene_scale={scene.scene_scale:.3f}")

    params = init_params(cfg.init_random, cfg.init_extent, cfg.sh_degree,
                         points=scene.init_points, colors=scene.init_colors)
    opts = make_optimizers(params, scene.scene_scale, cfg.lrs)
    cfg.strategy.check_sanity(params, opts)
    state = cfg.strategy.initialize_state(scene_scale=scene.scene_scale)

    # keep training images on GPU as uint8 (100 x 800x800x4 ~ 256 MB)
    imgs = [fr.image.cuda() for fr in scene.train]
    Ks = torch.stack([fr.K for fr in scene.train]).cuda()
    w2cs = torch.stack([fr.w2c for fr in scene.train]).cuda()
    W, H = scene.train[0].width, scene.train[0].height
    means_lr0 = opts["means"].param_groups[0]["lr"]
    cams = sorted({camera_of(fr.name) for fr in scene.train + scene.test})
    cam_idx = [cams.index(camera_of(fr.name)) for fr in scene.train]
    cam_M = torch.nn.Parameter(torch.eye(3, device="cuda").repeat(len(cams), 1, 1)) if cfg.camera_color else None
    opt_M = torch.optim.Adam([cam_M], lr=1e-3) if cam_M is not None else None
    mats = lambda: None if cam_M is None else {c: cam_M[j].detach() for j, c in enumerate(cams)}

    log, results = [], {}
    pbar = trange(1, cfg.max_steps + 1, dynamic_ncols=True)
    t_start = time.time()
    for step in pbar:
        i = np.random.randint(len(imgs))
        step_bg = torch.rand(3, device="cuda") if cfg.random_bg else bg
        gt, gt_a = composite(imgs[i], step_bg)
        sh_deg = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        if cam_M is None:
            img, alpha, info = render(params, w2cs[i : i + 1], Ks[i : i + 1], W, H, sh_deg, step_bg)
            img = img[0]
        else:
            img, alpha, info = render(params, w2cs[i : i + 1], Ks[i : i + 1], W, H, sh_deg, None)
            img = img[0] @ cam_M[cam_idx[i]].T + (1 - alpha[0]) * step_bg
        cfg.strategy.step_pre_backward(params, opts, state, step, info)
        l1 = F.l1_loss(img, gt)
        s = ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - s)
        if cfg.alpha_lambda > 0 and gt_a is not None:
            loss = loss + cfg.alpha_lambda * F.l1_loss(alpha[0], gt_a)
        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        if opt_M is not None:
            opt_M.step(); opt_M.zero_grad(set_to_none=True)
        opts["means"].param_groups[0]["lr"] = means_lr0 * exp_lr_factor(step, cfg.max_steps)
        cfg.strategy.step_post_backward(params, opts, state, step, info, packed=False)

        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", n=params["means"].shape[0])
            log.append({"step": step, "loss": loss.item(), "n": int(params["means"].shape[0])})
        if step in cfg.eval_steps or step == cfg.max_steps:
            r = evaluate(params, scene, sh_deg, bg, cfg.out / f"renders_{step}", cam_mats=mats())
            r["train_minutes"] = (time.time() - t_start) / 60
            results[step] = r
            print(f"[eval step {step}] " + json.dumps(r))
            save_checkpoint(cfg.out / f"ckpt_{step}.pt", params, {"step": step, "sh_degree": sh_deg,
                            "scene_scale": scene.scene_scale, "white_bg": cfg.white_bg,
                            "camera_color": None if cam_M is None else {c: m.cpu().tolist() for c, m in mats().items()}})

    export_ply(cfg.out / "point_cloud.ply", params)
    (cfg.out / "results.json").write_text(json.dumps(results, indent=2))
    (cfg.out / "train_log.json").write_text(json.dumps(log))
    cfg_dump = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    (cfg.out / "config.json").write_text(json.dumps(cfg_dump, indent=2, default=str))
    return results


def main() -> None:
    train(tyro.cli(Config))


if __name__ == "__main__":
    main()
