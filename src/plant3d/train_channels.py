"""Stage 2: freeze 3DGS geometry, learn per-Gaussian linear spectral channels (reflectance) from one or more
camera sources, each with its own poses/intrinsics (e.g. RGB camera + 850 nm NIR camera, or one multispectral camera).

    uv run python -m plant3d.train_channels --geometry outputs/houseplant_rgb_30k/ckpt_30000.pt \
        --data data/greenhouse/houseplant_prep/undistorted --out outputs/houseplant_spectral
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from tqdm import trange

from plant3d.data import Frame, load_colmap
from plant3d.gaussians import exp_lr_factor, export_ply, load_checkpoint, render, rgb_to_sh0, save_checkpoint
from plant3d.losses import psnr, ssim
from plant3d.spectral import index_colormap, ndre, ndvi


@dataclass
class ChannelFrame:
    name: str
    K: torch.Tensor
    w2c: torch.Tensor
    width: int
    height: int
    values: torch.Tensor  # [H, W, C_src] float32, linear
    alpha: torch.Tensor | None  # [H, W, 1] in [0,1], or None (then a background level is learned)


@dataclass
class ChannelSource:
    name: str
    channels: list[str]  # global channel names observed by this source, in the order of `values`
    train: list[ChannelFrame]
    test: list[ChannelFrame] = field(default_factory=list)


def bands_source_from_colmap(root: Path, channels: list[str], test_every: int, bands_dir: str = "bands",
                             use_masks: bool = True, name: str = "ms", test_group: int = 1,
                             folders: list[str] | None = None, file_bands: list[str] | None = None) -> ChannelSource:
    """Frames from a COLMAP model whose per-frame bands are stored as bands/<image stem>.npy [C,H,W].

    `file_bands` names the bands stored in the .npy files (default: `channels`, in that order); `channels` may then be
    any subset, e.g. ["nir"] for a NIR-only camera. `folders` keeps only frames from those camera folders."""
    root = Path(root)
    scene = load_colmap(root, test_every, masks="masks" if use_masks else None, test_group=test_group)
    sel = None if file_bands is None else [file_bands.index(c) for c in channels]
    keep = (lambda fr: True) if folders is None else (lambda fr: fr.name.split("/")[0] in folders)

    def conv(fr: Frame) -> ChannelFrame:
        arr = np.load(root / bands_dir / f"{fr.name}.npy").astype(np.float32)
        v = torch.from_numpy(arr if sel is None else arr[sel]).permute(1, 2, 0)
        a = fr.image[..., 3:].float() / 255.0 if fr.image.shape[-1] == 4 else None
        return ChannelFrame(fr.name, fr.K, fr.w2c, fr.width, fr.height, v.contiguous(), a)

    return ChannelSource(name, channels, [conv(f) for f in scene.train if keep(f)], [conv(f) for f in scene.test if keep(f)])


@torch.no_grad()
def _to_dev(fr: ChannelFrame):
    return (fr.values.cuda(), None if fr.alpha is None else fr.alpha.cuda(), fr.w2c[None].cuda(), fr.K[None].cuda())


def blend_weights(geom, frames: list[ChannelFrame]) -> torch.Tensor:
    """Per-Gaussian sum of alpha-blending weights over all pixels of `frames` (exact, via d(sum img)/d(color))."""
    n = geom["means"].shape[0]
    acc = torch.zeros(n, device="cuda")
    for fr in frames:
        ones = torch.ones(n, 1, device="cuda", requires_grad=True)
        img, _, _ = render(geom, fr.w2c[None].cuda(), fr.K[None].cuda(), fr.width, fr.height, None, None, colors=ones)
        img.sum().backward()
        acc += ones.grad[:, 0]
    return acc


def sh_basis(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """Real SH basis without the constant (l=0) term, [..., K-1] for K = (degree+1)^2, degree <= 2."""
    x, y, z = dirs.unbind(-1)
    b = [y, z, x]  # l = 1 (constant factors are absorbed into the coefficients)
    if degree >= 2:
        b += [x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y]
    return torch.stack(b, -1)


def view_colors(extra: torch.Tensor, extra_sh: torch.Tensor | None, means: torch.Tensor, w2c: torch.Tensor, degree: int):
    """Per-Gaussian channel values seen from each camera: DC reflectance + view-dependent SH term, clamped >= 0.
    extra [N, C], extra_sh [N, K-1, C] or None, w2c [B, 4, 4] -> [B, N, C] (or [N, C] when extra_sh is None)."""
    if extra_sh is None:
        return extra
    cam = torch.linalg.inv(w2c)[:, :3, 3]  # [B, 3] camera centres
    dirs = F.normalize(means[None] - cam[:, None], dim=-1)  # [B, N, 3]
    return (extra[None] + torch.einsum("bnk,nkc->bnc", sh_basis(dirs, degree), extra_sh)).clamp_min(0)


@dataclass
class Config:
    geometry: Path = Path("outputs/houseplant_rgb_30k/ckpt_30000.pt")
    data: Path = Path("data/greenhouse/houseplant_prep/undistorted")
    out: Path = Path("outputs/houseplant_spectral")
    channels: list[str] = field(default_factory=lambda: ["green", "red", "red_edge", "nir"])
    test_every: int = 8
    test_group: int = 1  # hold out groups of this many consecutive frames (Greenhouse: 4 = a whole azimuth stop)
    steps: int = 5000
    lr: float = 5e-3
    init_value: float = 0.2
    ssim_lambda: float = 0.2
    loss: str = "l1"  # "l1" fits the per-view median, "l2" the mean: band sums (whole-plant NDVI) need the mean
    min_weight: float = 1.0  # Gaussians with fewer than this many "pixels" of blending weight are low-confidence
    view_sh: int = 0  # degree of a view-dependent SH term per channel (0 = view-independent reflectance only)
    sh_lr_scale: float = 0.1  # SH coefficients learn at lr * this
    sh_prior: float = 1e-3  # L2 weight on the SH coefficients (sparse views: keeps the view term small unless needed)
    shading: bool = False  # leaf-orientation shading (plant3d.shading): reflectance x (ambient + direct + transmitted)
    shading_lr: float = 1e-2
    shading_normal_k: int = 0  # smooth Gaussian normals over this many neighbours (0 = raw normals)
    frame_gain: bool = False  # learn a per-frame, per-band log-gain (exposure / illumination level), zero-mean
    gain_lr: float = 1e-2
    gain_prior: float = 1e-3  # L2 weight on the log-gains
    n_boot: int = 2000  # bootstrap resamples (over views) for confidence intervals
    batch_views: int = 1  # views rendered per optimizer step; 4-8 gives ~2.5-3x the views/s of 1 (kernels are tiny)
    seed: int = 0


def train_channels(geom, sources: list[ChannelSource], cfg: Config, callback=None):
    """`callback(step, extra)` is called after every optimizer step (e.g. to snapshot convergence).

    Returns (extra [N, C], channel names, aux = {"bg": {source: level}, "gain": {source: [n_train, C_src] log-gain},
    "extra_sh": [N, K-1, C] or None}, log)."""
    all_ch = list(dict.fromkeys(c for s in sources for c in s.channels))
    n = geom["means"].shape[0]
    for p in geom.values():
        p.requires_grad_(False)
    extra = torch.nn.Parameter(torch.full((n, len(all_ch)), cfg.init_value, device="cuda"))
    k_sh = (cfg.view_sh + 1) ** 2 - 1
    extra_sh = torch.nn.Parameter(torch.zeros(n, k_sh, len(all_ch), device="cuda")) if cfg.view_sh > 0 else None
    bgs = {s.name: torch.nn.Parameter(torch.zeros(len(s.channels), device="cuda")) for s in sources}
    gains = {s.name: torch.nn.Parameter(torch.zeros(len(s.train), len(s.channels), device="cuda")) for s in sources}
    groups = [{"params": [extra], "lr": cfg.lr}, {"params": list(bgs.values()), "lr": 1e-2}]
    shade = None
    if cfg.shading:
        from plant3d.shading import Shading

        frames_all = [f for s_ in sources for f in s_.train + s_.test]
        cams = sorted({f.name.split("/")[0] for f in frames_all})
        shade = Shading(geom, torch.stack([f.w2c for f in frames_all]).cuda(), cams, len(all_ch), cfg.shading_normal_k).cuda()
        groups.append({"params": list(shade.parameters()), "lr": cfg.shading_lr})
    if extra_sh is not None:
        groups.append({"params": [extra_sh], "lr": cfg.lr * cfg.sh_lr_scale})
    if cfg.frame_gain:
        groups.append({"params": list(gains.values()), "lr": cfg.gain_lr})
    opt = torch.optim.Adam(groups, eps=1e-15, fused=True)
    idx = {s.name: torch.tensor([all_ch.index(c) for c in s.channels], device="cuda") for s in sources}
    data = {s.name: [_to_dev(f) for f in s.train] for s in sources}
    sizes = np.array([len(s.train) for s in sources], dtype=float)
    rng = np.random.default_rng(cfg.seed)
    log = []
    pbar = trange(1, cfg.steps + 1, dynamic_ncols=True)
    for step in pbar:
        s = sources[rng.choice(len(sources), p=sizes / sizes.sum())]
        nb = cfg.batch_views
        fis = [int(rng.integers(len(s.train)))] if nb == 1 else [int(i) for i in rng.integers(len(s.train), size=nb)]
        fr = s.train[fis[0]]  # all frames of a source share one image size
        d = [data[s.name][i] for i in fis]
        v = torch.stack([x[0] for x in d])  # [B, H, W, C]
        a = None if d[0][1] is None else torch.stack([x[1] for x in d])
        w2c, K = torch.cat([x[2] for x in d]), torch.cat([x[3] for x in d])
        c = len(s.channels)
        ci = idx[s.name]
        cols = view_colors(extra[:, ci], None if extra_sh is None else extra_sh[:, :, ci], geom["means"], w2c, cfg.view_sh)
        if shade is not None:
            cols = cols * shade.factor(w2c, shade.cam_index([s.train[i].name for i in fis]), geom["means"])[..., ci]
        img, alpha, _ = render(geom, w2c, K, fr.width, fr.height, None, None, colors=cols)
        if cfg.frame_gain:
            img = img * torch.exp(gains[s.name][fis])[:, None, None, :]
        if a is not None:  # known foreground: composite both on a random background (prevents fog-like bias)
            bg = torch.rand(len(fis), 1, 1, c, device="cuda")
            gt = v * a + bg * (1 - a)
        else:  # unknown backdrop: learn its level
            bg = bgs[s.name]
            gt = v
        pred = img + (1 - alpha) * bg
        l1 = F.l1_loss(pred, gt) if cfg.loss == "l1" else F.mse_loss(pred, gt)
        ss = ssim(pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ss)
        if cfg.frame_gain:
            loss = loss + cfg.gain_prior * gains[s.name].pow(2).mean()
        if extra_sh is not None:
            loss = loss + cfg.sh_prior * extra_sh.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            if cfg.frame_gain:  # keep log-gains zero-mean per band: their mean is just the reflectance scale
                for sn, g in gains.items():
                    m = g.mean(0)
                    g.sub_(m)
                    extra[:, idx[sn]] *= torch.exp(m)
            extra.clamp_(min=0)  # reflectance is non-negative
        if callback is not None:
            callback(step, extra.detach())
        fac = exp_lr_factor(step, cfg.steps, 0.1)
        opt.param_groups[0]["lr"] = cfg.lr * fac
        if extra_sh is not None:
            opt.param_groups[2]["lr"] = cfg.lr * cfg.sh_lr_scale * fac
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            log.append({"step": step, "loss": loss.item()})
    aux = {"bg": {k: v.detach() for k, v in bgs.items()}, "gain": {k: v.detach() for k, v in gains.items()}, "shading": shade,
           "extra_sh": None if extra_sh is None else extra_sh.detach()}
    return extra.detach(), all_ch, aux, log


BLOCKS = (8, 16, 32)


@torch.no_grad()
def _ndvi_view(geom, cols, fr: ChannelFrame, jn: int, jr: int, sigma: tuple | None):
    """Render one view with per-Gaussian colours `cols` ([N, C] or [1, N, C]); NDVI errors at pixel / block /
    whole-plant scale on symmetrically selected vegetation."""
    v, a, w2c, K = _to_dev(fr)
    img, alpha, _ = render(geom, w2c, K, fr.width, fr.height, None, None, colors=cols)
    img, alpha = img[0], alpha[0, ..., 0]
    fg = (a[..., 0] > 0.5) if a is not None else torch.ones_like(alpha, dtype=torch.bool)
    nd_gt, nd_pr = ndvi(v[..., jn], v[..., jr]), ndvi(img[..., jn], img[..., jr])
    # vegetation pixels, selected symmetrically from measured AND rendered values
    # (selecting on measured NDVI alone biases the rendered NDVI low: regression to the mean)
    leaf = fg & (alpha > 0.5) & ((nd_gt + nd_pr) / 2 > 0.3) & ((v[..., jn] + img[..., jn]) / 2 > 0.2)
    L = leaf.float()
    rec = {"name": fr.name, "n_px": int(leaf.sum()), "px_err": (nd_pr - nd_gt)[leaf].cpu().numpy(), "block": {}}
    for B in BLOCKS:  # block NDVI from band sums = what a coarser sensor pixel would measure
        def bs(x):
            return F.avg_pool2d((x * L)[None, None], B, B)[0, 0]
        ok = bs(torch.ones_like(L)) >= 0.5
        gt_b = ndvi(bs(v[..., jn]), bs(v[..., jr]))[ok]
        rec["block"][B] = ((ndvi(bs(img[..., jn]), bs(img[..., jr]))[ok] - gt_b).cpu().numpy(), gt_b.cpu().numpy())
    rec["whole_gt"] = ndvi((v[..., jn] * L).sum(), (v[..., jr] * L).sum()).item()
    rec["whole_pr"] = ndvi((img[..., jn] * L).sum(), (img[..., jr] * L).sum()).item()
    if sigma is not None:  # NDVI noise of the *measurement* from read noise (lower bound: no shot noise)
        sn, sr = sigma
        N, R = v[..., jn][leaf], v[..., jr][leaf]
        sd = 2 * torch.sqrt((R * sn) ** 2 + (N * sr) ** 2) / (N + R).clamp_min(1e-6) ** 2
        rec["noise_mae"] = float((sd * np.sqrt(2 / np.pi)).mean())  # E|e| of a Gaussian error with std sd
    rec["centre"] = np.linalg.inv(fr.w2c.numpy())[:3, 3]
    return rec, (v, a, img, alpha, fg, nd_gt, nd_pr, leaf)


def _agg(recs: list[dict], n_boot: int, seed: int = 0) -> dict:
    """Error statistics pooled over views, with 95% cluster-bootstrap CIs (views are resampled, not pixels)."""
    rng = np.random.default_rng(seed)
    bi = rng.integers(0, len(recs), (n_boot, len(recs)))

    def st(errs):  # errs: per-view arrays
        n = np.array([e.size for e in errs], float)
        sa = np.array([np.abs(e).sum() for e in errs])
        sb = np.array([e.sum() for e in errs])
        ok = n[bi].sum(1) > 0
        mae_b = sa[bi].sum(1)[ok] / n[bi].sum(1)[ok]
        bias_b = sb[bi].sum(1)[ok] / n[bi].sum(1)[ok]
        return {"mae": float(sa.sum() / n.sum()), "bias": float(sb.sum() / n.sum()),
                "mae_ci95": np.percentile(mae_b, [2.5, 97.5]).tolist(),
                "bias_ci95": np.percentile(bias_b, [2.5, 97.5]).tolist()}

    out = {"pixel": st([r["px_err"] for r in recs])}
    for B in BLOCKS:
        out[f"block{B}px"] = st([r["block"][B][0] for r in recs])
    out["whole_plant_per_view"] = st([np.array([r["whole_pr"] - r["whole_gt"]]) for r in recs])
    if "noise_mae" in recs[0]:
        out["pixel_noise_floor_mae"] = float(np.average([r["noise_mae"] for r in recs], weights=[r["n_px"] for r in recs]))
    return out


@torch.no_grad()
def evaluate(geom, extra, all_ch, sources, out_dir: Path, save_n: int = 4, noise_rho: dict | None = None,
             n_boot: int = 2000, extra_sh: torch.Tensor | None = None, view_sh: int = 0, shading=None) -> dict:
    """Per-channel PSNR / MAE and vegetation NDVI errors on the held-out views of each source, plus:
    the same NDVI statistics on the training views, two baselines (constant = mean measured whole-plant NDVI of the
    training views; nearest = measured whole-plant NDVI of the training view whose camera is closest), the read-noise
    NDVI floor, 95% bootstrap CIs over views, and per-view records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in sources:
        if not s.test:
            continue
        cidx = [all_ch.index(c) for c in s.channels]
        ex = extra[:, cidx]
        sh = None if extra_sh is None else extra_sh[:, :, cidx]

        def colors_for(fr):
            w2c = fr.w2c[None].cuda()
            cols = view_colors(ex, sh, geom["means"], w2c, view_sh)
            if shading is not None:
                cols = cols * shading.factor(w2c, shading.cam_index([fr.name]), geom["means"])[..., cidx]
            return cols

        per_ch = {c: [] for c in s.channels}
        for fr in s.test:
            v, a, w2c, K = _to_dev(fr)
            cols = colors_for(fr)
            img = render(geom, w2c, K, fr.width, fr.height, None, None, colors=cols)[0][0]  # black background
            gt = v * a if a is not None else v
            fg = (a[..., 0] > 0.5) if a is not None else torch.ones_like(v[..., 0], dtype=torch.bool)
            for j, c in enumerate(s.channels):
                per_ch[c].append({"psnr": psnr(img[..., j].clamp(0, 1.5), gt[..., j].clamp(0, 1.5)).item(),
                                  "mae_fg": (img[..., j] - gt[..., j]).abs()[fg].mean().item()})
        r = {c: {k: float(np.mean([d[k] for d in v])) for k in v[0]} for c, v in per_ch.items()}
        if "nir" in s.channels and "red" in s.channels:
            jn, jr = s.channels.index("nir"), s.channels.index("red")
            sigma = (noise_rho["nir"], noise_rho["red"]) if noise_rho else None
            test, train = [], []
            for i, fr in enumerate(s.test):
                rec, (v, a, img, alpha, fg, nd_gt, nd_pr, leaf) = _ndvi_view(geom, colors_for(fr), fr, jn, jr, sigma)
                test.append(rec)
                if i < save_n:
                    def cm(x, m):
                        rgb = index_colormap(x)
                        rgb[~m] = 0
                        return rgb
                    err = (nd_pr - nd_gt).abs().clamp(0, 0.3) / 0.3
                    err_rgb = torch.stack([err, err * 0.3, 1 - err], -1)
                    err_rgb[~leaf] = 0
                    row = torch.cat([cm(nd_gt, fg), cm(nd_pr, alpha > 0.5), err_rgb], 1)
                    iio.imwrite(out_dir / f"ndvi_{fr.name.replace('/', '_')}.png", (row.cpu().numpy() * 255).astype(np.uint8))
            train = [_ndvi_view(geom, colors_for(fr), fr, jn, jr, sigma)[0] for fr in s.train]
            r["ndvi_vegetation"] = _agg(test, n_boot)
            r["ndvi_vegetation_train_views"] = _agg(train, n_boot)
            # baselines on the same held-out vegetation pixels / blocks / views
            const = float(np.mean([t["whole_gt"] for t in train]))
            tc = np.stack([t["centre"] for t in train])
            near = [train[int(np.argmin(np.linalg.norm(tc - t["centre"], axis=1)))]["whole_gt"] for t in test]
            whole_gt = np.array([t["whole_gt"] for t in test])
            blk = {B: np.concatenate([t["block"][B][1] for t in test]) for B in BLOCKS}
            r["baselines"] = {
                "constant_ndvi": const,
                "whole_plant_constant": {"mae": float(np.abs(const - whole_gt).mean()), "bias": float((const - whole_gt).mean())},
                "whole_plant_nearest_train_view": {"mae": float(np.abs(np.array(near) - whole_gt).mean()),
                                                   "bias": float((np.array(near) - whole_gt).mean())},
                **{f"block{B}px_constant": {"mae": float(np.abs(const - g).mean())} for B, g in blk.items()},
                "measured_whole_plant_std_over_train_views": float(np.std([t["whole_gt"] for t in train])),
            }
            r["per_view"] = [{"split": sp, "name": t["name"], "n_px": t["n_px"], "whole_gt": t["whole_gt"],
                              "whole_pr": t["whole_pr"], "pixel_mae": float(np.abs(t["px_err"]).mean()) if t["n_px"] else None}
                             for sp, recs in (("test", test), ("train", train)) for t in recs]
        res[s.name] = r
    return res


def export(geom, extra, all_ch, weights: torch.Tensor, cfg: Config, out: Path) -> dict:
    fields = {c: extra[:, i].cpu().numpy() for i, c in enumerate(all_ch)}
    fields["weight"] = weights.cpu().numpy()
    conf = weights >= cfg.min_weight
    stats = {"n_gaussians": int(weights.numel()), "n_confident": int(conf.sum())}
    if "nir" in all_ch and "red" in all_ch:
        nd = ndvi(extra[:, all_ch.index("nir")], extra[:, all_ch.index("red")])
        fields["ndvi"] = nd.cpu().numpy()
        veg = conf & (extra[:, all_ch.index("nir")] > 0.2) & (nd > 0.3)
        w = weights[veg]
        stats.update({"n_vegetation": int(veg.sum()), "ndvi_veg_weighted_mean": float((nd[veg] * w).sum() / w.sum()),
                      "ndvi_veg_p10_p50_p90": torch.quantile(nd[veg], torch.tensor([0.1, 0.5, 0.9], device="cuda")).tolist()})
        if "red_edge" in all_ch:
            fields["ndre"] = ndre(extra[:, all_ch.index("nir")], extra[:, all_ch.index("red_edge")]).cpu().numpy()
        # NDVI-coloured copy for viewers (SuperSplat etc.): DC colour = colormap(NDVI), low-confidence -> grey
        vis = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in geom.items()})
        col = index_colormap(nd)
        col[~conf] = 0.5
        vis["sh0"].data = rgb_to_sh0(col)[:, None, :]
        vis["shN"].data.zero_()
        export_ply(out / "point_cloud_ndvi.ply", vis)
    export_ply(out / "point_cloud_spectral.ply", geom, extra_fields=fields)
    return stats


def main() -> None:
    cfg = tyro.cli(Config)
    torch.manual_seed(cfg.seed)
    cfg.out.mkdir(parents=True, exist_ok=True)
    geom, meta = load_checkpoint(cfg.geometry)
    sources = [bands_source_from_colmap(cfg.data, cfg.channels, cfg.test_every, test_group=cfg.test_group)]
    noise_rho = None
    meta_path = cfg.data.parent / "meta.json"  # written by prep_greenhouse
    if meta_path.exists():
        pm = json.loads(meta_path.read_text())
        if "read_noise_dn" in pm:
            noise_rho = {c: pm["read_noise_dn"][c] / pm["white_dn"][c] for c in cfg.channels}
    print(f"geometry: {geom['means'].shape[0]} Gaussians | sources: " +
          ", ".join(f"{s.name}({len(s.train)} train / {len(s.test)} test, {s.channels})" for s in sources))
    t0 = time.time()
    extra, all_ch, aux, log = train_channels(geom, sources, cfg)
    weights = blend_weights(geom, [f for s in sources for f in s.train])
    results = {"train_minutes": (time.time() - t0) / 60, "config": {k: str(v) for k, v in vars(cfg).items()},
               "eval": evaluate(geom, extra, all_ch, sources, cfg.out / "renders", noise_rho=noise_rho, n_boot=cfg.n_boot,
                                extra_sh=aux["extra_sh"], view_sh=cfg.view_sh, shading=aux["shading"])}
    if aux["shading"] is not None:
        results["shading"] = aux["shading"].summary()
        torch.save(aux["shading"].state_dict() | {"cams": aux["shading"].cams}, cfg.out / "shading.pt")
    if aux["extra_sh"] is not None:  # how much of the appearance is view-dependent, per channel
        rms = aux["extra_sh"].pow(2).sum(1).sqrt()  # [N, C]
        results["view_sh"] = {"degree": cfg.view_sh, "rms_sh_over_dc": {c: float((rms[:, i] / extra[:, i].clamp_min(1e-3)).median())
                                                                           for i, c in enumerate(all_ch)}}
    if cfg.frame_gain:
        results["frame_gain"] = {s.name: {"frames": [f.name for f in s.train], "channels": s.channels,
                                          "log_gain": aux["gain"][s.name].cpu().tolist()} for s in sources}
    results["export"] = export(geom, extra, all_ch, weights, cfg, cfg.out)
    ck = {**geom, "extra": torch.nn.Parameter(extra)}
    if aux["extra_sh"] is not None:
        ck["extra_sh"] = torch.nn.Parameter(aux["extra_sh"])
    save_checkpoint(cfg.out / "spectral.pt", torch.nn.ParameterDict(ck), {**meta, "channels": all_ch, "view_sh": cfg.view_sh})
    torch.save({"weights": weights.cpu(), "channels": all_ch}, cfg.out / "weights.pt")
    (cfg.out / "results.json").write_text(json.dumps(results, indent=2))
    (cfg.out / "train_log.json").write_text(json.dumps(log))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
