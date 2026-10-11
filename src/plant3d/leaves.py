"""Per-leaf statistics: segment leaves with SAM in a few training views, lift the 2D masks onto the frozen Gaussians,
merge them across views into 3D leaves, then measure each leaf's NDVI in every view.

    uv run python -m plant3d.leaves --spectral outputs/v3/houseplant_spec_s0/spectral.pt \
        --data data/greenhouse/houseplant_prep/undistorted_v3 --out outputs/v3/leaves

Lifting uses the same exact-gradient trick as `blend_weights`: rendering colours = ones [N, M] and back-propagating
sum(image * one_hot_labels) gives, for every Gaussian, its blending weight inside each of the M masks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import tyro

from plant3d.gaussians import load_checkpoint, render
from plant3d.spectral import ndvi
from plant3d.train_channels import ChannelFrame, bands_source_from_colmap


@dataclass
class Args:
    spectral: Path = Path("outputs/v3/houseplant_spec_s0/spectral.pt")
    data: Path = Path("data/greenhouse/houseplant_prep/undistorted_v3")
    out: Path = Path("outputs/v3/leaves")
    test_every: int = 4
    test_group: int = 4
    ref_height_idx: int = 1  # reference views: training frames at this height index within each azimuth stop
    frames_per_stop: int = 4
    sam_model: str = "facebook/sam-vit-base"
    min_mask_px: int = 150
    max_mask_px: int = 30000
    min_veg_frac: float = 0.6  # a SAM mask counts as a leaf if this share of its pixels is vegetation
    assign_frac: float = 0.6  # a Gaussian joins a mask if this share of its in-view weight falls inside it
    merge_overlap: float = 0.3  # merge a view's leaf into a 3D leaf if this share of its weight is already there
    min_leaf_px: int = 200  # per-view leaf statistics need at least this many vegetation pixels


def vegetation(v: torch.Tensor, jn: int, jr: int) -> torch.Tensor:
    return (ndvi(v[..., jn], v[..., jr]) > 0.3) & (v[..., jn] > 0.2)


def sam_leaf_labels(gen, image: np.ndarray, veg: np.ndarray, a: Args) -> np.ndarray:
    """SAM automatic masks -> label map (0 = none). Larger valid masks win where masks overlap (parts -> leaf)."""
    from PIL import Image

    out = gen(Image.fromarray(image), points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.92)
    masks = [np.asarray(m, bool) for m in out["masks"]]
    keep = [m for m in masks if a.min_mask_px <= m.sum() <= a.max_mask_px and veg[m].mean() >= a.min_veg_frac]
    lab = np.zeros(image.shape[:2], np.int32)
    for i, m in enumerate(sorted(keep, key=lambda m: m.sum())):
        lab[m] = i + 1
    return lab


def mask_weights(geom, fr: ChannelFrame, lab: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, M] blending weight of every Gaussian inside each label of `lab`, and [N] its total weight in the view."""
    n, m = geom["means"].shape[0], int(lab.max())
    onehot = torch.nn.functional.one_hot(torch.from_numpy(lab).long().cuda(), m + 1).float()  # [H, W, M+1]
    ones = torch.ones(n, m + 1, device="cuda", requires_grad=True)
    with torch.enable_grad():
        img, _, _ = render(geom, fr.w2c[None].cuda(), fr.K[None].cuda(), fr.width, fr.height, None, None, colors=ones)
        (img[0] * onehot).sum().backward()
    g = ones.grad
    total = g.sum(1)  # channel 0 = unlabelled pixels, so the sum over channels is the weight in the whole view
    return g[:, 1:], total


def lift(geom, frames, labs, a: Args):
    """Greedy merge of per-view leaves into 3D leaves. Returns per-Gaussian leaf id (-1 none) and weights [N, L]."""
    leaves: list[torch.Tensor] = []  # accumulated per-Gaussian weight of each 3D leaf
    for fr, lab in zip(frames, labs):
        if lab.max() == 0:
            continue
        w, total = mask_weights(geom, fr, lab)
        best = w.max(1)
        own = (best.values >= a.assign_frac * total.clamp_min(1e-6)) & (best.values >= 0.05)
        for j in range(w.shape[1]):
            wj = torch.where(own & (best.indices == j), w[:, j], torch.zeros_like(w[:, j]))
            if wj.sum() <= 0:
                continue
            ov = [((wj > 0) & (L > 0)).float().mul(wj).sum() / wj.sum() for L in leaves]
            k = int(np.argmax([o.item() for o in ov])) if ov else -1
            if k >= 0 and ov[k] >= a.merge_overlap:
                leaves[k] = leaves[k] + wj
            else:
                leaves.append(wj.clone())
    W = torch.stack(leaves, 1)  # [N, L]
    has = W.sum(1) > 0
    label = torch.where(has & (W.max(1).values >= 0.5 * W.sum(1)), W.argmax(1), torch.full_like(has, -1, dtype=torch.long))
    return label, W


@torch.no_grad()
def label_map(geom, fr: ChannelFrame, label: torch.Tensor, n_leaves: int) -> torch.Tensor:
    """Render the 3D leaf labels into a view: per pixel the dominant leaf (-1 if none covers > 50%)."""
    oh = torch.zeros(label.shape[0], n_leaves, device="cuda")
    ok = label >= 0
    oh[ok, label[ok]] = 1
    img, _, _ = render(geom, fr.w2c[None].cuda(), fr.K[None].cuda(), fr.width, fr.height, None, None, colors=oh)
    v, i = img[0].max(-1)
    return torch.where(v > 0.5, i, torch.full_like(i, -1))


@torch.no_grad()
def per_leaf_table(geom, extra, channels, frames_split, label, n_leaves, a: Args) -> pd.DataFrame:
    """One row per (leaf, view): measured and rendered NDVI from band sums over the leaf's vegetation pixels.
    Pixels are selected symmetrically (measured AND rendered vegetation), as in train_channels.evaluate."""
    jn, jr = channels.index("nir"), channels.index("red")
    rows = []
    for split, frames in frames_split.items():
        for fr in frames:
            v, a_ = fr.values.cuda(), fr.alpha.cuda()[..., 0]
            img, alpha, _ = render(geom, fr.w2c[None].cuda(), fr.K[None].cuda(), fr.width, fr.height, None, None,
                                   colors=extra)
            img, alpha = img[0], alpha[0, ..., 0]
            nd_gt, nd_pr = ndvi(v[..., jn], v[..., jr]), ndvi(img[..., jn], img[..., jr])
            sel = (a_ > 0.5) & (alpha > 0.5) & ((nd_gt + nd_pr) / 2 > 0.3) & ((v[..., jn] + img[..., jn]) / 2 > 0.2)
            lm = label_map(geom, fr, label, n_leaves)
            for k in range(n_leaves):
                px = sel & (lm == k)
                n = int(px.sum())
                if n < a.min_leaf_px:
                    continue
                rows.append({"leaf": k, "view": fr.name, "split": split, "n_px": n,
                             "ndvi_measured": ndvi(v[..., jn][px].sum(), v[..., jr][px].sum()).item(),
                             "ndvi_rendered": ndvi(img[..., jn][px].sum(), img[..., jr][px].sum()).item()})
    return pd.DataFrame(rows)


def main() -> None:
    a = tyro.cli(Args)
    from transformers import pipeline

    a.out.mkdir(parents=True, exist_ok=True)
    sp, meta = load_checkpoint(a.spectral)
    extra = sp["extra"].data
    geom = {k: v.detach().requires_grad_(False) for k, v in sp.items() if k not in ("extra", "extra_sh")}  # DC term only
    ch = meta["channels"]
    jn, jr = ch.index("nir"), ch.index("red")
    src = bands_source_from_colmap(a.data, ch, a.test_every, test_group=a.test_group)
    refs = [fr for fr in src.train if (int(fr.name[-6:]) - 1) % a.frames_per_stop == a.ref_height_idx]
    gen = pipeline("mask-generation", model=a.sam_model, device=0)
    labs = []
    for fr in refs:
        cir = iio.imread(a.data / "images" / f"{fr.name}.png")
        veg = (vegetation(fr.values, jn, jr) & (fr.alpha[..., 0] > 0.5)).numpy()
        labs.append(sam_leaf_labels(gen, cir, veg, a))
        print(f"{fr.name}: {labs[-1].max()} leaf masks")
    label, W = lift(geom, refs, labs, a)
    n_leaves = W.shape[1]
    table = per_leaf_table(geom, extra, ch, {"train": src.train, "test": src.test}, label, n_leaves, a)
    # 3D per-leaf NDVI straight from the Gaussians: band sums weighted by each Gaussian's weight in its leaf
    g3d = []
    for k in range(n_leaves):
        m = label == k
        if m.sum() == 0:
            continue
        w = W[m, k]
        g3d.append({"leaf": k, "n_gaussians": int(m.sum()),
                    "ndvi_gaussians": ndvi((extra[m, jn] * w).sum(), (extra[m, jr] * w).sum()).item()})
    np.savez_compressed(a.out / "ref_labels.npz", **{fr.name.replace("/", "_"): lab for fr, lab in zip(refs, labs)})
    torch.save({"label": label.cpu(), "weights": W.cpu(), "refs": [fr.name for fr in refs]}, a.out / "gaussian_leaves.pt")
    table.to_csv(a.out / "per_leaf_view.csv", index=False)
    pd.DataFrame(g3d).to_csv(a.out / "per_leaf_gaussians.csv", index=False)
    # the per-leaf product: measured NDVI pooled over every view the leaf is seen in (3D model = pixel association)
    agg = table.groupby("leaf").agg(views=("view", "size"), median_px=("n_px", "median"),
                                    ndvi_measured_mean=("ndvi_measured", "mean"), ndvi_measured_std=("ndvi_measured", "std"))
    agg["ndvi_measured_se"] = agg.ndvi_measured_std / np.sqrt(agg.views)
    agg = agg.join(pd.DataFrame(g3d).set_index("leaf"))
    agg.to_csv(a.out / "per_leaf_summary.csv")
    summary = {"n_leaves_3d": n_leaves, "n_gaussians_labelled": int((label >= 0).sum()),
               "n_leaf_views": len(table), "leaves_seen_in_test": int(table[table.split == "test"].leaf.nunique())}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
