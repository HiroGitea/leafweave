"""Preprocess the Multispectral Greenhouse dataset (MSIS-AGRI-1-A, 4-lens snapshot camera) for 3DGS.

Raw TIFF = 1024x1024 uint16, four 512x512 quadrants, one lens per band (verified against the
dataset's own per-band PNGs, R^2 = 1.0):  TL 580 nm green | TR 660 nm red | BL 735 nm red-edge | BR 820 nm NIR.
Values are 10-bit DN stored << 6.  Inter-lens parallax is a constant sub-pixel shift per camera (~0.6 px, measured by
phase correlation against NIR); all bands share NIR's pose and are resampled onto the NIR grid during undistortion.

Radiometry:  rho_b = (DN_b - dark_b) / W_b
  dark_b : per-frame 1st percentile (sensor black + roughly uniform veiling glare)
  W_b    : white reference from the white turntable disk / pot, robust median over the whole sequence
           (LED illumination is fixed), assumed spectrally flat across 580-820 nm.  The leaves' whitish veins are
           also bright in every band but have ~3x lower red/NIR than white surfaces; they are excluded, otherwise
           W_red is pulled down by a third in frames where veins outnumber white pixels.

Outputs (under --out):
  images/cam{1,2}/NNNNNN.png   CIR false colour (R=NIR, G=red, B=green), gamma 1/2.2, for COLMAP + geometry
  masks/cam{1,2}/NNNNNN.png    foreground (plant + pot + turntable); also copied as COLMAP masks
  bands/cam{1,2}/NNNNNN.npy    float16 [4, H, W] relative reflectance, band order green, red, red_edge, nir
  sparse/0                     COLMAP reconstruction of all frames (SIMPLE_RADIAL), largest model only
  undistorted/{images,masks,bands,sparse/0}   undistorted to PINHOLE, bands aligned to NIR -- what training reads
  meta.json

    uv run python -m plant3d.prep_greenhouse --src data/greenhouse/houseplant --out data/greenhouse/houseplant_prep
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import tifffile
import tyro

BANDS = ["green", "red", "red_edge", "nir"]
WAVELENGTHS_NM = [580, 660, 735, 820]


@dataclass
class Args:
    src: Path = Path("data/greenhouse/houseplant")
    out: Path = Path("data/greenhouse/houseplant_prep")
    dark_percentile: float = 1.0
    white_candidates_percentile: float = 98.0  # white-reference candidates: top 2% "whiteness" pixels per frame
    white_flat_frac: float = 0.6  # keep candidates with red/NIR > this x their 90th percentile (drops leaf veins)
    mask_nir_thresh: float = 0.28  # NIR reflectance rel. to the white pot: black backdrop ~0.07, leaves ~1.6
    corner_radius: float = 0.62  # drop mask beyond this radius (x image width) from centre: NIR lens vignetting halo
    min_component_px: int = 150
    skip_sfm: bool = False
    camera_model: str = "SIMPLE_RADIAL"
    sift_peak_threshold: float = 0.002  # COLMAP default 0.0067; images are dark and the plant covers ~10% of the frame
    abs_pose_min_inliers: int = 15  # COLMAP default 30
    abs_pose_min_inlier_ratio: float = 0.15  # COLMAP default 0.25
    two_view_min_inliers: int = 15  # COLMAP default 15; thin plants (cotton) get < 15 matches between azimuth stops
    max_num_features: int = 8192
    align_bands: bool = True  # resample green / red / red-edge onto the NIR grid (removes ~0.6 px lens parallax)
    undistorted_dir: str = "undistorted"
    band_interp: str = "cubic"  # resampling of the reflectance bands: "cubic" keeps vein contrast, "linear" smooths


def read_bands(path: Path) -> np.ndarray:
    raw = tifffile.imread(path).astype(np.float32) / 64.0
    h, w = raw.shape[0] // 2, raw.shape[1] // 2
    return np.stack([raw[:h, :w], raw[:h, w:], raw[h:, :w], raw[h:, w:]])  # green, red, red_edge, nir


def foreground_mask(nir_rho: np.ndarray, thresh: float, min_px: int, corner_radius: float = 0.62) -> np.ndarray:
    sm = cv2.GaussianBlur(nir_rho, (0, 0), 1.0)
    h, w = sm.shape
    yy, xx = np.mgrid[:h, :w]
    corner = np.hypot(yy - h / 2, xx - w / 2) > corner_radius * w
    m = ((sm > thresh) & ~corner).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h = stats[big, :4]
        pad = int(0.15 * max(w, h))
        for i in range(1, n):  # largest blob + any sizeable blob near it (detached leaves)
            xi, yi, wi, hi, a = stats[i]
            near = xi < x + w + pad and xi + wi > x - pad and yi < y + h + pad and yi + hi > y - pad
            if i == big or (a >= min_px and near):
                keep[lab == i] = 1
    # fill holes
    ff = keep.copy()
    cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 1)
    return (keep | (1 - ff)).astype(bool)


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def read_noise(dn: np.ndarray) -> np.ndarray:
    """Per-band temporal-noise proxy (DN) from the black backdrop: std of horizontal neighbour differences / sqrt 2.
    Backdrop = the darker half of the frame by NIR inside the vignetting circle. Shot noise on bright pixels is not
    included, so NDVI noise derived from this is a lower bound."""
    h, w = dn.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    bg = (np.hypot(yy - h / 2, xx - w / 2) < 0.45 * w) & (dn[3] < np.percentile(dn[3], 50))
    bg = bg[:, :-1] & bg[:, 1:]
    d = np.diff(dn, axis=2)
    out = []
    for b in range(4):  # trimmed std (DN are integers, so MAD would be quantised)
        v = d[b][bg]
        lo, hi = np.percentile(v, [0.5, 99.5])
        out.append(v[(v >= lo) & (v <= hi)].std() / np.sqrt(2))
    return np.array(out)


def band_shift(dn: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Sub-pixel shift d_b of each band relative to NIR, i.e. band_b(x) = nir(x - d_b). Returns [3, 2] (x, y)."""
    sig = np.clip(dn - dark[:, None, None], 0, None)
    bright = sig[3] > 0.25 * np.percentile(sig[3], 99.9)
    ys, xs = np.nonzero(bright)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    win = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)

    def norm(x):
        x = np.log1p(x[y0:y1, x0:x1])
        return ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)

    ref = norm(sig[3])
    return np.array([cv2.phaseCorrelate(ref, norm(sig[b]), win)[0] for b in range(3)])


def estimate_band_shifts(frames, darks) -> dict:
    """Median per-camera band shift (the lenses are rigid, so it is a camera constant)."""
    out = {}
    for cam in sorted({c for c, _, _ in frames}):
        sh = np.array([band_shift(read_bands(p), d) for (c, _, p), d in zip(frames, darks) if c == cam])
        out[cam] = {b: np.median(sh[:, i], axis=0).round(3).tolist() for i, b in enumerate(BANDS[:3])}
        out[cam]["frame_std_px"] = sh.std(0).mean().round(3).item()
    return out


def prep(a: Args) -> dict:
    frames = []
    for cam in (1, 2):
        for p in sorted((a.src / "TIFF" / f"Camera{cam}").glob("*.tif")):
            frames.append((f"cam{cam}", p.stem.split("_")[-1], p))
    print(f"{len(frames)} frames")

    # pass 1: dark levels and white reference
    darks, whites, flat_frac, noise = [], [], [], []
    for _, _, p in frames:
        dn = read_bands(p)
        dark = np.percentile(dn.reshape(4, -1), a.dark_percentile, axis=1)
        noise.append(read_noise(dn))
        sig = dn - dark[:, None, None]
        scale = np.percentile(sig.reshape(4, -1), 99.9, axis=1)
        whiteness = (sig / scale[:, None, None]).min(0)  # bright in every band -> white surface or leaf vein
        cand = sig[:, whiteness > np.percentile(whiteness, a.white_candidates_percentile)]
        rn = cand[1] / np.maximum(cand[3], 1.0)
        flat = rn > a.white_flat_frac * np.percentile(rn, 90)  # spectrally flat: white pot / disk, not veins
        darks.append(dark)
        whites.append(np.median(cand[:, flat], axis=1))
        flat_frac.append(flat.mean())
    darks, whites, noise = np.array(darks), np.array(whites), np.array(noise)
    W = np.median(whites, axis=0)
    print("dark DN  mean per band:", dict(zip(BANDS, darks.mean(0).round(1))))
    print("white DN median per band:", dict(zip(BANDS, W.round(1))), " frame-to-frame CV:",
          dict(zip(BANDS, (whites.std(0) / whites.mean(0)).round(3))), f" | white candidates kept {np.mean(flat_frac):.0%}")

    # pass 2: write outputs
    for sub in ("images", "masks", "bands", "colmap_masks"):
        shutil.rmtree(a.out / sub, ignore_errors=True)
    fg_frac = []
    for (cam, idx, p), dark in zip(frames, darks):
        rho = (read_bands(p) - dark[:, None, None]) / W[:, None, None]
        mask = foreground_mask(rho[3], a.mask_nir_thresh, a.min_component_px, a.corner_radius)
        fg_frac.append(mask.mean())
        cir = np.stack([rho[3], rho[1], rho[0]], -1)
        for sub, arr in (("images", to_u8(cir)), ("masks", mask.astype(np.uint8) * 255)):
            (a.out / sub / cam).mkdir(parents=True, exist_ok=True)
            iio.imwrite(a.out / sub / cam / f"{idx}.png", arr)
        (a.out / "colmap_masks" / cam).mkdir(parents=True, exist_ok=True)
        iio.imwrite(a.out / "colmap_masks" / cam / f"{idx}.png.png", mask.astype(np.uint8) * 255)
        (a.out / "bands" / cam).mkdir(parents=True, exist_ok=True)
        np.save(a.out / "bands" / cam / f"{idx}.npy", rho.astype(np.float16))
    meta = {
        "source": "Multispectral Greenhouse (LGDWT-GS), doi:10.5281/zenodo.18838172, CC-BY-4.0",
        "bands": BANDS, "wavelengths_nm": WAVELENGTHS_NM,
        "quadrant_layout": {"TL": "green", "TR": "red", "BL": "red_edge", "BR": "nir"},
        "white_dn": dict(zip(BANDS, W.tolist())),
        "dark_dn_mean": dict(zip(BANDS, darks.mean(0).tolist())),
        "white_cv": dict(zip(BANDS, (whites.std(0) / whites.mean(0)).tolist())),
        "n_frames": len(frames), "fg_fraction_mean": float(np.mean(fg_frac)),
        "white_candidates_kept": float(np.mean(flat_frac)),
        "read_noise_dn": dict(zip(BANDS, np.median(noise, axis=0).tolist())),
        "band_shift_px": estimate_band_shifts(frames, darks),
        "note": ("White reference = vertical pot + horizontal turntable disk; both give the same red/NIR, so the "
                 "band LEDs light them alike. Whitish leaf veins (red/NIR ~3x lower) are excluded from W."),
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"foreground fraction mean {np.mean(fg_frac):.3f} (min {np.min(fg_frac):.3f}, max {np.max(fg_frac):.3f})")
    return meta


def run_sfm(a: Args) -> None:
    import pycolmap

    db = a.out / "database.db"
    db.unlink(missing_ok=True)
    shutil.rmtree(a.out / "sparse", ignore_errors=True)
    reader = pycolmap.ImageReaderOptions(camera_model=a.camera_model, mask_path=str(a.out / "colmap_masks"))
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = a.max_num_features
    ext.sift.estimate_affine_shape = True
    ext.sift.domain_size_pooling = True
    ext.sift.peak_threshold = a.sift_peak_threshold
    ext.use_gpu = False
    pycolmap.extract_features(db, a.out / "images", camera_mode=pycolmap.CameraMode.PER_FOLDER,
                              reader_options=reader, extraction_options=ext)
    match = pycolmap.FeatureMatchingOptions()
    match.guided_matching = True
    match.use_gpu = False
    verify = pycolmap.TwoViewGeometryOptions()
    verify.min_num_inliers = a.two_view_min_inliers
    pycolmap.match_exhaustive(db, matching_options=match, verification_options=verify)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.mapper.abs_pose_min_num_inliers = a.abs_pose_min_inliers
    opts.mapper.abs_pose_min_inlier_ratio = a.abs_pose_min_inlier_ratio
    recs = pycolmap.incremental_mapping(db, a.out / "images", a.out / "sparse", options=opts)
    for k, r in recs.items():
        print(f"model {k}: {r.num_reg_images()} images, {r.num_points3D()} points, "
              f"mean reproj err {r.compute_mean_reprojection_error():.3f} px")
        for c in r.cameras.values():
            print("   camera", c.camera_id, c.model, c.params.round(4))
    # keep only the largest model as sparse/0
    best = max(recs, key=lambda k: recs[k].num_reg_images())
    tmp = a.out / "_sparse_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (a.out / "sparse" / str(best)).rename(tmp)
    shutil.rmtree(a.out / "sparse")
    (a.out / "sparse").mkdir()
    tmp.rename(a.out / "sparse" / "0")


def undistort(a: Args) -> None:
    """SIMPLE_RADIAL/RADIAL/OPENCV -> PINHOLE with the same focal length and image size (barrel: crops edges)."""
    import pycolmap

    rec = pycolmap.Reconstruction(a.out / "sparse" / "0")
    dst = a.out / a.undistorted_dir
    shifts = json.loads((a.out / "meta.json").read_text())["band_shift_px"] if a.align_bands else {}
    shutil.rmtree(dst, ignore_errors=True)
    maps, new_cams = {}, {}
    for cid, cam in rec.cameras.items():
        p = cam.params
        if cam.model.name == "SIMPLE_RADIAL":
            f, cx, cy, k1 = p
            K, dist = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), np.array([k1, 0, 0, 0])
        elif cam.model.name == "OPENCV":
            fx, fy, cx, cy, k1, k2, p1, p2 = p
            K, dist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array([k1, k2, p1, p2])
        elif cam.model.name == "PINHOLE":
            K, dist = cam.calibration_matrix(), np.zeros(4)
        else:
            raise NotImplementedError(cam.model.name)
        maps[cid] = cv2.initUndistortRectifyMap(K, dist, None, K, (cam.width, cam.height), cv2.CV_32FC1)
        new_cams[cid] = pycolmap.Camera(model="PINHOLE", width=cam.width, height=cam.height,
                                        params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]], camera_id=cid)
    for img in rec.images.values():
        mx, my = maps[img.camera_id]
        stem = img.name.removesuffix(".png")
        cam = Path(img.name).parent.name
        mask = iio.imread(a.out / "masks" / img.name)
        rho = np.load(a.out / "bands" / f"{stem}.npy").astype(np.float32)
        # one resampling per band: undistortion + lens parallax (band_b(x) = nir(x - d_b) -> sample at x + d_b)
        rho_u = []
        for i, band in enumerate(rho):
            dx, dy = shifts[cam][BANDS[i]] if (shifts and i < 3) else (0.0, 0.0)
            interp = cv2.INTER_CUBIC if a.band_interp == "cubic" else cv2.INTER_LINEAR
            rho_u.append(cv2.remap(band, mx + dx, my + dy, interp, borderMode=cv2.BORDER_CONSTANT))
        rho_u = np.clip(np.stack(rho_u), 0, None)
        out = {
            "images": to_u8(np.stack([rho_u[3], rho_u[1], rho_u[0]], -1)),  # CIR rebuilt from the aligned bands
            "masks": cv2.remap(mask, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
        }
        for sub, arr in out.items():
            (dst / sub / Path(img.name).parent).mkdir(parents=True, exist_ok=True)
            iio.imwrite(dst / sub / img.name, arr)
        (dst / "bands" / Path(img.name).parent).mkdir(parents=True, exist_ok=True)
        np.save(dst / "bands" / f"{stem}.npy", rho_u.astype(np.float16))
    for cid, cam in new_cams.items():
        rec.cameras[cid] = cam
    (dst / "sparse" / "0").mkdir(parents=True)
    rec.write(dst / "sparse" / "0")
    print(f"undistorted {rec.num_images()} images -> {dst}" + (f", bands aligned to NIR: {shifts}" if shifts else ""))


def main() -> None:
    a = tyro.cli(Args)
    prep(a)
    if not a.skip_sfm:
        run_sfm(a)
    undistort(a)


if __name__ == "__main__":
    main()
