"""Camera + image datasets. All poses are stored in OpenCV convention (x right, y down, z forward),
which is what gsplat expects for `viewmats` (world-to-camera)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


@dataclass
class Frame:
    name: str
    K: torch.Tensor  # [3, 3] float32
    w2c: torch.Tensor  # [4, 4] float32, OpenCV convention
    width: int
    height: int
    image: torch.Tensor  # [H, W, C] uint8 (C = 3 or 4; alpha last if present)


@dataclass
class SceneData:
    train: list[Frame]
    test: list[Frame]
    scene_scale: float  # ~radius of camera positions, used for LR / densification scaling
    init_points: np.ndarray | None = None  # [M, 3] optional SfM points
    init_colors: np.ndarray | None = None  # [M, 3] in [0, 1]


_BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])  # flip y and z camera axes


def _load_blender_split(root: Path, split: str, downscale: int) -> list[Frame]:
    meta = json.loads((root / f"transforms_{split}.json").read_text())
    frames: list[Frame] = []
    for fr in meta["frames"]:
        path = root / (fr["file_path"].removeprefix("./") + ".png")
        img = iio.imread(path)  # [H, W, 4] uint8
        if downscale > 1:
            img = img[::downscale, ::downscale]  # cheap; fine for smoke tests
        h, w = img.shape[:2]
        focal = 0.5 * w / np.tan(0.5 * meta["camera_angle_x"])
        K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float32)
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64) @ _BLENDER_TO_OPENCV
        w2c = np.linalg.inv(c2w).astype(np.float32)
        frames.append(
            Frame(path.stem, torch.from_numpy(K), torch.from_numpy(w2c), w, h, torch.from_numpy(np.ascontiguousarray(img)))
        )
    return frames


def load_blender(root: str | Path, downscale: int = 1, test_every: int = 1) -> SceneData:
    """NeRF-synthetic format (transforms_{train,test}.json, RGBA PNGs)."""
    root = Path(root)
    train = _load_blender_split(root, "train", downscale)
    test = _load_blender_split(root, "test", downscale)[::test_every]
    cam_pos = np.stack([np.linalg.inv(f.w2c.numpy())[:3, 3] for f in train])
    scene_scale = float(np.linalg.norm(cam_pos - cam_pos.mean(0), axis=1).max()) * 1.1
    return SceneData(train=train, test=test, scene_scale=scene_scale)


def composite(image_u8: torch.Tensor, bg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """uint8 [..., H, W, C] -> float RGB composited on `bg` [C_rgb], plus alpha (or None)."""
    img = image_u8.float() / 255.0
    if img.shape[-1] == 4:
        rgb, a = img[..., :3], img[..., 3:]
        return rgb * a + bg * (1 - a), a
    return img, None


def load_colmap(
    root: str | Path,
    test_every: int = 8,
    images: str = "images",
    masks: str | None = "masks",
    sparse: str = "sparse/0",
    test_group: int = 1,
) -> SceneData:
    """COLMAP PINHOLE reconstruction + images (+ optional foreground masks -> alpha channel).

    Image names in the model (e.g. "cam1/000010.png") are resolved under `root/images` and `root/masks`.
    Frames are grouped by (folder, (frame number - 1) // test_group), using the last number in the file name, so groups
    stay aligned with the capture even when some frames failed to register; every `test_every`-th group (in name
    order) is held out for testing (test_group=1: every k-th frame; test_every=0: all frames train).  For a turntable
    that shoots several heights per azimuth stop (Greenhouse houseplant/tomato: 4, cotton: 5), test_group = frames per
    stop holds out whole stops = genuinely new azimuths.
    """
    import pycolmap

    root = Path(root)
    rec = pycolmap.Reconstruction(root / sparse)
    frames: list[Frame] = []
    for img in sorted(rec.images.values(), key=lambda im: im.name):
        cam = rec.cameras[img.camera_id]
        if cam.model.name not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError(f"{img.name}: camera model {cam.model.name}; undistort first")
        K = cam.calibration_matrix().astype(np.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3] = img.cam_from_world().matrix()
        rgb = iio.imread(root / images / img.name)[..., :3]
        if masks is not None:
            a = iio.imread(root / masks / img.name)
            rgb = np.concatenate([rgb, (a if a.ndim == 2 else a[..., 0])[..., None]], -1)
        frames.append(Frame(img.name.removesuffix(".png"), torch.from_numpy(K), torch.from_numpy(w2c),
                            cam.width, cam.height, torch.from_numpy(np.ascontiguousarray(rgb))))
    if test_every > 0:
        def group(i, name):
            nums = re.findall(r"\d+", Path(name).name)
            return (str(Path(name).parent), (int(nums[-1]) - 1) // test_group) if nums else ("", i // test_group)
        keys = [group(i, f.name) for i, f in enumerate(frames)]
        rank = {k: j for j, k in enumerate(dict.fromkeys(keys))}
        is_test = [rank[k] % test_every == 0 for k in keys]
        test = [f for f, t in zip(frames, is_test) if t]
        train = [f for f, t in zip(frames, is_test) if not t]
    else:
        train, test = frames, []
    cam_pos = np.stack([np.linalg.inv(f.w2c.numpy())[:3, 3] for f in train])
    scene_scale = float(np.linalg.norm(cam_pos - cam_pos.mean(0), axis=1).max()) * 1.1
    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    cols = np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0
    return SceneData(train=train, test=test, scene_scale=scene_scale, init_points=pts, init_colors=cols)
