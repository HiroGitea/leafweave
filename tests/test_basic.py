from pathlib import Path

import numpy as np
import pytest
import torch

from plant3d.gaussians import export_ply, init_params, render

FICUS = Path(__file__).parents[1] / "data/nerf_synthetic/ficus"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.skipif(not FICUS.exists(), reason="run scripts/download_ficus.sh")
def test_blender_poses_are_opencv():
    from plant3d.data import load_blender

    s = load_blender(FICUS, downscale=8, test_every=100)
    for fr in s.train[:5]:
        p = fr.w2c @ torch.tensor([0.0, 0, 0, 1])
        assert p[2] > 0  # object in front of camera
        uv = fr.K @ p[:3]
        uv = uv[:2] / uv[2]
        assert torch.allclose(uv, torch.tensor([fr.width / 2, fr.height / 2]), atol=2)
        up = fr.w2c @ torch.tensor([0.0, 0, 1, 1])  # blender world +z is up -> image y decreases
        assert (fr.K @ up[:3])[1] / up[2] < uv[1]


@cuda
def test_render_extra_channels_and_grad():
    params = init_params(2000, 0.5, sh_degree=0)
    extra = torch.rand(2000, 1, device="cuda", requires_grad=True)  # e.g. NIR
    w2c = torch.eye(4, device="cuda")[None]
    w2c[0, 2, 3] = 3.0
    K = torch.tensor([[200.0, 0, 64], [0, 200.0, 64], [0, 0, 1]], device="cuda")[None]
    img, alpha, _ = render(params, w2c, K, 128, 128, None, None, colors=extra)
    assert img.shape == (1, 128, 128, 1)
    img.sum().backward()
    assert extra.grad.abs().sum() > 0


@cuda
def test_export_ply(tmp_path):
    from plyfile import PlyData

    params = init_params(100, 1.0, sh_degree=3)
    params["extra"] = torch.nn.Parameter(torch.rand(100, 2, device="cuda"))
    export_ply(tmp_path / "x.ply", params)
    names = PlyData.read(str(tmp_path / "x.ply"))["vertex"].data.dtype.names
    assert "f_rest_44" in names and "extra_1" in names and "rot_3" in names


@cuda
def test_blend_weights_sum_to_alpha():
    from plant3d.train_channels import ChannelFrame, blend_weights

    params = init_params(3000, 0.5, sh_degree=0)
    for p in params.values():
        p.requires_grad_(False)
    w2c = torch.eye(4)
    w2c[2, 3] = 3.0
    K = torch.tensor([[150.0, 0, 48], [0, 150.0, 48], [0, 0, 1]])
    fr = ChannelFrame("x", K, w2c, 96, 96, torch.zeros(96, 96, 1), None)
    w = blend_weights(params, [fr])
    _, alpha, _ = render(params, w2c[None].cuda(), K[None].cuda(), 96, 96, None, None,
                         colors=torch.ones(3000, 1, device="cuda"))
    assert torch.allclose(w.sum(), alpha.sum(), rtol=1e-3)
    assert (w >= 0).all()


def test_band_shift_recovers_known_parallax():
    """band_shift returns d_b with band_b(x) = nir(x - d_b); undistort samples band_b at x + d_b to align it."""
    import cv2
    from plant3d.prep_greenhouse import band_shift

    rng = np.random.default_rng(0)
    h = w = 256
    yy, xx = np.mgrid[:h, :w]
    plant = (np.hypot(yy - 128, xx - 128) < 70).astype(np.float32)
    nir = 20 + 300 * plant * cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5) * 2
    shifts = [(0.6, 0.3), (0.1, 0.37), (0.45, -0.05)]
    bands = [cv2.warpAffine(nir, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_CUBIC) for dx, dy in shifts]
    dn = np.stack(bands + [nir])
    est = band_shift(dn, np.zeros(4))
    assert np.abs(est - np.array(shifts)).max() < 0.1  # phase-correlation sub-pixel peak: ~0.05-0.07 px accurate


def test_plane_homography_identity_and_shift():
    """Same camera -> identity; a camera translated sideways maps the plane's centre to the image centre shift."""
    from types import SimpleNamespace

    from plant3d.fusion import plane_homography

    K = torch.tensor([[300.0, 0, 256], [0, 300.0, 256], [0, 0, 1]])
    w2c = torch.eye(4)
    fa = SimpleNamespace(w2c=w2c, K=K)
    centre = np.array([0.0, 0.0, 4.0])  # 4 units in front of camera a
    H = plane_homography(fa, fa, centre)
    assert np.allclose(H / H[2, 2], np.eye(3), atol=1e-9)
    w2c_b = torch.eye(4)
    w2c_b[0, 3] = -0.4  # camera b sits 0.4 to the right: points move left in its image
    fb = SimpleNamespace(w2c=w2c_b, K=K)
    p = plane_homography(fa, fb, centre) @ np.array([256.0, 256.0, 1.0])
    assert np.allclose(p[:2] / p[2], [256 - 300 * 0.4 / 4, 256], atol=1e-6)
