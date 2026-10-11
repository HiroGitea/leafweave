"""Leaf-orientation shading for turntable captures: observed value = reflectance x shading(normal, light, view).

The lamps and cameras are fixed in the lab while the turntable rotates the plant, so in the plant's (world) frame the
light direction turns with the turntable. Each camera stays at one lab azimuth (it only moves up and down), so writing
the light in the frame (h, b, axis) -- h = horizontal unit vector from the turntable axis towards that camera,
b = axis x h -- makes it a constant per camera (cameras need not be synchronised). Leaves are thin and two-sided:

    s = ambient_b + max(0, n . l) + transmission_b * max(0, -n . l)

with the normal n of each (flat) Gaussian flipped to face the camera, so "-n . l > 0" means the leaf is lit from behind
and the camera sees light transmitted through it (leaves transmit a lot of NIR and little red: transmission_b is
learned per band, as is the ambient level). Reflectance then carries no orientation or lighting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_normals(geom) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit normal = axis of the smallest scale, and flatness = smallest / middle scale (0 = flat disc)."""
    q = F.normalize(geom["quats"], dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)
    s = geom["scales"]
    order = s.argsort(-1)
    n = torch.gather(R, 2, order[:, None, :1].expand(-1, 3, 1))[..., 0]
    ss = torch.gather(s, 1, order).exp()
    return F.normalize(n, dim=-1), ss[:, 0] / ss[:, 1].clamp_min(1e-9)


@torch.no_grad()
def smooth_normals(normals: torch.Tensor, means: torch.Tensor, k: int = 16, chunk: int = 4096) -> torch.Tensor:
    """Average each normal with its k nearest Gaussians (signs aligned first). Per-Gaussian normals of a 3DGS fit are
    noisy (neighbours on one leaf disagree by ~25 deg); a leaf is smooth at that scale."""
    out = torch.empty_like(normals)
    for i in range(0, len(means), chunk):
        nn = torch.cdist(means[i:i + chunk], means).topk(k + 1, largest=False).indices  # includes itself
        nb = normals[nn]  # [c, k+1, 3]
        sgn = torch.sign((nb * normals[i:i + chunk, None]).sum(-1, keepdim=True) + 1e-9)
        out[i:i + chunk] = F.normalize((nb * sgn).sum(1), dim=-1)
    return out


def turntable_frame(w2cs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Turntable axis (pointing up, i.e. against the cameras' image-down direction) and a point on it."""
    c2w = torch.linalg.inv(w2cs.double())
    C, D, down = c2w[:, :3, 3], c2w[:, :3, 2], c2w[:, :3, 1]
    axis = torch.linalg.svd(C - C.mean(0))[2][2]
    axis = axis * torch.sign(-(down @ axis).mean())
    P = torch.eye(3, dtype=C.dtype, device=C.device)[None] - D[:, :, None] * D[:, None, :]
    pivot = torch.linalg.solve(P.sum(0), (P @ C[:, :, None]).sum(0))[:, 0]
    return axis.float(), pivot.float()


class Shading(torch.nn.Module):
    def __init__(self, geom, w2cs_all: torch.Tensor, cams: list[str], n_channels: int, normal_k: int = 0):
        super().__init__()
        n, flat = gaussian_normals(geom)
        if normal_k > 0:
            n = smooth_normals(n, geom["means"].detach(), normal_k)
        self.register_buffer("normals", n)
        self.register_buffer("flatness", flat)
        axis, pivot = turntable_frame(w2cs_all)
        self.register_buffer("axis", axis)
        self.register_buffer("pivot", pivot)
        self.cams = list(cams)
        # light in each camera's (h, b, axis) frame; start above and towards the camera
        self.light = torch.nn.Parameter(torch.tensor([[0.8, 0.0, 0.6]] * len(cams)))
        self.log_ambient = torch.nn.Parameter(torch.zeros(n_channels))  # ambient = 1 x direct at start
        self.log_trans = torch.nn.Parameter(torch.full((n_channels,), -1.2))  # transmission ~0.3 at start

    def cam_index(self, names) -> torch.Tensor:
        return torch.tensor([self.cams.index(nm.split("/")[0]) for nm in names], device=self.light.device)

    def light_dirs(self, w2c: torch.Tensor, cam_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World light direction per view [B, 3] and camera centres [B, 3]."""
        c = torch.linalg.inv(w2c)[:, :3, 3]
        rel = c - self.pivot
        h = F.normalize(rel - (rel @ self.axis)[:, None] * self.axis, dim=-1)
        b = torch.cross(self.axis.expand_as(h), h, dim=-1)
        L = F.normalize(self.light[cam_idx], dim=-1)
        return L[:, :1] * h + L[:, 1:2] * b + L[:, 2:] * self.axis, c

    def factor(self, w2c: torch.Tensor, cam_idx: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        """Shading factor [B, N, C] for every Gaussian and channel."""
        l, c = self.light_dirs(w2c, cam_idx)
        view = c[:, None] - means[None]  # [B, N, 3]
        n = self.normals[None] * torch.sign((self.normals[None] * view).sum(-1, keepdim=True) + 1e-9)
        cos = (n * l[:, None]).sum(-1, keepdim=True)  # [B, N, 1]
        a, t = self.log_ambient.exp(), self.log_trans.exp()
        return a + F.relu(cos) + t * F.relu(-cos)

    def summary(self) -> dict:
        return {"light_in_camera_frame": {c: F.normalize(self.light[i], dim=-1).tolist() for i, c in enumerate(self.cams)},
                "ambient": self.log_ambient.exp().tolist(), "transmission": self.log_trans.exp().tolist()}
