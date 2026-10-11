"""3D maps of leaf chlorophyll and leaf structure on the frozen Gaussians, for notebooks and SuperSplat.

Indices (all from the view-averaged / DC reflectance of each Gaussian):
  chlorophyll  CIre = R_nir / R_red_edge - 1   (red-edge chlorophyll index, Gitelson et al. 2003; ~linear in leaf
               chlorophyll content where NDVI saturates). Bands: 820 nm and 735 nm.
  structure    R_nir (820 nm): the NIR plateau is set by scattering at cell wall / air interfaces in the mesophyll,
               i.e. internal leaf structure and thickness (and by how leaves stack).

Only Gaussians that belong to a 3D leaf (plant3d.leaves) and pass the vegetation test are mapped; stems, soil and the
pot stay grey. Single Gaussians are noisy (most are seen in few pixels), so maps are built the way the evaluation says
is reliable: band values are pooled over the k nearest Gaussians *of the same leaf*, weighted by visibility, and the
index is computed from the pooled bands. `*_leaf` maps paint every Gaussian of a leaf with that leaf's pooled index.
"""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from plant3d.export_splat import quat_mul, rot_to_quat, upright_frame
from plant3d.gaussians import export_ply, load_checkpoint, render, rgb_to_sh0
from plant3d.spectral import ndvi

# sequential palettes (position in [0, 1], RGB in [0, 1])
CHL_STOPS = (np.array([0.0, 0.3, 0.6, 1.0]), np.array([[0.93, 0.89, 0.62], [0.72, 0.82, 0.38], [0.30, 0.62, 0.26], [0.04, 0.33, 0.17]]))
STRUCT_STOPS = (np.array([0.0, 0.3, 0.65, 1.0]), np.array([[0.13, 0.11, 0.24], [0.38, 0.22, 0.47], [0.80, 0.42, 0.42], [0.99, 0.88, 0.70]]))
GREY = 0.32


def ramp(x: np.ndarray, vmin: float, vmax: float, stops) -> np.ndarray:
    pos, cols = stops
    t = np.clip((np.asarray(x) - vmin) / (vmax - vmin), 0, 1)
    return np.stack([np.interp(t, pos, cols[:, c]) for c in range(3)], -1).astype(np.float32)


def load(spectral: Path, weights: Path, leaves: Path | None = None) -> dict:
    sp, meta = load_checkpoint(spectral, device="cuda")
    ch = meta["channels"]
    geom = {k: v.detach() for k, v in sp.items() if k not in ("extra", "extra_sh")}
    ex = sp["extra"].detach()
    w = torch.load(weights, weights_only=True)["weights"].cuda()
    out = {"geom": geom, "extra": ex, "weights": w, "channels": ch, "meta": meta}
    if leaves is not None and (Path(leaves) / "gaussian_leaves.pt").exists():
        gl = torch.load(Path(leaves) / "gaussian_leaves.pt", weights_only=False)
        out["leaf_label"] = gl["label"].cuda()
        out["leaf_weight"] = gl["weights"].cuda()
    return out


@torch.no_grad()
def pooled_bands(xyz: torch.Tensor, bands: torch.Tensor, w: torch.Tensor, mask: torch.Tensor, k: int = 32,
                 chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Visibility-weighted band sums over the k nearest masked Gaussians (each masked Gaussian; others untouched).
    Returns pooled bands [N, C] and the mean neighbour distance [N] (the effective map resolution)."""
    idx = torch.nonzero(mask)[:, 0]
    P, B, W = xyz[idx], bands[idx], w[idx].clamp_min(0)[:, None]
    out = bands.clone()
    rad = torch.zeros(len(xyz), device=xyz.device)
    for i in range(0, len(idx), chunk):
        d = torch.cdist(P[i:i + chunk], P)
        dk, nn = d.topk(min(k, len(idx)), largest=False)
        ww = W[nn]  # [c, k, 1]
        out[idx[i:i + chunk]] = (B[nn] * ww).sum(1) / ww.sum(1).clamp_min(1e-9)
        rad[idx[i:i + chunk]] = dk.mean(1)
    return out, rad


@torch.no_grad()
def compute(m: dict, k: int = 32, min_weight: float = 1.0) -> dict:
    """Per-Gaussian chlorophyll / structure / NDVI maps (pooled within each leaf), the mapped mask, per-leaf values."""
    ex, w, ch = m["extra"], m["weights"], m["channels"]
    N, R, RE = (ex[:, ch.index(c)] for c in ("nir", "red", "red_edge"))
    veg = (w >= min_weight) & (ndvi(N, R) > 0.3) & (N > 0.2)
    lab = m.get("leaf_label")
    if lab is not None:
        veg = veg & (lab >= 0)
        pb, rad = ex.clone(), torch.zeros_like(N)
        for j in torch.unique(lab[veg]).tolist():
            pj, rj = pooled_bands(m["geom"]["means"], ex, w, veg & (lab == j), k)
            sel = veg & (lab == j)
            pb[sel], rad[sel] = pj[sel], rj[sel]
    else:
        pb, rad = pooled_bands(m["geom"]["means"], ex, w, veg, k)
    Np, Rp, REp = (pb[:, ch.index(c)] for c in ("nir", "red", "red_edge"))
    maps = {"cire": Np / REp.clamp_min(1e-3) - 1, "nir": Np, "ndvi": ndvi(Np, Rp),
            "cire_raw": N / RE.clamp_min(1e-3) - 1, "nir_raw": N, "veg": veg, "radius": rad}
    if "leaf_label" in m:
        lab, LW = m["leaf_label"], m["leaf_weight"]
        rows, leaf_cire, leaf_nir = [], torch.full_like(N, float("nan")), torch.full_like(N, float("nan"))
        for j in range(LW.shape[1]):
            sel = (lab == j) & veg
            if sel.sum() < 20:
                continue
            ww = (LW[sel, j] * w[sel]).clamp_min(1e-9)  # leaf membership x visibility
            sN, sRE, sR = ((ww * x[sel]).sum() for x in (N, RE, R))
            c, nir = (sN / sRE - 1).item(), (sN / ww.sum()).item()
            rows.append({"leaf": j, "n_gaussians": int(sel.sum()), "cire": c, "nir": nir, "ndvi": ndvi(sN, sR).item()})
            leaf_cire[lab == j], leaf_nir[lab == j] = c, nir
        maps["leaves"] = pd.DataFrame(rows).set_index("leaf")
        maps["cire_leaf"], maps["nir_leaf"] = leaf_cire, leaf_nir
    return maps


def colors(maps: dict, key: str, vmin: float, vmax: float, stops, context: np.ndarray | None = None) -> np.ndarray:
    """RGB per Gaussian: palette on vegetation (and on leaves for *_leaf keys), grey elsewhere."""
    v = maps[key].cpu().numpy()
    ok = maps["veg"].cpu().numpy() & np.isfinite(v)
    rgb = np.full((len(v), 3), GREY, np.float32) if context is None else context.copy()
    rgb[ok] = ramp(v[ok], vmin, vmax, stops)
    return rgb


def orbit_w2c(data: Path, n: int = 6, elev_offset: float = 0.0) -> list[torch.Tensor]:
    """Camera poses orbiting the turntable axis, starting from a training camera (same construction as notebook 03)."""
    from plant3d.data import load_colmap

    sc = load_colmap(data, 0)
    C = np.stack([np.linalg.inv(f.w2c.numpy())[:3, 3] for f in sc.train])
    e1, e2, axis = np.linalg.svd(C - C.mean(0))[2]
    P = C - C.mean(0)
    x, y = P @ e1, P @ e2
    cx, cy, _ = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones_like(x)], x ** 2 + y ** 2, rcond=None)[0]
    ctr = C.mean(0) + cx * e1 + cy * e2
    base = np.linalg.inv(sc.train[1].w2c.numpy())
    out = []
    for t in np.linspace(0, 2 * math.pi, n, endpoint=False):
        a = axis / np.linalg.norm(axis)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        Rm = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = ctr - Rm @ ctr
        out.append(torch.tensor(np.linalg.inv(T @ base), dtype=torch.float32))
    return out, sc.train[0].K, sc.train[0].width, sc.train[0].height


@torch.no_grad()
def render_rgb(geom: dict, rgb: np.ndarray, w2c: torch.Tensor, K: torch.Tensor, width: int, height: int, bg=(1, 1, 1)) -> np.ndarray:
    img, _, _ = render(geom, w2c[None].cuda(), K[None].cuda(), width, height, None,
                       torch.tensor(bg, dtype=torch.float32, device="cuda"), colors=torch.from_numpy(rgb).cuda())
    return img[0].clamp(0, 1).cpu().numpy()


def export_plys(m: dict, layers: dict[str, np.ndarray], out: Path) -> list[Path]:
    """One standard 3DGS .ply per colouring (DC colour = map colour, no view-dependent colour) for SuperSplat etc."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rgb in layers.items():
        g = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in m["geom"].items()})
        g["sh0"].data = rgb_to_sh0(torch.from_numpy(rgb).cuda())[:, None, :]
        g["shN"].data.zero_()
        p = out / f"plant_{name}.ply"
        export_ply(p, g)
        paths.append(p)
    return paths


VIEWER = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#16191b;color:#e4e2dc;font:13px system-ui,"Noto Sans CJK SC",sans-serif}
canvas{width:100%;height:100%;display:block;touch-action:none}
.bar{position:absolute;left:10px;top:10px;display:flex;flex-wrap:wrap;gap:4px;max-width:calc(100% - 20px)}
button{font:500 12px system-ui,"Noto Sans CJK SC",sans-serif;color:#b8bdb6;background:#22272a;border:1px solid #333a3f;border-radius:4px;padding:5px 9px;cursor:pointer}
button[aria-pressed=true]{background:#d9d6cc;color:#16191b;border-color:#d9d6cc}
.leg{position:absolute;left:10px;bottom:10px;background:rgba(34,39,42,.92);border:1px solid #333a3f;border-radius:5px;padding:7px 9px;width:230px}
.leg .g{height:9px;border-radius:2px}.leg .t{display:flex;justify-content:space-between;font:11px ui-monospace,monospace;color:#9aa19a;margin-top:3px}
.leg .n{font-size:11px;color:#9aa19a;margin-bottom:4px}
.msg{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);color:#9aa19a;font:12px ui-monospace,monospace}
.hint{position:absolute;right:10px;bottom:10px;font-size:11px;color:#8a908a}
</style></head><body>
<canvas id="c"></canvas><div class="bar" id="bar"></div><div class="leg" id="leg"><div class="n" id="ln"></div><div class="g" id="lg"></div><div class="t" id="lt"></div></div>
<div class="msg" id="msg">loading...</div><div class="hint">drag: rotate / wheel: zoom / right-drag: pan</div>
<script type="application/json" id="data">__DATA__</script>
<script type="module">
import * as SPLAT from "https://cdn.jsdelivr.net/npm/gsplat@1.2.4/dist/index.js";
const D=JSON.parse(document.getElementById("data").textContent);
const b64=s=>{const b=atob(s),o=new Uint8Array(b.length);for(let i=0;i<b.length;i++)o[i]=b.charCodeAt(i);return o};
const msg=document.getElementById("msg");
try{
const renderer=new SPLAT.WebGLRenderer(document.getElementById("c")),scene=new SPLAT.Scene(),camera=new SPLAT.Camera();
const controls=new SPLAT.OrbitControls(camera,document.getElementById("c"));
const splat=SPLAT.Loader.LoadFromArrayBuffer(b64(D.splat).buffer,scene);
const layers=Object.fromEntries(D.layers.map(l=>[l.key,b64(l.rgb)]));
const bar=document.getElementById("bar");
function show(key){const L=D.layers.find(l=>l.key===key),src=layers[key],dst=splat.data.colors,n=splat.data.vertexCount;
 for(let i=0;i<n;i++){dst[4*i]=src[3*i];dst[4*i+1]=src[3*i+1];dst[4*i+2]=src[3*i+2];}
 splat.data.changed=true;
 for(const b of bar.children)b.setAttribute("aria-pressed",String(b.dataset.key===key));
 const leg=document.getElementById("leg");leg.hidden=!L.legend;
 if(L.legend){document.getElementById("ln").textContent=L.legend.name;document.getElementById("lg").style.background=L.legend.gradient;
  document.getElementById("lt").innerHTML=L.legend.ticks.map(t=>"<span>"+t+"</span>").join("");}}
for(const l of D.layers){const b=document.createElement("button");b.textContent=l.label;b.dataset.key=l.key;b.onclick=()=>show(l.key);bar.appendChild(b);}
show(D.layers[0].key);msg.hidden=true;
const frame=()=>{controls.update();renderer.render(scene,camera);requestAnimationFrame(frame)};frame();
}catch(e){msg.textContent="viewer failed: "+e}
</script></body></html>"""


def gradient_css(stops) -> str:
    pos, cols = stops
    return "linear-gradient(90deg," + ",".join(f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)}) {p*100:.0f}%" for p, c in zip(pos, cols)) + ")"


def viewer_document(m: dict, data: Path, layers: list[dict], min_opacity: float = 0.02, scale_clip: float = 0.2) -> str:
    """Self-contained HTML document (gsplat.js from jsDelivr + inline data) with one button per colour layer.
    layers: [{"key", "label", "rgb": [N,3] float, "legend": {"name", "gradient", "ticks"} or None}]"""
    g = {k: v.cpu().numpy() for k, v in m["geom"].items()}
    opa = 1 / (1 + np.exp(-g["opacities"]))
    keep = opa >= min_opacity
    R, pivot = upright_frame(data)
    xyz = (g["means"] - pivot) @ R.T
    q = g["quats"] / np.linalg.norm(g["quats"], axis=1, keepdims=True)
    q = quat_mul(rot_to_quat(R)[None], q)
    buf = np.zeros(int(keep.sum()), dtype=[("xyz", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    buf["xyz"] = xyz[keep]
    buf["scale"] = np.minimum(np.exp(g["scales"]), scale_clip)[keep]
    buf["rgba"][:, 3] = (opa[keep] * 255).astype(np.uint8)
    buf["rot"] = np.clip(q[keep] * 128 + 128, 0, 255).astype(np.uint8)
    enc = lambda b: base64.b64encode(b).decode()
    payload = {"splat": enc(buf.tobytes()), "layers": [
        {"key": L["key"], "label": L["label"], "legend": L.get("legend"),
         "rgb": enc((np.clip(L["rgb"][keep], 0, 1) * 255).astype(np.uint8).tobytes())} for L in layers]}
    return VIEWER.replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))


def viewer_iframe(doc: str, height: int = 640) -> str:
    return (f'<iframe srcdoc="{html.escape(doc, quote=True)}" style="width:100%;height:{height}px;border:0;border-radius:6px" '
            'allow="fullscreen"></iframe>')


def turntable_camera(data: Path, az_deg: float, el_deg: float, radius: float, target=(0.0, 0.0, 0.0)):
    """World-to-camera (OpenCV) for a camera on a sphere of `radius` around `target` (a point in the upright frame:
    y down, turntable axis = -y): azimuth around the vertical, elevation above the horizontal."""
    R, pivot = upright_frame(data)
    az, el = math.radians(az_deg), math.radians(el_deg)
    t = np.asarray(target, dtype=float)
    c = t + radius * np.array([math.cos(el) * math.sin(az), -math.sin(el), math.cos(el) * math.cos(az)])
    z = (t - c) / np.linalg.norm(t - c)
    down = np.array([0.0, 1.0, 0.0])
    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = R.T @ np.stack([x, y, z], 1)
    c2w[:3, 3] = R.T @ c + pivot
    return torch.tensor(np.linalg.inv(c2w), dtype=torch.float32)
