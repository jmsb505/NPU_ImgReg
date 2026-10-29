import json
import os
import sys

import cv2
import numpy as np
import torch

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
sys.path.insert(0, os.path.join(HOME, "vendor"))
sys.path.insert(0, HOME)
from starbench_detectors import XFeatDetector

OUT = os.path.join(HOME, "results", "pixelcheck")
os.makedirs(OUT, mode=0o700, exist_ok=True)
MODS = ["Intra_blue-green", "Intra_blue-red", "Intra_green-red", "OPT-OPT", "SAR_VV-VV", "SAR_VH-VH", "SAR_VH-VV", "SAR_VH-RGB", "SAR_VV-RGB"]

with open(os.path.join(HOME, "results", "nominal-intra_blue-green-gt.json")) as f:
    intra = json.load(f)
for r in intra:
    r["modality"] = "Intra_blue-green"
with open(os.path.join(HOME, "results", "nominal-8mod-gt.json")) as f:
    recs = intra + json.load(f)["records"]

GRID = np.stack(np.meshgrid(np.arange(16, 256, 32), np.arange(16, 256, 32)), -1).reshape(-1, 2).astype(np.float64)


def apply(H, p):
    return p @ H[:, :2].T + H[:, 2]


def dense_px(M, T):
    return float(np.linalg.norm(apply(M, GRID) - apply(T, GRID), axis=1).mean())


def ncc(x, y, mask):
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else None


def warp(img, M):
    h, w = img.shape
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones_like(img), M, (w, h), flags=cv2.INTER_NEAREST) > 0
    return out, valid


print("cuda available", torch.cuda.is_available(), flush=True)
detector = XFeatDetector("cuda", ransac_thresh=5.0)
print("detector device", detector.device, flush=True)
for r in recs:
    a = cv2.imread(r["moving"], cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(r["fixed"], cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(r["moving"][:-5] + "g.png", cv2.IMREAD_GRAYSCALE)
    T = np.asarray(r["T"], dtype=np.float64)[:2, :]
    res = detector.register(r["fixed"], r["moving"], ransac_thresh=5.0)
    M = None if res["transform"] is None else np.asarray(res["transform"], dtype=np.float64)[:2, :]
    r["M"] = None if M is None else M.tolist()
    r["ncc_raw"] = ncc(a, b, np.ones(a.shape, dtype=bool))
    r["ncc_xfeat"] = None
    r["dense_px"] = None
    if M is not None:
        wa, valid = warp(a, M)
        r["ncc_xfeat"] = ncc(wa, b, valid)
        r["dense_px"] = dense_px(M, T)
    wg, validg = warp(a, T)
    r["ncc_gt_vs_g"] = ncc(wg, g, validg)
print("gpu MB allocated", round(torch.cuda.memory_allocated() / 2**20, 1), flush=True)


def med(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 3) if vals else None


for mod in MODS:
    rs = [r for r in recs if r["modality"] == mod]
    ok = [r for r in rs if r["dense_px"] is not None]
    improved = sum(1 for r in ok if r["ncc_raw"] is not None and r["ncc_xfeat"] is not None and r["ncc_xfeat"] > r["ncc_raw"])
    print("SUM", mod, "sets", len(rs), "ok", len(ok), "ncc_raw", med([r["ncc_raw"] for r in rs]), "ncc_xfeat", med([r["ncc_xfeat"] for r in ok]), "improved", improved, "dense_med", med([r["dense_px"] for r in ok]), "dense_le5", sum(r["dense_px"] <= 5 for r in ok), "gt_vs_g", med([r["ncc_gt_vs_g"] for r in rs]), flush=True)

keep = ["modality", "set", "ncc_raw", "ncc_xfeat", "dense_px", "ncc_gt_vs_g", "M"]
with open(os.path.join(OUT, "pixelcheck.json"), "w") as f:
    json.dump([{k: r[k] for k in keep} for r in recs], f)


def fmt(v, d=3):
    return None if v is None else round(v, d)


def panel(r):
    a = cv2.imread(r["moving"], cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(r["fixed"], cv2.IMREAD_GRAYSCALE)
    wa = np.zeros_like(a)
    if r["M"] is not None:
        wa, _ = warp(a, np.asarray(r["M"], dtype=np.float64))
    row = np.hstack([np.dstack([b, b, b]), np.dstack([wa, wa, wa]), np.dstack([wa, wa, b])])
    label = f"{r['modality']} {r['set']} dense {fmt(r['dense_px'], 1)} px"
    cv2.putText(row, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return row


def pick(mod, cond):
    return next(r for r in recs if r["modality"] == mod and cond(r))


examples = [
    pick("Intra_blue-green", lambda r: True),
    pick("OPT-OPT", lambda r: True),
    pick("SAR_VH-RGB", lambda r: r["dense_px"] is not None and r["dense_px"] <= 5),
    pick("SAR_VH-RGB", lambda r: r["dense_px"] is not None and r["dense_px"] > 20),
    pick("SAR_VH-RGB", lambda r: r["dense_px"] is None),
]
for r in examples:
    print("EX", r["modality"], r["set"], "dense", fmt(r["dense_px"], 2), "ncc_raw", fmt(r["ncc_raw"]), "ncc_xfeat", fmt(r["ncc_xfeat"]), flush=True)
cv2.imwrite(os.path.join(OUT, "overlays.png"), np.vstack([panel(r) for r in examples]))
print("saved", os.path.join(OUT, "overlays.png"), flush=True)
