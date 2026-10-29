import json
import os

import cv2
import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
OUT = os.path.join(HOME, "results", "pixelcheck")
with open(os.path.join(HOME, "results", "nominal-intra_blue-green-gt.json")) as f:
    intra = json.load(f)
for r in intra:
    r["modality"] = "Intra_blue-green"
with open(os.path.join(HOME, "results", "nominal-8mod-gt.json")) as f:
    recs = intra + json.load(f)["records"]
with open(os.path.join(OUT, "pixelcheck.json")) as f:
    chk = json.load(f)
assert len(chk) == len(recs)
MODS = ["Intra_blue-green", "Intra_blue-red", "Intra_green-red", "OPT-OPT", "SAR_VV-VV", "SAR_VH-VH", "SAR_VH-VV", "SAR_VH-RGB", "SAR_VV-RGB"]


def mi(x, y, mask, bins=32):
    h, _, _ = np.histogram2d(x[mask].ravel(), y[mask].ravel(), bins=bins)
    p = h / h.sum()
    nz = p > 0
    outer = p.sum(1)[:, None] * p.sum(0)[None, :]
    return float((p[nz] * np.log(p[nz] / outer[nz])).sum())


def warp(img, M):
    h, w = img.shape
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones_like(img), M, (w, h), flags=cv2.INTER_NEAREST) > 0
    return out, valid


ALL = np.ones((256, 256), dtype=bool)
rows = []
for r, c in zip(recs, chk):
    a = cv2.imread(r["moving"], cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(r["fixed"], cv2.IMREAD_GRAYSCALE)
    T = np.asarray(r["T"], dtype=np.float64)[:2, :]
    wg, vg = warp(a, T)
    m_x = None
    if c["M"] is not None:
        wx, vx = warp(a, np.asarray(c["M"], dtype=np.float64))
        m_x = mi(wx, b, vx)
    rows.append((c["modality"], mi(a, b, ALL), mi(wg, b, vg), m_x))


def med(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 3) if vals else None


for mod in MODS:
    rs = [row for row in rows if row[0] == mod]
    gt_gain = sum(1 for row in rs if row[2] > row[1])
    x_gain = sum(1 for row in rs if row[3] is not None and row[3] > row[1])
    print("MI", mod, "sets", len(rs), "raw", med([row[1] for row in rs]), "gt", med([row[2] for row in rs]), "xfeat", med([row[3] for row in rs]), "gt_beats_raw", gt_gain, "xfeat_beats_raw", x_gain, flush=True)
