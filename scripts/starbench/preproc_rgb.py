import json
import os
import sys

import cv2
import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
sys.path.insert(0, os.path.join(HOME, "vendor"))
sys.path.insert(0, HOME)
from starbench_detectors import XFeatDetector

ROOT = os.path.join(HOME, "results", "preproc")
MODS = ["SAR_VH-RGB", "SAR_VV-RGB"]

with open(os.path.join(HOME, "results", "nominal-8mod-gt.json")) as f:
    recs = [r for r in json.load(f)["records"] if r["modality"] in MODS]
with open(os.path.join(ROOT, "summary.json")) as f:
    prev = json.load(f)
print("sets", len(recs), flush=True)


def median5(img):
    return cv2.medianBlur(img, 5)


def clahe(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def median5_clahe(img):
    return clahe(median5(img))


VARIANTS = {
    "sar_mc_rgb_mc": (median5_clahe, median5_clahe),
    "sar_mc_rgb_c": (median5_clahe, clahe),
}
detector = XFeatDetector("cuda", ransac_thresh=5.0)


def tre(M, T, lm):
    est = lm @ M[:, :2].T + M[:, 2]
    gt = lm @ T[:, :2].T + T[:, 2]
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))


summary = {}
for name, (f_mov, f_fix) in VARIANTS.items():
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    out = []
    for r in recs:
        tag = r["modality"] + "_" + r["set"][:-4].replace("/", "_")
        moving = os.path.join(d, tag + "a.png")
        fixed = os.path.join(d, tag + "b.png")
        src_m = cv2.imread(r["moving"], cv2.IMREAD_GRAYSCALE)
        src_f = cv2.imread(r["fixed"], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(moving, f_mov(src_m))
        cv2.imwrite(fixed, f_fix(src_f))
        if r is recs[0]:
            dm = np.abs(cv2.imread(moving, cv2.IMREAD_GRAYSCALE).astype(float) - src_m.astype(float)).mean()
            df = np.abs(cv2.imread(fixed, cv2.IMREAD_GRAYSCALE).astype(float) - src_f.astype(float)).mean()
            print("CHK", name, "moving change", round(float(dm), 2), "fixed change", round(float(df), 2), flush=True)
        res = detector.register(fixed, moving, ransac_thresh=5.0)
        t = None
        if res["transform"] is not None:
            M = np.asarray(res["transform"], dtype=np.float64)[:2, :]
            t = tre(M, np.asarray(r["T"], dtype=np.float64)[:2, :], np.asarray(r["landmarks_mov"], dtype=np.float64))
        out.append({"modality": r["modality"], "set": r["set"], "tre_px": t, "inliers": int(res["matches_count"])})
    summary[name] = out


def stats(rs):
    ok = [o["tre_px"] for o in rs if o["tre_px"] is not None]
    inl = [o["inliers"] for o in rs if o["tre_px"] is not None]
    if not ok:
        return {"sets": len(rs), "no_transform": len(rs), "within5": 0}
    return {"sets": len(rs), "no_transform": len(rs) - len(ok), "tre_median": round(float(np.median(ok)), 2), "tre_p90": round(float(np.percentile(ok, 90)), 2), "within5": int(np.sum(np.array(ok) <= 5)), "inliers_median": int(np.median(inl))}


def ok5(t):
    return t is not None and t <= 5


def paired(a, b):
    gained = sum(1 for x, y in zip(a, b) if not ok5(x["tre_px"]) and ok5(y["tre_px"]))
    lost = sum(1 for x, y in zip(a, b) if ok5(x["tre_px"]) and not ok5(y["tre_px"]))
    return gained, lost


for name in VARIANTS:
    for mod in MODS:
        cur = [o for o in summary[name] if o["modality"] == mod]
        raw = [o for o in prev["raw"] if o["modality"] == mod]
        sar = [o for o in prev["median5_clahe"] if o["modality"] == mod]
        g1, l1 = paired(raw, cur)
        g2, l2 = paired(sar, cur)
        print("RES", name, mod, json.dumps(stats(cur)), "vs_raw gained", g1, "lost", l1, "vs_sar_clahe gained", g2, "lost", l2, flush=True)
with open(os.path.join(ROOT, "summary_rgb.json"), "w") as f:
    json.dump(summary, f)
