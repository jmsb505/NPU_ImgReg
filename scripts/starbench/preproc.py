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

ROOT = os.path.join(HOME, "results", "preproc")
os.makedirs(ROOT, mode=0o700, exist_ok=True)
MODS = ["SAR_VH-RGB", "SAR_VV-RGB"]

with open(os.path.join(HOME, "results", "nominal-8mod-gt.json")) as f:
    recs = [r for r in json.load(f)["records"] if r["modality"] in MODS]
print("sets", len(recs), flush=True)


def median5(img):
    return cv2.medianBlur(img, 5)


def median5_clahe(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.medianBlur(img, 5))


def median5_gradient(img):
    d = cv2.medianBlur(img, 5).astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.normalize(np.sqrt(gx * gx + gy * gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


VARIANTS = {"raw": None, "median5": median5, "median5_clahe": median5_clahe, "median5_gradient": median5_gradient}
detector = XFeatDetector("cuda", ransac_thresh=5.0)


def tre(M, T, lm):
    est = lm @ M[:, :2].T + M[:, 2]
    gt = lm @ T[:, :2].T + T[:, 2]
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))


summary = {}
for name, fn in VARIANTS.items():
    out = []
    for r in recs:
        moving = r["moving"]
        if fn is not None:
            tag = r["modality"] + "_" + r["set"][:-4].replace("/", "_")
            os.makedirs(os.path.join(ROOT, name), exist_ok=True)
            moving = os.path.join(ROOT, name, tag + "a.png")
            src = cv2.imread(r["moving"], cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(moving, fn(src))
            if r is recs[0]:
                changed = np.abs(cv2.imread(moving, cv2.IMREAD_GRAYSCALE).astype(float) - src.astype(float)).mean()
                print("CHK", name, "mean abs change on first set", round(float(changed), 2), flush=True)
        res = detector.register(r["fixed"], moving, ransac_thresh=5.0)
        t = None
        if res["transform"] is not None:
            M = np.asarray(res["transform"], dtype=np.float64)[:2, :]
            t = tre(M, np.asarray(r["T"], dtype=np.float64)[:2, :], np.asarray(r["landmarks_mov"], dtype=np.float64))
        out.append({"modality": r["modality"], "set": r["set"], "tre_px": t, "inliers": int(res["matches_count"])})
    summary[name] = out
print("gpu MB allocated", round(torch.cuda.memory_allocated() / 2**20, 1), flush=True)


def stats(rs):
    ok = [o["tre_px"] for o in rs if o["tre_px"] is not None]
    inl = [o["inliers"] for o in rs if o["tre_px"] is not None]
    if not ok:
        return {"sets": len(rs), "no_transform": len(rs), "within5": 0}
    return {
        "sets": len(rs),
        "no_transform": len(rs) - len(ok),
        "tre_median": round(float(np.median(ok)), 2),
        "tre_p90": round(float(np.percentile(ok, 90)), 2),
        "within5": int(np.sum(np.array(ok) <= 5)),
        "inliers_median": int(np.median(inl)),
    }


for name in VARIANTS:
    for mod in MODS:
        print("RES", name, mod, json.dumps(stats([o for o in summary[name] if o["modality"] == mod])), flush=True)
with open(os.path.join(ROOT, "summary.json"), "w") as f:
    json.dump(summary, f)
