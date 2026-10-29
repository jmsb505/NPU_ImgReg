import json
import os
import shutil
import sys

import cv2
import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
sys.path.insert(0, os.path.join(HOME, "vendor"))
sys.path.insert(0, HOME)
from starbench_detectors import XFeatDetector

OUT = os.path.join(HOME, "results", "heldout")
os.makedirs(OUT, mode=0o700, exist_ok=True)
with open(os.path.join(HOME, "results", "heldout-gt.json")) as f:
    recs = json.load(f)["records"]
print("heldout sets", len(recs), flush=True)


def median5(img):
    return cv2.medianBlur(img, 5)


def clahe(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def median5_clahe(img):
    return clahe(median5(img))


def gradient(img):
    d = median5(img).astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.normalize(np.sqrt(gx * gx + gy * gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


VARIANTS = {
    "baseline": (None, None),
    "sar_median5": (median5, None),
    "sar_median5_clahe": (median5_clahe, None),
    "sar_median5_gradient": (gradient, None),
    "both_median5_clahe": (median5_clahe, median5_clahe),
    "sar_clahe_rgb_clahe": (median5_clahe, clahe),
}
detector = XFeatDetector("cuda", ransac_thresh=5.0)


def tre(M, T, lm):
    est = lm @ M[:, :2].T + M[:, 2]
    gt = lm @ T[:, :2].T + T[:, 2]
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))


results = {}
for name, (f_mov, f_fix) in VARIANTS.items():
    d = os.path.join(OUT, name)
    if f_mov is not None or f_fix is not None:
        os.makedirs(d, exist_ok=True)
    out = []
    for r in recs:
        tag = r["tier"] + "_" + r["modality"] + "_" + r["set"][:-4].replace("/", "_")
        moving, fixed = r["moving"], r["fixed"]
        if f_mov is not None:
            moving = os.path.join(d, tag + "a.png")
            cv2.imwrite(moving, f_mov(cv2.imread(r["moving"], cv2.IMREAD_GRAYSCALE)))
        if f_fix is not None:
            fixed = os.path.join(d, tag + "b.png")
            cv2.imwrite(fixed, f_fix(cv2.imread(r["fixed"], cv2.IMREAD_GRAYSCALE)))
        res = detector.register(fixed, moving, ransac_thresh=5.0)
        t = None
        if res["transform"] is not None:
            M = np.asarray(res["transform"], dtype=np.float64)[:2, :]
            t = tre(M, np.asarray(r["T"], dtype=np.float64)[:2, :], np.asarray(r["landmarks_mov"], dtype=np.float64))
        out.append({"tier": r["tier"], "modality": r["modality"], "set": r["set"], "tre_px": t})
    results[name] = out
    print("done", name, flush=True)

with open(os.path.join(HOME, "results", "nominal_configs.json")) as f:
    nominal = json.load(f)
for name in VARIANTS:
    nominal_rs = nominal[name]
    tiers = {"nominal": nominal_rs, "offnominal": [o for o in results[name] if o["tier"] == "offnominal"], "anomaly": [o for o in results[name] if o["tier"] == "anomaly"]}
    base = {"nominal": nominal["baseline"], "offnominal": [o for o in results["baseline"] if o["tier"] == "offnominal"], "anomaly": [o for o in results["baseline"] if o["tier"] == "anomaly"]}
    for tier, rs in tiers.items():
        t = [o["tre_px"] for o in rs]
        got = np.array([x for x in t if x is not None])
        pen = np.array([256.0 if x is None else x for x in t])
        g5 = sum(1 for x, y in zip(base[tier], rs) if not (x["tre_px"] is not None and x["tre_px"] <= 5) and (y["tre_px"] is not None and y["tre_px"] <= 5))
        l5 = sum(1 for x, y in zip(base[tier], rs) if (x["tre_px"] is not None and x["tre_px"] <= 5) and not (y["tre_px"] is not None and y["tre_px"] <= 5))
        print("RES", name, tier, "n", len(rs), "fail", int(sum(x is None for x in t)), "w5", int(np.sum(got <= 5)), "w11", int(np.sum(got <= 11.1)), "median", round(float(np.median(got)), 2), "p90", round(float(np.percentile(got, 90)), 2), "mean", round(float(got.mean()), 2), "mean256", round(float(pen.mean()), 2), "vs_base_5px_gained", g5, "lost", l5, flush=True)
with open(os.path.join(HOME, "results", "heldout-eval.json"), "w") as f:
    json.dump(results, f)
shutil.rmtree(OUT)
print("removed scratch PNGs", flush=True)
