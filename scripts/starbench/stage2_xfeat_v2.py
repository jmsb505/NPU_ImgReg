import json
import os
import sys

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
sys.path.insert(0, os.path.join(HOME, "vendor"))
sys.path.insert(0, HOME)
from starbench_detectors import XFeatDetector


def tre(est, gt, lm):
    return float(np.sqrt(np.mean(np.sum((lm @ est[:, :2].T + est[:, 2] - (lm @ gt[:, :2].T + gt[:, 2])) ** 2, axis=1))))


src = os.path.join(HOME, "results", "nominal-intra_blue-green-gt.json")
dst = os.path.join(HOME, "results", "nominal-intra_blue-green-xfeat.json")
with open(src) as f:
    records = json.load(f)

detector = XFeatDetector("cuda", ransac_thresh=5.0)
for r in records:
    result = detector.register(r["fixed"], r["moving"], ransac_thresh=5.0)
    r["inliers"] = int(result["matches_count"])
    r["seconds"] = float(result["time"])
    r["tre_px"] = None
    r["tre_inverse_px"] = None
    if result["transform"] is not None:
        M = np.asarray(result["transform"], dtype=np.float64)[:2, :]
        T3 = np.asarray(r["T"], dtype=np.float64)
        lm = np.asarray(r["landmarks_mov"], dtype=np.float64)
        r["tre_px"] = tre(M, T3[:2, :], lm)
        r["tre_inverse_px"] = tre(M, np.linalg.inv(T3)[:2, :], lm)

with open(dst, "w") as f:
    json.dump(records, f)

ok = np.array([r["tre_px"] for r in records if r["tre_px"] is not None])
inv = np.array([r["tre_inverse_px"] for r in records if r["tre_inverse_px"] is not None])
secs = np.array([r["seconds"] for r in records])
print("sets", len(records), "no transform", len(records) - len(ok))
if ok.size:
    print("TRE px (harness T_reg_gt) median", round(float(np.median(ok)), 2), "p90", round(float(np.percentile(ok, 90)), 2), "mean", round(float(ok.mean()), 2))
    print("fraction <= 5 px", round(float(np.mean(ok <= 5)), 3))
    print("diagnostic TRE px vs inv(T_reg_gt) median", round(float(np.median(inv)), 2))
print("inliers median", int(np.median([r["inliers"] for r in records])))
print("seconds median", round(float(np.median(secs)), 3), "total", round(float(secs.sum()), 1))
