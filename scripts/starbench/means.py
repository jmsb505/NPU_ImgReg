import json
import os

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
ROOT = os.path.join(HOME, "results", "preproc")
with open(os.path.join(ROOT, "summary.json")) as f:
    prev = json.load(f)
with open(os.path.join(ROOT, "summary_rgb.json")) as f:
    rgb = json.load(f)
configs = {
    "baseline": prev["raw"],
    "sar_median5": prev["median5"],
    "sar_median5_clahe": prev["median5_clahe"],
    "sar_median5_gradient": prev["median5_gradient"],
    "both_median5_clahe": rgb["sar_mc_rgb_mc"],
    "sar_clahe_rgb_clahe": rgb["sar_mc_rgb_c"],
}
for name, rs in configs.items():
    t = [o["tre_px"] for o in rs]
    got = np.array([x for x in t if x is not None])
    pen = np.array([256.0 if x is None else x for x in t])
    print("MEAN", name, "mean_over_transforms", round(float(got.mean()), 2), "mean_failures_as_256", round(float(pen.mean()), 2), "median", round(float(np.median(got)), 2), flush=True)
