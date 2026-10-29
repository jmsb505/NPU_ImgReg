import json
import logging
import os
import sys

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
sys.path.insert(0, HOME)
import starbench_utils as utils

logging.disable(logging.INFO)
root = os.path.join(HOME, "data", "nominal", "Nominal", "Intra_blue-green")
records = []
for land in sorted(os.listdir(root)):
    folder = os.path.join(root, land)
    if not os.path.isdir(folder):
        continue
    for s in utils.find_matching_files_in_folder(folder):
        _, _, lf, lm, T = utils.load_ground_truth(s["gt_file"])
        lf = np.asarray(lf, dtype=np.float64)
        lm = np.asarray(lm, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        check = lm @ T[:2, :2].T + T[:2, 2]
        records.append({
            "set": os.path.relpath(s["gt_file"], root),
            "moving": s["moving_file"],
            "fixed": s["fixed_file"],
            "T": T.tolist(),
            "landmarks_mov": lm.tolist(),
            "gt_self_check_px": float(np.sqrt(np.mean(np.sum((check - lf) ** 2, axis=1)))),
        })

os.makedirs(os.path.join(HOME, "results"), mode=0o700, exist_ok=True)
with open(os.path.join(HOME, "results", "nominal-intra_blue-green-gt.json"), "w") as f:
    json.dump(records, f)
check = np.array([r["gt_self_check_px"] for r in records])
print("sets", len(records), "gt self-check px median", round(float(np.median(check)), 3), "max", round(float(check.max()), 3))
