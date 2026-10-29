import json
import logging
import os
import sys

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
sys.path.insert(0, HOME)
import starbench_utils as utils

logging.disable(logging.ERROR)
MODS = ["SAR_VH-RGB", "SAR_VV-RGB"]
TIERS = {"offnominal": os.path.join(HOME, "data", "offnominal"), "anomaly": os.path.join(HOME, "data", "anomaly")}
records, errors = [], []
for tier, root in TIERS.items():
    for dirpath, dirnames, _ in os.walk(root):
        mod = os.path.basename(dirpath)
        if mod not in MODS:
            continue
        for land in sorted(dirnames):
            for s in utils.find_matching_files_in_folder(os.path.join(dirpath, land)):
                name = os.path.relpath(s["gt_file"], dirpath)
                try:
                    _, _, lf, lm, T = utils.load_ground_truth(s["gt_file"])
                except Exception as e:
                    errors.append({"tier": tier, "modality": mod, "set": name, "error": repr(e)[:200]})
                    continue
                records.append({"tier": tier, "modality": mod, "set": name, "moving": s["moving_file"], "fixed": s["fixed_file"], "T": np.asarray(T, dtype=np.float64).tolist(), "landmarks_mov": np.asarray(lm, dtype=np.float64).tolist()})
os.makedirs(os.path.join(HOME, "results"), mode=0o700, exist_ok=True)
with open(os.path.join(HOME, "results", "heldout-gt.json"), "w") as f:
    json.dump({"records": records, "errors": errors}, f)
for tier in TIERS:
    for mod in MODS:
        print("GT", tier, mod, sum(1 for r in records if r["tier"] == tier and r["modality"] == mod), "errors", sum(1 for e in errors if e["tier"] == tier and e["modality"] == mod), flush=True)
