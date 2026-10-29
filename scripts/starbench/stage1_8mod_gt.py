import json
import logging
import os
import sys

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
sys.path.insert(0, HOME)
import starbench_utils as utils

logging.disable(logging.ERROR)
MODALITIES = ["Intra_blue-red", "Intra_green-red", "OPT-OPT", "SAR_VH-RGB", "SAR_VH-VH", "SAR_VH-VV", "SAR_VV-RGB", "SAR_VV-VV"]
base = os.path.join(HOME, "data", "nominal", "Nominal")
records, errors = [], []
for mod in MODALITIES:
    root = os.path.join(base, mod)
    for land in sorted(os.listdir(root)):
        folder = os.path.join(root, land)
        if not os.path.isdir(folder):
            continue
        for s in utils.find_matching_files_in_folder(folder):
            name = os.path.relpath(s["gt_file"], root)
            try:
                _, _, lf, lm, T = utils.load_ground_truth(s["gt_file"])
            except Exception as e:
                errors.append({"modality": mod, "set": name, "error": repr(e)[:200]})
                continue
            lf = np.asarray(lf, dtype=np.float64)
            lm = np.asarray(lm, dtype=np.float64)
            records.append({
                "modality": mod,
                "set": name,
                "moving": s["moving_file"],
                "fixed": s["fixed_file"],
                "T": np.asarray(T, dtype=np.float64).tolist(),
                "landmarks_mov": lm.tolist(),
                "landmarks_fix_equal_mov": bool(np.array_equal(lf, lm)),
            })

os.makedirs(os.path.join(HOME, "results"), mode=0o700, exist_ok=True)
with open(os.path.join(HOME, "results", "nominal-8mod-gt.json"), "w") as f:
    json.dump({"records": records, "errors": errors}, f)
for mod in MODALITIES:
    rs = [r for r in records if r["modality"] == mod]
    print(mod, "sets", len(rs), "gt errors", sum(e["modality"] == mod for e in errors), "fix==mov", all(r["landmarks_fix_equal_mov"] for r in rs))
print("total sets", len(records), "gt errors", len(errors))
