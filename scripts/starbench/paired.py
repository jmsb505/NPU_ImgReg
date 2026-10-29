import json
import os

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
with open(os.path.join(HOME, "results", "preproc", "summary.json")) as f:
    s = json.load(f)
raw = s["raw"]
for name in ["median5", "median5_clahe", "median5_gradient"]:
    for mod in ["SAR_VH-RGB", "SAR_VV-RGB"]:
        a = [o for o in raw if o["modality"] == mod]
        b = [o for o in s[name] if o["modality"] == mod]
        assert [x["set"] for x in a] == [y["set"] for y in b]
        ok5 = lambda t: t is not None and t <= 5
        gained = sum(1 for x, y in zip(a, b) if not ok5(x["tre_px"]) and ok5(y["tre_px"]))
        lost = sum(1 for x, y in zip(a, b) if ok5(x["tre_px"]) and not ok5(y["tre_px"]))
        diffs = [y["tre_px"] - x["tre_px"] for x, y in zip(a, b) if x["tre_px"] is not None and y["tre_px"] is not None]
        better = sum(1 for d in diffs if d < 0)
        print("PAIRED", name, mod, "gained", gained, "lost", lost, "both_pairs", len(diffs), "median_change_px", round(float(np.median(diffs)), 2), "better_pairs", better, flush=True)
