import json
import os

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
with open(os.path.join(HOME, "results", "heldout-eval.json")) as f:
    ho = json.load(f)
TIERS = ["offnominal", "anomaly"]


def ok(t, thr):
    return t is not None and t <= thr


base = [o for tier in TIERS for o in ho["baseline"] if o["tier"] == tier]
for name in ho:
    rs = [o for tier in TIERS for o in ho[name] if o["tier"] == tier]
    got = [o["tre_px"] for o in rs if o["tre_px"] is not None]
    pen = [256.0 if o["tre_px"] is None else o["tre_px"] for o in rs]
    g5 = sum(1 for x, y in zip(base, rs) if not ok(x["tre_px"], 5) and ok(y["tre_px"], 5))
    l5 = sum(1 for x, y in zip(base, rs) if ok(x["tre_px"], 5) and not ok(y["tre_px"], 5))
    g11 = sum(1 for x, y in zip(base, rs) if not ok(x["tre_px"], 11.1) and ok(y["tre_px"], 11.1))
    l11 = sum(1 for x, y in zip(base, rs) if ok(x["tre_px"], 11.1) and not ok(y["tre_px"], 11.1))
    print("HELDOUT", name, "pairs", len(rs), "w5", sum(1 for v in got if v <= 5), "w11", sum(1 for v in got if v <= 11.1), "median", round(float(np.median(got)), 2), "p90", round(float(np.percentile(got, 90)), 2), "mean256", round(float(np.mean(pen)), 2), "5px g/l", g5, l5, "11.1px g/l", g11, l11, flush=True)
