import json
import os

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
with open(os.path.join(HOME, "results", "heldout-eval.json")) as f:
    ho = json.load(f)
with open(os.path.join(HOME, "results", "nominal_configs.json")) as f:
    nom = json.load(f)
TIERS = ["offnominal", "anomaly"]


def ok(t, thr):
    return t is not None and t <= thr


def rows(name, tier):
    return nom[name] if tier == "nominal" else [o for o in ho[name] if o["tier"] == tier]


base = {tier: rows("baseline", tier) for tier in ["nominal"] + TIERS}
for name in ho:
    pooled = {"w5": 0, "w11": 0, "pairs": 0, "sum256": 0.0, "g11": 0, "l11": 0, "g5": 0, "l5": 0}
    for tier in ["nominal"] + TIERS:
        rs = rows(name, tier)
        bs = base[tier]
        vals = [o["tre_px"] for o in rs]
        got = [v for v in vals if v is not None]
        big = sum(1 for v in got if v > 100)
        pooled["w5"] += sum(1 for v in got if v <= 5)
        pooled["w11"] += sum(1 for v in got if v <= 11.1)
        pooled["pairs"] += len(rs)
        pooled["sum256"] += sum(256.0 if v is None else v for v in vals)
        pooled["g11"] += sum(1 for x, y in zip(bs, rs) if not ok(x["tre_px"], 11.1) and ok(y["tre_px"], 11.1))
        pooled["l11"] += sum(1 for x, y in zip(bs, rs) if ok(x["tre_px"], 11.1) and not ok(y["tre_px"], 11.1))
        pooled["g5"] += sum(1 for x, y in zip(bs, rs) if not ok(x["tre_px"], 5) and ok(y["tre_px"], 5))
        pooled["l5"] += sum(1 for x, y in zip(bs, rs) if ok(x["tre_px"], 5) and not ok(y["tre_px"], 5))
        if tier in TIERS:
            print("TIER", name, tier, "pairs over 100px", big, "max", round(max(got), 1), flush=True)
    print("POOLED", name, "pairs", pooled["pairs"], "w5", pooled["w5"], "w11", pooled["w11"], "mean256", round(pooled["sum256"] / pooled["pairs"], 2), "5px gained/lost vs base", pooled["g5"], pooled["l5"], "11.1px gained/lost vs base", pooled["g11"], pooled["l11"], flush=True)
