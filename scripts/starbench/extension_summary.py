import json
import os

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
R = os.path.join(HOME, "results")
with open(os.path.join(R, "preproc", "summary.json")) as f:
    prev = json.load(f)
with open(os.path.join(R, "preproc", "summary_rgb.json")) as f:
    rgb = json.load(f)
with open(os.path.join(R, "heldout-eval.json")) as f:
    ho = json.load(f)
with open(os.path.join(R, "nominal-8mod-provenance.json")) as f:
    prov = json.load(f)


def ok(t, thr):
    return t is not None and t <= thr


base = {"nominal": prev["raw"], "offnominal": [o for o in ho["baseline"] if o["tier"] == "offnominal"], "anomaly": [o for o in ho["baseline"] if o["tier"] == "anomaly"]}
ext = {"nominal": rgb["sar_mc_rgb_c"], "offnominal": [o for o in ho["sar_clahe_rgb_clahe"] if o["tier"] == "offnominal"], "anomaly": [o for o in ho["sar_clahe_rgb_clahe"] if o["tier"] == "anomaly"]}


def stats(rs, bs):
    t = [o["tre_px"] for o in rs]
    got = np.array([x for x in t if x is not None])
    pen = np.array([256.0 if x is None else x for x in t])
    pairs = list(zip(bs, rs))
    return {
        "pairs": len(rs),
        "no_transform": [o["modality"] + "/" + o["set"] for o in rs if o["tre_px"] is None],
        "within_5px": int(np.sum(got <= 5)),
        "within_11_1px": int(np.sum(got <= 11.1)),
        "median_px": round(float(np.median(got)), 2),
        "p90_px": round(float(np.percentile(got, 90)), 2),
        "mean_px_over_transforms": round(float(got.mean()), 2),
        "mean_px_failures_256": round(float(pen.mean()), 2),
        "pairs_over_100px": int(np.sum(got > 100)),
        "max_px": round(float(got.max()), 1),
        "paired_5px_gained": sum(1 for x, y in pairs if not ok(x["tre_px"], 5) and ok(y["tre_px"], 5)),
        "paired_5px_lost": sum(1 for x, y in pairs if ok(x["tre_px"], 5) and not ok(y["tre_px"], 5)),
        "paired_11_1px_gained": sum(1 for x, y in pairs if not ok(x["tre_px"], 11.1) and ok(y["tre_px"], 11.1)),
        "paired_11_1px_lost": sum(1 for x, y in pairs if ok(x["tre_px"], 11.1) and not ok(y["tre_px"], 11.1)),
    }


out = {}
for tier in ["nominal", "offnominal", "anomaly"]:
    out[tier] = {"baseline": stats(base[tier], base[tier]), "extension": stats(ext[tier], base[tier])}
with open(os.path.join(R, "extension-summary.json"), "w") as f:
    json.dump(out, f)
for tier in out:
    for arm in ["baseline", "extension"]:
        print("STAT", tier, arm, json.dumps(out[tier][arm]), flush=True)
print("PROV code", json.dumps(prov["code"]), flush=True)
print("PROV runtime", json.dumps(prov["runtime"]), flush=True)
print("PROV dataset", json.dumps(prov["dataset"]), flush=True)
print("PROV metric", prov["metric"], flush=True)
