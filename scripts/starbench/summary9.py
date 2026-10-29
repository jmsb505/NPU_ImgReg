import json
import os

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
with open(os.path.join(HOME, "results", "nominal-intra_blue-green-xfeat.json")) as f:
    intra = json.load(f)
for r in intra:
    r["modality"] = "Intra_blue-green"
with open(os.path.join(HOME, "results", "nominal-8mod-xfeat.json")) as f:
    recs = intra + json.load(f)["records"]
order = ["Intra_blue-green", "Intra_blue-red", "Intra_green-red", "OPT-OPT", "SAR_VV-VV", "SAR_VH-VH", "SAR_VH-VV", "SAR_VH-RGB", "SAR_VV-RGB"]
rows = {}
for mod in order:
    rs = [r for r in recs if r["modality"] == mod]
    ok = [r for r in rs if r["tre_px"] is not None]
    tre = np.array([r["tre_px"] for r in ok])
    rows[mod] = {
        "sets": len(rs),
        "no_transform": len(rs) - len(ok),
        "within_5px_of_all_sets": int(np.sum(tre <= 5)),
        "tre_px_median": round(float(np.median(tre)), 3),
        "tre_px_p90": round(float(np.percentile(tre, 90)), 3),
        "inliers_median": int(np.median([r["inliers"] for r in ok])),
        "seconds_median": round(float(np.median([r["seconds"] for r in rs if r["seconds"] is not None])), 4),
    }
sar_rgb_by_land = {}
for mod in ["SAR_VH-RGB", "SAR_VV-RGB"]:
    rs = [r for r in recs if r["modality"] == mod]
    sar_rgb_by_land[mod] = {}
    for land in sorted({r["set"].split("/")[0] for r in rs}):
        sub = [r for r in rs if r["set"].split("/")[0] == land]
        sar_rgb_by_land[mod][land] = f"{sum(1 for r in sub if r['tre_px'] is not None and r['tre_px'] <= 5)}/{len(sub)}"
failures = [{"modality": r["modality"], "set": r["set"]} for r in recs if r["tre_px"] is None]
summary = {
    "definitions": {
        "within_5px_of_all_sets": "count of sets with TRE <= 5 px; sets without a transform count as misses",
        "tre_px": "computed over sets that returned a transform",
        "inliers_median": "over sets that returned a transform",
        "seconds_median": "XFeatDetector.register() time, all sets",
    },
    "modalities": rows,
    "total_sets": len(recs),
    "total_within_5px_of_all_sets": sum(v["within_5px_of_all_sets"] for v in rows.values()),
    "no_transform_sets": failures,
    "sar_rgb_within_5px_by_land": sar_rgb_by_land,
}
with open(os.path.join(HOME, "results", "nominal-9mod-summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
for mod in order:
    print(mod, json.dumps(rows[mod]))
print("total", summary["total_sets"], "within 5 px", summary["total_within_5px_of_all_sets"])
print("no transform", failures)
print(sar_rgb_by_land)
