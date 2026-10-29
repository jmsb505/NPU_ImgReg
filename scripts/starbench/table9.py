import json
import os

import numpy as np

HOME = os.path.expanduser(os.environ["STARBENCH_HOME"])
with open(os.path.join(HOME, "results", "nominal-intra_blue-green-xfeat.json")) as f:
    intra = json.load(f)
for r in intra:
    r["modality"] = "Intra_blue-green"
with open(os.path.join(HOME, "results", "nominal-8mod-xfeat.json")) as f:
    eight = json.load(f)["records"]
recs = intra + eight
order = ["Intra_blue-green", "Intra_blue-red", "Intra_green-red", "OPT-OPT", "SAR_VV-VV", "SAR_VH-VH", "SAR_VH-VV", "SAR_VH-RGB", "SAR_VV-RGB"]
print("modality sets fail within5_all tre_med tre_p90 inliers_med sec_med")
for mod in order:
    rs = [r for r in recs if r["modality"] == mod]
    ok = np.array([r["tre_px"] for r in rs if r["tre_px"] is not None])
    within = int(np.sum(ok <= 5))
    inl = [r["inliers"] for r in rs if r["tre_px"] is not None]
    sec = [r["seconds"] for r in rs if r["seconds"] is not None]
    print(f"{mod} {len(rs)} {len(rs) - ok.size} {within}/{len(rs)} ({100 * within / len(rs):.1f}%) {np.median(ok):.2f} {np.percentile(ok, 90):.2f} {int(np.median(inl))} {np.median(sec):.4f}")
within_all = sum(int(np.sum(np.array([r["tre_px"] for r in recs if r["modality"] == m and r["tre_px"] is not None]) <= 5)) for m in order)
print("all 9 within 5 px:", within_all, "of", len(recs))
print("== SAR-RGB by land (within 5 px / sets)")
for mod in ["SAR_VH-RGB", "SAR_VV-RGB"]:
    rs = [r for r in recs if r["modality"] == mod]
    lands = sorted({r["set"].split("/")[0] for r in rs})
    parts = []
    for land in lands:
        sub = [r for r in rs if r["set"].split("/")[0] == land]
        hit = sum(1 for r in sub if r["tre_px"] is not None and r["tre_px"] <= 5)
        parts.append(f"{land} {hit}/{len(sub)}")
    print(mod, ", ".join(parts))
