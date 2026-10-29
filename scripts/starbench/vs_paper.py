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

groups = [
    ("intra-band (1), paper thr 0.22 px", ["Intra_blue-green", "Intra_blue-red", "Intra_green-red"], 0.22),
    ("monomodal (2), paper thr 2 px", ["OPT-OPT"], 2.0),
    ("multimodal (3), paper thr 11.1 px", ["SAR_VH-RGB", "SAR_VV-RGB"], 11.1),
    ("multi-polarity (4), paper thr 5.83 px", ["SAR_VH-VV"], 5.83),
]
for name, mods, thr in groups:
    rs = [r for r in recs if r["modality"] in mods]
    ok = np.array([r["tre_px"] for r in rs if r["tre_px"] is not None])
    print(f"{name}: sets {len(rs)}, mean {ok.mean():.2f}, median {np.median(ok):.2f}, p90 {np.percentile(ok, 90):.2f}, within thr {int(np.sum(ok <= thr))}/{len(rs)} ({100 * np.sum(ok <= thr) / len(rs):.1f}%)")

sar = [r for r in recs if r["modality"] in ["SAR_VH-RGB", "SAR_VV-RGB"]]
ok_all = np.array([r["tre_px"] for r in sar if r["tre_px"] is not None])
print("multimodal all SAR-RGB mean TRE (successful sets)", round(float(ok_all.mean()), 2))
for land in ["Urbano", "Agri", "Barrenland", "Grassland"]:
    sub = np.array([r["tre_px"] for r in sar if r["set"].split("/")[0] == land and r["tre_px"] is not None])
    print(land, "mean", round(float(sub.mean()), 2), "median", round(float(np.median(sub)), 2), "n_ok", sub.size)
